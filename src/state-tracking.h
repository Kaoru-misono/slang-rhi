#pragma once

#include "rhi-shared.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

namespace rhi {

/// Open-addressing map from a resource pointer to its tracked state. Recording looks up
/// every bound resource on every state change, and a node-based map allocates per
/// resource and probes slowly, which dominated recording in checked builds.
template<typename Key, typename Value>
class ResourceStateMap
{
public:
    struct Entry
    {
        Key* key;
        Value value;
    };

    Value* find(Key* key)
    {
        if (m_index.empty())
            return nullptr;
        size_t const mask = m_index.size() - 1;
        for (size_t slot = slotOf(key, mask);; slot = (slot + 1) & mask)
        {
            uint32_t const entry = m_index[slot];
            if (entry == kEmpty)
                return nullptr;
            if (m_entries[entry].key == key)
                return &m_entries[entry].value;
        }
    }

    Value& insertOrAssign(Key* key, Value value)
    {
        if (Value* existing = find(key))
        {
            *existing = std::move(value);
            return *existing;
        }
        if ((m_entries.size() + 1) * 2 > m_index.size())
            rebuild(std::max<size_t>(64, m_index.size() * 2));
        m_entries.emplace_back(Entry{key, std::move(value)});
        place(key, uint32_t(m_entries.size() - 1));
        return m_entries.back().value;
    }

    void erase(Key* key)
    {
        auto it = std::find_if(m_entries.begin(), m_entries.end(), [key](const Entry& entry) { return entry.key == key; });
        if (it == m_entries.end())
            return;
        *it = std::move(m_entries.back());
        m_entries.pop_back();
        rebuild(m_index.size());
    }

    void clear()
    {
        m_entries.clear();
        std::fill(m_index.begin(), m_index.end(), kEmpty);
    }

    // Iteration must not insert: that can move the entries.
    Entry* begin() { return m_entries.data(); }
    Entry* end() { return m_entries.data() + m_entries.size(); }

private:
    static constexpr uint32_t kEmpty = ~0u;

    static size_t slotOf(Key* key, size_t mask)
    {
        return size_t((uint64_t(uintptr_t(key)) * 0x9E3779B97F4A7C15ull) >> 32) & mask;
    }

    void place(Key* key, uint32_t entry)
    {
        size_t const mask = m_index.size() - 1;
        size_t slot = slotOf(key, mask);
        while (m_index[slot] != kEmpty)
            slot = (slot + 1) & mask;
        m_index[slot] = entry;
    }

    void rebuild(size_t slotCount)
    {
        m_index.assign(slotCount, kEmpty);
        for (uint32_t entry = 0; entry < uint32_t(m_entries.size()); ++entry)
            place(m_entries[entry].key, entry);
    }

    std::vector<Entry> m_entries;
    std::vector<uint32_t> m_index;
};

static_assert(uint32_t(ResourceState::MicromapWrite) < 64);

class BufferStateSet
{
public:
    explicit BufferStateSet(ResourceState state) { add(state); }

    bool contains(ResourceState state) const { return (m_bits & stateBit(state)) != 0; }
    bool isExactly(ResourceState state) const { return m_bits == stateBit(state); }
    bool hasSingleState() const { return m_bits != 0 && (m_bits & (m_bits - 1)) == 0; }
    void add(ResourceState state) { m_bits |= stateBit(state); }

    template<typename Function>
    void forEach(Function&& function) const
    {
        uint64_t remainingStates = m_bits;
        while (remainingStates != 0)
        {
            uint32_t const stateIndex = uint32_t(std::countr_zero(remainingStates));
            function(ResourceState(stateIndex));
            remainingStates &= remainingStates - 1;
        }
    }

    bool operator==(const BufferStateSet& other) const = default;

private:
    static uint64_t stateBit(ResourceState state)
    {
        uint32_t const stateIndex = uint32_t(state);
        if (stateIndex >= 64)
        {
            SLANG_RHI_ASSERT_FAILURE("ResourceState does not fit in BufferStateSet");
            return 0;
        }
        return uint64_t(1) << stateIndex;
    }

    uint64_t m_bits = 0;
};

enum class BufferReadStatePolicy
{
    None,
    Graphics,
    Compute,
};

inline BufferReadStatePolicy getBufferReadStatePolicy(QueueType queueType)
{
    switch (queueType)
    {
    case QueueType::Graphics:
        return BufferReadStatePolicy::Graphics;
    case QueueType::Compute:
        return BufferReadStatePolicy::Compute;
    default:
        return BufferReadStatePolicy::None;
    }
}

inline bool isMergeableBufferReadState(ResourceState state, BufferReadStatePolicy policy)
{
    switch (state)
    {
    case ResourceState::VertexBuffer:
    case ResourceState::IndexBuffer:
        return policy == BufferReadStatePolicy::Graphics;
    case ResourceState::ConstantBuffer:
    case ResourceState::ShaderResource:
    case ResourceState::IndirectArgument:
        return policy == BufferReadStatePolicy::Graphics || policy == BufferReadStatePolicy::Compute;
    default:
        return false;
    }
}

inline bool containsOnlyMergeableBufferReadStates(const BufferStateSet& states, BufferReadStatePolicy policy)
{
    bool allMergeable = true;
    states.forEach(
        [&](ResourceState state)
        {
            allMergeable = allMergeable && isMergeableBufferReadState(state, policy);
        }
    );
    return allMergeable;
}

inline bool isValidBufferStateSet(const BufferStateSet& states)
{
    return states.hasSingleState() ||
           containsOnlyMergeableBufferReadStates(states, BufferReadStatePolicy::Graphics);
}

struct BufferState
{
    BufferStateSet states{ResourceState::Undefined};
};

struct TextureState
{
    ResourceState state = ResourceState::Undefined;
    std::vector<ResourceState> subresourceStates;
};

struct BufferBarrier
{
    Buffer* buffer;
    BufferStateSet stateBefore;
    BufferStateSet stateAfter;
};

struct TextureBarrier
{
    Texture* texture;
    bool entireTexture;
    uint32_t mip;
    uint32_t layer;
    ResourceState stateBefore;
    ResourceState stateAfter;
};

class StateTracking
{
public:
    void setBufferState(Buffer* buffer, ResourceState state)
    {
        if (buffer->m_desc.memoryType != MemoryType::DeviceLocal)
            return;

        BufferState* bufferState = getBufferState(buffer);
        transitionBufferState(
            buffer,
            *bufferState,
            BufferStateSet(state),
            state == ResourceState::UnorderedAccess
        );
    }

    // Returns whether the effective state changed, excluding same-state UAV ordering barriers.
    bool requireBufferState(Buffer* buffer, ResourceState state, BufferReadStatePolicy policy)
    {
        if (buffer->m_desc.memoryType != MemoryType::DeviceLocal)
            return false;

        BufferState* bufferState = getBufferState(buffer);
        BufferStateSet requiredStates(state);
        if (isMergeableBufferReadState(state, policy) &&
            containsOnlyMergeableBufferReadStates(bufferState->states, policy))
        {
            requiredStates = bufferState->states;
            requiredStates.add(state);
        }

        const bool changed = requiredStates != bufferState->states;
        transitionBufferState(
            buffer,
            *bufferState,
            requiredStates,
            state == ResourceState::UnorderedAccess
        );
        return changed;
    }

    // Returns whether any requested subresource changed state, excluding UAV ordering barriers.
    bool setTextureState(Texture* texture, SubresourceRange subresourceRange, ResourceState state)
    {
        // Cannot change state of upload/readback textures.
        if (texture->m_desc.memoryType != MemoryType::DeviceLocal)
        {
            return false;
        }

        subresourceRange = texture->resolveSubresourceRange(subresourceRange);
        bool isEntireTexture = texture->isEntireTexture(subresourceRange);
        TextureState* textureState = getTextureState(texture);
        bool changed = false;

        if (isEntireTexture && textureState->subresourceStates.empty())
        {
            changed = state != textureState->state;
            // Transition entire texture.
            if (state != textureState->state || state == ResourceState::UnorderedAccess)
            {
                // Try to merge with an existing entire-texture barrier for the same texture.
                bool merged = false;
                for (auto& existing : m_textureBarriers)
                {
                    if (existing.texture == texture && existing.entireTexture)
                    {
                        existing.stateAfter = state;
                        merged = true;
                        break;
                    }
                }
                if (!merged)
                {
                    m_textureBarriers.push_back({texture, true, 0, 0, textureState->state, state});
                }
                textureState->state = state;
            }
        }
        else
        {
            // Transition subresources.
            if (textureState->subresourceStates.empty())
            {
                textureState->subresourceStates.resize(texture->m_desc.getSubresourceCount(), textureState->state);
                textureState->state = ResourceState::Undefined;
            }
            const uint32_t layerEnd = subresourceRange.layer + subresourceRange.layerCount;
            for (uint32_t layer = subresourceRange.layer; layer < layerEnd; layer++)
            {
                for (uint32_t mip = subresourceRange.mip; mip < subresourceRange.mip + subresourceRange.mipCount; mip++)
                {
                    uint32_t subresourceIndex = layer * texture->m_desc.mipCount + mip;
                    changed |= state != textureState->subresourceStates[subresourceIndex];
                    if (state != textureState->subresourceStates[subresourceIndex] ||
                        state == ResourceState::UnorderedAccess)
                    {
                        m_textureBarriers.push_back({
                            texture,
                            false,
                            mip,
                            layer,
                            textureState->subresourceStates[subresourceIndex],
                            state,
                        });
                        textureState->subresourceStates[subresourceIndex] = state;
                    }
                }
            }

            // Check if all subresource states are equal and we can represent them as a single texture state.
            ResourceState commonState = textureState->subresourceStates[0];
            bool allEqual = true;
            for (ResourceState subresourceState : textureState->subresourceStates)
            {
                if (subresourceState != commonState)
                {
                    allEqual = false;
                    break;
                }
            }
            if (allEqual)
            {
                textureState->state = commonState;
                textureState->subresourceStates.clear();
            }
        }
        return changed;
    }

    void requireDefaultStates()
    {
        for (auto& bufferState : m_bufferStates)
        {
            if (!bufferState.value.states.isExactly(bufferState.key->m_desc.defaultState))
            {
                setBufferState(bufferState.key, bufferState.key->m_desc.defaultState);
            }
        }
        for (auto& textureState : m_textureStates)
        {
            if (textureState.value.state != textureState.key->m_desc.defaultState)
            {
                setTextureState(textureState.key, kEntireTexture, textureState.key->m_desc.defaultState);
            }
        }
    }

    const std::vector<BufferBarrier>& getBufferBarriers() const { return m_bufferBarriers; }

    const std::vector<TextureBarrier>& getTextureBarriers() const { return m_textureBarriers; }

    void forgetBufferState(Buffer* buffer)
    {
        assertNoPendingBufferBarrier(buffer);
        m_bufferStates.erase(buffer);
    }

    void assumeBufferState(Buffer* buffer, ResourceState state)
    {
        if (buffer->m_desc.memoryType != MemoryType::DeviceLocal)
            return;

        assertNoPendingBufferBarrier(buffer);
        m_bufferStates.insertOrAssign(buffer, BufferState{BufferStateSet(state)});
    }

    void clearBarriers()
    {
        m_bufferBarriers.clear();
        m_textureBarriers.clear();
    }

    void clear()
    {
        m_bufferStates.clear();
        m_textureStates.clear();
        clearBarriers();
    }

private:
    ResourceStateMap<Buffer, BufferState> m_bufferStates;
    ResourceStateMap<Texture, TextureState> m_textureStates;
    std::vector<BufferBarrier> m_bufferBarriers;
    std::vector<TextureBarrier> m_textureBarriers;

    void assertNoPendingBufferBarrier(Buffer* buffer) const
    {
        for (const auto& barrier : m_bufferBarriers)
            SLANG_RHI_ASSERT(barrier.buffer != buffer);
    }

    void transitionBufferState(
        Buffer* buffer,
        BufferState& currentState,
        BufferStateSet requiredStates,
        bool forceBarrier
    )
    {
        if (requiredStates == currentState.states && !forceBarrier)
            return;

        bool merged = false;
        for (auto& existing : m_bufferBarriers)
        {
            if (existing.buffer == buffer)
            {
                existing.stateAfter = requiredStates;
                merged = true;
                break;
            }
        }
        if (!merged)
        {
            m_bufferBarriers.emplace_back(BufferBarrier{buffer, currentState.states, requiredStates});
        }
        currentState.states = requiredStates;
    }

    BufferState* getBufferState(Buffer* buffer)
    {
        if (BufferState* state = m_bufferStates.find(buffer))
            return state;
        return &m_bufferStates.insertOrAssign(buffer, BufferState{BufferStateSet(buffer->m_desc.defaultState)});
    }

    TextureState* getTextureState(Texture* texture)
    {
        if (TextureState* state = m_textureStates.find(texture))
            return state;
        return &m_textureStates.insertOrAssign(texture, TextureState{texture->m_desc.defaultState, {}});
    }
};

} // namespace rhi
