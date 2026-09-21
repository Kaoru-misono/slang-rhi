#pragma once

#include "rhi-shared.h"

#include <bit>
#include <vector>
#include <unordered_map>

namespace rhi {

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

    void requireBufferState(Buffer* buffer, ResourceState state, BufferReadStatePolicy policy)
    {
        if (buffer->m_desc.memoryType != MemoryType::DeviceLocal)
            return;

        BufferState* bufferState = getBufferState(buffer);
        BufferStateSet requiredStates(state);
        if (isMergeableBufferReadState(state, policy) &&
            containsOnlyMergeableBufferReadStates(bufferState->states, policy))
        {
            requiredStates = bufferState->states;
            requiredStates.add(state);
        }

        transitionBufferState(
            buffer,
            *bufferState,
            requiredStates,
            state == ResourceState::UnorderedAccess
        );
    }

    void setTextureState(Texture* texture, SubresourceRange subresourceRange, ResourceState state)
    {
        // Cannot change state of upload/readback buffers.
        if (texture->m_desc.memoryType != MemoryType::DeviceLocal)
        {
            return;
        }

        subresourceRange = texture->resolveSubresourceRange(subresourceRange);
        bool isEntireTexture = texture->isEntireTexture(subresourceRange);
        TextureState* textureState = getTextureState(texture);

        if (isEntireTexture && textureState->subresourceStates.empty())
        {
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
            uint32_t layerCount = texture->m_desc.getLayerCount();
            for (uint32_t layer = subresourceRange.layer; layer < layerCount; layer++)
            {
                for (uint32_t mip = subresourceRange.mip; mip < subresourceRange.mip + subresourceRange.mipCount; mip++)
                {
                    uint32_t subresourceIndex = layer * texture->m_desc.mipCount + mip;
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
    }

    void requireDefaultStates()
    {
        for (auto& bufferState : m_bufferStates)
        {
            if (!bufferState.second.states.isExactly(bufferState.first->m_desc.defaultState))
            {
                setBufferState(bufferState.first, bufferState.first->m_desc.defaultState);
            }
        }
        for (auto& textureState : m_textureStates)
        {
            if (textureState.second.state != textureState.first->m_desc.defaultState)
            {
                setTextureState(textureState.first, kEntireTexture, textureState.first->m_desc.defaultState);
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
        m_bufferStates.insert_or_assign(buffer, BufferState{BufferStateSet(state)});
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
    std::unordered_map<Buffer*, BufferState> m_bufferStates;
    std::unordered_map<Texture*, TextureState> m_textureStates;
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
        auto it = m_bufferStates.find(buffer);
        if (it != m_bufferStates.end())
            return &it->second;
        m_bufferStates[buffer] = {BufferStateSet(buffer->m_desc.defaultState)};
        return &m_bufferStates[buffer];
    }

    TextureState* getTextureState(Texture* texture)
    {
        auto it = m_textureStates.find(texture);
        if (it != m_textureStates.end())
            return &it->second;
        m_textureStates[texture] = {texture->m_desc.defaultState};
        return &m_textureStates[texture];
    }
};

} // namespace rhi
