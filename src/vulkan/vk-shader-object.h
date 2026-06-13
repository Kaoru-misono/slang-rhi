#pragma once

#include "vk-base.h"
#include "vk-shader-object-layout.h"
#include "vk-constant-buffer-pool.h"

#include "core/short_vector.h"

#include <vector>

namespace rhi::vk {

struct BindingDataBuilder
{
    DeviceImpl* m_device;
    ArenaAllocator* m_allocator;
    BindingCache* m_bindingCache;
    BindingDataImpl* m_bindingData;
    ConstantBufferPool* m_constantBufferPool;
    DescriptorSetAllocator* m_descriptorSetAllocator;

    // TODO remove
    std::span<const VkPushConstantRange> m_pushConstantRanges;


    /// Bind this object as a root shader object
    Result bindAsRoot(
        RootShaderObject* shaderObject,
        RootShaderObjectLayoutImpl* specializedLayout,
        BindingDataImpl*& outBindingData
    );

    /// Bind this shader object as an entry point
    Result bindAsEntryPoint(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        EntryPointLayout* specializedLayout,
        uint32_t entryPointIndex
    );

    /// Bind the ordinary data buffer if needed.
    Result bindOrdinaryDataBufferIfNeeded(
        ShaderObject* shaderObject,
        BindingOffset& ioOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind this shader object as a "value"
    ///
    /// This is the mode used for binding sub-objects for existential-type
    /// fields, and is also used as part of the implementation of the
    /// parameter-block and constant-buffer cases.
    ///
    Result bindAsValue(
        ShaderObject* shaderObject,
        const BindingOffset& offset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Allocate the descriptor sets needed for binding this object (but not nested parameter
    /// blocks)
    Result allocateDescriptorSets(
        ShaderObject* shaderObject,
        const BindingOffset& offset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind this object as a `ParameterBlock<X>`.
    Result bindAsParameterBlock(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind this object as a `ConstantBuffer<X>`.
    Result bindAsConstantBuffer(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );
};

struct BindingDataImpl : BindingData
{
public:
    struct BufferState
    {
        BufferImpl* buffer;
        ResourceState state;
    };
    struct TextureState
    {
        TextureViewImpl* textureView;
        ResourceState state;
    };
    /// Entry point data for copying to shader binding table (ray tracing)
    struct EntryPointData
    {
        // Host memory pointer to entry point uniform data
        void* data;
        // Size of the data in bytess
        size_t size;
    };

    /// Required buffer states.
    BufferState* bufferStates;
    uint32_t bufferStateCapacity;
    uint32_t bufferStateCount;
    /// Required texture states.
    TextureState* textureStates;
    uint32_t textureStateCapacity;
    uint32_t textureStateCount;

    /// Pipeline layout.
    VkPipelineLayout pipelineLayout;

    /// Descriptor sets.
    VkDescriptorSet* descriptorSets;
    uint32_t descriptorSetCount;

    /// Push constants.
    VkPushConstantRange* pushConstantRanges;
    void** pushConstantData;
    uint32_t pushConstantCount;

    /// Entry point data (for ray tracing SBT).
    EntryPointData* entryPointData;
    uint32_t entryPointCount;
};

/// Cache for parameter block descriptor sets to avoid redundant allocation and writing.
/// Vulkan descriptor sets are live GPU objects — reusing a set that has already been bound
/// to a command buffer and then writing new content is undefined behavior. Therefore we
/// cache at the `bindAsParameterBlock` level: if the entire sub-object tree is unchanged
/// (same `getSubTreeVersion()`), we skip allocation, descriptor writing, AND constant
/// buffer creation, and reuse the exact same descriptor sets from the previous draw.
struct BindingCache
{
    static constexpr uint32_t kMaxEntries = 16;
    static constexpr uint32_t kMaxDescriptorSetsPerEntry = 4;

    struct Entry
    {
        uint32_t objectUid;
        uint32_t subTreeVersion;
        ShaderObjectLayoutImpl* layout;
        VkDescriptorSet descriptorSets[kMaxDescriptorSetsPerEntry];
        uint32_t descriptorSetCount;
    };

    Entry entries[kMaxEntries] = {};
    uint32_t entryCount = 0;

    Entry* lookup(uint32_t objectUid, uint32_t subTreeVersion, ShaderObjectLayoutImpl* layout)
    {
        for (uint32_t i = 0; i < entryCount; ++i)
        {
            if (entries[i].objectUid == objectUid && entries[i].subTreeVersion == subTreeVersion &&
                entries[i].layout == layout)
                return &entries[i];
        }
        return nullptr;
    }

    void store(
        uint32_t objectUid,
        uint32_t subTreeVersion,
        ShaderObjectLayoutImpl* layout,
        VkDescriptorSet* sets,
        uint32_t count
    )
    {
        if (entryCount < kMaxEntries && count <= kMaxDescriptorSetsPerEntry)
        {
            auto& entry = entries[entryCount++];
            entry.objectUid = objectUid;
            entry.subTreeVersion = subTreeVersion;
            entry.layout = layout;
            entry.descriptorSetCount = count;
            for (uint32_t i = 0; i < count; ++i)
                entry.descriptorSets[i] = sets[i];
        }
    }

    void reset() { entryCount = 0; }
};

} // namespace rhi::vk
