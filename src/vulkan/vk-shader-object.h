#pragma once

#include "vk-base.h"
#include "vk-shader-object-layout.h"
#include "../transient-buffer-heap.h"
#include "../binding-table-cache.h"

#include "core/short_vector.h"

#include <vector>

namespace rhi::vk {

struct BindingTableEntry;

struct BindingDataBuilder
{
    DeviceImpl* m_device;
    ArenaAllocator* m_allocator;
    BindingCache* m_bindingCache;
    BindingDataImpl* m_bindingData;
    std::vector<RefPtr<RefObject>>* m_retained;
    BindingTableEntry* m_table = nullptr;
    bool m_tableHit = false;

    TransientBufferArena* m_constantBufferArena;
    DescriptorSetAllocator* m_descriptorSetAllocator;

    void recordDescriptorSet(uint32_t setIndex, VkDescriptorSet set);

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

    /// Bind this object as a `PushConstantBuffer<X>`.
    Result bindAsPushConstantBuffer(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind this object as a push-constant block that holds no descriptors.
    Result bindAsPushConstantBlock(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Copy this object's ordinary data into the next push-constant range.
    Result writePushConstantData(
        ShaderObject* shaderObject,
        BindingOffset& ioOffset,
        ShaderObjectLayoutImpl* specializedLayout
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
    /// The exact subresource range whose state is required by this binding.
    struct TextureState
    {
        TextureImpl* texture;
        SubresourceRange subresourceRange;
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

    /// The resource states one table or one push-constant block needs. A table owns its lists; a block's lists live
    /// in the command allocator, because nothing about a block is cached.
    struct StateSlice
    {
        const BufferState* buffers;
        uint32_t bufferCount;
        const TextureState* textures;
        uint32_t textureCount;
        StateSlice* next;
    };
    StateSlice* stateSlices;
    StateSlice* lastStateSlice;

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

struct BindingTableEntry
{
    BindingClosureKey key;
    std::vector<BindingDataImpl::BufferState> buffers;
    std::vector<BindingDataImpl::TextureState> textures;
    struct Set { uint32_t offset; VkDescriptorSet set; };
    std::vector<Set> sets;
};

struct BindingCache : BindingTableCache<BindingTableEntry>
{};

} // namespace rhi::vk
