#pragma once

#include "d3d12-base.h"
#include "d3d12-shader-object-layout.h"
#include "../transient-buffer-heap.h"
#include "../binding-table-cache.h"

#include "core/short_vector.h"

#include <vector>

namespace rhi::d3d12 {

/// A reprsentation of an allocated descriptor set, consisting of an option resource table and
/// an optional sampler table
struct DescriptorSet
{
    GPUDescriptorRange resources;
    D3D12_CPU_DESCRIPTOR_HANDLE* samplerDescriptors = nullptr;
    uint32_t samplerCount = 0;
    uint32_t samplerRootParameter = 0;
};


struct BindingTableEntry;

struct BindingDataBuilder
{
    DeviceImpl* m_device;
    ArenaAllocator* m_allocator;
    BindingCache* m_bindingCache;
    BindingDataImpl* m_bindingData;
    std::vector<InternalRefPtr<RefObject>>* m_retained;
    BindingTableEntry* m_table = nullptr;
    bool m_tableHit = false;
    uint32_t m_rootDescriptorCursor = 0;

    TransientBufferArena* m_constantBufferArena;
    GPUDescriptorArena* m_cbvSrvUavArena;
    GPUDescriptorArena* m_samplerArena;

    Result bindAsRoot(
        RootShaderObject* shaderObject,
        RootShaderObjectLayoutImpl* specializedLayout,
        BindingDataImpl*& outBindingData
    );

    /// Prepare to bind this object as a parameter block.
    ///
    /// This involves allocating and binding any descriptor tables necessary
    /// to to store the state of the object. The function returns a descriptor
    /// set formed from any table(s) allocated. In addition, the `ioOffset`
    /// parameter will be adjusted to be correct for binding values into
    /// the resulting descriptor set.
    ///
    /// Returns:
    ///   SLANG_OK when successful,
    ///   SLANG_E_OUT_OF_MEMORY when descriptor heap is full.
    ///
    Result allocateDescriptorSets(
        ShaderObject* shaderObject,
        BindingOffset& ioOffset,
        ShaderObjectLayoutImpl* specializedLayout,
        DescriptorSet& outDescriptorSet
    );

    void writeRootParameter(uint32_t index, uint64_t address, bool uav);

    Result finishSamplerTable(const DescriptorSet& descriptorSet, ShaderObjectLayoutImpl* layout);

    /// Bind this object as a `ConstantBuffer<X>`
    Result bindAsConstantBuffer(
        ShaderObject* shaderObject,
        const DescriptorSet& descriptorSet,
        const BindingOffset& inOffset,
        uint32_t& rootParamIndex,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind this object as a `ParameterBlock<X>`
    Result bindAsParameterBlock(
        ShaderObject* shaderObject,
        const BindingOffset& offset,
        uint32_t& rootParamIndex,
        ShaderObjectLayoutImpl* specializedLayout
    );

    Result bindAsValue(
        ShaderObject* shaderObject,
        const DescriptorSet& descriptorSet,
        const BindingOffset& offset,
        uint32_t& rootParamIndex,
        ShaderObjectLayoutImpl* specializedLayout
    );

    Result bindOrdinaryDataBufferIfNeeded(
        ShaderObject* shaderObject,
        const DescriptorSet& descriptorSet,
        BindingOffset& ioOffset,
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
        TextureImpl* texture;
        SubresourceRange range;
        ResourceState state;
    };

    struct RootParameter
    {
        enum Type
        {
            CBV,
            UAV,
            SRV,
            DescriptorTable,
        };
        Type type;
        UINT index;
        union
        {
            D3D12_GPU_VIRTUAL_ADDRESS bufferLocation;
            D3D12_GPU_DESCRIPTOR_HANDLE baseDescriptor;
        };
    };

    struct StateSlice
    {
        const std::vector<BufferState>* buffers;
        const std::vector<TextureState>* textures;
        StateSlice* next;
    };
    StateSlice* stateSlices;
    StateSlice* lastStateSlice;

    /// Root parameters.
    RootParameter* rootParameters;
    uint32_t rootParameterCount;
};

struct BindingTableEntry
{
    BindingClosureKey key;
    std::vector<BindingDataImpl::BufferState> buffers;
    std::vector<BindingDataImpl::TextureState> textures;
    GPUDescriptorRange resources;
    GPUDescriptorRange samplers;
    std::vector<BindingDataImpl::RootParameter> rootDescriptors;
};

// Descriptor tables are immutable once referenced by a recorded draw.
struct BindingCache : BindingTableCache<BindingTableEntry>
{
    struct SamplerEntry
    {
        ShaderObjectLayoutImpl* layout;
        size_t hash;
        std::vector<SIZE_T> descriptors;
        GPUDescriptorRange samplers;
    };


    std::vector<SamplerEntry> samplerEntries;

    static size_t hashSamplers(const D3D12_CPU_DESCRIPTOR_HANDLE* descriptors, uint32_t count)
    {
        size_t hash = count;
        for (uint32_t i = 0; i < count; ++i)
            hash ^= descriptors[i].ptr + size_t(0x9e3779b9) + (hash << 6) + (hash >> 2);
        return hash;
    }

    GPUDescriptorRange lookupSamplers(
        ShaderObjectLayoutImpl* layout,
        const D3D12_CPU_DESCRIPTOR_HANDLE* descriptors,
        uint32_t count,
        size_t hash
    ) const
    {
        for (const auto& entry : samplerEntries)
        {
            if (entry.layout != layout || entry.hash != hash || entry.descriptors.size() != count)
                continue;
            bool equal = true;
            for (uint32_t i = 0; i < count; ++i)
                if (entry.descriptors[i] != descriptors[i].ptr)
                {
                    equal = false;
                    break;
                }
            if (equal)
                return entry.samplers;
        }
        return {};
    }

    void storeSamplers(
        ShaderObjectLayoutImpl* layout,
        const D3D12_CPU_DESCRIPTOR_HANDLE* descriptors,
        uint32_t count,
        size_t hash,
        GPUDescriptorRange samplers
    )
    {
        SamplerEntry entry{layout, hash, {}, samplers};
        entry.descriptors.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
            entry.descriptors.push_back(descriptors[i].ptr);
        samplerEntries.push_back(std::move(entry));
    }

    void reset()
    {
        samplerEntries.clear();
        BindingTableCache<BindingTableEntry>::reset();
    }
};

} // namespace rhi::d3d12
