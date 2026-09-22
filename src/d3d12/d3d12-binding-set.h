#pragma once

#include "d3d12-base.h"
#include "../binding-set.h"
#include "../pipeline-layout.h"

#include <vector>

namespace rhi::d3d12 {

// A set layout and a pipeline layout need no deferred destruction here: what an in-flight
// recording reads from them is the root signature, which the pipeline holds a reference to,
// and the descriptors a set copied out of them at creation time.
class BindingSetLayoutImpl : public BindingSetLayout
{
public:
    BindingSetLayoutImpl(Device* device, const BindingSetLayoutDesc& desc);

    /// Sorts the entries into the two descriptor tables a set is bound through.
    virtual Result initNative() override;

    D3D12_DESCRIPTOR_RANGE_TYPE getRangeType(uint32_t entryIndex) const { return m_rangeTypes[entryIndex]; }
    /// Index of the first descriptor of `entryIndex` within its own table.
    uint32_t getTableOffset(uint32_t entryIndex) const { return m_tableOffsets[entryIndex]; }

    /// Descriptors in the CBV/SRV/UAV table and in the sampler table.
    uint32_t m_resourceCount = 0;
    uint32_t m_samplerCount = 0;
    std::vector<D3D12_DESCRIPTOR_RANGE_TYPE> m_rangeTypes;
    std::vector<uint32_t> m_tableOffsets;
};

class BindingSetImpl : public BindingSet
{
public:
    BindingSetImpl(Device* device, BindingSetLayoutImpl* layout);
    ~BindingSetImpl();
    virtual void deleteThis() override;

    /// Allocates the set's CPU descriptor ranges and writes one descriptor per binding.
    virtual Result initNative() override;

    BindingSetLayoutImpl* getLayoutImpl() const { return static_cast<BindingSetLayoutImpl*>(m_layout.get()); }

    CPUDescriptorRangeAllocation m_resources;
    CPUDescriptorRangeAllocation m_samplers;
};

/// Root parameter index of a table or root descriptor the layout does not use.
static constexpr uint32_t kInvalidRootParameterIndex = 0xffffffff;

class PipelineLayoutImpl : public PipelineLayout
{
public:
    /// Root parameter indices of one binding set's two tables.
    struct SetInfo
    {
        uint32_t resourceRootParameterIndex = kInvalidRootParameterIndex;
        uint32_t samplerRootParameterIndex = kInvalidRootParameterIndex;
    };

    PipelineLayoutImpl(Device* device, const PipelineLayoutDesc& desc);

    /// Builds the root signature: an optional root CBV for the execution constants, then one resource
    /// table and one sampler table per binding set.
    virtual Result initNative() override;

    ComPtr<ID3D12RootSignature> m_rootSignature;
    std::vector<SetInfo> m_sets;
    uint32_t m_constantsRootParameterIndex = kInvalidRootParameterIndex;
};

} // namespace rhi::d3d12
