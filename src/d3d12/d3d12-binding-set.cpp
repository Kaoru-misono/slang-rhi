#include "d3d12-binding-set.h"
#include "d3d12-acceleration-structure.h"
#include "d3d12-buffer.h"
#include "d3d12-device.h"
#include "d3d12-sampler.h"
#include "d3d12-shader-object-layout.h"
#include "d3d12-texture.h"

#include "../shader.h"

namespace rhi::d3d12 {

static D3D12_DESCRIPTOR_RANGE_TYPE getBindingKindRangeType(BindingKind kind)
{
    switch (kind)
    {
    case BindingKind::SamplerState:
    case BindingKind::SamplerComparisonState:
        return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    case BindingKind::Texture1D:
    case BindingKind::Texture2D:
    case BindingKind::Texture2DArray:
    case BindingKind::Texture3D:
    case BindingKind::TextureCube:
    case BindingKind::TextureCubeArray:
    case BindingKind::Buffer:
    case BindingKind::StructuredBuffer:
    case BindingKind::ByteAddressBuffer:
    case BindingKind::RaytracingAccelerationStructure:
        return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case BindingKind::RWTexture1D:
    case BindingKind::RWTexture2D:
    case BindingKind::RWTexture2DArray:
    case BindingKind::RWTexture3D:
    case BindingKind::RWBuffer:
    case BindingKind::RWStructuredBuffer:
    case BindingKind::RWByteAddressBuffer:
        return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    }
    SLANG_RHI_UNREACHABLE("Invalid binding kind");
    return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
}

BindingSetLayoutImpl::BindingSetLayoutImpl(Device* device, const BindingSetLayoutDesc& desc)
    : BindingSetLayout(device, desc)
{
}

Result BindingSetLayoutImpl::initNative()
{
    m_rangeTypes.resize(m_desc.entryCount);
    m_tableOffsets.resize(m_desc.entryCount);
    for (uint32_t i = 0; i < m_desc.entryCount; ++i)
    {
        m_rangeTypes[i] = getBindingKindRangeType(m_desc.entries[i].kind);
        uint32_t& count = m_rangeTypes[i] == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER ? m_samplerCount : m_resourceCount;
        m_tableOffsets[i] = count;
        count += m_desc.entries[i].count;
    }
    return SLANG_OK;
}

BindingSetImpl::BindingSetImpl(Device* device, BindingSetLayoutImpl* layout)
    : BindingSet(device, layout)
{
}

BindingSetImpl::~BindingSetImpl()
{
    DeviceImpl* device = getDevice<DeviceImpl>();
    if (m_resources.isValid())
        device->m_cpuCbvSrvUavHeap->free(m_resources);
    if (m_samplers.isValid())
        device->m_cpuSamplerHeap->free(m_samplers);
}

void BindingSetImpl::deleteThis()
{
    // A set bound by an in-flight command buffer must outlive the submission.
    getDevice<DeviceImpl>()->deferDelete(this);
}

Result BindingSetImpl::initNative()
{
    DeviceImpl* device = getDevice<DeviceImpl>();
    BindingSetLayoutImpl* layout = getLayoutImpl();
    if (layout->m_resourceCount)
    {
        m_resources = device->m_cpuCbvSrvUavHeap->allocate(layout->m_resourceCount);
        if (!m_resources.isValid())
        {
            device->printError(
                "Binding set: CPU descriptor allocation failed (%u resource descriptors).",
                layout->m_resourceCount
            );
            return SLANG_E_OUT_OF_MEMORY;
        }
    }
    if (layout->m_samplerCount)
    {
        m_samplers = device->m_cpuSamplerHeap->allocate(layout->m_samplerCount);
        if (!m_samplers.isValid())
        {
            device->printError(
                "Binding set: CPU descriptor allocation failed (%u sampler descriptors).",
                layout->m_samplerCount
            );
            return SLANG_E_OUT_OF_MEMORY;
        }
    }
    for (uint32_t slot = 0; slot < layout->m_desc.entryCount; ++slot)
    {
        const auto& entry = layout->m_desc.entries[slot];
        for (uint32_t arrayIndex = 0; arrayIndex < entry.count; ++arrayIndex)
        {
            const BindingSetBinding& binding = m_bindings[layout->getBindingOffset(slot) + arrayIndex];
            uint32_t tableIndex = layout->getTableOffset(slot) + arrayIndex;
            D3D12_CPU_DESCRIPTOR_HANDLE source = {};
            switch (entry.kind)
            {
            case BindingKind::SamplerState:
            case BindingKind::SamplerComparisonState:
                source = checked_cast<SamplerImpl*>(binding.resource.get())->m_descriptor.cpuHandle;
                break;
            case BindingKind::Texture1D:
            case BindingKind::Texture2D:
            case BindingKind::Texture2DArray:
            case BindingKind::Texture3D:
            case BindingKind::TextureCube:
            case BindingKind::TextureCubeArray:
                source = checked_cast<TextureViewImpl*>(binding.resource.get())->getSRV();
                break;
            case BindingKind::RWTexture1D:
            case BindingKind::RWTexture2D:
            case BindingKind::RWTexture2DArray:
            case BindingKind::RWTexture3D:
                source = checked_cast<TextureViewImpl*>(binding.resource.get())->getUAV();
                break;
            case BindingKind::Buffer:
            case BindingKind::RWBuffer:
            {
                auto* buffer = checked_cast<BufferImpl*>(binding.resource.get());
                source = entry.kind == BindingKind::Buffer
                             ? buffer->getSRV(buffer->m_desc.format, 0, binding.bufferRange)
                             : buffer->getUAV(buffer->m_desc.format, 0, binding.bufferRange);
                break;
            }
            case BindingKind::StructuredBuffer:
            case BindingKind::RWStructuredBuffer:
            {
                auto* buffer = checked_cast<BufferImpl*>(binding.resource.get());
                uint32_t stride = layout->m_reflection[slot].structuredBufferStride;
                if (!stride)
                    stride = buffer->m_desc.elementSize;
                if (!stride)
                {
                    device->printError(
                        "Binding set '%s', slot %u: structured buffer stride is unknown; "
                        "create the pipeline layout that uses this set layout first.",
                        m_label ? m_label : "<unnamed>",
                        slot
                    );
                    return SLANG_E_INVALID_ARG;
                }
                source = entry.kind == BindingKind::StructuredBuffer
                             ? buffer->getSRV(Format::Undefined, stride, binding.bufferRange)
                             : buffer->getUAV(Format::Undefined, stride, binding.bufferRange);
                break;
            }
            case BindingKind::ByteAddressBuffer:
            case BindingKind::RWByteAddressBuffer:
            {
                auto* buffer = checked_cast<BufferImpl*>(binding.resource.get());
                source = entry.kind == BindingKind::ByteAddressBuffer
                             ? buffer->getSRV(Format::Undefined, 0, binding.bufferRange)
                             : buffer->getUAV(Format::Undefined, 0, binding.bufferRange);
                break;
            }
            case BindingKind::RaytracingAccelerationStructure:
                source = checked_cast<AccelerationStructureImpl*>(binding.resource.get())
                             ->m_descriptor.cpuHandle;
                break;
            }
            bool isSampler = isSamplerBindingKind(entry.kind);
            device->m_device->CopyDescriptorsSimple(
                1,
                (isSampler ? m_samplers : m_resources).getCpuHandle(tableIndex),
                source,
                isSampler ? D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER : D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
            );
        }
    }
    return SLANG_OK;
}

PipelineLayoutImpl::PipelineLayoutImpl(Device* device, const PipelineLayoutDesc& desc)
    : PipelineLayout(device, desc)
{
}

/// True when the reflected ranges cover exactly one descriptor per expected type, in order. The
/// expectation is the order a binding set writes its CPU descriptors in, so a match also proves the
/// table offsets the set relies on are the ones the shader reads.
static bool descriptorRangesMatch(
    const std::vector<D3D12_DESCRIPTOR_RANGE1>& ranges,
    const std::vector<D3D12_DESCRIPTOR_RANGE_TYPE>& expectedTypes
)
{
    size_t cursor = 0;
    for (const auto& range : ranges)
    {
        if (range.NumDescriptors > expectedTypes.size() - cursor)
            return false;
        for (uint32_t i = 0; i < range.NumDescriptors; ++i)
        {
            if (expectedTypes[cursor++] != range.RangeType)
                return false;
        }
    }
    return cursor == expectedTypes.size();
}

Result PipelineLayoutImpl::initNative()
{
    DeviceImpl* device = getDevice<DeviceImpl>();
    slang::ProgramLayout* programLayout = m_program->linkedProgram->getLayout();
    const char* label = m_desc.label ? m_desc.label : "<unnamed>";

    RootShaderObjectLayoutImpl::RootSignatureDescBuilder builder(device);
    uint32_t rootSetIndex = builder.addDescriptorSet();
    builder.addAsValue(programLayout->getGlobalParamsVarLayout(), rootSetIndex);
    for (uint32_t i = 0; i < programLayout->getEntryPointCount(); ++i)
        builder.addAsValue(programLayout->getEntryPointByIndex(i)->getVarLayout(), rootSetIndex);

    if (!builder.m_rootParameters.empty())
    {
        device->printError(
            "Pipeline layout '%s': root descriptor parameters are not supported by a pipeline layout",
            label
        );
        return SLANG_E_INVALID_ARG;
    }
    if (builder.m_descriptorSets.size() != m_setLayouts.size() + 1)
    {
        device->printError(
            "Pipeline layout '%s': the program's parameter blocks do not match the layout's binding sets",
            label
        );
        return SLANG_E_INVALID_ARG;
    }
    const auto& rootSet = builder.m_descriptorSets[rootSetIndex];
    const auto& rootRanges = rootSet.m_resourceRanges;
    bool hasGlobalResources = !rootSet.m_samplerRanges.empty();
    for (const auto& range : rootRanges)
        hasGlobalResources |= range.RangeType != D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    if (hasGlobalResources)
    {
        device->printError(
            "Pipeline layout '%s': global-scope resources are not supported; resources must live in a ParameterBlock",
            label
        );
        return SLANG_E_INVALID_ARG;
    }

    // Both Inline and Buffered execution constants become one root CBV on D3D12.
    if (m_constantsProfile == ConstantsProfile::None)
    {
        if (!rootRanges.empty())
        {
            device->printError(
                "Pipeline layout '%s': the program declares a constant buffer the layout does not",
                label
            );
            return SLANG_E_INVALID_ARG;
        }
    }
    else if (rootRanges.size() != 1 || rootRanges[0].NumDescriptors != 1)
    {
        device->printError(
            "Pipeline layout '%s': the program's execution constants do not map to a single constant buffer",
            label
        );
        return SLANG_E_INVALID_ARG;
    }

    for (uint32_t i = 0; i < m_setLayouts.size(); ++i)
    {
        auto* setLayout = static_cast<BindingSetLayoutImpl*>(m_setLayouts[i].get());
        const auto& descriptorSet = builder.m_descriptorSets[i + 1];
        std::vector<D3D12_DESCRIPTOR_RANGE_TYPE> resourceTypes;
        std::vector<D3D12_DESCRIPTOR_RANGE_TYPE> samplerTypes;
        for (uint32_t e = 0; e < setLayout->m_desc.entryCount; ++e)
        {
            auto type = setLayout->getRangeType(e);
            auto& types = type == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER ? samplerTypes : resourceTypes;
            types.insert(types.end(), setLayout->m_desc.entries[e].count, type);
        }
        if (!descriptorRangesMatch(descriptorSet.m_resourceRanges, resourceTypes) ||
            !descriptorRangesMatch(descriptorSet.m_samplerRanges, samplerTypes))
        {
            device->printError(
                "Pipeline layout '%s': binding set %u does not match the reflected parameter block",
                label,
                i
            );
            return SLANG_E_INVALID_ARG;
        }
    }

    m_sets.resize(m_setLayouts.size());
    std::vector<D3D12_ROOT_PARAMETER1> rootParameters;
    uint32_t rootSignatureCost = 0;
    if (m_constantsProfile != ConstantsProfile::None)
    {
        D3D12_ROOT_PARAMETER1 parameter = {};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameter.Descriptor.ShaderRegister = rootRanges[0].BaseShaderRegister;
        parameter.Descriptor.RegisterSpace = rootRanges[0].RegisterSpace;
        // Consecutive binds write further constants into the same transient upload page after this descriptor is set.
        parameter.Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
        m_constantsRootParameterIndex = uint32_t(rootParameters.size());
        rootParameters.push_back(parameter);
        rootSignatureCost += 2;
    }
    for (uint32_t i = 0; i < m_setLayouts.size(); ++i)
    {
        auto* setLayout = static_cast<BindingSetLayoutImpl*>(m_setLayouts[i].get());
        const auto& descriptorSet = builder.m_descriptorSets[i + 1];
        if (setLayout->m_resourceCount)
        {
            D3D12_ROOT_PARAMETER1 parameter = {};
            parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            parameter.DescriptorTable.NumDescriptorRanges = UINT(descriptorSet.m_resourceRanges.size());
            parameter.DescriptorTable.pDescriptorRanges = descriptorSet.m_resourceRanges.data();
            m_sets[i].resourceRootParameterIndex = uint32_t(rootParameters.size());
            rootParameters.push_back(parameter);
            rootSignatureCost += 1;
        }
        if (setLayout->m_samplerCount)
        {
            D3D12_ROOT_PARAMETER1 parameter = {};
            parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            parameter.DescriptorTable.NumDescriptorRanges = UINT(descriptorSet.m_samplerRanges.size());
            parameter.DescriptorTable.pDescriptorRanges = descriptorSet.m_samplerRanges.data();
            m_sets[i].samplerRootParameterIndex = uint32_t(rootParameters.size());
            rootParameters.push_back(parameter);
            rootSignatureCost += 1;
        }
    }
    if (rootSignatureCost > 64)
    {
        device->printError(
            "Pipeline layout '%s': the root signature needs %u DWORDs, the limit is 64.",
            label,
            rootSignatureCost
        );
        return SLANG_E_INVALID_ARG;
    }

    D3D12_ROOT_SIGNATURE_DESC1 rootSignatureDesc = {};
    rootSignatureDesc.NumParameters = UINT(rootParameters.size());
    rootSignatureDesc.pParameters = rootParameters.data();
    rootSignatureDesc.NumStaticSamplers = 0;
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    if (m_bindless)
    {
        rootSignatureDesc.Flags |= D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
                                   D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC versionedDesc = {};
    versionedDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    versionedDesc.Desc_1_1 = rootSignatureDesc;
    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    if (SLANG_FAILED(
            device->m_D3D12SerializeVersionedRootSignature(&versionedDesc, signature.writeRef(), error.writeRef())
        ))
    {
        device->handleMessage(
            DebugMessageType::Error,
            DebugMessageSource::Layer,
            "error: D3D12SerializeRootSignature failed"
        );
        if (error)
        {
            device->handleMessage(
                DebugMessageType::Error,
                DebugMessageSource::Driver,
                (const char*)error->GetBufferPointer()
            );
        }
        return SLANG_FAIL;
    }
    SLANG_D3D_RETURN_ON_FAIL_REPORT(
        device->m_device->CreateRootSignature(
            0,
            signature->GetBufferPointer(),
            signature->GetBufferSize(),
            IID_PPV_ARGS(m_rootSignature.writeRef())
        ),
        device
    );
    return SLANG_OK;
}

Result DeviceImpl::createBindingSetLayoutImpl(const BindingSetLayoutDesc& desc, RefPtr<BindingSetLayout>& outLayout)
{
    outLayout = new BindingSetLayoutImpl(this, desc);
    return SLANG_OK;
}

Result DeviceImpl::createBindingSetImpl(BindingSetLayout* layout, RefPtr<BindingSet>& outBindingSet)
{
    outBindingSet = new BindingSetImpl(this, checked_cast<BindingSetLayoutImpl*>(layout));
    return SLANG_OK;
}

Result DeviceImpl::createPipelineLayoutImpl(const PipelineLayoutDesc& desc, RefPtr<PipelineLayout>& outLayout)
{
    outLayout = new PipelineLayoutImpl(this, desc);
    return SLANG_OK;
}

} // namespace rhi::d3d12
