#include "binding-set.h"

#include "rhi-shared.h"
#include "device.h"

namespace rhi {

// ----------------------------------------------------------------------------
// BindingSetLayout
// ----------------------------------------------------------------------------

IBindingSetLayout* BindingSetLayout::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == IBindingSetLayout::getTypeGuid())
        return static_cast<IBindingSetLayout*>(this);
    return nullptr;
}

BindingSetLayout::BindingSetLayout(Device* device, const BindingSetLayoutDesc& desc)
    : DeviceChild(device)
    , m_desc(desc)
{
    auto* entries = const_cast<BindingSetLayoutEntry*>(m_desc.entries);
    m_descHolder.holdList(entries, m_desc.entryCount);
    for (uint32_t i = 0; i < m_desc.entryCount; ++i)
        m_descHolder.holdString(entries[i].name);
    m_desc.entries = entries;
    m_descHolder.holdString(m_desc.label);

    for (uint32_t i = 0; i < m_desc.entryCount; ++i)
    {
        m_bindingOffsets.emplace_back(m_bindingCount);
        m_bindingCount += entries[i].count;
    }
    m_reflection.resize(m_desc.entryCount);
}

BindingSetLayout::~BindingSetLayout()
{
}

bool isSamplerBindingKind(BindingKind kind)
{
    switch (kind)
    {
    case BindingKind::SamplerState:
    case BindingKind::SamplerComparisonState:
        return true;
    case BindingKind::Texture1D:
    case BindingKind::Texture2D:
    case BindingKind::Texture2DArray:
    case BindingKind::Texture3D:
    case BindingKind::TextureCube:
    case BindingKind::TextureCubeArray:
    case BindingKind::RWTexture1D:
    case BindingKind::RWTexture2D:
    case BindingKind::RWTexture2DArray:
    case BindingKind::RWTexture3D:
    case BindingKind::Buffer:
    case BindingKind::RWBuffer:
    case BindingKind::StructuredBuffer:
    case BindingKind::RWStructuredBuffer:
    case BindingKind::ByteAddressBuffer:
    case BindingKind::RWByteAddressBuffer:
    case BindingKind::RaytracingAccelerationStructure:
        return false;
    }
    SLANG_RHI_UNREACHABLE("Invalid binding kind");
    return false;
}

bool isTextureBindingKind(BindingKind kind)
{
    switch (kind)
    {
    case BindingKind::Texture1D:
    case BindingKind::Texture2D:
    case BindingKind::Texture2DArray:
    case BindingKind::Texture3D:
    case BindingKind::TextureCube:
    case BindingKind::TextureCubeArray:
    case BindingKind::RWTexture1D:
    case BindingKind::RWTexture2D:
    case BindingKind::RWTexture2DArray:
    case BindingKind::RWTexture3D:
        return true;
    case BindingKind::SamplerState:
    case BindingKind::SamplerComparisonState:
    case BindingKind::Buffer:
    case BindingKind::RWBuffer:
    case BindingKind::StructuredBuffer:
    case BindingKind::RWStructuredBuffer:
    case BindingKind::ByteAddressBuffer:
    case BindingKind::RWByteAddressBuffer:
    case BindingKind::RaytracingAccelerationStructure:
        return false;
    }
    SLANG_RHI_UNREACHABLE("Invalid binding kind");
    return false;
}

bool isBufferBindingKind(BindingKind kind)
{
    switch (kind)
    {
    case BindingKind::Buffer:
    case BindingKind::RWBuffer:
    case BindingKind::StructuredBuffer:
    case BindingKind::RWStructuredBuffer:
    case BindingKind::ByteAddressBuffer:
    case BindingKind::RWByteAddressBuffer:
        return true;
    case BindingKind::SamplerState:
    case BindingKind::SamplerComparisonState:
    case BindingKind::Texture1D:
    case BindingKind::Texture2D:
    case BindingKind::Texture2DArray:
    case BindingKind::Texture3D:
    case BindingKind::TextureCube:
    case BindingKind::TextureCubeArray:
    case BindingKind::RWTexture1D:
    case BindingKind::RWTexture2D:
    case BindingKind::RWTexture2DArray:
    case BindingKind::RWTexture3D:
    case BindingKind::RaytracingAccelerationStructure:
        return false;
    }
    SLANG_RHI_UNREACHABLE("Invalid binding kind");
    return false;
}

ResourceState getBindingKindResourceState(BindingKind kind)
{
    switch (kind)
    {
    case BindingKind::Texture1D:
    case BindingKind::Texture2D:
    case BindingKind::Texture2DArray:
    case BindingKind::Texture3D:
    case BindingKind::TextureCube:
    case BindingKind::TextureCubeArray:
    case BindingKind::Buffer:
    case BindingKind::StructuredBuffer:
    case BindingKind::ByteAddressBuffer:
        return ResourceState::ShaderResource;
    case BindingKind::RWTexture1D:
    case BindingKind::RWTexture2D:
    case BindingKind::RWTexture2DArray:
    case BindingKind::RWTexture3D:
    case BindingKind::RWBuffer:
    case BindingKind::RWStructuredBuffer:
    case BindingKind::RWByteAddressBuffer:
        return ResourceState::UnorderedAccess;
    case BindingKind::SamplerState:
    case BindingKind::SamplerComparisonState:
    case BindingKind::RaytracingAccelerationStructure:
        return ResourceState::Undefined;
    }
    SLANG_RHI_UNREACHABLE("Invalid binding kind");
    return ResourceState::Undefined;
}

// ----------------------------------------------------------------------------
// BindingSet
// ----------------------------------------------------------------------------

IBindingSet* BindingSet::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == IBindingSet::getTypeGuid())
        return static_cast<IBindingSet*>(this);
    return nullptr;
}

BindingSet::BindingSet(Device* device, BindingSetLayout* layout)
    : DeviceChild(device)
    , m_layout(layout)
{
    m_bindings.resize(layout->getBindingCount());
}

BindingSet::~BindingSet()
{
}

Result BindingSet::init(const BindingSetDesc& desc)
{
    auto fail = [&](uint32_t slot, uint32_t arrayIndex, const char* reason) -> Result
    {
        getDevice()->printError(
            "Binding set '%s', slot %u[%u]: %s", desc.label ? desc.label : "<unnamed>", slot, arrayIndex, reason
        );
        return SLANG_E_INVALID_ARG;
    };
    if (!desc.entryCount || !desc.entries)
        return fail(0, 0, "entries must be non-null and entryCount must be nonzero");

    for (uint32_t i = 0; i < desc.entryCount; ++i)
    {
        const auto& entry = desc.entries[i];
        auto failEntry = [&](const char* reason) { return fail(entry.slot, entry.arrayIndex, reason); };
        if (entry.slot >= m_layout->m_desc.entryCount)
            return failEntry("slot is out of bounds");
        const auto& layoutEntry = m_layout->m_desc.entries[entry.slot];
        if (entry.arrayIndex >= layoutEntry.count)
            return failEntry("array index is out of bounds");
        if (int(entry.sampler != nullptr) + int(entry.textureView != nullptr) + int(entry.buffer != nullptr) +
                int(entry.accelerationStructure != nullptr) !=
            1)
            return failEntry("exactly one resource must be specified");
        auto& binding = m_bindings[m_layout->getBindingOffset(entry.slot) + entry.arrayIndex];
        if (binding.sampler || binding.textureView || binding.buffer || binding.accelerationStructure)
            return failEntry("duplicate binding");

        auto* sampler = checked_cast<Sampler*>(entry.sampler);
        auto* textureView = checked_cast<TextureView*>(entry.textureView);
        auto* buffer = checked_cast<Buffer*>(entry.buffer);
        auto* accelerationStructure = checked_cast<AccelerationStructure*>(entry.accelerationStructure);
        if ((sampler && sampler->getDevice() != getDevice()) ||
            (textureView && textureView->getDevice() != getDevice()) ||
            (buffer && buffer->getDevice() != getDevice()) ||
            (accelerationStructure && accelerationStructure->getDevice() != getDevice()))
            return failEntry("resource belongs to another device");

        BindingKind kind = layoutEntry.kind;
        ResourceState state = getBindingKindResourceState(kind);
        BufferRange bufferRange = kEntireBuffer;
        if (isSamplerBindingKind(kind))
        {
            if (!sampler)
                return failEntry("expected a sampler");
            if ((sampler->getDesc().reductionOp == TextureReductionOp::Comparison) !=
                (kind == BindingKind::SamplerComparisonState))
                return failEntry("sampler comparison mode does not match the binding kind");
        }
        else if (isTextureBindingKind(kind))
        {
            if (!textureView)
                return failEntry("expected a texture view");
            TextureType expectedType;
            switch (kind)
            {
            case BindingKind::Texture1D:
            case BindingKind::RWTexture1D:
                expectedType = TextureType::Texture1D;
                break;
            case BindingKind::Texture2D:
            case BindingKind::RWTexture2D:
                expectedType = TextureType::Texture2D;
                break;
            case BindingKind::Texture2DArray:
            case BindingKind::RWTexture2DArray:
                expectedType = TextureType::Texture2DArray;
                break;
            case BindingKind::Texture3D:
            case BindingKind::RWTexture3D:
                expectedType = TextureType::Texture3D;
                break;
            case BindingKind::TextureCube:
                expectedType = TextureType::TextureCube;
                break;
            case BindingKind::TextureCubeArray:
                expectedType = TextureType::TextureCubeArray;
                break;
            default:
                return failEntry("invalid texture binding kind");
            }
            const auto& textureDesc = textureView->getTexture()->getDesc();
            if (textureDesc.type != expectedType)
                return failEntry("texture type does not match the binding kind");
            TextureUsage usage = state == ResourceState::ShaderResource ? TextureUsage::ShaderResource
                                                                       : TextureUsage::UnorderedAccess;
            if (!is_set(textureDesc.usage, usage))
                return failEntry("texture usage does not support the binding kind");
        }
        else if (isBufferBindingKind(kind))
        {
            if (!buffer)
                return failEntry("expected a buffer");
            const auto& bufferDesc = buffer->getDesc();
            const auto& requested = entry.bufferRange;
            bufferRange = buffer->resolveBufferRange(requested);
            // resolveBufferRange clamps, so an out-of-bounds request has to be caught before clamping.
            const bool clamped = requested.offset > bufferDesc.size ||
                                 (requested.size != kEntireBuffer.size &&
                                  requested.size > bufferDesc.size - requested.offset);
            if (clamped || !bufferRange.size)
                return failEntry("buffer range is empty or out of bounds");
            BufferUsage usage = state == ResourceState::ShaderResource ? BufferUsage::ShaderResource
                                                                      : BufferUsage::UnorderedAccess;
            if (!is_set(bufferDesc.usage, usage))
                return failEntry("buffer usage does not support the binding kind");
            uint32_t stride = m_layout->m_reflection[entry.slot].structuredBufferStride;
            if ((kind == BindingKind::StructuredBuffer || kind == BindingKind::RWStructuredBuffer) && stride &&
                bufferRange.size % stride != 0)
                return failEntry("buffer range size is not a multiple of the structured buffer stride");
        }
        else if (kind == BindingKind::RaytracingAccelerationStructure)
        {
            if (!accelerationStructure)
                return failEntry("expected an acceleration structure");
        }

        binding.sampler = sampler;
        binding.textureView = textureView;
        binding.buffer = buffer;
        binding.bufferRange = bufferRange;
        binding.accelerationStructure = accelerationStructure;
    }

    for (uint32_t slot = 0; slot < m_layout->m_desc.entryCount; ++slot)
    {
        for (uint32_t arrayIndex = 0; arrayIndex < m_layout->m_desc.entries[slot].count; ++arrayIndex)
        {
            const auto& binding = m_bindings[m_layout->getBindingOffset(slot) + arrayIndex];
            if (!binding.sampler && !binding.textureView && !binding.buffer && !binding.accelerationStructure)
                return fail(slot, arrayIndex, "binding is missing");
        }
    }

    for (uint32_t slot = 0; slot < m_layout->m_desc.entryCount; ++slot)
    {
        const auto& layoutEntry = m_layout->m_desc.entries[slot];
        ResourceState state = getBindingKindResourceState(layoutEntry.kind);
        if (state == ResourceState::Undefined)
            continue;
        for (uint32_t arrayIndex = 0; arrayIndex < layoutEntry.count; ++arrayIndex)
        {
            const auto& binding = m_bindings[m_layout->getBindingOffset(slot) + arrayIndex];
            ResourceAccess access;
            access.state = state;
            if (binding.buffer)
                access.buffer = binding.buffer;
            else
            {
                access.texture = binding.textureView->getTexture();
                access.subresourceRange = binding.textureView->getDesc().subresourceRange;
            }
            m_accesses.emplace_back(access);
        }
    }

    m_label = desc.label;
    m_descHolder.holdString(m_label);
    return SLANG_OK;
}

} // namespace rhi
