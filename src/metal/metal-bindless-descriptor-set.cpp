#include "metal-bindless-descriptor-set.h"

#include "metal-buffer.h"
#include "metal-sampler.h"
#include "metal-texture.h"

namespace rhi::metal {

BindlessDescriptorSet::BindlessDescriptorSet(DeviceImpl* device, const BindlessDesc& desc)
    : m_desc(desc)
{
    SLANG_UNUSED(device);
}

Result BindlessDescriptorSet::initialize()
{
    m_bufferAllocator.capacity = m_desc.bufferCount;
    m_textureAllocator.capacity = m_desc.textureCount;
    m_samplerAllocator.capacity = m_desc.samplerCount;
    m_combinedTextureSamplerAllocator.capacity = m_desc.combinedTextureSamplerCount;

    m_buffers.resize(m_desc.bufferCount);
    m_textures.resize(m_desc.textureCount);
    m_samplers.resize(m_desc.samplerCount);
    m_combinedTextureSamplers.resize(m_desc.combinedTextureSamplerCount);

    return SLANG_OK;
}

Result BindlessDescriptorSet::allocBufferHandle(
    IBuffer* buffer,
    DescriptorHandleAccess access,
    Format format,
    BufferRange range,
    DescriptorHandle* outHandle
)
{
    uint32_t slot = 0;
    SLANG_RETURN_ON_FAIL(m_bufferAllocator.allocate(&slot));

    DescriptorHandleType type = DescriptorHandleType::Undefined;
    switch (access)
    {
    case DescriptorHandleAccess::Read:
        type = DescriptorHandleType::Buffer;
        break;
    case DescriptorHandleAccess::ReadWrite:
        type = DescriptorHandleType::RWBuffer;
        break;
    default:
        return SLANG_E_INVALID_ARG;
    }

    m_buffers[slot] = {type, checked_cast<BufferImpl*>(buffer), access, format, range};
    outHandle->type = type;
    outHandle->value = slot;
    return SLANG_OK;
}

Result BindlessDescriptorSet::allocTextureHandle(
    ITextureView* textureView,
    DescriptorHandleAccess access,
    DescriptorHandle* outHandle
)
{
    uint32_t slot = 0;
    SLANG_RETURN_ON_FAIL(m_textureAllocator.allocate(&slot));

    DescriptorHandleType type = DescriptorHandleType::Undefined;
    switch (access)
    {
    case DescriptorHandleAccess::Read:
        type = DescriptorHandleType::Texture;
        break;
    case DescriptorHandleAccess::ReadWrite:
        type = DescriptorHandleType::RWTexture;
        break;
    default:
        return SLANG_E_INVALID_ARG;
    }

    m_textures[slot] = {type, checked_cast<TextureViewImpl*>(textureView)};
    outHandle->type = type;
    outHandle->value = slot;
    return SLANG_OK;
}

Result BindlessDescriptorSet::allocSamplerHandle(ISampler* sampler, DescriptorHandle* outHandle)
{
    uint32_t slot = 0;
    SLANG_RETURN_ON_FAIL(m_samplerAllocator.allocate(&slot));

    m_samplers[slot] = {DescriptorHandleType::Sampler, checked_cast<SamplerImpl*>(sampler)};
    outHandle->type = DescriptorHandleType::Sampler;
    outHandle->value = slot;
    return SLANG_OK;
}

Result BindlessDescriptorSet::allocCombinedTextureSamplerHandle(
    ITextureView* textureView,
    ISampler* sampler,
    DescriptorHandle* outHandle
)
{
    uint32_t slot = 0;
    SLANG_RETURN_ON_FAIL(m_combinedTextureSamplerAllocator.allocate(&slot));

    m_combinedTextureSamplers[slot] = {
        DescriptorHandleType::CombinedTextureSampler,
        checked_cast<TextureViewImpl*>(textureView),
        checked_cast<SamplerImpl*>(sampler),
    };
    outHandle->type = DescriptorHandleType::CombinedTextureSampler;
    outHandle->value = slot;
    return SLANG_OK;
}

Result BindlessDescriptorSet::freeHandle(const DescriptorHandle& handle)
{
    switch (handle.type)
    {
    case DescriptorHandleType::Buffer:
    case DescriptorHandleType::RWBuffer:
        if (handle.value >= m_buffers.size())
            return SLANG_E_INVALID_ARG;
        m_buffers[uint32_t(handle.value)] = {};
        return m_bufferAllocator.free(uint32_t(handle.value));
    case DescriptorHandleType::Texture:
    case DescriptorHandleType::RWTexture:
        if (handle.value >= m_textures.size())
            return SLANG_E_INVALID_ARG;
        m_textures[uint32_t(handle.value)] = {};
        return m_textureAllocator.free(uint32_t(handle.value));
    case DescriptorHandleType::Sampler:
        if (handle.value >= m_samplers.size())
            return SLANG_E_INVALID_ARG;
        m_samplers[uint32_t(handle.value)] = {};
        return m_samplerAllocator.free(uint32_t(handle.value));
    case DescriptorHandleType::CombinedTextureSampler:
        if (handle.value >= m_combinedTextureSamplers.size())
            return SLANG_E_INVALID_ARG;
        m_combinedTextureSamplers[uint32_t(handle.value)] = {};
        return m_combinedTextureSamplerAllocator.free(uint32_t(handle.value));
    default:
        return SLANG_E_INVALID_ARG;
    }
}

Result BindlessDescriptorSet::resolveBufferHandle(
    const DescriptorHandle& handle,
    BufferImpl** outBuffer,
    BufferRange* outRange,
    DescriptorHandleAccess* outAccess,
    Format* outFormat
)
{
    const BufferEntry* entry = nullptr;
    SLANG_RETURN_ON_FAIL(resolveEntry(m_buffers, handle, &entry));
    if (!entry->buffer)
        return SLANG_E_INVALID_ARG;
    *outBuffer = entry->buffer;
    *outRange = entry->range;
    if (outAccess)
        *outAccess = entry->access;
    if (outFormat)
        *outFormat = entry->format;
    return SLANG_OK;
}

Result BindlessDescriptorSet::resolveTextureHandle(const DescriptorHandle& handle, TextureViewImpl** outTextureView)
{
    const TextureEntry* entry = nullptr;
    SLANG_RETURN_ON_FAIL(resolveEntry(m_textures, handle, &entry));
    if (!entry->textureView)
        return SLANG_E_INVALID_ARG;
    *outTextureView = entry->textureView;
    return SLANG_OK;
}

Result BindlessDescriptorSet::resolveSamplerHandle(const DescriptorHandle& handle, SamplerImpl** outSampler)
{
    const SamplerEntry* entry = nullptr;
    SLANG_RETURN_ON_FAIL(resolveEntry(m_samplers, handle, &entry));
    if (!entry->sampler)
        return SLANG_E_INVALID_ARG;
    *outSampler = entry->sampler;
    return SLANG_OK;
}

Result BindlessDescriptorSet::resolveCombinedTextureSamplerHandle(
    const DescriptorHandle& handle,
    TextureViewImpl** outTextureView,
    SamplerImpl** outSampler
)
{
    const CombinedTextureSamplerEntry* entry = nullptr;
    SLANG_RETURN_ON_FAIL(resolveEntry(m_combinedTextureSamplers, handle, &entry));
    if (!entry->textureView || !entry->sampler)
        return SLANG_E_INVALID_ARG;
    *outTextureView = entry->textureView;
    *outSampler = entry->sampler;
    return SLANG_OK;
}

} // namespace rhi::metal
