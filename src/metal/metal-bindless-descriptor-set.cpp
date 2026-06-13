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

    auto* impl = checked_cast<TextureViewImpl*>(textureView);
    uint64_t resourceId = impl->m_textureView->gpuResourceID()._impl;
    m_textures[slot] = {type, impl};
    m_textureIdToSlot[resourceId] = slot;
    outHandle->type = type;
    outHandle->value = resourceId; // Metal: the GPU reads this inline as an MTL::ResourceID
    return SLANG_OK;
}

Result BindlessDescriptorSet::allocSamplerHandle(ISampler* sampler, DescriptorHandle* outHandle)
{
    uint32_t slot = 0;
    SLANG_RETURN_ON_FAIL(m_samplerAllocator.allocate(&slot));

    auto* impl = checked_cast<SamplerImpl*>(sampler);
    uint64_t resourceId = impl->m_samplerState->gpuResourceID()._impl;
    m_samplers[slot] = {DescriptorHandleType::Sampler, impl};
    m_samplerIdToSlot[resourceId] = slot;
    outHandle->type = DescriptorHandleType::Sampler;
    outHandle->value = resourceId; // Metal: inline sampler MTL::ResourceID
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

    auto* texImpl = checked_cast<TextureViewImpl*>(textureView);
    auto* sampImpl = checked_cast<SamplerImpl*>(sampler);
    // Key on the texture's resource id (what a combined .Handle dereference reads).
    uint64_t resourceId = texImpl->m_textureView->gpuResourceID()._impl;
    m_combinedTextureSamplers[slot] = {
        DescriptorHandleType::CombinedTextureSampler, texImpl, sampImpl,
    };
    m_combinedIdToSlot[resourceId] = slot;
    outHandle->type = DescriptorHandleType::CombinedTextureSampler;
    outHandle->value = resourceId;
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
    {
        auto it = m_textureIdToSlot.find(handle.value);
        if (it == m_textureIdToSlot.end())
            return SLANG_E_INVALID_ARG;
        uint32_t slot = it->second;
        m_textureIdToSlot.erase(it);
        m_textures[slot] = {};
        return m_textureAllocator.free(slot);
    }
    case DescriptorHandleType::Sampler:
    {
        auto it = m_samplerIdToSlot.find(handle.value);
        if (it == m_samplerIdToSlot.end())
            return SLANG_E_INVALID_ARG;
        uint32_t slot = it->second;
        m_samplerIdToSlot.erase(it);
        m_samplers[slot] = {};
        return m_samplerAllocator.free(slot);
    }
    case DescriptorHandleType::CombinedTextureSampler:
    {
        auto it = m_combinedIdToSlot.find(handle.value);
        if (it == m_combinedIdToSlot.end())
            return SLANG_E_INVALID_ARG;
        uint32_t slot = it->second;
        m_combinedIdToSlot.erase(it);
        m_combinedTextureSamplers[slot] = {};
        return m_combinedTextureSamplerAllocator.free(slot);
    }
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
    auto it = m_textureIdToSlot.find(handle.value);
    if (it == m_textureIdToSlot.end() || it->second >= m_textures.size())
        return SLANG_E_INVALID_ARG;
    const TextureEntry& entry = m_textures[it->second];
    if (entry.type != handle.type || !entry.textureView)
        return SLANG_E_INVALID_ARG;
    *outTextureView = entry.textureView;
    return SLANG_OK;
}

Result BindlessDescriptorSet::resolveSamplerHandle(const DescriptorHandle& handle, SamplerImpl** outSampler)
{
    auto it = m_samplerIdToSlot.find(handle.value);
    if (it == m_samplerIdToSlot.end() || it->second >= m_samplers.size())
        return SLANG_E_INVALID_ARG;
    const SamplerEntry& entry = m_samplers[it->second];
    if (!entry.sampler)
        return SLANG_E_INVALID_ARG;
    *outSampler = entry.sampler;
    return SLANG_OK;
}

Result BindlessDescriptorSet::resolveCombinedTextureSamplerHandle(
    const DescriptorHandle& handle,
    TextureViewImpl** outTextureView,
    SamplerImpl** outSampler
)
{
    auto it = m_combinedIdToSlot.find(handle.value);
    if (it == m_combinedIdToSlot.end() || it->second >= m_combinedTextureSamplers.size())
        return SLANG_E_INVALID_ARG;
    const CombinedTextureSamplerEntry& entry = m_combinedTextureSamplers[it->second];
    if (!entry.textureView || !entry.sampler)
        return SLANG_E_INVALID_ARG;
    *outTextureView = entry.textureView;
    *outSampler = entry.sampler;
    return SLANG_OK;
}

void BindlessDescriptorSet::collectResidentResources(
    std::vector<MTL::Resource*>& outRead,
    std::vector<MTL::Resource*>& outReadWrite
) const
{
    // Walk the slot-indexed arrays directly (freed slots are zeroed -> skipped).
    // Textures referenced indirectly by id from an argument buffer must be made
    // resident or Metal samples zero. Samplers are not MTL::Resources.
    for (const TextureEntry& entry : m_textures)
    {
        if (!entry.textureView)
            continue;
        MTL::Resource* res = entry.textureView->m_textureView.get();
        if (entry.type == DescriptorHandleType::RWTexture)
            outReadWrite.push_back(res);
        else
            outRead.push_back(res);
    }
    for (const CombinedTextureSamplerEntry& entry : m_combinedTextureSamplers)
    {
        if (entry.textureView)
            outRead.push_back(entry.textureView->m_textureView.get());
    }
}

} // namespace rhi::metal
