#include "metal-buffer.h"
#include "metal-bindless-descriptor-set.h"
#include "metal-device.h"
#include "metal-utils.h"

namespace rhi::metal {

BufferImpl::BufferImpl(Device* device, const BufferDesc& desc)
    : Buffer(device, desc)
{
}

BufferImpl::~BufferImpl()
{
    DeviceImpl* device = getDevice<DeviceImpl>();
    if (device && device->m_bindlessDescriptorSet)
    {
        for (auto& handle : m_descriptorHandles)
        {
            SLANG_RHI_ASSERT(SLANG_SUCCEEDED(device->m_bindlessDescriptorSet->freeHandle(handle.second)));
        }
    }
}

void BufferImpl::deleteThis()
{
    getDevice<DeviceImpl>()->deferDelete(this);
}

DeviceAddress BufferImpl::getDeviceAddress()
{
    return m_buffer->gpuAddress();
}

Result BufferImpl::getNativeHandle(NativeHandle* outHandle)
{
    outHandle->type = NativeHandleType::MTLBuffer;
    outHandle->value = (uint64_t)m_buffer.get();
    return SLANG_OK;
}

Result BufferImpl::getSharedHandle(NativeHandle* outHandle)
{
    *outHandle = {};
    return SLANG_E_NOT_AVAILABLE;
}

Result BufferImpl::getDescriptorHandle(
    DescriptorHandleAccess access,
    Format format,
    BufferRange range,
    DescriptorHandle* outHandle
)
{
    DeviceImpl* device = getDevice<DeviceImpl>();
    if (!device->m_bindlessDescriptorSet)
        return SLANG_E_NOT_AVAILABLE;

    range = resolveBufferRange(range);
    DescriptorHandleKey key = {access, format, range};
    DescriptorHandle& handle = m_descriptorHandles[key];
    if (!handle)
    {
        SLANG_RETURN_ON_FAIL(device->m_bindlessDescriptorSet->allocBufferHandle(this, access, format, range, &handle));
    }

    *outHandle = handle;
    return SLANG_OK;
}

Result BufferImpl::getTextureBufferView(
    DescriptorHandleAccess access,
    Format format,
    BufferRange range,
    MTL::Texture** outTexture
)
{
    if (format == Format::Undefined)
        return SLANG_E_INVALID_ARG;

    MTL::PixelFormat pixelFormat = translatePixelFormat(format);
    if (pixelFormat == MTL::PixelFormatInvalid)
        return SLANG_E_INVALID_ARG;

    range = resolveBufferRange(range);

    const FormatInfo& formatInfo = getFormatInfo(format);
    if (formatInfo.pixelsPerBlock != 1 || range.size % formatInfo.blockSizeInBytes != 0)
        return SLANG_E_INVALID_ARG;

    DescriptorHandleKey key = {access, format, range};
    NS::SharedPtr<MTL::Texture>& texture = m_textureBufferViews[key];
    if (!texture)
    {
        MTL::TextureUsage usage = MTL::TextureUsageShaderRead;
        if (access == DescriptorHandleAccess::ReadWrite)
        {
            usage = MTL::TextureUsage(usage | MTL::TextureUsageShaderWrite);
        }
        else if (access != DescriptorHandleAccess::Read)
        {
            return SLANG_E_INVALID_ARG;
        }

        NS::UInteger width = NS::UInteger(range.size / formatInfo.blockSizeInBytes);
        MTL::ResourceOptions resourceOptions = MTL::ResourceOptions(m_buffer->resourceOptions());
        NS::SharedPtr<MTL::TextureDescriptor> textureDesc =
            NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
        textureDesc->setTextureType(MTL::TextureTypeTextureBuffer);
        textureDesc->setPixelFormat(pixelFormat);
        textureDesc->setWidth(width);
        textureDesc->setHeight(1);
        textureDesc->setDepth(1);
        textureDesc->setMipmapLevelCount(1);
        textureDesc->setArrayLength(1);
        textureDesc->setResourceOptions(resourceOptions);
        textureDesc->setUsage(usage);

        NS::UInteger bytesPerRow = NS::UInteger(range.size);
        texture = NS::TransferPtr(m_buffer->newTexture(textureDesc.get(), NS::UInteger(range.offset), bytesPerRow));
        if (!texture)
            return SLANG_FAIL;
    }

    *outTexture = texture.get();
    return SLANG_OK;
}

Result DeviceImpl::createBuffer(const BufferDesc& desc_, const void* initData, IBuffer** outBuffer)
{
    AUTORELEASEPOOL

    BufferDesc desc = fixupBufferDesc(desc_);

    const Size bufferSize = desc.size;

    MTL::ResourceOptions resourceOptions = MTL::ResourceOptions(0);
    switch (desc.memoryType)
    {
    case MemoryType::DeviceLocal:
        resourceOptions = MTL::ResourceStorageModePrivate;
        break;
    case MemoryType::Upload:
    case MemoryType::ReadBack:
        resourceOptions = MTL::ResourceStorageModeManaged;
        break;
    }

    RefPtr<BufferImpl> buffer(new BufferImpl(this, desc));
    buffer->m_buffer = NS::TransferPtr(m_device->newBuffer(bufferSize, resourceOptions));
    if (!buffer->m_buffer)
    {
        return SLANG_FAIL;
    }

    if (desc.label)
        buffer->m_buffer->addDebugMarker(createString(desc.label).get(), NS::Range(0, desc.size));

    if (initData)
    {
        NS::SharedPtr<MTL::Buffer> stagingBuffer =
            NS::TransferPtr(m_device->newBuffer(initData, bufferSize, MTL::ResourceStorageModeManaged));
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
        if (!stagingBuffer || !commandBuffer || !encoder)
        {
            return SLANG_FAIL;
        }
        encoder->copyFromBuffer(stagingBuffer.get(), 0, buffer->m_buffer.get(), 0, bufferSize);
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }

    returnComPtr(outBuffer, buffer);
    return SLANG_OK;
}

Result DeviceImpl::createBufferFromNativeHandle(NativeHandle handle, const BufferDesc& desc, IBuffer** outBuffer)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::mapBuffer(IBuffer* buffer, CpuAccessMode mode, void** outData)
{
    BufferImpl* bufferImpl = checked_cast<BufferImpl*>(buffer);
    bufferImpl->m_lastCpuAccessMode = mode;
    if (mode == CpuAccessMode::Read)
    {
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
        encoder->synchronizeResource(bufferImpl->m_buffer.get());
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }
    *outData = bufferImpl->m_buffer->contents();
    return SLANG_OK;
}

Result DeviceImpl::unmapBuffer(IBuffer* buffer)
{
    BufferImpl* bufferImpl = checked_cast<BufferImpl*>(buffer);
    if (bufferImpl->m_lastCpuAccessMode == CpuAccessMode::Write)
    {
        bufferImpl->m_buffer->didModifyRange(NS::Range(0, bufferImpl->m_desc.size));
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
        encoder->synchronizeResource(bufferImpl->m_buffer.get());
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }
    return SLANG_OK;
}

} // namespace rhi::metal
