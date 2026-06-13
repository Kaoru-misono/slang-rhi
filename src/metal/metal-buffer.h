#pragma once

#include "metal-base.h"

#include <unordered_map>

namespace rhi::metal {

class BufferImpl : public Buffer
{
public:
    BufferImpl(Device* device, const BufferDesc& desc);
    ~BufferImpl();

    virtual void deleteThis() override;

    // IResource implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;

    // IBuffer implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(NativeHandle* outHandle) override;
    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getDescriptorHandle(
        DescriptorHandleAccess access,
        Format format,
        BufferRange range,
        DescriptorHandle* outHandle
    ) override;

    /// Lower this buffer to a Metal texture-buffer view (used to bind a
    /// TypedBuffer, which Metal represents as a texture). Cached per
    /// (access, format, range).
    Result getTextureBufferView(
        DescriptorHandleAccess access,
        Format format,
        BufferRange range,
        MTL::Texture** outTexture
    );

public:
    NS::SharedPtr<MTL::Buffer> m_buffer;
    DeviceAddress m_deviceAddress = 0;

private:
    struct DescriptorHandleKey
    {
        DescriptorHandleAccess access;
        Format format;
        BufferRange range;

        bool operator==(const DescriptorHandleKey& other) const
        {
            return access == other.access && format == other.format && range.offset == other.range.offset &&
                   range.size == other.range.size;
        }
    };

    struct DescriptorHandleKeyHasher
    {
        size_t operator()(const DescriptorHandleKey& key) const
        {
            size_t hash = 0;
            hash_combine(hash, key.access);
            hash_combine(hash, key.format);
            hash_combine(hash, key.range.offset);
            hash_combine(hash, key.range.size);
            return hash;
        }
    };

    std::unordered_map<DescriptorHandleKey, DescriptorHandle, DescriptorHandleKeyHasher> m_descriptorHandles;
    std::unordered_map<DescriptorHandleKey, NS::SharedPtr<MTL::Texture>, DescriptorHandleKeyHasher> m_textureBufferViews;
};

} // namespace rhi::metal
