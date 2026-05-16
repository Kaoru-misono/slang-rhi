#pragma once

#include "metal-base.h"

#include <vector>

namespace rhi::metal {

class BindlessDescriptorSet : public RefObject
{
public:
    BindlessDescriptorSet(DeviceImpl* device, const BindlessDesc& desc);

    Result initialize();

    Result allocBufferHandle(
        IBuffer* buffer,
        DescriptorHandleAccess access,
        Format format,
        BufferRange range,
        DescriptorHandle* outHandle
    );
    Result allocTextureHandle(ITextureView* textureView, DescriptorHandleAccess access, DescriptorHandle* outHandle);
    Result allocSamplerHandle(ISampler* sampler, DescriptorHandle* outHandle);
    Result allocCombinedTextureSamplerHandle(ITextureView* textureView, ISampler* sampler, DescriptorHandle* outHandle);
    Result freeHandle(const DescriptorHandle& handle);

    Result resolveBufferHandle(
        const DescriptorHandle& handle,
        BufferImpl** outBuffer,
        BufferRange* outRange,
        DescriptorHandleAccess* outAccess = nullptr,
        Format* outFormat = nullptr
    );
    Result resolveTextureHandle(const DescriptorHandle& handle, TextureViewImpl** outTextureView);
    Result resolveSamplerHandle(const DescriptorHandle& handle, SamplerImpl** outSampler);
    Result resolveCombinedTextureSamplerHandle(
        const DescriptorHandle& handle,
        TextureViewImpl** outTextureView,
        SamplerImpl** outSampler
    );

private:
    struct SlotAllocator
    {
        uint32_t capacity = 0;
        uint32_t count = 0;
        std::vector<uint32_t> freeSlots;

        Result allocate(uint32_t* outSlot)
        {
            if (!freeSlots.empty())
            {
                *outSlot = freeSlots.back();
                freeSlots.pop_back();
                return SLANG_OK;
            }
            if (count < capacity)
            {
                *outSlot = count++;
                return SLANG_OK;
            }
            return SLANG_E_OUT_OF_MEMORY;
        }

        Result free(uint32_t slot)
        {
            if (slot >= capacity)
                return SLANG_E_INVALID_ARG;
            freeSlots.push_back(slot);
            return SLANG_OK;
        }
    };

    struct BufferEntry
    {
        DescriptorHandleType type = DescriptorHandleType::Undefined;
        BufferImpl* buffer = nullptr;
        DescriptorHandleAccess access = DescriptorHandleAccess::Read;
        Format format = Format::Undefined;
        BufferRange range = kEntireBuffer;
    };

    struct TextureEntry
    {
        DescriptorHandleType type = DescriptorHandleType::Undefined;
        TextureViewImpl* textureView = nullptr;
    };

    struct SamplerEntry
    {
        DescriptorHandleType type = DescriptorHandleType::Undefined;
        SamplerImpl* sampler = nullptr;
    };

    struct CombinedTextureSamplerEntry
    {
        DescriptorHandleType type = DescriptorHandleType::Undefined;
        TextureViewImpl* textureView = nullptr;
        SamplerImpl* sampler = nullptr;
    };

    template<typename T>
    Result resolveEntry(const std::vector<T>& entries, const DescriptorHandle& handle, const T** outEntry)
    {
        uint32_t slot = uint32_t(handle.value);
        if (slot >= entries.size())
            return SLANG_E_INVALID_ARG;
        const T& entry = entries[slot];
        if (entry.type != handle.type)
            return SLANG_E_INVALID_ARG;
        *outEntry = &entry;
        return SLANG_OK;
    }

    BindlessDesc m_desc;

    SlotAllocator m_bufferAllocator;
    SlotAllocator m_textureAllocator;
    SlotAllocator m_samplerAllocator;
    SlotAllocator m_combinedTextureSamplerAllocator;

    std::vector<BufferEntry> m_buffers;
    std::vector<TextureEntry> m_textures;
    std::vector<SamplerEntry> m_samplers;
    std::vector<CombinedTextureSamplerEntry> m_combinedTextureSamplers;
};

} // namespace rhi::metal
