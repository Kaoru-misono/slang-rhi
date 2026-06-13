#pragma once

#include "metal-base.h"

#include <unordered_map>
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

    /// Append every live bindless texture's MTL::Resource to the residency lists so
    /// the caller can useResources() them on the encoder. Metal does not auto-page
    /// resources referenced indirectly by id from an argument buffer, so an embedded
    /// .Handle texture samples zero unless made resident. RWTexture entries go to
    /// outReadWrite, the rest to outRead. Samplers are not MTL::Resources (skipped).
    void collectResidentResources(
        std::vector<MTL::Resource*>& outRead,
        std::vector<MTL::Resource*>& outReadWrite
    ) const;

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

    // On this Slang/Metal target an embedded `.Handle` lowers to an inline 64-bit
    // MTL::ResourceID, so a texture/sampler/combined handle's `value` is the
    // gpuResourceID (what the GPU reads from the argument buffer), NOT the slot.
    // These map that resource id back to the internal slot for resolve/free.
    // (Buffer handles keep slot-as-value; embedded buffer bindless is unchanged.)
    std::unordered_map<uint64_t, uint32_t> m_textureIdToSlot;
    std::unordered_map<uint64_t, uint32_t> m_samplerIdToSlot;
    std::unordered_map<uint64_t, uint32_t> m_combinedIdToSlot;
};

} // namespace rhi::metal
