#pragma once

#include "vk-base.h"
#include "vk-memory-allocator.h"

namespace rhi::vk {

// Combined buffer and memory class.
// Supports two initialization paths:
// - init(): Direct allocation via VMA with dedicated memory (for shared/external memory)
// - initSubAllocated(): Sub-allocated via VMA (default path)
class VKBufferHandleRAII
{
public:
    /// Initialize with dedicated allocation (for shared/external memory resources).
    Result init(
        const VulkanApi& api,
        Size bufferSize,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags reqMemoryProperties,
        VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleTypeFlags = 0
    );

    /// Initialize with VMA sub-allocated memory (default path).
    Result initSubAllocated(
        VulkanMemoryAllocator& allocator,
        const VulkanApi& api,
        Size bufferSize,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags reqMemoryProperties
    );

    /// Returns true if has been initialized
    bool isInitialized() const { return m_api != nullptr; }

    /// Whether this buffer uses VMA-managed memory
    bool isSubAllocated() const { return m_allocator != nullptr; }

    VKBufferHandleRAII() = default;

    ~VKBufferHandleRAII()
    {
        if (m_allocator)
        {
            // VMA-managed: destroys both VkBuffer and frees memory
            m_allocator->destroyBuffer(m_buffer, m_allocation);
        }
        else if (m_api)
        {
            // Legacy direct allocation
            m_api->vkDestroyBuffer(m_api->m_device, m_buffer, nullptr);
            m_api->vkFreeMemory(m_api->m_device, m_memory, nullptr);
        }
    }

    /// Get the persistently mapped pointer (for host-visible VMA allocations).
    /// Returns nullptr for non-mapped or non-VMA allocations.
    void* getMappedPtr() const
    {
        return m_allocInfo.pMappedData;
    }

    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE; // For direct alloc: owned; for VMA: from VmaAllocationInfo
    const VulkanApi* m_api = nullptr;

    // VMA tracking
    VulkanMemoryAllocator* m_allocator = nullptr;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VmaAllocationInfo m_allocInfo = {};
};

class BufferImpl : public Buffer
{
public:
    BufferImpl(Device* device, const BufferDesc& desc);
    ~BufferImpl();

    virtual void deleteThis() override;

    VKBufferHandleRAII m_buffer;
    VKBufferHandleRAII m_uploadBuffer;

    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(NativeHandle* outHandle) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getDescriptorHandle(
        DescriptorHandleAccess access,
        Format format,
        BufferRange range,
        DescriptorHandle* outHandle
    ) override;

    struct ViewKey
    {
        Format format;
        BufferRange range;
        bool operator==(const ViewKey& other) const { return format == other.format && range == other.range; }
    };

    struct ViewKeyHasher
    {
        size_t operator()(const ViewKey& key) const
        {
            size_t hash = 0;
            hash_combine(hash, key.format);
            hash_combine(hash, key.range.offset);
            hash_combine(hash, key.range.size);
            return hash;
        }
    };

    std::mutex m_mutex;
    std::unordered_map<ViewKey, VkBufferView, ViewKeyHasher> m_views;

    VkBufferView getView(Format format, const BufferRange& range);

    struct DescriptorHandleKey
    {
        DescriptorHandleAccess access;
        Format format;
        BufferRange range;
        bool operator==(const DescriptorHandleKey& other) const
        {
            return access == other.access && format == other.format && range == other.range;
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
};

} // namespace rhi::vk
