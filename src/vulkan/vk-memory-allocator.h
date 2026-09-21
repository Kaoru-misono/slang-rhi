#pragma once

#include "vk-base.h"

// VMA configuration: use dynamic function loading since slang-rhi uses VK_NO_PROTOTYPES
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

namespace rhi::vk {

/// Thin wrapper around VMA (Vulkan Memory Allocator).
/// Handles VmaAllocator lifetime and provides convenience methods
/// matching the slang-rhi buffer/texture creation patterns.
class VulkanMemoryAllocator
{
public:
    VulkanMemoryAllocator() = default;
    ~VulkanMemoryAllocator();

    /// Initialize VMA with the Vulkan API handles.
    Result init(const VulkanApi* api);

    /// Destroy the VMA allocator. Must be called before VkDevice destruction.
    void destroy();

    /// Get the underlying VMA allocator handle.
    VmaAllocator getAllocator() const { return m_allocator; }

    /// Create a VkBuffer with sub-allocated memory.
    /// @param bufferCreateInfo The VkBufferCreateInfo for the buffer
    /// @param memProps Required memory property flags
    /// @param outBuffer Output VkBuffer handle
    /// @param outAllocation Output VMA allocation handle
    /// @param outAllocInfo Output allocation info (memory, offset, mapped pointer, etc.)
    Result createBuffer(
        const VkBufferCreateInfo& bufferCreateInfo,
        VkMemoryPropertyFlags memProps,
        VkBuffer* outBuffer,
        VmaAllocation* outAllocation,
        VmaAllocationInfo* outAllocInfo
    );

    /// Create a VkBuffer with dedicated memory (for shared/external memory).
    /// Uses VMA's dedicated allocation flag.
    Result createBufferDedicated(
        const VkBufferCreateInfo& bufferCreateInfo,
        VkMemoryPropertyFlags memProps,
        VkBuffer* outBuffer,
        VmaAllocation* outAllocation,
        VmaAllocationInfo* outAllocInfo
    );

    /// Allocate memory for an existing VkImage.
    /// VMA handles alignment, dedicated allocation detection, and memory type selection.
    Result allocateMemoryForImage(
        VkImage image,
        VkMemoryPropertyFlags memProps,
        VmaAllocation* outAllocation,
        VmaAllocationInfo* outAllocInfo
    );

    /// Destroy a buffer and free its memory.
    void destroyBuffer(VkBuffer buffer, VmaAllocation allocation);

    /// Free image memory.
    void freeMemory(VmaAllocation allocation);

private:
    const VulkanApi* m_api = nullptr;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
};

} // namespace rhi::vk
