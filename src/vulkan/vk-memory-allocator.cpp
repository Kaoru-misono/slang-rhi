// Suppress warnings in VMA (third-party header-only library)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4189) // local variable initialized but not referenced
#pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif

#define VMA_IMPLEMENTATION
#include "vk-memory-allocator.h"
#include "vk-device.h"

namespace rhi::vk {

VulkanMemoryAllocator::~VulkanMemoryAllocator()
{
    destroy();
}

Result VulkanMemoryAllocator::init(const VulkanApi* api)
{
    m_api = api;

    // Provide vkGetInstanceProcAddr and vkGetDeviceProcAddr so VMA can load
    // all other Vulkan functions dynamically (since we use VK_NO_PROTOTYPES).
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = api->vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = api->vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo createInfo = {};
    createInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    createInfo.physicalDevice = api->m_physicalDevice;
    createInfo.device = api->m_device;
    createInfo.instance = api->m_instance;
    createInfo.pVulkanFunctions = &vulkanFunctions;

    // Enable buffer device address if supported
    if (api->m_extendedFeatures.vulkan12Features.bufferDeviceAddress)
    {
        createInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }

    VkResult result = vmaCreateAllocator(&createInfo, &m_allocator);
    if (result != VK_SUCCESS)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

void VulkanMemoryAllocator::destroy()
{
    if (m_allocator)
    {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }
    m_api = nullptr;
}

Result VulkanMemoryAllocator::createBuffer(
    const VkBufferCreateInfo& bufferCreateInfo,
    VkMemoryPropertyFlags memProps,
    VkBuffer* outBuffer,
    VmaAllocation* outAllocation,
    VmaAllocationInfo* outAllocInfo
)
{
    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.requiredFlags = memProps;

    // Persistently map host-visible memory
    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
        allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VkResult result = vmaCreateBuffer(
        m_allocator,
        &bufferCreateInfo,
        &allocCreateInfo,
        outBuffer,
        outAllocation,
        outAllocInfo
    );

    if (result != VK_SUCCESS)
    {
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

Result VulkanMemoryAllocator::createBufferDedicated(
    const VkBufferCreateInfo& bufferCreateInfo,
    VkMemoryPropertyFlags memProps,
    VkBuffer* outBuffer,
    VmaAllocation* outAllocation,
    VmaAllocationInfo* outAllocInfo
)
{
    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.requiredFlags = memProps;
    allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
        allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VkResult result = vmaCreateBuffer(
        m_allocator,
        &bufferCreateInfo,
        &allocCreateInfo,
        outBuffer,
        outAllocation,
        outAllocInfo
    );

    if (result != VK_SUCCESS)
    {
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

Result VulkanMemoryAllocator::allocateMemoryForImage(
    VkImage image,
    VkMemoryPropertyFlags memProps,
    VmaAllocation* outAllocation,
    VmaAllocationInfo* outAllocInfo
)
{
    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.requiredFlags = memProps;

    VkResult result = vmaAllocateMemoryForImage(
        m_allocator,
        image,
        &allocCreateInfo,
        outAllocation,
        outAllocInfo
    );

    if (result != VK_SUCCESS)
    {
        return SLANG_FAIL;
    }

    // Bind the image to the allocated memory
    result = vmaBindImageMemory(m_allocator, *outAllocation, image);
    if (result != VK_SUCCESS)
    {
        vmaFreeMemory(m_allocator, *outAllocation);
        *outAllocation = VK_NULL_HANDLE;
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

void VulkanMemoryAllocator::destroyBuffer(VkBuffer buffer, VmaAllocation allocation)
{
    if (m_allocator && buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(m_allocator, buffer, allocation);
    }
}

void VulkanMemoryAllocator::freeMemory(VmaAllocation allocation)
{
    if (m_allocator && allocation != VK_NULL_HANDLE)
    {
        vmaFreeMemory(m_allocator, allocation);
    }
}

} // namespace rhi::vk

#ifdef _MSC_VER
#pragma warning(pop)
#endif
