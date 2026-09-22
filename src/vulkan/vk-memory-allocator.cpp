// Suppress warnings in VMA (third-party header-only library)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4189) // local variable initialized but not referenced
#pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif

// NOTE(april): the VMA implementation (VMA_IMPLEMENTATION) is compiled once by
// upstream's dedicated slang-rhi-vma static library (external/vma/vk_mem_alloc.cpp,
// force-including vulkan/vma-config.h). This translation unit is a pure consumer
// of that implementation; defining VMA_IMPLEMENTATION here too would duplicate
// every VMA symbol and fail to link. The VMA config (dynamic functions) in
// vk-memory-allocator.h matches vma-config.h, so the declarations are ABI-compatible.
#include "vk-memory-allocator.h"
#include "vk-device.h"
#include "vk-utils.h"

namespace rhi::vk {

VulkanMemoryAllocator::~VulkanMemoryAllocator()
{
    destroy();
}

Result VulkanMemoryAllocator::init(DeviceImpl* device)
{
    m_device = device;
    m_api = &device->m_api;

    // Provide vkGetInstanceProcAddr and vkGetDeviceProcAddr so VMA can load
    // all other Vulkan functions dynamically (since we use VK_NO_PROTOTYPES).
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = m_api->vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = m_api->vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo createInfo = {};
    createInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    createInfo.physicalDevice = m_api->m_physicalDevice;
    createInfo.device = m_api->m_device;
    createInfo.instance = m_api->m_instance;
    createInfo.pVulkanFunctions = &vulkanFunctions;

    // Enable buffer device address if supported
    if (m_api->m_extendedFeatures.vulkan12Features.bufferDeviceAddress)
    {
        createInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }

    VkResult result = vmaCreateAllocator(&createInfo, &m_allocator);
    if (result != VK_SUCCESS)
    {
        reportVulkanError(result, "vmaCreateAllocator", SLANG_RHI_SOURCE_LOCATION(), m_device);
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
    m_device = nullptr;
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

    // Cooperative-vector conversion requires 64-byte device addresses even when
    // vkGetBufferMemoryRequirements permits a smaller allocation alignment.
    const VkDeviceSize minAlignment =
        m_api->m_extendedFeatures.cooperativeVectorFeatures.cooperativeVector &&
                (bufferCreateInfo.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
            ? 64
            : 1;
    VkResult result = vmaCreateBufferWithAlignment(
        m_allocator,
        &bufferCreateInfo,
        &allocCreateInfo,
        minAlignment,
        outBuffer,
        outAllocation,
        outAllocInfo
    );

    if (result != VK_SUCCESS)
    {
        reportVulkanError(result, "vmaCreateBufferWithAlignment", SLANG_RHI_SOURCE_LOCATION(), m_device);
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
        reportVulkanError(result, "vmaCreateBuffer", SLANG_RHI_SOURCE_LOCATION(), m_device);
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
        reportVulkanError(result, "vmaAllocateMemoryForImage", SLANG_RHI_SOURCE_LOCATION(), m_device);
        return SLANG_FAIL;
    }

    // Bind the image to the allocated memory
    result = vmaBindImageMemory(m_allocator, *outAllocation, image);
    if (result != VK_SUCCESS)
    {
        reportVulkanError(result, "vmaBindImageMemory", SLANG_RHI_SOURCE_LOCATION(), m_device);
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
