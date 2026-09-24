#include "vk-descriptor-allocator.h"
#include "vk-utils.h"

#include "core/static_vector.h"

namespace rhi::vk {
namespace {
Result descriptorAllocationResult(VkResult result)
{
    return result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                   result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL
               ? SLANG_E_OUT_OF_MEMORY
               : SLANG_FAIL;
}
} // namespace

Result DescriptorSetAllocator::newPool(VkDescriptorPool* outPool)
{
    VkDescriptorPoolCreateInfo descriptorPoolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    static_vector<VkDescriptorPoolSize, 32> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1024});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4096});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4096});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 4096});
    poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 16});
    if (m_api->m_extendedFeatures.inlineUniformBlockFeatures.inlineUniformBlock)
    {
        poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT, 16});
    }
    if (m_api->m_extendedFeatures.accelerationStructureFeatures.accelerationStructure)
    {
        poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 256});
    }
    descriptorPoolInfo.maxSets = 4096;
    descriptorPoolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    descriptorPoolInfo.pPoolSizes = poolSizes.data();
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VkDescriptorPoolInlineUniformBlockCreateInfo inlineUniformBlockInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO
    };
    inlineUniformBlockInfo.maxInlineUniformBlockBindings = 16;
    descriptorPoolInfo.pNext = &inlineUniformBlockInfo;

    *outPool = VK_NULL_HANDLE;
    // Reserve host ownership before creating the native pool.
    pools.reserve(pools.size() + 1);
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    const VkResult result = m_api->vkCreateDescriptorPool(m_api->m_device, &descriptorPoolInfo, nullptr, &descriptorPool);
    if (result != VK_SUCCESS)
        return descriptorAllocationResult(result);
    pools.push_back(descriptorPool);
    *outPool = descriptorPool;
    return SLANG_OK;
}

Result DescriptorSetAllocator::allocate(VkDescriptorSetLayout layout, VulkanDescriptorSet* outSet)
{
    *outSet = {};
    SLANG_RETURN_ON_FAIL(m_resetResult);
    if (m_liveSets == UINT64_MAX || (m_maxSets && m_liveSets >= m_maxSets))
        return SLANG_E_OUT_OF_MEMORY;
    VkDescriptorSetAllocateInfo info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;
    // Pool exhaustion permits trying another pool; native device/host errors must propagate.
    for (auto it = pools.rbegin(); it != pools.rend(); ++it)
    {
        info.descriptorPool = *it;
        VkDescriptorSet handle = VK_NULL_HANDLE;
        const VkResult result = m_api->vkAllocateDescriptorSets(m_api->m_device, &info, &handle);
        if (result == VK_SUCCESS)
        {
            ++m_liveSets;
            *outSet = {handle, *it};
            return SLANG_OK;
        }
        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL)
            return descriptorAllocationResult(result);
    }
    SLANG_RETURN_ON_FAIL(newPool(&info.descriptorPool));
    VkDescriptorSet handle = VK_NULL_HANDLE;
    const VkResult result = m_api->vkAllocateDescriptorSets(m_api->m_device, &info, &handle);
    if (result != VK_SUCCESS)
    {
        // No set has ever used this new pool; do not retain it across failed retries.
        m_api->vkDestroyDescriptorPool(m_api->m_device, info.descriptorPool, nullptr);
        pools.pop_back();
        return descriptorAllocationResult(result);
    }
    ++m_liveSets;
    *outSet = {handle, info.descriptorPool};
    return SLANG_OK;
}

void DescriptorSetAllocator::free(VulkanDescriptorSet set)
{
    if (set.handle && m_api->vkFreeDescriptorSets(m_api->m_device, set.pool, 1, &set.handle) == VK_SUCCESS)
        --m_liveSets;
}

Result DescriptorSetAllocator::reset()
{
    for (auto pool : pools)
    {
        const VkResult result = m_api->vkResetDescriptorPool(m_api->m_device, pool, 0);
        if (result != VK_SUCCESS)
        {
            m_resetResult = descriptorAllocationResult(result);
            return m_resetResult;
        }
    }
    m_liveSets = 0;
    m_resetResult = SLANG_OK;
    return SLANG_OK;
}

void DescriptorSetAllocator::close()
{
    for (auto pool : pools)
        m_api->vkDestroyDescriptorPool(m_api->m_device, pool, nullptr);
    pools.clear();
    m_liveSets = 0;
    m_resetResult = SLANG_OK;
}

} // namespace rhi::vk
