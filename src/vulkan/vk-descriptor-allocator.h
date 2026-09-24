#pragma once

#include "vk-api.h"
#include "core/common.h"

#include <vector>

namespace rhi::vk {

struct VulkanDescriptorSet
{
    VkDescriptorSet handle = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
};

class DescriptorSetAllocator
{
public:
    std::vector<VkDescriptorPool> pools;
    const VulkanApi* m_api = nullptr;

    void init(VulkanApi* api, uint32_t maxSets = 0)
    {
        m_api = api;
        m_maxSets = maxSets;
    }
    Result allocate(VkDescriptorSetLayout layout, VulkanDescriptorSet* outSet);
    void free(VulkanDescriptorSet set);
    Result reset();
    void close();

private:
    Result newPool(VkDescriptorPool* outPool);

    uint32_t m_maxSets = 0;
    uint64_t m_liveSets = 0;
    Result m_resetResult = SLANG_OK;
};

} // namespace rhi::vk
