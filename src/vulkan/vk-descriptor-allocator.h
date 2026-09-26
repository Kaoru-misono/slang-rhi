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

    void init(VulkanApi* api)
    {
        m_api = api;
    }
    Result allocate(VkDescriptorSetLayout layout, VulkanDescriptorSet* outSet);
    void free(VulkanDescriptorSet set);
    Result reset();
    void close();

private:
    Result newPool(VkDescriptorPool* outPool);

    Result m_resetResult = SLANG_OK;
};

} // namespace rhi::vk
