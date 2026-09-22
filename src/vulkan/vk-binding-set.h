#pragma once

#include "vk-base.h"

#include "../binding-set.h"
#include "../pipeline-layout.h"

#include "core/short_vector.h"

#include <mutex>
#include <span>
#include <vector>

namespace rhi::vk {

/// Descriptor set index of a set a pipeline layout does not declare.
static constexpr uint32_t kInvalidDescriptorSetIndex = 0xffffffff;

/// Growable pool list for the descriptor set a BindingSet owns for its whole lifetime. Kept
/// apart from the per-command-buffer DescriptorSetAllocator, whose pools are reset when a
/// recording is reused.
class BindingSetDescriptorPool
{
public:
    void initialize(DeviceImpl* device);
    void release();

    Result allocate(
        VkDescriptorSetLayout setLayout,
        std::span<const VkDescriptorPoolSize> poolSizes,
        VkDescriptorSet* outDescriptorSet,
        VkDescriptorPool* outPool
    );
    void free(VkDescriptorSet descriptorSet, VkDescriptorPool pool);

private:
    /// A set may only be allocated from a pool that reserved every descriptor type its layout binds,
    /// so each pool carries the sizes it was created with.
    struct Pool
    {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::vector<VkDescriptorPoolSize> sizes;
    };

    Result createPool(std::span<const VkDescriptorPoolSize> poolSizes, VkDescriptorPool* outPool);

    DeviceImpl* m_device = nullptr;
    std::mutex m_mutex;
    std::vector<Pool> m_pools;
};

class BindingSetLayoutImpl : public BindingSetLayout
{
public:
    BindingSetLayoutImpl(Device* device, const BindingSetLayoutDesc& desc);
    ~BindingSetLayoutImpl();
    virtual void deleteThis() override;

    /// Creates the VkDescriptorSetLayout, where binding i is entry i, visible to every stage.
    Result initNative();

    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    /// Descriptor counts by type, which size the pool a set of this layout is allocated from.
    std::vector<VkDescriptorPoolSize> m_poolSizes;
};

class BindingSetImpl : public BindingSet
{
public:
    BindingSetImpl(Device* device, BindingSetLayoutImpl* layout);
    ~BindingSetImpl();
    virtual void deleteThis() override;

    /// Allocates the persistent descriptor set and writes every binding once.
    Result initNative();

    BindingSetLayoutImpl* getLayoutImpl() const { return static_cast<BindingSetLayoutImpl*>(m_layout.get()); }

    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
};

class PipelineLayoutImpl : public PipelineLayout
{
public:
    PipelineLayoutImpl(Device* device, const PipelineLayoutDesc& desc);
    ~PipelineLayoutImpl();
    virtual void deleteThis() override;

    /// Places the binding sets, the execution constants and the bindless table at the descriptor
    /// set indices the program reflects them to, and creates the VkPipelineLayout.
    Result initNative();

    BindingSetLayoutImpl* getSetLayoutImpl(uint32_t index) const
    {
        return static_cast<BindingSetLayoutImpl*>(getSetLayout(index));
    }

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    /// Descriptor set index of m_setLayouts[i].
    short_vector<uint32_t, 8> m_setIndices;
    /// Number of descriptor sets the layout declares, counting the ones no parameter occupies.
    uint32_t m_descriptorSetCount = 0;
    uint32_t m_constantsSetIndex = kInvalidDescriptorSetIndex;
    uint32_t m_constantsBinding = 0;
    /// Owned; the single uniform buffer binding a Buffered constants block is written to.
    VkDescriptorSetLayout m_constantsSetLayout = VK_NULL_HANDLE;
    uint32_t m_bindlessSetIndex = kInvalidDescriptorSetIndex;
    /// Owned; bound at the set indices no parameter occupies, which Vulkan still requires a set for.
    VkDescriptorSetLayout m_emptySetLayout = VK_NULL_HANDLE;
};

} // namespace rhi::vk
