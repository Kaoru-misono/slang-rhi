#include "vk-binding-set.h"
#include "vk-acceleration-structure.h"
#include "vk-bindless-descriptor-set.h"
#include "vk-buffer.h"
#include "vk-device.h"
#include "vk-sampler.h"
#include "vk-texture.h"

#include "../command-buffer.h"
#include "../shader.h"

#include <algorithm>
#include <vector>

namespace rhi::vk {

static VkDescriptorType translateBindingKind(BindingKind kind)
{
    switch (kind)
    {
    case BindingKind::SamplerState:
    case BindingKind::SamplerComparisonState:
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    case BindingKind::Texture1D:
    case BindingKind::Texture2D:
    case BindingKind::Texture2DArray:
    case BindingKind::Texture3D:
    case BindingKind::TextureCube:
    case BindingKind::TextureCubeArray:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case BindingKind::RWTexture1D:
    case BindingKind::RWTexture2D:
    case BindingKind::RWTexture2DArray:
    case BindingKind::RWTexture3D:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case BindingKind::Buffer:
        return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    case BindingKind::RWBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    case BindingKind::StructuredBuffer:
    case BindingKind::RWStructuredBuffer:
    case BindingKind::ByteAddressBuffer:
    case BindingKind::RWByteAddressBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case BindingKind::RaytracingAccelerationStructure:
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    }
    SLANG_RHI_UNREACHABLE("Invalid binding kind");
    return VK_DESCRIPTOR_TYPE_SAMPLER;
}

void BindingSetDescriptorPool::initialize(DeviceImpl* device)
{
    m_device = device;
}

void BindingSetDescriptorPool::release()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (VkDescriptorPool pool : m_pools)
        m_device->m_api.vkDestroyDescriptorPool(m_device->m_api.m_device, pool, nullptr);
    m_pools.clear();
}

Result BindingSetDescriptorPool::createPool(std::span<const VkDescriptorPoolSize> poolSizes, VkDescriptorPool* outPool)
{
    std::vector<VkDescriptorPoolSize> sizes(poolSizes.begin(), poolSizes.end());
    for (auto& size : sizes)
        size.descriptorCount = std::max(1u, size.descriptorCount * 256u);
    if (sizes.empty())
        sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER, 1});
    VkDescriptorPoolCreateInfo createInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    createInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    createInfo.maxSets = 256;
    createInfo.poolSizeCount = uint32_t(sizes.size());
    createInfo.pPoolSizes = sizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    auto& api = m_device->m_api;
    if (api.vkCreateDescriptorPool(api.m_device, &createInfo, nullptr, &pool) != VK_SUCCESS)
    {
        m_device->printError("Failed to create a Vulkan descriptor pool.");
        return SLANG_FAIL;
    }
    m_pools.push_back(pool);
    *outPool = pool;
    return SLANG_OK;
}

Result BindingSetDescriptorPool::allocate(
    VkDescriptorSetLayout setLayout,
    std::span<const VkDescriptorPoolSize> poolSizes,
    VkDescriptorSet* outDescriptorSet,
    VkDescriptorPool* outPool
)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& api = m_device->m_api;
    VkDescriptorSetAllocateInfo allocateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &setLayout;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    for (auto it = m_pools.rbegin(); it != m_pools.rend(); ++it)
    {
        allocateInfo.descriptorPool = *it;
        if (api.vkAllocateDescriptorSets(api.m_device, &allocateInfo, &descriptorSet) == VK_SUCCESS)
        {
            *outDescriptorSet = descriptorSet;
            *outPool = *it;
            return SLANG_OK;
        }
    }
    SLANG_RETURN_ON_FAIL(createPool(poolSizes, &allocateInfo.descriptorPool));
    if (api.vkAllocateDescriptorSets(api.m_device, &allocateInfo, &descriptorSet) != VK_SUCCESS)
    {
        m_device->printError("Failed to allocate a Vulkan descriptor set for a binding set.");
        return SLANG_E_OUT_OF_MEMORY;
    }
    *outDescriptorSet = descriptorSet;
    *outPool = allocateInfo.descriptorPool;
    return SLANG_OK;
}

void BindingSetDescriptorPool::free(VkDescriptorSet descriptorSet, VkDescriptorPool pool)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (descriptorSet != VK_NULL_HANDLE && pool != VK_NULL_HANDLE)
        m_device->m_api.vkFreeDescriptorSets(m_device->m_api.m_device, pool, 1, &descriptorSet);
}

BindingSetLayoutImpl::BindingSetLayoutImpl(Device* device, const BindingSetLayoutDesc& desc)
    : BindingSetLayout(device, desc)
{
}

BindingSetLayoutImpl::~BindingSetLayoutImpl()
{
    auto& api = getDevice<DeviceImpl>()->m_api;
    if (m_descriptorSetLayout != VK_NULL_HANDLE)
        api.vkDestroyDescriptorSetLayout(api.m_device, m_descriptorSetLayout, nullptr);
}

void BindingSetLayoutImpl::deleteThis()
{
    // A layout a pipeline layout or an in-flight recording still names must outlive the submission.
    getDevice<DeviceImpl>()->deferDelete(this);
}

Result BindingSetLayoutImpl::initNative()
{
    auto* device = getDevice<DeviceImpl>();
    const char* label = m_desc.label ? m_desc.label : "<unnamed>";
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(m_desc.entryCount);
    for (uint32_t i = 0; i < m_desc.entryCount; ++i)
    {
        const auto& entry = m_desc.entries[i];
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = i;
        binding.descriptorType = translateBindingKind(entry.kind);
        binding.descriptorCount = entry.count;
        binding.stageFlags = VK_SHADER_STAGE_ALL;
        binding.pImmutableSamplers = nullptr;
        bindings.push_back(binding);
        auto found = std::find_if(
            m_poolSizes.begin(),
            m_poolSizes.end(),
            [&](const VkDescriptorPoolSize& size)
            {
                return size.type == binding.descriptorType;
            }
        );
        if (found != m_poolSizes.end())
            found->descriptorCount += entry.count;
        else
            m_poolSizes.push_back({binding.descriptorType, entry.count});
    }
    VkDescriptorSetLayoutCreateInfo createInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    createInfo.bindingCount = uint32_t(bindings.size());
    createInfo.pBindings = bindings.data();
    auto& api = device->m_api;
    if (api.vkCreateDescriptorSetLayout(api.m_device, &createInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS)
    {
        device->printError("Binding set layout '%s': failed to create the Vulkan descriptor set layout.", label);
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

BindingSetImpl::BindingSetImpl(Device* device, BindingSetLayoutImpl* layout)
    : BindingSet(device, layout)
{
}

BindingSetImpl::~BindingSetImpl()
{
    getDevice<DeviceImpl>()->m_bindingSetDescriptorPool.free(m_descriptorSet, m_descriptorPool);
}

void BindingSetImpl::deleteThis()
{
    // A set bound by an in-flight command buffer and its resource references must outlive the submission.
    getDevice<DeviceImpl>()->deferDelete(this);
}

Result BindingSetImpl::initNative()
{
    auto* device = getDevice<DeviceImpl>();
    auto* layout = getLayoutImpl();
    const char* label = m_label ? m_label : "<unnamed>";
    SLANG_RETURN_ON_FAIL(
        device->m_bindingSetDescriptorPool
            .allocate(layout->m_descriptorSetLayout, layout->m_poolSizes, &m_descriptorSet, &m_descriptorPool)
    );

    // Every info list is reserved for the whole set up front: the writes point into them, so a single
    // reallocation would leave dangling pointers behind.
    const uint32_t bindingCount = layout->getBindingCount();
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkBufferView> bufferViews;
    std::vector<VkWriteDescriptorSetAccelerationStructureKHR> accelerationStructureInfos;
    writes.reserve(bindingCount);
    imageInfos.reserve(bindingCount);
    bufferInfos.reserve(bindingCount);
    bufferViews.reserve(bindingCount);
    accelerationStructureInfos.reserve(bindingCount);
    for (uint32_t slot = 0; slot < layout->m_desc.entryCount; ++slot)
    {
        const auto& entry = layout->m_desc.entries[slot];
        for (uint32_t arrayIndex = 0; arrayIndex < entry.count; ++arrayIndex)
        {
            const BindingSetBinding& binding = m_bindings[layout->getBindingOffset(slot) + arrayIndex];
            VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = m_descriptorSet;
            write.dstBinding = slot;
            write.dstArrayElement = arrayIndex;
            write.descriptorCount = 1;
            write.descriptorType = translateBindingKind(entry.kind);
            switch (write.descriptorType)
            {
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                imageInfos.push_back({
                    checked_cast<SamplerImpl*>(binding.resource.get())->m_sampler,
                    VK_NULL_HANDLE,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                });
                write.pImageInfo = &imageInfos.back();
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                imageInfos.push_back({
                    VK_NULL_HANDLE,
                    checked_cast<TextureViewImpl*>(binding.resource.get())->getView().imageView,
                    write.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                                             : VK_IMAGE_LAYOUT_GENERAL,
                });
                write.pImageInfo = &imageInfos.back();
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            {
                auto* buffer = checked_cast<BufferImpl*>(binding.resource.get());
                if (buffer->m_desc.format == Format::Undefined)
                {
                    device->printError(
                        "Binding set '%s', slot %u: a typed buffer binding needs a buffer created with a format.",
                        label,
                        slot
                    );
                    return SLANG_E_INVALID_ARG;
                }
                bufferViews.push_back(buffer->getView(buffer->m_desc.format, binding.bufferRange));
                write.pTexelBufferView = &bufferViews.back();
                break;
            }
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            {
                auto* buffer = checked_cast<BufferImpl*>(binding.resource.get());
                bufferInfos.push_back({
                    buffer->m_buffer.m_buffer,
                    binding.bufferRange.offset,
                    binding.bufferRange.size,
                });
                write.pBufferInfo = &bufferInfos.back();
                break;
            }
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            {
                VkWriteDescriptorSetAccelerationStructureKHR info = {
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR
                };
                info.accelerationStructureCount = 1;
                info.pAccelerationStructures =
                    &checked_cast<AccelerationStructureImpl*>(binding.resource.get())->m_vkHandle;
                accelerationStructureInfos.push_back(info);
                write.pNext = &accelerationStructureInfos.back();
                break;
            }
            default:
                SLANG_RHI_UNREACHABLE("Invalid binding kind");
            }
            writes.push_back(write);
        }
    }
    device->m_api.vkUpdateDescriptorSets(device->m_api.m_device, uint32_t(writes.size()), writes.data(), 0, nullptr);
    return SLANG_OK;
}

PipelineLayoutImpl::PipelineLayoutImpl(Device* device, const PipelineLayoutDesc& desc)
    : PipelineLayout(device, desc)
{
}

PipelineLayoutImpl::~PipelineLayoutImpl()
{
    auto& api = getDevice<DeviceImpl>()->m_api;
    if (m_pipelineLayout != VK_NULL_HANDLE)
        api.vkDestroyPipelineLayout(api.m_device, m_pipelineLayout, nullptr);
    if (m_constantsSetLayout != VK_NULL_HANDLE)
        api.vkDestroyDescriptorSetLayout(api.m_device, m_constantsSetLayout, nullptr);
    if (m_emptySetLayout != VK_NULL_HANDLE)
        api.vkDestroyDescriptorSetLayout(api.m_device, m_emptySetLayout, nullptr);
}

void PipelineLayoutImpl::deleteThis()
{
    // An in-flight recording binds descriptor sets through this layout.
    getDevice<DeviceImpl>()->deferDelete(this);
}

Result PipelineLayoutImpl::initNative()
{
    auto* device = getDevice<DeviceImpl>();
    auto& api = device->m_api;
    auto* program = m_program.get();
    slang::ProgramLayout* programLayout = program->linkedProgram->getLayout();
    const char* label = m_desc.label ? m_desc.label : "<unnamed>";

    using Kind = slang::TypeReflection::Kind;
    using Category = slang::ParameterCategory;
    auto collectSet = [&](auto&& self, slang::VariableLayoutReflection* varLayout, uint32_t parentSetIndex) -> void
    {
        uint32_t setIndex =
            parentSetIndex + uint32_t(varLayout->getOffset(SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE));
        m_setIndices.push_back(setIndex);
        auto* elementTypeLayout = varLayout->getTypeLayout()->getElementTypeLayout();
        for (unsigned i = 0; i < elementTypeLayout->getFieldCount(); ++i)
        {
            auto* field = elementTypeLayout->getFieldByIndex(i);
            if (field->getTypeLayout()->getKind() == Kind::ParameterBlock)
                self(self, field, setIndex);
        }
    };
    // Inline constants reach the shader as push constants, so only a Buffered block occupies a set.
    auto collectConstants = [&](slang::VariableLayoutReflection* varLayout)
    {
        if (!varLayout || m_constantsProfile != ConstantsProfile::Buffered)
            return;
        m_constantsSetIndex = uint32_t(varLayout->getBindingSpace(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
        m_constantsBinding = uint32_t(varLayout->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
    };
    for (unsigned i = 0; i < programLayout->getParameterCount(); ++i)
    {
        auto* param = programLayout->getParameterByIndex(i);
        if (param->getTypeLayout()->getKind() == Kind::ParameterBlock)
            collectSet(collectSet, param, 0);
        else if (param->getTypeLayout()->getKind() == Kind::ConstantBuffer)
            collectConstants(param);
    }
    for (unsigned i = 0; i < programLayout->getEntryPointCount(); ++i)
    {
        auto* entryPoint = programLayout->getEntryPointByIndex(i);
        bool hasUniformBlock = false;
        for (unsigned j = 0; j < entryPoint->getParameterCount(); ++j)
        {
            auto* param = entryPoint->getParameterByIndex(j);
            if (param->getTypeLayout()->getKind() == Kind::ParameterBlock)
            {
                collectSet(collectSet, param, 0);
            }
            else if (param->getCategoryCount() == 1 && param->getCategoryByIndex(0) == Category::Uniform)
            {
                if (!hasUniformBlock)
                {
                    collectConstants(entryPoint->getVarLayout());
                    hasUniformBlock = true;
                }
            }
        }
    }
    if (m_setIndices.size() != m_setLayouts.size())
    {
        device->printError("Pipeline layout '%s': reflected parameter block count does not match set count.", label);
        return SLANG_FAIL;
    }

    uint32_t descriptorSetCount = 0;
    uint64_t samplers = 0;
    uint64_t sampledImages = 0;
    uint64_t storageImages = 0;
    uint64_t storageBuffers = 0;
    uint64_t uniformBuffers = 0;
    for (uint32_t i = 0; i < uint32_t(m_setLayouts.size()); ++i)
    {
        auto* setLayout = getSetLayoutImpl(i);
        descriptorSetCount = std::max(descriptorSetCount, m_setIndices[i] + 1);
        for (uint32_t j = 0; j < setLayout->m_desc.entryCount; ++j)
        {
            const auto& entry = setLayout->m_desc.entries[j];
            switch (translateBindingKind(entry.kind))
            {
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                samplers += entry.count;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                sampledImages += entry.count;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                storageImages += entry.count;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                storageBuffers += entry.count;
                break;
            default:
                break;
            }
        }
    }
    if (m_constantsProfile == ConstantsProfile::Buffered)
    {
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = m_constantsBinding;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_ALL;
        VkDescriptorSetLayoutCreateInfo createInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        createInfo.bindingCount = 1;
        createInfo.pBindings = &binding;
        if (api.vkCreateDescriptorSetLayout(api.m_device, &createInfo, nullptr, &m_constantsSetLayout) != VK_SUCCESS)
        {
            device->printError("Pipeline layout '%s': failed to create the constants descriptor set layout.", label);
            return SLANG_FAIL;
        }
        descriptorSetCount = std::max(descriptorSetCount, m_constantsSetIndex + 1);
        uniformBuffers = 1;
    }
    if (m_bindless && device->m_bindlessDescriptorSet)
    {
        // Slang reserves a space for the bindless heap; without one it follows every other set.
        const SlangInt reservedSpace = programLayout->getBindlessSpaceIndex();
        m_bindlessSetIndex = reservedSpace >= 0 ? uint32_t(reservedSpace) : descriptorSetCount;
        descriptorSetCount = std::max(descriptorSetCount, m_bindlessSetIndex + 1);
    }
    m_descriptorSetCount = descriptorSetCount;
    std::vector<VkDescriptorSetLayout> setLayouts(descriptorSetCount, VK_NULL_HANDLE);
    auto placeSetLayout = [&](uint32_t setIndex, VkDescriptorSetLayout setLayout) -> Result
    {
        if (setLayouts[setIndex] != VK_NULL_HANDLE)
        {
            device->printError("Pipeline layout '%s': two parameters reflect to descriptor set %u.", label, setIndex);
            return SLANG_FAIL;
        }
        setLayouts[setIndex] = setLayout;
        return SLANG_OK;
    };
    for (size_t i = 0; i < m_setIndices.size(); ++i)
        SLANG_RETURN_ON_FAIL(placeSetLayout(m_setIndices[i], getSetLayoutImpl(uint32_t(i))->m_descriptorSetLayout));
    if (m_constantsProfile == ConstantsProfile::Buffered)
        SLANG_RETURN_ON_FAIL(placeSetLayout(m_constantsSetIndex, m_constantsSetLayout));
    if (m_bindlessSetIndex != kInvalidDescriptorSetIndex)
        SLANG_RETURN_ON_FAIL(
            placeSetLayout(m_bindlessSetIndex, device->m_bindlessDescriptorSet->m_descriptorSetLayout)
        );

    for (auto& setLayout : setLayouts)
    {
        if (setLayout != VK_NULL_HANDLE)
            continue;
        if (m_emptySetLayout == VK_NULL_HANDLE)
        {
            VkDescriptorSetLayoutCreateInfo createInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            if (api.vkCreateDescriptorSetLayout(api.m_device, &createInfo, nullptr, &m_emptySetLayout) != VK_SUCCESS)
            {
                device->printError("Pipeline layout '%s': failed to create the empty descriptor set layout.", label);
                return SLANG_FAIL;
            }
        }
        setLayout = m_emptySetLayout;
    }

    const auto& limits = api.m_deviceProperties.limits;
    if (m_constantsProfile == ConstantsProfile::Inline)
    {
        if (m_constantsSize > limits.maxPushConstantsSize)
        {
            device->printError(
                "Pipeline layout '%s': inline constants size (%u) exceeds maxPushConstantsSize (%u).",
                label,
                m_constantsSize,
                limits.maxPushConstantsSize
            );
            return SLANG_FAIL;
        }
    }
    auto validateDescriptorCount = [&](uint64_t count, uint32_t limit, const char* limitName) -> Result
    {
        if (count > limit)
        {
            device->printError(
                "Pipeline layout '%s': descriptor count (%llu) exceeds %s (%u).",
                label,
                static_cast<unsigned long long>(count),
                limitName,
                limit
            );
            return SLANG_FAIL;
        }
        return SLANG_OK;
    };
    SLANG_RETURN_ON_FAIL(
        validateDescriptorCount(samplers, limits.maxPerStageDescriptorSamplers, "maxPerStageDescriptorSamplers")
    );
    SLANG_RETURN_ON_FAIL(validateDescriptorCount(
        sampledImages,
        limits.maxPerStageDescriptorSampledImages,
        "maxPerStageDescriptorSampledImages"
    ));
    SLANG_RETURN_ON_FAIL(validateDescriptorCount(
        storageImages,
        limits.maxPerStageDescriptorStorageImages,
        "maxPerStageDescriptorStorageImages"
    ));
    SLANG_RETURN_ON_FAIL(validateDescriptorCount(
        storageBuffers,
        limits.maxPerStageDescriptorStorageBuffers,
        "maxPerStageDescriptorStorageBuffers"
    ));
    SLANG_RETURN_ON_FAIL(validateDescriptorCount(
        uniformBuffers,
        limits.maxPerStageDescriptorUniformBuffers,
        "maxPerStageDescriptorUniformBuffers"
    ));
    VkPushConstantRange pushConstantRange = {VK_SHADER_STAGE_ALL, 0, m_constantsSize};
    VkPipelineLayoutCreateInfo createInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    createInfo.setLayoutCount = uint32_t(setLayouts.size());
    createInfo.pSetLayouts = setLayouts.data();
    if (m_constantsProfile == ConstantsProfile::Inline)
    {
        createInfo.pushConstantRangeCount = 1;
        createInfo.pPushConstantRanges = &pushConstantRange;
    }
    if (api.vkCreatePipelineLayout(api.m_device, &createInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
    {
        device->printError("Pipeline layout '%s': failed to create the Vulkan pipeline layout.", label);
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

Result DeviceImpl::createBindingSetLayoutImpl(const BindingSetLayoutDesc& desc, RefPtr<BindingSetLayout>& outLayout)
{
    outLayout = new BindingSetLayoutImpl(this, desc);
    return SLANG_OK;
}

Result DeviceImpl::createBindingSetImpl(BindingSetLayout* layout, RefPtr<BindingSet>& outBindingSet)
{
    outBindingSet = new BindingSetImpl(this, checked_cast<BindingSetLayoutImpl*>(layout));
    return SLANG_OK;
}

Result DeviceImpl::createPipelineLayoutImpl(const PipelineLayoutDesc& desc, RefPtr<PipelineLayout>& outLayout)
{
    outLayout = new PipelineLayoutImpl(this, desc);
    return SLANG_OK;
}

uint32_t DeviceImpl::getInlineConstantsSizeLimit() const
{
    return std::min(uint32_t(kMaxInlineConstantsSize), m_api.m_deviceProperties.limits.maxPushConstantsSize);
}

} // namespace rhi::vk
