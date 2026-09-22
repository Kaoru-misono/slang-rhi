#include "testing.h"
#include "flat-binding-test-data.h"
#include "constant-layout.h"

#if SLANG_RHI_ENABLE_VULKAN
#include "vulkan/vk-api.h"

using namespace rhi;
using namespace rhi::testing;

namespace {
using NativeParams = flat::Params;

struct NativeVulkanConstants
{
    vk::VulkanModule module;
    vk::VulkanApi api{};
    VkDescriptorSetLayout setLayouts[2]{};
    VkDescriptorSet sets[2]{};
    VkDescriptorPool pool{};
    VkPipelineLayout layout{};
    VkPipeline compute{};
    VkPipeline graphics{};
    std::vector<VkShaderModule> shaders;
    VkImageView target{};
    bool inlineConstants{};

    ~NativeVulkanConstants()
    {
        if (compute)
            api.vkDestroyPipeline(api.m_device, compute, nullptr);
        if (graphics)
            api.vkDestroyPipeline(api.m_device, graphics, nullptr);
        for (auto shader : shaders)
            api.vkDestroyShaderModule(api.m_device, shader, nullptr);
        if (layout)
            api.vkDestroyPipelineLayout(api.m_device, layout, nullptr);
        if (pool)
            api.vkDestroyDescriptorPool(api.m_device, pool, nullptr);
        for (auto setLayout : setLayouts)
            if (setLayout)
                api.vkDestroyDescriptorSetLayout(api.m_device, setLayout, nullptr);
    }

    void init(IDevice* device, bool useInline)
    {
        inlineConstants = useInline;
        DeviceNativeHandles handles{};
        REQUIRE_CALL(device->getNativeDeviceHandles(&handles));
        REQUIRE_CALL(module.init());
        REQUIRE_CALL(api.initGlobalProcs(module));
        REQUIRE_CALL(api.initInstanceProcs(reinterpret_cast<VkInstance>(handles.handles[0].value)));
        REQUIRE_CALL(api.initPhysicalDevice(reinterpret_cast<VkPhysicalDevice>(handles.handles[1].value)));
        REQUIRE_CALL(api.initDeviceProcs(reinterpret_cast<VkDevice>(handles.handles[2].value)));
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr};
        bindings[1] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL, nullptr};
        for (uint32_t i = 0; i < (inlineConstants ? 1u : 2u); ++i)
        {
            VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            desc.bindingCount = 1;
            desc.pBindings = &bindings[i];
            REQUIRE(api.vkCreateDescriptorSetLayout(api.m_device, &desc, nullptr, &setLayouts[i]) == VK_SUCCESS);
        }
        VkPushConstantRange push{VK_SHADER_STAGE_ALL, 0, sizeof(NativeParams)};
        VkPipelineLayoutCreateInfo layoutDesc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutDesc.setLayoutCount = inlineConstants ? 1 : 2;
        layoutDesc.pSetLayouts = setLayouts;
        layoutDesc.pushConstantRangeCount = inlineConstants ? 1 : 0;
        layoutDesc.pPushConstantRanges = inlineConstants ? &push : nullptr;
        REQUIRE(api.vkCreatePipelineLayout(api.m_device, &layoutDesc, nullptr, &layout) == VK_SUCCESS);
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1}
        };
        VkDescriptorPoolCreateInfo poolDesc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolDesc.maxSets = layoutDesc.setLayoutCount;
        poolDesc.poolSizeCount = layoutDesc.setLayoutCount;
        poolDesc.pPoolSizes = sizes;
        REQUIRE(api.vkCreateDescriptorPool(api.m_device, &poolDesc, nullptr, &pool) == VK_SUCCESS);
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool;
        allocation.descriptorSetCount = layoutDesc.setLayoutCount;
        allocation.pSetLayouts = setLayouts;
        REQUIRE(api.vkAllocateDescriptorSets(api.m_device, &allocation, sets) == VK_SUCCESS);
    }

    VkShaderModule compile(slang::IComponentType* program, uint32_t entry)
    {
        ComPtr<slang::IBlob> code, diagnostics;
        auto result = program->getEntryPointCode(entry, 0, code.writeRef(), diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        REQUIRE_CALL(result);
        VkShaderModuleCreateInfo desc{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        desc.codeSize = code->getBufferSize();
        desc.pCode = static_cast<const uint32_t*>(code->getBufferPointer());
        VkShaderModule shader{};
        REQUIRE(api.vkCreateShaderModule(api.m_device, &desc, nullptr, &shader) == VK_SUCCESS);
        shaders.emplace_back(shader);
        return shader;
    }

    void createPipelines(slang::IComponentType* program)
    {
        VkComputePipelineCreateInfo computeDesc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        computeDesc.layout = layout;
        computeDesc.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        computeDesc.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeDesc.stage.module = compile(program, 0);
        computeDesc.stage.pName = "main";
        REQUIRE(
            api.vkCreateComputePipelines(api.m_device, VK_NULL_HANDLE, 1, &computeDesc, nullptr, &compute) == VK_SUCCESS
        );

        VkPipelineShaderStageCreateInfo stages[2]{};
        for (uint32_t i = 0; i < 2; ++i)
        {
            stages[i] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stages[i].stage = i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[i].module = compile(program, i + 1);
            stages[i].pName = "main";
        }
        VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.lineWidth = 1.f;
        VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState color{};
        color.colorWriteMask = 15;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &color;
        VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamicStates;
        VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &format;
        VkGraphicsPipelineCreateInfo desc{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        desc.pNext = &rendering;
        desc.stageCount = 2;
        desc.pStages = stages;
        desc.pVertexInputState = &vertex;
        desc.pInputAssemblyState = &assembly;
        desc.pViewportState = &viewport;
        desc.pRasterizationState = &raster;
        desc.pMultisampleState = &samples;
        desc.pColorBlendState = &blend;
        desc.pDynamicState = &dynamic;
        desc.layout = layout;
        REQUIRE(
            api.vkCreateGraphicsPipelines(api.m_device, VK_NULL_HANDLE, 1, &desc, nullptr, &graphics) == VK_SUCCESS
        );
    }

    struct Snapshot
    {
        NativeParams params;
        uint32_t offset;
        bool render;
    };

    static void SLANG_MCALL record(const ExecuteCallbackContext* context, void* object, const void* data, Size size)
    {
        REQUIRE(size == sizeof(Snapshot));
        Snapshot snapshot;
        std::memcpy(&snapshot, data, sizeof(snapshot));
        auto& self = *static_cast<NativeVulkanConstants*>(object);
        auto& api = self.api;
        auto commands = reinterpret_cast<VkCommandBuffer>(context->nativeHandle.value);
        const auto point = snapshot.render ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE;
        api.vkCmdBindPipeline(commands, point, snapshot.render ? self.graphics : self.compute);
        api.vkCmdBindDescriptorSets(
            commands,
            point,
            self.layout,
            0,
            self.inlineConstants ? 1 : 2,
            self.sets,
            self.inlineConstants ? 0 : 1,
            self.inlineConstants ? nullptr : &snapshot.offset
        );
        if (self.inlineConstants)
            api.vkCmdPushConstants(
                commands,
                self.layout,
                VK_SHADER_STAGE_ALL,
                0,
                sizeof(snapshot.params),
                &snapshot.params
            );
        if (!snapshot.render)
        {
            api.vkCmdDispatch(commands, 2, 1, 1);
            return;
        }
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = self.target;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea.extent = {2, 2};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color;
        api.vkCmdBeginRenderingKHR(commands, &rendering);
        VkViewport viewport{0.f, 0.f, 2.f, 2.f, 0.f, 1.f};
        VkRect2D scissor{{0, 0}, {2, 2}};
        api.vkCmdSetViewport(commands, 0, 1, &viewport);
        api.vkCmdSetScissor(commands, 0, 1, &scissor);
        api.vkCmdDraw(commands, 3, 1, 0, 0);
        api.vkCmdEndRenderingKHR(commands);
    }
};

ComPtr<slang::IComponentType> compileNativeProgram(
    IDevice* device,
    bool inlineConstants,
    ComPtr<slang::ISession>& session
)
{
    auto* global = device->getSlangSession()->getGlobalSession();
    slang::TargetDesc target{};
    target.format = SLANG_SPIRV;
    target.forceGLSLScalarBufferLayout = true;
    slang::PreprocessorMacroDesc macros[] = {{"NATIVE_LAYOUT", "1"}, {"INLINE_CONSTANTS", inlineConstants ? "1" : "0"}};
    auto paths = getSlangSearchPaths();
    slang::SessionDesc desc{};
    desc.targets = &target;
    desc.targetCount = 1;
    desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
    desc.searchPaths = paths.data();
    desc.searchPathCount = paths.size();
    desc.preprocessorMacros = macros;
    desc.preprocessorMacroCount = 2;
    REQUIRE_CALL(global->createSession(desc, session.writeRef()));
    ComPtr<slang::IBlob> diagnostics;
    auto module = session->loadModule("test-flat-binding-abi", diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    REQUIRE(module);
    ComPtr<slang::IEntryPoint> entries[3];
    const char* names[] = {"csMain", "vsMain", "psMain"};
    slang::IComponentType* components[4] = {module};
    for (uint32_t i = 0; i < 3; ++i)
    {
        REQUIRE_CALL(module->findEntryPointByName(names[i], entries[i].writeRef()));
        components[i + 1] = entries[i];
    }
    ComPtr<slang::IComponentType> composite, linked;
    REQUIRE_CALL(session->createCompositeComponentType(components, 4, composite.writeRef(), diagnostics.writeRef()));
    REQUIRE_CALL(composite->link(linked.writeRef(), diagnostics.writeRef()));
    auto* globals = linked->getLayout()->getGlobalParamsTypeLayout();
    auto* parameters = globals->getFieldByIndex(globals->findFieldIndexByName("parameters"));
    auto category =
        inlineConstants ? slang::ParameterCategory::PushConstantBuffer : slang::ParameterCategory::DescriptorTableSlot;
    CHECK(parameters->getOffset(category) == 0);
    CHECK(parameters->getBindingSpace(category) == (inlineConstants ? 0u : 1u));
    CHECK(parameters->getTypeLayout()->getElementTypeLayout()->getSize() == sizeof(NativeParams));
    std::string diagnostic;
    REQUIRE_CALL(validateConstantLayout(parameters->getTypeLayout()->getElementTypeLayout(), diagnostic));
    return linked;
}

void runNativeVulkanConstants(IDevice* device, bool inlineConstants)
{
    NativeVulkanConstants fixture;
    fixture.init(device, inlineConstants);
    ComPtr<slang::ISession> session;
    auto program = compileNativeProgram(device, inlineConstants, session);
    fixture.createPipelines(program);
    const NativeParams first{
        {1, 2, 3},
        2,
        {{1, 2, 3, 4}, {-2, 1, 0, 3}},
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}},
        {17, {5, 7, 11}}
    };
    auto second = first;
    second.scale = 3;
    const auto stride = std::max<VkDeviceSize>(
        sizeof(NativeParams),
        fixture.api.m_deviceProperties.limits.minUniformBufferOffsetAlignment
    );
    REQUIRE(stride <= UINT32_MAX);
    REQUIRE(sizeof(NativeParams) <= fixture.api.m_deviceProperties.limits.maxUniformBufferRange);
    std::vector<std::byte> storage(size_t(stride + sizeof(NativeParams)));
    std::memcpy(storage.data(), &first, sizeof(first));
    std::memcpy(storage.data() + stride, &second, sizeof(second));
    BufferDesc constantsDesc{};
    constantsDesc.size = storage.size();
    constantsDesc.usage = BufferUsage::ConstantBuffer;
    constantsDesc.memoryType = MemoryType::Upload;
    constantsDesc.defaultState = ResourceState::ConstantBuffer;
    auto constants = device->createBuffer(constantsDesc, storage.data());
    REQUIRE(constants);
    BufferDesc outputDesc{};
    outputDesc.size = 32;
    outputDesc.elementSize = 16;
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    outputDesc.defaultState = ResourceState::UnorderedAccess;
    auto output = device->createBuffer(outputDesc);
    REQUIRE(output);
    NativeHandle nativeConstants, nativeOutput;
    REQUIRE_CALL(constants->getNativeHandle(&nativeConstants));
    REQUIRE_CALL(output->getNativeHandle(&nativeOutput));
    VkDescriptorBufferInfo infos[] = {
        {reinterpret_cast<VkBuffer>(nativeOutput.value), 0, 32},
        {reinterpret_cast<VkBuffer>(nativeConstants.value), 0, sizeof(NativeParams)}
    };
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < (inlineConstants ? 1u : 2u); ++i)
    {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = fixture.sets[i];
        writes[i].descriptorCount = 1;
        writes[i].descriptorType =
            i == 0 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[i].pBufferInfo = &infos[i];
    }
    fixture.api.vkUpdateDescriptorSets(fixture.api.m_device, inlineConstants ? 1 : 2, writes, 0, nullptr);
    TextureDesc textureDesc{};
    textureDesc.type = TextureType::Texture2D;
    textureDesc.size = {2, 2, 1};
    textureDesc.format = Format::RGBA32Float;
    textureDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    textureDesc.defaultState = ResourceState::RenderTarget;
    auto texture = device->createTexture(textureDesc);
    auto view = device->createTextureView(texture, {});
    REQUIRE(view);
    NativeHandle nativeView;
    REQUIRE_CALL(view->getNativeHandle(&nativeView));
    fixture.target = reinterpret_cast<VkImageView>(nativeView.value);
    auto queue = device->getQueue(QueueType::Graphics);
    for (bool render : {false, true})
    {
        ComPtr<ICommandBuffer> commands[2];
        for (uint32_t i = 0; i < 2; ++i)
        {
            auto encoder = queue->createCommandEncoder();
            encoder->setBufferState(output, ResourceState::UnorderedAccess);
            if (render)
                encoder->setTextureState(texture, ResourceState::RenderTarget);
            NativeVulkanConstants::Snapshot snapshot{i == 0 ? first : second, uint32_t(i * stride), render};
            ExecuteCallbackDesc callback{};
            callback.callback = NativeVulkanConstants::record;
            callback.userObject = &fixture;
            callback.retainUserObject = [](void*)
            {
            };
            callback.releaseUserObject = [](void*)
            {
            };
            callback.userData = &snapshot;
            callback.userDataSize = sizeof(snapshot);
            encoder->executeCallback(callback);
            snapshot = {};
            commands[i] = encoder->finish();
            REQUIRE(commands[i]);
        }
        // Both recordings share immutable descriptors, but their constants stay independent.
        for (uint32_t i : {1u, 0u})
        {
            REQUIRE_CALL(queue->submit(commands[i]));
            REQUIRE_CALL(queue->waitOnHost());
            if (!render)
            {
                auto expected = makeArray<float>(97, 111, 127, 138, 49, 55, 63, 66);
                if (i == 1)
                    for (uint32_t j = 0; j < 8; ++j)
                        expected[j] += j % 4 == 3 ? 0.f : float(j % 4 + 1);
                compareComputeResult(device, output, expected);
            }
            else
            {
                ComPtr<ISlangBlob> pixels;
                SubresourceLayout pixelLayout{};
                REQUIRE_CALL(device->readTexture(texture, 0, 0, pixels.writeRef(), &pixelLayout));
                auto expected = makeArray<float>(146, 166, 190, 204);
                if (i == 1)
                    for (uint32_t j = 0; j < 3; ++j)
                        expected[j] += float(2 * (j + 1));
                for (uint32_t y = 0; y < 2; ++y)
                    for (uint32_t x = 0; x < 2; ++x)
                    {
                        float pixel[4];
                        std::memcpy(
                            pixel,
                            static_cast<const std::byte*>(pixels->getBufferPointer()) + y * pixelLayout.rowPitch +
                                x * sizeof(pixel),
                            sizeof(pixel)
                        );
                        compareResultFuzzy(pixel, expected.data(), 4);
                    }
            }
        }
    }
}
} // namespace

GPU_TEST_CASE("flat-native-constants-dynamic-ubo", Vulkan)
{
    runNativeVulkanConstants(device, false);
}
GPU_TEST_CASE("flat-native-constants-push", Vulkan)
{
    runNativeVulkanConstants(device, true);
}
#endif
