#include "testing.h"
#include "test-ray-tracing-common.h"
#include "debug-layer/debug-command-encoder.h"
#include "../src/state-tracking.h"
#if SLANG_RHI_ENABLE_VULKAN
#include "vulkan/vk-api.h"
#endif

#include <array>
#include <cstddef>
#include <cstring>
#include <mutex>
// doctest stringifies the string_view comparisons below, which needs the stream operator.
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace rhi;
using namespace rhi::testing;

namespace {
const DebugLayerOptions kRootDebugOptions{.coreValidation = true};
struct alignas(16) BindingConstants
{
    uint32_t offset;
    uint32_t multiplier;
    uint32_t bias;
    uint32_t tag;
};

struct alignas(16) BindingWideConstants
{
    uint32_t offset;
    uint32_t multiplier;
    uint32_t bias;
    uint32_t tag;
    uint32_t extra[4];
};

static_assert(std::is_trivially_copyable_v<BindingConstants> && std::is_standard_layout_v<BindingConstants>);
static_assert(sizeof(BindingConstants) == 16 && alignof(BindingConstants) == 16);
static_assert(offsetof(BindingConstants, offset) == 0 && offsetof(BindingConstants, multiplier) == 4);
static_assert(offsetof(BindingConstants, bias) == 8 && offsetof(BindingConstants, tag) == 12);
static_assert(std::is_trivially_copyable_v<BindingWideConstants> && std::is_standard_layout_v<BindingWideConstants>);
static_assert(sizeof(BindingWideConstants) == 32 && alignof(BindingWideConstants) == 16);
static_assert(offsetof(BindingWideConstants, offset) == 0 && offsetof(BindingWideConstants, multiplier) == 4);
static_assert(offsetof(BindingWideConstants, bias) == 8 && offsetof(BindingWideConstants, tag) == 12);
static_assert(offsetof(BindingWideConstants, extra) == 16 && sizeof(BindingWideConstants::extra) == 16);

struct alignas(16) BindingBindlessConstants
{
    uint32_t offset;
    uint32_t multiplier;
    uint32_t bias;
    uint32_t tag;
    uint32_t handle[4];
};

static_assert(std::is_trivially_copyable_v<BindingBindlessConstants> && std::is_standard_layout_v<BindingBindlessConstants>);
static_assert(sizeof(BindingBindlessConstants) == 32 && alignof(BindingBindlessConstants) == 16);
static_assert(offsetof(BindingBindlessConstants, offset) == 0 && offsetof(BindingBindlessConstants, multiplier) == 4);
static_assert(offsetof(BindingBindlessConstants, bias) == 8 && offsetof(BindingBindlessConstants, tag) == 12);
static_assert(offsetof(BindingBindlessConstants, handle) == 16 && sizeof(BindingBindlessConstants::handle) == 16);

struct BindingVariant
{
    bool inlineConstants = false;
    bool wideConstants = false;
    bool bindlessHandle = false;
};

auto createBindingSession(IDevice* device, const BindingVariant& variant, const char* profile = "sm_6_0")
    -> ComPtr<slang::ISession>
{
    auto deviceSession = device->getSlangSession();
    auto* globalSession = deviceSession->getGlobalSession();
    slang::TargetDesc target{};
    target.format = device->getDeviceType() == DeviceType::Vulkan ? SLANG_SPIRV : SLANG_DXIL;
    if (target.format == SLANG_DXIL)
    {
        // ResourceDescriptorHeap, which a descriptor handle lowers to, needs SM 6.6.
        target.profile = globalSession->findProfile(variant.bindlessHandle ? "sm_6_6" : profile);
    }
    target.forceGLSLScalarBufferLayout = true;
    slang::CompilerOptionEntry options[2]{};
    options[0].name = slang::CompilerOptionName::EmitSpirvDirectly;
    options[0].value.kind = slang::CompilerOptionValueKind::Int;
    options[0].value.intValue0 = 1;
    options[1].name = slang::CompilerOptionName::MatrixLayoutRow;
    options[1].value.kind = slang::CompilerOptionValueKind::Int;
    options[1].value.intValue0 = 1;
    const slang::PreprocessorMacroDesc macros[]{
        {"INLINE_CONSTANTS", variant.inlineConstants ? "1" : "0"},
        {"WIDE_CONSTANTS", variant.wideConstants ? "1" : "0"},
        {"BINDLESS_HANDLE", variant.bindlessHandle ? "1" : "0"},
    };
    auto searchPaths = getSlangSearchPaths();
    slang::SessionDesc desc{};
    desc.targets = &target;
    desc.targetCount = 1;
    desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    desc.searchPaths = searchPaths.data();
    desc.searchPathCount = static_cast<SlangInt>(searchPaths.size());
    desc.preprocessorMacros = macros;
    desc.preprocessorMacroCount = static_cast<SlangInt>(std::size(macros));
    desc.compilerOptionEntries = options;
    desc.compilerOptionEntryCount = 2;
    ComPtr<slang::ISession> session;
    REQUIRE_CALL(globalSession->createSession(desc, session.writeRef()));
    return session;
}

struct BindingScene
{
    ComPtr<IBuffer> input;
    ComPtr<IBuffer> output;
    ComPtr<ITexture> texture;
    ComPtr<ITextureView> textureView;
    ComPtr<ISampler> sampler;
    ComPtr<IShaderProgram> program;
    ComPtr<IShaderObject> set;
};

constexpr uint32_t kSampledValue = 3;

auto createBindingSceneResources(IDevice* device, std::span<const uint32_t> input, uint32_t outputCount, BindingScene& scene)
    -> void
{
    const BufferDesc inputDesc{
        .size = input.size_bytes(),
        .elementSize = 4,
        .usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination,
        .defaultState = ResourceState::ShaderResource,
        .label = "flat-scene-input",
    };
    REQUIRE_CALL(device->createBuffer(inputDesc, input.data(), scene.input.writeRef()));
    const BufferDesc outputDesc{
        .size = static_cast<uint64_t>(outputCount) * 4,
        .elementSize = 4,
        .usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination,
        .defaultState = ResourceState::UnorderedAccess,
    };
    const std::vector<uint32_t> zeros(outputCount, 0);
    REQUIRE_CALL(device->createBuffer(outputDesc, zeros.data(), scene.output.writeRef()));
    TextureDesc textureDesc{};
    textureDesc.type = TextureType::Texture2D;
    textureDesc.size = {1, 1, 1};
    textureDesc.format = Format::RGBA32Float;
    textureDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
    textureDesc.defaultState = ResourceState::ShaderResource;
    textureDesc.label = "flat-scene-texture";
    const std::array<float, 4> texel{static_cast<float>(kSampledValue), 0.f, 0.f, 0.f};
    const SubresourceData data{.data = texel.data(), .rowPitch = 16, .slicePitch = 16};
    REQUIRE_CALL(device->createTexture(textureDesc, &data, scene.texture.writeRef()));
    scene.textureView = scene.texture->getDefaultView();
    REQUIRE(scene.textureView != nullptr);
    scene.sampler = device->createSampler({});
    REQUIRE(scene.sampler != nullptr);
}

auto createBindingSceneSet(
    IDevice* device,
    const BindingScene& scene,
    IBuffer* input,
    IBuffer* output,
    const char* label
) -> ComPtr<IShaderObject>
{
    auto root = device->createRootShaderObject(scene.program);
    REQUIRE(root != nullptr);
    ShaderCursor cursor(root);
    REQUIRE_CALL(cursor["scene"]["input"].setBinding(input));
    REQUIRE_CALL(cursor["scene"]["output"].setBinding(output));
    REQUIRE_CALL(cursor["scene"]["texture"].setBinding(scene.textureView));
    REQUIRE_CALL(cursor["scene"]["sampler"].setBinding(scene.sampler));
    ComPtr<IShaderObject> block;
    REQUIRE_CALL(root->getObject(cursor["scene"].m_offset, block.writeRef()));
    return block;
}

auto createSceneBlock(IDevice* device, BindingScene& scene) -> void
{
    scene.set = createBindingSceneSet(device, scene, scene.input, scene.output, "scene");
}

auto bindingExpected(uint32_t previous, uint32_t input, const BindingConstants& constants) -> uint32_t
{
    return previous * constants.tag + input * constants.multiplier + constants.bias + kSampledValue;
}

struct SceneArguments
{
    const void* constants;
    size_t constantsSize;
    IShaderObject* scene;
};

auto sceneArguments(const void* constants, size_t size, std::span<IShaderObject* const> scenes) -> SceneArguments
{
    REQUIRE(scenes.size() == 1);
    return {constants, size, scenes[0]};
}

template<typename Pass, typename Pipeline>
Result bindScene(Pass* pass, Pipeline* pipeline, const SceneArguments& args)
{
    auto* root = pass->bindPipeline(pipeline);
    if (!root) return SLANG_FAIL;
    SLANG_RETURN_ON_FAIL(ShaderCursor(root)["constants"].getDereferenced().setData(args.constants, args.constantsSize));
    return ShaderCursor(root)["scene"].setObject(args.scene);
}

/// Resources created with initial data leave an upload in flight that owns the device, so a case
/// that aborts before reaching its own submit would keep the device alive past the end of the run.
struct BindingDeviceDrain
{
    IDevice* device;

    ~BindingDeviceDrain() { device->getQueue(QueueType::Graphics)->waitOnHost(); }
};

/// Devices these cases create themselves are not wired to the suite's debug callback, so they would
/// fail with no explanation at all.
struct BindingMessageSink final : IDebugCallback
{
    std::string messages;

    virtual SLANG_NO_THROW void SLANG_MCALL handleMessage(
        DebugMessageType,
        DebugMessageSource,
        const char* message
    ) override
    {
        messages += message;
        messages += '\n';
    }
};

// Both backends and both profiles use the same CPU-computed expectation table.
auto runBindingCompute(IDevice* device, bool inlineConstants) -> void
{
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {.inlineConstants = inlineConstants});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 4, scene);
    scene.program = program;
    createSceneBlock(device, scene);
    const ComputePipelineDesc desc{.program = program};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    const BindingConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    IShaderObject* const sets[]{scene.set};
    const auto bindings = sceneArguments(&constants, sizeof(constants), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
    pass->dispatchCompute(4, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[i] = bindingExpected(0, input[i], constants);
    }
    compareComputeResult(device, scene.output, expected);
}
} // namespace

GPU_TEST_CASE("shader-object-binding-compute-buffered", D3D12 | Vulkan)
{
    runBindingCompute(device, false);
}

GPU_TEST_CASE("shader-object-binding-compute-inline", D3D12 | Vulkan)
{
    runBindingCompute(device, true);
}

GPU_TEST_CASE_EX("shader-object-binding-graphics-vs-ps", D3D12 | Vulkan, kRootDebugOptions)
{
#if !SLANG_RHI_DEBUG
    // The debug encoder wrapper exists only with RHI validation, which the harness enables in debug builds.
    SKIP("requires the RHI validation layer of debug builds");
#endif
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(
        loadAndLinkProgram(device, session, "test-flat-binding-cases", {"vsMain", "psMain"}, program.writeRef())
    );
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 4, scene);
    scene.program = program;
    createSceneBlock(device, scene);
    ColorTargetDesc target{};
    target.format = Format::RGBA32Float;
    RenderPipelineDesc desc{.program = program, .targets = &target, .targetCount = 1};
    desc.depthStencil.depthTestEnable = false;
    desc.depthStencil.depthWriteEnable = false;
    desc.rasterizer.cullMode = CullMode::None;
    ComPtr<IRenderPipeline> pipeline;
    REQUIRE_CALL(device->createRenderPipeline(desc, pipeline.writeRef()));
    TextureDesc textureDesc{};
    textureDesc.type = TextureType::Texture2D;
    textureDesc.size = {2, 2, 1};
    textureDesc.format = Format::RGBA32Float;
    textureDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    textureDesc.defaultState = ResourceState::RenderTarget;
    auto texture = device->createTexture(textureDesc);
    REQUIRE(texture != nullptr);
    auto view = device->createTextureView(texture, {});
    REQUIRE(view != nullptr);
    // The vertex stage consumes multiplier and the pixel stage bias/tag, detecting constants lost in either stage.
    BindingConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 11};
    IShaderObject* const sets[]{scene.set};
    const auto bindings = sceneArguments(&constants, sizeof(constants), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto* debugEncoder = static_cast<debug::DebugCommandEncoder*>(encoder.get());
    RenderPassColorAttachment attachment{};
    attachment.view = view;
    attachment.loadOp = LoadOp::Clear;
    attachment.storeOp = StoreOp::Store;
    RenderPassDesc passDesc{};
    passDesc.colorAttachments = &attachment;
    passDesc.colorAttachmentCount = 1;
    auto pass = encoder->beginRenderPass(passDesc);
    REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
    RenderState state{};
    state.viewports[0] = Viewport::fromSize(2, 2);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(2, 2);
    state.scissorRectCount = 1;
    pass->setRenderState(state);
    DrawArguments draw{};
    draw.vertexCount = 3;
    pass->draw(draw);
    debug::DebugRootShaderObject* firstRoot = nullptr;
    const auto bindShaderObject = [&](bool suppliedRoot)
    {
        auto legacyDesc = desc;
        ComPtr<IRenderPipeline> legacyPipeline;
        REQUIRE_CALL(device->createRenderPipeline(legacyDesc, legacyPipeline.writeRef()));
        ComPtr<IShaderObject> root;
        if (suppliedRoot)
        {
            REQUIRE_CALL(device->createRootShaderObject(program, root.writeRef()));
            pass->bindPipeline(legacyPipeline, root);
        }
        else
        {
            root = pass->bindPipeline(legacyPipeline);
            auto* wrapper = debugEncoder->m_renderPassEncoder.m_rootObject.get();
            REQUIRE(wrapper != nullptr);
            CHECK(root.get() == wrapper);
            if (firstRoot)
                CHECK(wrapper == firstRoot);
            firstRoot = wrapper;
            CHECK(wrapper->m_objects.empty());
            CHECK(wrapper->m_entryPoints.empty());
        }
        REQUIRE(root != nullptr);
        constants.bias = 19;
        ShaderCursor cursor(root);
        REQUIRE_CALL(cursor["constants"].getDereferenced().setData(&constants, sizeof(constants)));
        REQUIRE_CALL(cursor["scene"]["input"].setBinding(scene.input));
        REQUIRE_CALL(cursor["scene"]["output"].setBinding(scene.output));
        REQUIRE_CALL(cursor["scene"]["texture"].setBinding(scene.textureView));
        REQUIRE_CALL(cursor["scene"]["sampler"].setBinding(scene.sampler));
        if (!suppliedRoot)
        {
            CHECK_FALSE(firstRoot->m_objects.empty());
            ComPtr<IShaderObject> entryPoint;
            REQUIRE_CALL(root->getEntryPoint(0, entryPoint.writeRef()));
            CHECK_FALSE(firstRoot->m_entryPoints.empty());
        }
        pass->setRenderState(state);
        pass->draw(draw);
    };
    SUBCASE("shader object only") {}
    SUBCASE("switch to encoder-created root") { bindShaderObject(false); }
    SUBCASE("switch to supplied root") { bindShaderObject(true); }
    SUBCASE("encoder root reused across repeated root binds")
    {
        bindShaderObject(false);
        bindShaderObject(false);
        constants.bias = 23;
        REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
        pass->draw(draw);
        bindShaderObject(false);
    }
    SUBCASE("return from encoder root to shared block")
    {
        bindShaderObject(false);
        constants.bias = 23;
        REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
        pass->draw(draw);
    }
    SUBCASE("new pass has no inherited binding")
    {
        pass->end();
        pass = encoder->beginRenderPass(passDesc);
        pass->setRenderState(state);
        bindShaderObject(false);
    }
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    const std::array<float, 4> expected{
        static_cast<float>(input[0] * constants.multiplier + constants.bias + constants.tag + kSampledValue),
        0.f,
        0.f,
        1.f,
    };
    ComPtr<ISlangBlob> pixels;
    SubresourceLayout readbackLayout{};
    REQUIRE_CALL(device->readTexture(texture, 0, 0, pixels.writeRef(), &readbackLayout));
    REQUIRE(readbackLayout.rowPitch >= sizeof(float) * 8);
    REQUIRE(pixels->getBufferSize() >= readbackLayout.rowPitch + sizeof(float) * 8);
    for (uint32_t y = 0; y < 2; ++y)
    {
        for (uint32_t x = 0; x < 2; ++x)
        {
            std::array<float, 4> pixel{};
            const auto* source = static_cast<const std::byte*>(pixels->getBufferPointer());
            std::memcpy(pixel.data(), source + y * readbackLayout.rowPitch + x * sizeof(pixel), sizeof(pixel));
            compareResultFuzzy(pixel.data(), expected.data(), pixel.size());
        }
    }
}

GPU_TEST_CASE_EX("shader-object-binding-debug-root-reuse-compute", D3D12 | Vulkan, kRootDebugOptions)
{
#if !SLANG_RHI_DEBUG
    // The debug encoder wrapper exists only with RHI validation, which the harness enables in debug builds.
    SKIP("requires the RHI validation layer of debug builds");
#endif
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 20, scene);
    scene.program = program;
    createSceneBlock(device, scene);
    ComputePipelineDesc desc{.program = program};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    ComPtr<IComputePipeline> legacyPipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, legacyPipeline.writeRef()));
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto* debugEncoder = static_cast<debug::DebugCommandEncoder*>(encoder.get());
    auto pass = encoder->beginComputePass();
    IShaderObject* const sets[]{scene.set};
    debug::DebugRootShaderObject* firstRoot = nullptr;
    std::array<uint32_t, 20> expected{};
    // Alternate shared children and freshly populated roots, with disjoint output regions.
    for (uint32_t step = 0; step < 5; ++step)
    {
        const BindingConstants constants{.offset = step * 4, .multiplier = step + 2, .bias = step * 7, .tag = 0};
        if (step == 0 || step == 3)
        {
            REQUIRE_CALL(bindScene(pass, pipeline.get(), sceneArguments(&constants, sizeof(constants), sets)));
        }
        else
        {
            auto* root = pass->bindPipeline(legacyPipeline);
            REQUIRE(root != nullptr);
            auto* wrapper = debugEncoder->m_computePassEncoder.m_rootObject.get();
            REQUIRE(wrapper != nullptr);
            CHECK(root == wrapper);
            if (firstRoot)
                CHECK(wrapper == firstRoot);
            firstRoot = wrapper;
            CHECK(wrapper->m_objects.empty());
            CHECK(wrapper->m_entryPoints.empty());
            ShaderCursor cursor(root);
            REQUIRE_CALL(cursor["constants"].getDereferenced().setData(&constants, sizeof(constants)));
            REQUIRE_CALL(cursor["scene"]["input"].setBinding(scene.input));
            REQUIRE_CALL(cursor["scene"]["output"].setBinding(scene.output));
            REQUIRE_CALL(cursor["scene"]["texture"].setBinding(scene.textureView));
            REQUIRE_CALL(cursor["scene"]["sampler"].setBinding(scene.sampler));
            CHECK_FALSE(wrapper->m_objects.empty());
            ComPtr<IShaderObject> entryPoint;
            REQUIRE_CALL(root->getEntryPoint(0, entryPoint.writeRef()));
            CHECK_FALSE(wrapper->m_entryPoints.empty());
        }
        pass->dispatchCompute(4, 1, 1);
        for (uint32_t i = 0; i < 4; ++i)
            expected[step * 4 + i] = bindingExpected(0, input[i], constants);
    }
    CHECK(debugEncoder->m_renderPassEncoder.m_rootObject == nullptr);
    CHECK(debugEncoder->m_rayTracingPassEncoder.m_rootObject == nullptr);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE_EX("shader-object-binding-debug-root-reuse-ray-tracing", D3D12 | Vulkan, kRootDebugOptions)
{
#if !SLANG_RHI_DEBUG
    // The debug encoder wrapper exists only with RHI validation, which the harness enables in debug builds.
    SKIP("requires the RHI validation layer of debug builds");
#endif
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported; debug RT root allocation regression not executed");
    const BindingDeviceDrain drain{device};
    auto queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue);
    TLAS tlas(device, queue, blas.blas);
    auto session = createBindingSession(device, {}, "sm_6_3");
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-rt",
        {"rayGenMain", "missMain", "closestHitMain"}, program.writeRef()));
    struct RtConstants
    {
        uint32_t dims[2];
        float hitValue;
        float missValue;
    };
    static_assert(sizeof(RtConstants) == 16);
    HitGroupDesc hitGroup{};
    hitGroup.hitGroupName = "hitGroup";
    hitGroup.closestHitEntryPoint = "closestHitMain";
    RayTracingPipelineDesc pipelineDesc{};
    pipelineDesc.program = program;
    pipelineDesc.hitGroups = &hitGroup;
    pipelineDesc.hitGroupCount = 1;
    pipelineDesc.maxRayPayloadSize = 16;
    pipelineDesc.maxAttributeSizeInBytes = 8;
    pipelineDesc.maxRecursion = 1;
    ComPtr<IRayTracingPipeline> pipeline;
    REQUIRE_CALL(device->createRayTracingPipeline(pipelineDesc, pipeline.writeRef()));
    ComPtr<IRayTracingPipeline> legacyPipeline;
    REQUIRE_CALL(device->createRayTracingPipeline(pipelineDesc, legacyPipeline.writeRef()));
    const char* raygenNames[]{"rayGenMain"};
    const char* missNames[]{"missMain"};
    const char* hitNames[]{"hitGroup"};
    ShaderTableDesc tableDesc{};
    tableDesc.program = program;
    tableDesc.rayGenShaderEntryPointNames = raygenNames;
    tableDesc.rayGenShaderCount = 1;
    tableDesc.missShaderEntryPointNames = missNames;
    tableDesc.missShaderCount = 1;
    tableDesc.hitGroupNames = hitNames;
    tableDesc.hitGroupCount = 1;
    ComPtr<IShaderTable> table;
    REQUIRE_CALL(device->createShaderTable(tableDesc, table.writeRef()));
    std::array<ComPtr<ITexture>, 5> outputs;
    auto encoder = queue->createCommandEncoder();
    auto* debugEncoder = static_cast<debug::DebugCommandEncoder*>(encoder.get());
    auto pass = encoder->beginRayTracingPass();
    debug::DebugRootShaderObject* firstRoot = nullptr;
    for (uint32_t step = 0; step < outputs.size(); ++step)
    {
        TextureDesc textureDesc{};
        textureDesc.type = TextureType::Texture2D;
        textureDesc.size = {4, 4, 1};
        textureDesc.format = Format::RGBA32Float;
        textureDesc.usage = TextureUsage::UnorderedAccess | TextureUsage::CopySource;
        textureDesc.defaultState = ResourceState::UnorderedAccess;
        REQUIRE_CALL(device->createTexture(textureDesc, nullptr, outputs[step].writeRef()));
        const RtConstants constants{{4, 4}, 0.75f + float(step), 0.25f + float(step)};
        if (step == 0 || step == 3)
        {
            auto* root = pass->bindPipeline(pipeline, table);
            REQUIRE(root != nullptr);
            REQUIRE_CALL(ShaderCursor(root)["params"].getDereferenced().setData(&constants, sizeof(constants)));
            REQUIRE_CALL(ShaderCursor(root)["scene"]["tlas"].setBinding(tlas.tlas));
            REQUIRE_CALL(ShaderCursor(root)["scene"]["output"].setBinding(outputs[step]));
        }
        else
        {
            auto* root = pass->bindPipeline(legacyPipeline, table);
            REQUIRE(root != nullptr);
            auto* wrapper = debugEncoder->m_rayTracingPassEncoder.m_rootObject.get();
            REQUIRE(wrapper != nullptr);
            CHECK(root == wrapper);
            if (firstRoot)
                CHECK(wrapper == firstRoot);
            firstRoot = wrapper;
            CHECK(wrapper->m_objects.empty());
            CHECK(wrapper->m_entryPoints.empty());
            ShaderCursor cursor(root);
            REQUIRE_CALL(cursor["params"].getDereferenced().setData(&constants, sizeof(constants)));
            REQUIRE_CALL(cursor["scene"]["tlas"].setBinding(tlas.tlas));
            REQUIRE_CALL(cursor["scene"]["output"].setBinding(outputs[step]->getDefaultView()));
            CHECK_FALSE(wrapper->m_objects.empty());
            ComPtr<IShaderObject> entryPoint;
            REQUIRE_CALL(root->getEntryPoint(0, entryPoint.writeRef()));
            CHECK_FALSE(wrapper->m_entryPoints.empty());
        }
        pass->dispatchRays(0, 4, 4, 1);
    }
    CHECK(debugEncoder->m_computePassEncoder.m_rootObject == nullptr);
    CHECK(debugEncoder->m_renderPassEncoder.m_rootObject == nullptr);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    for (uint32_t step = 0; step < outputs.size(); ++step)
    {
        std::array<float, 64> expected{};
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 4; ++x)
            {
                const uint32_t offset = (y * 4 + x) * 4;
                const float value = (x + y <= 3 ? 0.75f : 0.25f) + float(step);
                expected[offset] = expected[offset + 1] = expected[offset + 2] = value;
                expected[offset + 3] = 1.f;
            }
        compareComputeResult(device, outputs[step], 0, 0, expected);
    }
}

GPU_TEST_CASE("shader-object-binding-block-early-release", D3D12 | Vulkan)
{
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 4, scene);
    scene.program = program;
    createSceneBlock(device, scene);
    const ComputePipelineDesc desc{.program = program};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    // Leave resource ownership to the set before recording; retain only the output for readback.
    scene.textureView.setNull();
    scene.texture.setNull();
    scene.sampler.setNull();
    scene.input.setNull();
    const BindingConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    IShaderObject* const sets[]{scene.set};
    const auto bindings = sceneArguments(&constants, sizeof(constants), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
    pass->dispatchCompute(4, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[i] = bindingExpected(0, input[i], constants);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("shader-object-binding-constants-copied-to-root", D3D12 | Vulkan)
{
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 4, scene);
    scene.program = program;
    createSceneBlock(device, scene);
    const ComputePipelineDesc desc{.program = program};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    BindingConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const BindingConstants expectedConstants = constants;
    IShaderObject* const sets[]{scene.set};
    const auto bindings = sceneArguments(&constants, sizeof(constants), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
    constants = {.offset = 2, .multiplier = 101, .bias = 999, .tag = 5};
    pass->dispatchCompute(4, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[i] = bindingExpected(0, input[i], expectedConstants);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("shader-object-binding-uav-order", D3D12 | Vulkan)
{
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 4, scene);
    scene.program = program;
    createSceneBlock(device, scene);
    const ComputePipelineDesc desc{.program = program};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    const BindingConstants first{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const BindingConstants second{.offset = 0, .multiplier = 5, .bias = 1, .tag = 2};
    IShaderObject* const sets[]{scene.set};
    auto bindings = sceneArguments(&first, sizeof(first), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
    pass->dispatchCompute(4, 1, 1);
    // The second dispatch must observe the first: recorded accesses must produce a UAV barrier between them.
    bindings = sceneArguments(&second, sizeof(second), sets);
    REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
    pass->dispatchCompute(4, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[i] = bindingExpected(bindingExpected(0, input[i], first), input[i], second);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("shader-object-binding-set-shared-by-two-command-buffers", D3D12 | Vulkan)
{
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 8, scene);
    scene.program = program;
    createSceneBlock(device, scene);
    const ComputePipelineDesc desc{.program = program};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    const BindingConstants constantsA{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const BindingConstants constantsB{.offset = 4, .multiplier = 5, .bias = 1, .tag = 0};
    IShaderObject* const sets[]{scene.set};
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoderA = queue->createCommandEncoder();
    ComPtr<ICommandBuffer> commandsA;
    {
        const auto bindings = sceneArguments(&constantsA, sizeof(constantsA), sets);
        auto pass = encoderA->beginComputePass();
        REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
        pass->dispatchCompute(4, 1, 1);
        pass->end();
        commandsA = encoderA->finish();
        REQUIRE(commandsA != nullptr);
    }
    auto encoderB = queue->createCommandEncoder();
    ComPtr<ICommandBuffer> commandsB;
    {
        const auto bindings = sceneArguments(&constantsB, sizeof(constantsB), sets);
        auto pass = encoderB->beginComputePass();
        REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
        pass->dispatchCompute(4, 1, 1);
        pass->end();
        commandsB = encoderB->finish();
        REQUIRE(commandsB != nullptr);
    }
    ICommandBuffer* buffersA[]{commandsA};
    ICommandBuffer* buffersB[]{commandsB};
    const SubmitDesc submitA{.commandBuffers = buffersA, .commandBufferCount = 1};
    REQUIRE_CALL(queue->submit(submitA));
    REQUIRE_CALL(queue->waitOnHost());
    // The empty submission retires A and reclaims its transient/descriptor allocations before B uses the shared set.
    const SubmitDesc empty{};
    REQUIRE_CALL(queue->submit(empty));
    REQUIRE_CALL(queue->waitOnHost());
    const SubmitDesc submitB{.commandBuffers = buffersB, .commandBufferCount = 1};
    REQUIRE_CALL(queue->submit(submitB));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 8> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[constantsA.offset + i] = bindingExpected(0, input[i], constantsA);
        expected[constantsB.offset + i] = bindingExpected(0, input[i], constantsB);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("shader-object-binding-block-reused-across-root-constants", D3D12 | Vulkan)
{
    const BindingDeviceDrain drain{device};
    auto narrowSession = createBindingSession(device, {});
    auto wideSession = createBindingSession(device, {.wideConstants = true});
    ComPtr<IShaderProgram> narrowProgram;
    ComPtr<IShaderProgram> wideProgram;
    REQUIRE_CALL(
        loadAndLinkProgram(device, narrowSession, "test-flat-binding-cases", "csMain", narrowProgram.writeRef())
    );
    REQUIRE_CALL(loadAndLinkProgram(device, wideSession, "test-flat-binding-cases", "csMain", wideProgram.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 8, scene);
    scene.program = narrowProgram;
    createSceneBlock(device, scene);
    const ComputePipelineDesc narrowDesc{.program = narrowProgram};
    const ComputePipelineDesc wideDesc{.program = wideProgram};
    ComPtr<IComputePipeline> narrowPipeline;
    ComPtr<IComputePipeline> widePipeline;
    REQUIRE_CALL(device->createComputePipeline(narrowDesc, narrowPipeline.writeRef()));
    REQUIRE_CALL(device->createComputePipeline(wideDesc, widePipeline.writeRef()));
    const BindingConstants narrow{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const BindingWideConstants wide{.offset = 4, .multiplier = 5, .bias = 1, .tag = 0, .extra = {2, 0, 0, 9}};
    IShaderObject* const sets[]{scene.set};
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    {
        const auto bindings = sceneArguments(&narrow, sizeof(narrow), sets);
        REQUIRE_CALL(bindScene(pass, narrowPipeline.get(), bindings));
        pass->dispatchCompute(4, 1, 1);
    }
    {
        const auto bindings = sceneArguments(&wide, sizeof(wide), sets);
        REQUIRE_CALL(bindScene(pass, widePipeline.get(), bindings));
        pass->dispatchCompute(4, 1, 1);
    }
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    const BindingConstants narrowView{
        .offset = wide.offset,
        .multiplier = wide.multiplier,
        .bias = wide.bias,
        .tag = wide.tag,
    };
    std::array<uint32_t, 8> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[narrow.offset + i] = bindingExpected(0, input[i], narrow);
        expected[wide.offset + i] = bindingExpected(0, input[i], narrowView) + wide.extra[0] + wide.extra[3];
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("shader-object-binding-reverse-order-submission", D3D12 | Vulkan)
{
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 8, scene);
    scene.program = program;
    createSceneBlock(device, scene);
    const ComputePipelineDesc desc{.program = program};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    const BindingConstants constantsA{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const BindingConstants constantsB{.offset = 4, .multiplier = 5, .bias = 1, .tag = 0};
    IShaderObject* const sets[]{scene.set};
    auto queue = device->getQueue(QueueType::Graphics);
    // Reverse submission requires constant bytes to be copied per command buffer, not per queue.
    auto encoderA = queue->createCommandEncoder();
    ComPtr<ICommandBuffer> commandsA;
    {
        const auto bindings = sceneArguments(&constantsA, sizeof(constantsA), sets);
        auto pass = encoderA->beginComputePass();
        REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
        pass->dispatchCompute(4, 1, 1);
        pass->end();
        commandsA = encoderA->finish();
        REQUIRE(commandsA != nullptr);
    }
    auto encoderB = queue->createCommandEncoder();
    ComPtr<ICommandBuffer> commandsB;
    {
        const auto bindings = sceneArguments(&constantsB, sizeof(constantsB), sets);
        auto pass = encoderB->beginComputePass();
        REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
        pass->dispatchCompute(4, 1, 1);
        pass->end();
        commandsB = encoderB->finish();
        REQUIRE(commandsB != nullptr);
    }
    ICommandBuffer* buffersB[]{commandsB};
    const SubmitDesc submitB{.commandBuffers = buffersB, .commandBufferCount = 1};
    REQUIRE_CALL(queue->submit(submitB));
    ICommandBuffer* buffersA[]{commandsA};
    const SubmitDesc submitA{.commandBuffers = buffersA, .commandBufferCount = 1};
    REQUIRE_CALL(queue->submit(submitA));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 8> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[constantsA.offset + i] = bindingExpected(0, input[i], constantsA);
        expected[constantsB.offset + i] = bindingExpected(0, input[i], constantsB);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("shader-object-binding-handle-access-keepalive", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::Bindless))
    {
        SKIP("Bindless is not supported");
    }
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {.bindlessHandle = true});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, input, 4, scene);
    scene.program = program;
    createSceneBlock(device, scene);
    const ComputePipelineDesc desc{.program = program};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    const std::array<uint32_t, 4> extraData{10u, 20u, 30u, 40u};
    const BufferDesc extraDesc{
        .size = 16,
        .elementSize = 0,
        .usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination,
        .defaultState = ResourceState::ShaderResource,
    };
    ComPtr<IBuffer> extra;
    REQUIRE_CALL(device->createBuffer(extraDesc, extraData.data(), extra.writeRef()));
    DescriptorHandle handle{};
    REQUIRE_CALL(extra->getDescriptorHandle(DescriptorHandleAccess::Read, Format::Undefined, kEntireBuffer, &handle));
    REQUIRE(handle.type == DescriptorHandleType::Buffer);
    BindingBindlessConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0, .handle = {}};
    std::memcpy(constants.handle, &handle.value, sizeof(handle.value));
    IShaderObject* const sets[]{scene.set};
    auto bindings = sceneArguments(&constants, sizeof(constants), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    encoder->setBufferState(extra, ResourceState::ShaderResource);
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
    // Only the recorded access can keep this buffer alive; no binding set owns it.
    extra.setNull();
    pass->dispatchCompute(4, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    const BindingConstants narrowView{
        .offset = constants.offset,
        .multiplier = constants.multiplier,
        .bias = constants.bias,
        .tag = constants.tag,
    };
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[i] = bindingExpected(0, input[i], narrowView) + extraData[i];
    }
    compareComputeResult(device, scene.output, expected);
}

namespace {
/// The buffer the two halves of the transition case hand between a set entry and a descriptor
/// handle. It carries both usages, and no element size, because a descriptor handle is always a raw
/// view while a set entry takes its stride from the reflected element type.
auto createBindingTransitionBuffer(IDevice* device, uint32_t count, const char* label) -> ComPtr<IBuffer>
{
    const BufferDesc desc{
        .size = static_cast<uint64_t>(count) * 4,
        .elementSize = 0,
        .usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopySource |
                 BufferUsage::CopyDestination,
        // ShaderResource so that every write below, through a set entry or through a handle, has to
        // be preceded by a transition the recorded accesses ask for.
        .defaultState = ResourceState::ShaderResource,
        .label = label,
    };
    const std::vector<uint32_t> zeros(count, 0);
    ComPtr<IBuffer> buffer;
    REQUIRE_CALL(device->createBuffer(desc, zeros.data(), buffer.writeRef()));
    return buffer;
}

auto bindlessConstants(const BindingConstants& values, const DescriptorHandle& handle) -> BindingBindlessConstants
{
    BindingBindlessConstants constants{
        .offset = values.offset,
        .multiplier = values.multiplier,
        .bias = values.bias,
        .tag = values.tag,
        .handle = {},
    };
    std::memcpy(constants.handle, &handle.value, sizeof(handle.value));
    return constants;
}
} // namespace

GPU_TEST_CASE("shader-object-binding-handle-access-transition", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::Bindless))
    {
        SKIP("Bindless is not supported");
    }
    const BindingDeviceDrain drain{device};
    auto session = createBindingSession(device, {.bindlessHandle = true});
    auto load = [&](const char* entryPoint)
    {
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", entryPoint, program.writeRef()));
        return program;
    };
    auto setWrite = load("csSetWrite");
    auto handleRead = load("csHandleRead");
    auto handleWrite = load("csHandleWrite");
    auto setRead = load("csSetRead");

    constexpr uint32_t kCount = 4;
    const std::array<uint32_t, kCount> seed{5, 6, 7, 8};
    BindingScene scene;
    createBindingSceneResources(device, seed, kCount, scene);
    auto bridgeToHandle = createBindingTransitionBuffer(device, kCount, "bridge-to-handle");
    auto bridgeToSet = createBindingTransitionBuffer(device, kCount, "bridge-to-set");
    auto handleReadResult = createBindingTransitionBuffer(device, kCount, "handle-read-result");
    auto setReadResult = createBindingTransitionBuffer(device, kCount, "set-read-result");

    scene.program = setWrite;
    auto setWriteSet = createBindingSceneSet(device, scene, scene.input, bridgeToHandle, "set-write");
    auto handleReadSet = createBindingSceneSet(device, scene, scene.input, handleReadResult, "handle-read");
    auto handleWriteSet = createBindingSceneSet(device, scene, scene.input, setReadResult, "handle-write");
    auto setReadSet = createBindingSceneSet(device, scene, bridgeToSet, setReadResult, "set-read");

    auto createPipeline = [&](IShaderProgram* program)
    {
        ComPtr<IComputePipeline> pipeline;
        const ComputePipelineDesc desc{.program = program};
        REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
        return pipeline;
    };
    auto setWritePipeline = createPipeline(setWrite);
    auto handleReadPipeline = createPipeline(handleRead);
    auto handleWritePipeline = createPipeline(handleWrite);
    auto setReadPipeline = createPipeline(setRead);

    DescriptorHandle readHandle{};
    REQUIRE_CALL(
        bridgeToHandle->getDescriptorHandle(DescriptorHandleAccess::Read, Format::Undefined, kEntireBuffer, &readHandle)
    );
    REQUIRE(readHandle.type == DescriptorHandleType::Buffer);
    DescriptorHandle writeHandle{};
    REQUIRE_CALL(bridgeToSet->getDescriptorHandle(
        DescriptorHandleAccess::ReadWrite,
        Format::Undefined,
        kEntireBuffer,
        &writeHandle
    ));
    REQUIRE(writeHandle.type == DescriptorHandleType::RWBuffer);

    const BindingConstants produceToHandle{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const BindingConstants consumeFromHandle{.offset = 0, .multiplier = 0, .bias = 0, .tag = 2};
    const BindingConstants produceToSet{.offset = 0, .multiplier = 5, .bias = 1, .tag = 0};
    const BindingConstants consumeFromSet{.offset = 0, .multiplier = 0, .bias = 0, .tag = 3};
    const auto setWriteConstants = bindlessConstants(produceToHandle, readHandle);
    const auto handleReadConstants = bindlessConstants(consumeFromHandle, readHandle);
    const auto handleWriteConstants = bindlessConstants(produceToSet, writeHandle);
    const auto setReadConstants = bindlessConstants(consumeFromSet, writeHandle);

    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto dispatch = [&](IComputePipeline* pipeline, IShaderObject* block,
                        const BindingBindlessConstants& constants, IBuffer* access, ResourceState state)
    {
        if (access) encoder->setBufferState(access, state);
        auto pass = encoder->beginComputePass();
        REQUIRE_CALL(bindScene(pass, pipeline, {&constants, sizeof(constants), block}));
        pass->dispatchCompute(kCount, 1, 1);
        pass->end();
    };
    dispatch(setWritePipeline, setWriteSet, setWriteConstants, nullptr, ResourceState::Undefined);
    dispatch(handleReadPipeline, handleReadSet, handleReadConstants, bridgeToHandle, ResourceState::ShaderResource);
    dispatch(handleWritePipeline, handleWriteSet, handleWriteConstants, bridgeToSet, ResourceState::UnorderedAccess);
    dispatch(setReadPipeline, setReadSet, setReadConstants, nullptr, ResourceState::Undefined);
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());

    std::array<uint32_t, kCount> expectedHandleRead{};
    std::array<uint32_t, kCount> expectedSetRead{};
    for (uint32_t i = 0; i < kCount; ++i)
    {
        expectedHandleRead[i] =
            (seed[i] * produceToHandle.multiplier + produceToHandle.bias) * consumeFromHandle.tag;
        expectedSetRead[i] = (seed[i] * produceToSet.multiplier + produceToSet.bias) * consumeFromSet.tag;
    }
    compareComputeResult(device, handleReadResult, expectedHandleRead);
    compareComputeResult(device, setReadResult, expectedSetRead);
}


// Independent regression for the partial-layer tracker fix: transitions must stop at layerCount.
GPU_TEST_CASE("texture-state-requested-layers", D3D12 | Vulkan)
{
    TextureDesc desc{};
    desc.type = TextureType::Texture2DArray;
    desc.size = {8, 8, 1};
    desc.arrayLength = 4;
    desc.mipCount = 3;
    desc.format = Format::RGBA8Unorm;
    desc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
    desc.defaultState = ResourceState::ShaderResource;
    auto texture = device->createTexture(desc);
    REQUIRE(texture != nullptr);
    auto* native = checked_cast<Texture*>(texture.get());
    StateTracking tracking;
    tracking.setTextureState(native, {1, 1, 1, 1}, ResourceState::CopyDestination);
    REQUIRE(tracking.getTextureBarriers().size() == 1);
    CHECK_FALSE(tracking.getTextureBarriers()[0].entireTexture);
    CHECK(tracking.getTextureBarriers()[0].layer == 1);
    CHECK(tracking.getTextureBarriers()[0].mip == 1);
    tracking.clearBarriers();
    // Checking every unrequested subresource catches both leading and trailing over-transition.
    for (uint32_t layer = 0; layer < 4; ++layer)
    {
        for (uint32_t mip = 0; mip < 3; ++mip)
        {
            if (layer == 1 && mip == 1)
                continue;
            tracking.setTextureState(native, {layer, 1, mip, 1}, ResourceState::ShaderResource);
        }
    }
    CHECK(tracking.getTextureBarriers().empty());
    tracking.setTextureState(native, {1, 1, 1, 1}, ResourceState::ShaderResource);
    REQUIRE(tracking.getTextureBarriers().size() == 1);
    CHECK(tracking.getTextureBarriers()[0].stateBefore == ResourceState::CopyDestination);
    tracking.clearBarriers();
    // Remaining-layer/mip sentinels normalize to exactly the requested suffix.
    tracking.setTextureState(native, {2, kAllLayers, 1, kAllMips}, ResourceState::CopyDestination);
    REQUIRE(tracking.getTextureBarriers().size() == 4);
    for (const auto& barrier : tracking.getTextureBarriers())
    {
        CHECK_FALSE(barrier.entireTexture);
        CHECK(barrier.layer >= 2);
        CHECK(barrier.layer < 4);
        CHECK(barrier.mip >= 1);
        CHECK(barrier.mip < 3);
    }
    tracking.clearBarriers();
    tracking.setTextureState(native, {0, 2, 0, kAllMips}, ResourceState::ShaderResource);
    tracking.setTextureState(native, {2, kAllLayers, 0, 1}, ResourceState::ShaderResource);
    CHECK(tracking.getTextureBarriers().empty());
}

namespace {
struct BindingTransitionMessageSink final : IDebugCallback
{
    std::mutex               mutex;
    std::string              errors;

    virtual SLANG_NO_THROW void SLANG_MCALL handleMessage(
        DebugMessageType type,
        DebugMessageSource source,
        const char* message
    ) override
    {
        const std::lock_guard lock(mutex);
        if (type == DebugMessageType::Error)
        {
            errors += message;
            errors += '\n';
        }
    }
};
} // namespace

GPU_TEST_CASE("shader-object-recorded-error-propagates-at-finish", D3D12 | DontCreateDevice)
{
    BindingTransitionMessageSink sink;
    DeviceDesc desc{.deviceType = ctx->deviceType, .adapter = getSelectedDeviceAdapter(ctx->deviceType)};
    desc.debugCallback = &sink;
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(desc, local.writeRef()));
    const BindingDeviceDrain drain{local};
    const char* source = R"(
Texture2D<float> tex;
SamplerState samp;
RWStructuredBuffer<float> output;
[shader("compute")][numthreads(1, 1, 1)]
void main() { output[0] = tex.SampleLevel(samp, float2(0.5), 0); }
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(local, source, program.writeRef()));
    auto pipeline = local->createComputePipeline({.program = program});
    REQUIRE(pipeline != nullptr);
    auto root = local->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    const float texels[]{1.f, 2.f, 3.f, 4.f};
    TextureDesc textureDesc{};
    textureDesc.size = {2, 2, 1};
    textureDesc.format = Format::R32Float;
    textureDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
    const SubresourceData data{texels, 2 * sizeof(float), 0};
    auto texture = local->createTexture(textureDesc, &data);
    REQUIRE(texture != nullptr);
    std::array<ComPtr<ISampler>, 1> samplers;
    for (uint32_t i = 0; i < samplers.size(); ++i)
    {
        SamplerDesc samplerDesc{};
        samplerDesc.minFilter = samplerDesc.magFilter = i ? TextureFilteringMode::Linear : TextureFilteringMode::Point;
        REQUIRE_CALL(local->createSampler(samplerDesc, samplers[i].writeRef()));
    }
    const uint32_t initial = 0;
    BufferDesc outputDesc{};
    outputDesc.size = outputDesc.elementSize = sizeof(float);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    auto output = local->createBuffer(outputDesc, &initial);
    REQUIRE(output != nullptr);
    ShaderCursor cursor(root);
    REQUIRE_CALL(cursor["tex"].setBinding(texture));
    REQUIRE_CALL(cursor["output"].setBinding(output));
    auto queue = local->getQueue(QueueType::Graphics);
    {
        auto encoder = queue->createCommandEncoder();
        auto* pass = encoder->beginComputePass();
        pass->bindPipeline(pipeline, nullptr);
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        ComPtr<ICommandBuffer> commands;
        const Result result = encoder->finish({}, commands.writeRef());
        INFO(sink.errors);
        CHECK(result == SLANG_E_INVALID_ARG);
        CHECK(commands == nullptr);
    }
    compareComputeResult(local, output, std::array<uint32_t, 1>{0});
    REQUIRE_CALL(queue->waitOnHost());
    auto encoder = queue->createCommandEncoder();
    auto* pass = encoder->beginComputePass();
    REQUIRE_CALL(cursor["samp"].setBinding(samplers[0]));
    pass->bindPipeline(pipeline, root);
    pass->dispatchCompute(1, 1, 1);
    pass->end();
    ComPtr<ICommandBuffer> commands;
    REQUIRE_CALL(encoder->finish({}, commands.writeRef()));
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(local, output, std::array<uint32_t, 1>{0x40800000});
}

GPU_TEST_CASE("shader-object-binding-state-transition-read-union", D3D12 | Vulkan | DontCreateDevice)
{
    BindingTransitionMessageSink sink;
    DeviceDesc deviceDesc{.deviceType = ctx->deviceType, .adapter = getSelectedDeviceAdapter(ctx->deviceType)};
    deviceDesc.enableValidation = true;
    deviceDesc.debugCallback = &sink;
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
    const BindingDeviceDrain drain{local};
    auto session = createBindingSession(local, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(local, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    BindingScene scene;
    scene.program = program;
    createBindingSceneResources(local, input, 4, scene);
    createSceneBlock(local, scene);
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(local->createComputePipeline({.program = program}, pipeline.writeRef()));
    const std::array<uint32_t, 3> dispatch{4, 1, 1};
    const BufferDesc argumentDesc{
        .size = sizeof(dispatch),
        .elementSize = 4,
        .usage = BufferUsage::ShaderResource | BufferUsage::IndirectArgument | BufferUsage::CopyDestination,
        .defaultState = ResourceState::IndirectArgument,
        .label = "read-union-arguments",
    };
    auto arguments = local->createBuffer(argumentDesc, dispatch.data());
    REQUIRE(arguments != nullptr);
    // SRV extends the already-known indirect state once. Subsequent SRV and indirect requests
    // are subsets of that effective union; both dispatches must still produce ordered UAV writes.
    IShaderObject* const sets[]{scene.set};
    const BindingConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 2};
    auto bindings = sceneArguments(&constants, sizeof(constants), sets);
    auto queue = local->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    encoder->setBufferState(arguments, ResourceState::ShaderResource);
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(bindScene(pass, pipeline.get(), bindings));
    // Reuse the same root, covering both the retained read union and UAV ordering.
    pass->dispatchComputeIndirect({arguments, 0});
    pass->dispatchComputeIndirect({arguments, 0});
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
        expected[i] = bindingExpected(bindingExpected(0, input[i], constants), input[i], constants);
    compareComputeResult(local, scene.output, expected);
    const std::lock_guard lock(sink.mutex);
    INFO(sink.errors);
    CHECK(sink.errors.empty());
}
