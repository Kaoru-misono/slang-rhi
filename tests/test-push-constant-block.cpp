#include "testing.h"
#include "test-ray-tracing-common.h"
#include "shader-object.h"
#include "debug-layer/debug-shader-object.h"
#include "debug-layer/debug-command-buffer.h"
#include "command-buffer.h"
#include "device.h"
#include "d3d12/d3d12-shader-object-layout.h"
#include "vulkan/vk-shader-object-layout.h"

#include <cstring>

using namespace rhi;
using namespace rhi::testing;

namespace {

RootShaderObject* getNativeRoot(IShaderObject* object)
{
    auto* wrapped = dynamic_cast<debug::DebugShaderObject*>(object);
    return checked_cast<RootShaderObject*>(wrapped ? wrapped->baseObject.get() : object);
}

#if SLANG_RHI_ENABLE_BINDING_COUNTERS
const BindingCounters& getBindingCounters(ICommandBuffer* commands)
{
    auto* wrapped = dynamic_cast<debug::DebugCommandBuffer*>(commands);
    return checked_cast<CommandBuffer*>(wrapped ? wrapped->baseObject.get() : commands)->m_bindingCounters;
}
#endif

// D3D12 holds the block in a root parameter instead of a constant-buffer view. On Vulkan it is the program's only
// push-constant range.
void checkBlockLayout(IDevice* device, IShaderProgram* program, uint32_t expectedSize)
{
    auto root = device->createRootShaderObject(program);
    REQUIRE(root);
    auto layout = getNativeRoot(root)->m_layout.get();
    uint32_t blockCount = 0;
    for (uint32_t i = 0; i < layout->getSubObjectRangeCount(); ++i)
    {
        if (!layout->getBindingRange(layout->getSubObjectRange(i).bindingRangeIndex).isPushConstantBlock)
            continue;
        ++blockCount;
        if (device->getDeviceType() == DeviceType::D3D12)
        {
            auto block = checked_cast<d3d12::ShaderObjectLayoutImpl*>(layout->getSubObjectRangeLayout(i));
            CHECK(block->getTotalOrdinaryDataSize() == expectedSize);
            CHECK(block->getOrdinaryDataBufferCount() == 0);
        }
    }
    CHECK(blockCount == 1);
    if (device->getDeviceType() == DeviceType::D3D12)
    {
        auto native = checked_cast<d3d12::RootShaderObjectLayoutImpl*>(layout);
        CHECK(native->m_pushConstantRootParameterIndex != UINT32_MAX);
    }
    else
    {
        auto native = checked_cast<vk::RootShaderObjectLayoutImpl*>(layout);
        REQUIRE(native->getAllPushConstantRanges().size() == 1);
        CHECK(native->getAllPushConstantRanges()[0].size == expectedSize);
    }
}

struct Resources
{
    IDevice* device;
    ComPtr<IBuffer> output;
    Resources(IDevice* device)
        : device(device)
    {
        BufferDesc desc = {};
        desc.size = 8;
        desc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
        desc.defaultState = ResourceState::UnorderedAccess;
        const uint32_t zero[2] = {};
        REQUIRE_CALL(device->createBuffer(desc, zero, output.writeRef()));
    }
    void bind(IShaderObject* root)
    {
        const uint32_t values[4] = {20, 30, 0, 0};
        REQUIRE_CALL(ShaderCursor(root)["params"]["values"].setData(values, sizeof(values)));
        REQUIRE_CALL(ShaderCursor(root)["params"]["output"].setBinding(output));
    }
};

void submit(IDevice* device, ICommandEncoder* encoder)
{
    ComPtr<ICommandBuffer> commands;
    REQUIRE_CALL(encoder->finish(commands.writeRef()));
    auto queue = device->getQueue(QueueType::Graphics);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
}

class ErrorRecorder : public IDebugCallback
{
public:
    std::vector<std::string> errors;

    virtual SLANG_NO_THROW void SLANG_MCALL handleMessage(
        DebugMessageType type,
        DebugMessageSource source,
        const char* message
    ) override
    {
        SLANG_UNUSED(source);
        if (type == DebugMessageType::Error)
            errors.emplace_back(message);
    }

    size_t count(const char* text) const
    {
        size_t result = 0;
        for (const auto& error : errors)
            result += error.find(text) != std::string::npos ? 1 : 0;
        return result;
    }
};

ComPtr<IBuffer> createUintBuffer(IDevice* device, BufferUsage usage, std::initializer_list<uint32_t> values)
{
    BufferDesc desc = {};
    desc.size = values.size() * sizeof(uint32_t);
    desc.elementSize = sizeof(uint32_t);
    desc.usage = usage | BufferUsage::CopySource | BufferUsage::CopyDestination;
    auto buffer = device->createBuffer(desc, values.begin());
    REQUIRE(buffer);
    return buffer;
}

} // namespace

// One root serves every dispatch of a pass. Only the block changes between them, so the tables are built once.
GPU_TEST_CASE("push-constant-block-compute", D3D12 | Vulkan)
{
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-push-constant-block", "computeMain", program.writeRef()));
    checkBlockLayout(device, program, 16);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    for (bool explicitRoot : {false, true})
    {
        CAPTURE(explicitRoot);
        Resources resources(device);
        auto queue = device->getQueue(QueueType::Graphics);
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        ComPtr<IShaderObject> root;
        if (explicitRoot)
        {
            REQUIRE_CALL(device->createRootShaderObject(program, root.writeRef()));
            pass->bindPipeline(pipeline, root);
        }
        else
            root = pass->bindPipeline(pipeline);
        REQUIRE(root);
        resources.bind(root);
        const std::array<uint32_t, 4> dispatches[] = {{0, 2, 0, 0}, {1, 9, 0, 0}};
        for (const auto& dispatch : dispatches)
        {
            REQUIRE_CALL(ShaderCursor(root)["root"]["dispatch"].setData(dispatch));
            pass->dispatchCompute(1, 1, 1);
        }
        pass->end();
        ComPtr<ICommandBuffer> commands;
        REQUIRE_CALL(encoder->finish(commands.writeRef()));
#if SLANG_RHI_ENABLE_BINDING_COUNTERS
        // The root and `params` tables each miss once; `params.values` is the only constant buffer.
        CHECK(getBindingCounters(commands).tableMisses == 2);
        CHECK(getBindingCounters(commands).tableHits == 2);
        CHECK(getBindingCounters(commands).cbAllocations == 1);
#endif
        REQUIRE_CALL(queue->submit(commands));
        REQUIRE_CALL(queue->waitOnHost());
        compareComputeResult(device, resources.output, std::array<uint32_t, 2>{22, 29});
    }
}

// A global constant buffer keeps b0 and its view. The block takes the next register without one, also when an
// entry-point parameter block claims another space.
GPU_TEST_CASE("push-constant-block-beside-constant-buffer", D3D12 | Vulkan)
{
    for (bool entryBlock : {false, true})
    {
        CAPTURE(entryBlock);
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(loadProgram(
            device,
            "test-push-constant-block-global-buffer",
            entryBlock ? "computeWithBlock" : "computeMain",
            program.writeRef()
        ));
        checkBlockLayout(device, program, 12);
        auto pipeline = device->createComputePipeline({.program = program});
        REQUIRE(pipeline);
        Resources resources(device);
        auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        auto root = pass->bindPipeline(pipeline);
        REQUIRE(root);
        ShaderCursor cursor(root);
        REQUIRE_CALL(cursor["globals"]["values"].setData(std::array<uint32_t, 4>{20, 0, 0, 0}));
        if (entryBlock)
            REQUIRE_CALL(ShaderCursor(root->getEntryPoint(0))["outputs"]["output"].setBinding(resources.output));
        else
            REQUIRE_CALL(cursor["output"].setBinding(resources.output));
        REQUIRE_CALL(cursor["step"]["slot"].setData(uint32_t(1)));
        REQUIRE_CALL(cursor["step"]["value"].setData(uint32_t(7)));
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        submit(device, encoder);
        compareComputeResult(device, resources.output, std::array<uint32_t, 2>{0, 27});
    }
}

// Both stages read the one block, and each draw of a pass sees the bytes written right before it.
GPU_TEST_CASE("push-constant-block-render", D3D12 | Vulkan)
{
    Resources resources(device);
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-push-constant-block", {"vertexMain", "fragmentMain"}, program.writeRef()));
    checkBlockLayout(device, program, 16);
    ColorTargetDesc colorTarget = {};
    colorTarget.format = Format::RGBA32Float;
    RenderPipelineDesc desc = {};
    desc.program = program;
    desc.targetCount = 1;
    desc.targets = &colorTarget;
    desc.rasterizer.cullMode = CullMode::None;
    auto pipeline = device->createRenderPipeline(desc);
    REQUIRE(pipeline);
    TextureDesc textureDesc = {};
    textureDesc.size = {4, 4, 1};
    textureDesc.format = Format::RGBA32Float;
    textureDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    auto texture = device->createTexture(textureDesc);
    REQUIRE(texture);

    auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
    RenderPassColorAttachment color = {};
    color.view = texture->getDefaultView();
    color.loadOp = LoadOp::Clear;
    RenderPassDesc passDesc = {};
    passDesc.colorAttachmentCount = 1;
    passDesc.colorAttachments = &color;
    auto pass = encoder->beginRenderPass(passDesc);
    auto root = pass->bindPipeline(pipeline);
    REQUIRE(root);
    resources.bind(root);
    REQUIRE_CALL(ShaderCursor(root->getEntryPoint(1))["extra"]["bias"].setData(uint32_t(5)));
    // Each draw covers one half of the target.
    const std::array<uint32_t, 4> dispatches[] = {{0, 2, 3, 0}, {0, 4, 7, 0}};
    for (uint32_t i = 0; i < 2; ++i)
    {
        REQUIRE_CALL(ShaderCursor(root)["root"]["dispatch"].setData(dispatches[i]));
        RenderState state = {};
        state.viewportCount = state.scissorRectCount = 1;
        state.viewports[0] = Viewport::fromSize(4, 4);
        state.scissorRects[0] = {i * 2, 0, i * 2 + 2, 4};
        pass->setRenderState(state);
        DrawArguments draw = {};
        draw.vertexCount = 3;
        pass->draw(draw);
    }
    pass->end();
    submit(device, encoder);

    ComPtr<ISlangBlob> pixels;
    SubresourceLayout layout = {};
    REQUIRE_CALL(device->readTexture(texture, 0, 0, pixels.writeRef(), &layout));
    for (uint32_t y = 0; y < 4; ++y)
    {
        const auto row = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(pixels->getBufferPointer()) + y * layout.rowPitch
        );
        for (uint32_t x = 0; x < 4; ++x)
        {
            CAPTURE(x);
            CHECK(row[x * 4] == (x < 2 ? 22.f : 24.f));
            CHECK(row[x * 4 + 1] == (x < 2 ? 38.f : 42.f));
        }
    }
}

GPU_TEST_CASE("push-constant-block-ray-tracing", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing unsupported");
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-push-constant-block-rt", "rayMain", program.writeRef()));
    checkBlockLayout(device, program, 16);
    RayTracingPipelineDesc desc = {};
    desc.program = program;
    auto pipeline = device->createRayTracingPipeline(desc);
    REQUIRE(pipeline);
    const char* entry = "rayMain";
    ShaderTableDesc tableDesc = {};
    tableDesc.program = program;
    tableDesc.rayGenShaderCount = 1;
    tableDesc.rayGenShaderEntryPointNames = &entry;
    ComPtr<IShaderTable> table;
    REQUIRE_CALL(device->createShaderTable(tableDesc, table.writeRef()));
    REQUIRE(table);
    Resources resources(device);
    auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
    auto pass = encoder->beginRayTracingPass();
    auto root = pass->bindPipeline(pipeline, table);
    REQUIRE(root);
    resources.bind(root);
    const std::array<uint32_t, 4> dispatches[] = {{0, 2, 0, 0}, {1, 9, 0, 0}};
    for (const auto& dispatch : dispatches)
    {
        REQUIRE_CALL(ShaderCursor(root)["root"]["dispatch"].setData(dispatch));
        pass->dispatchRays(0, 1, 1, 1);
    }
    pass->end();
    submit(device, encoder);
    compareComputeResult(device, resources.output, std::array<uint32_t, 2>{22, 29});
}

// No cached table covers the block, so every bind checks the state of the resources its handles name.
GPU_TEST_CASE("push-constant-block-handle-states", D3D12 | Vulkan)
{
    ErrorRecorder recorder;
    DeviceExtraOptions options;
    options.resourceStateMode = ResourceStateMode::Explicit;
    options.validateResourceStates = true;
    options.debugCallback = &recorder;
    device = createTestingDevice(ctx, ctx->deviceType, false, &options);
    REQUIRE(device != nullptr);
    if (!device->hasFeature(Feature::Bindless))
        SKIP("bindless unsupported");

    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(
        device,
        R"(
import push_constants;
struct Dispatch { RWStructuredBuffer<uint>.Handle output; uint value; };
[PushConstants] [[vk::push_constant]] ConstantBuffer<Dispatch> dispatch;
[shader("compute")][numthreads(1, 1, 1)]
void main()
{
    dispatch.output[0] = dispatch.value;
}
)",
        program.writeRef()
    ));
    checkBlockLayout(device, program, 12);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline != nullptr);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    BufferDesc outputDesc{};
    outputDesc.size = sizeof(uint32_t);
    outputDesc.elementSize = sizeof(uint32_t);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    outputDesc.label = "BlockHandleOutput";
    auto output = device->createBuffer(outputDesc);
    REQUIRE(output != nullptr);
    ShaderCursor block = ShaderCursor(root)["dispatch"];
    REQUIRE_CALL(block["output"]
                     .setDescriptorHandle(output, DescriptorHandleAccess::ReadWrite, Format::Undefined, 0, kEntireBuffer));
    REQUIRE_CALL(block["value"].setData(uint32_t(7)));

    auto queue = device->getQueue(QueueType::Graphics);
    auto dispatchTwice = [&](ResourceState declared)
    {
        auto encoder = queue->createCommandEncoder();
        encoder->setBufferState(output, declared);
        auto* pass = encoder->beginComputePass();
        pass->bindPipeline(pipeline, root);
        // The second dispatch finds the root table cached.
        pass->dispatchCompute(1, 1, 1);
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        REQUIRE_CALL(queue->submit(encoder->finish()));
        REQUIRE_CALL(queue->waitOnHost());
    };

    dispatchTwice(ResourceState::UnorderedAccess);
    CHECK(recorder.errors.empty());

    dispatchTwice(ResourceState::ShaderResource);
    CHECK(recorder.errors.size() == 2);
    CHECK(recorder.count("BlockHandleOutput") == 2);
    CHECK(recorder.count(enumToString(ResourceState::UnorderedAccess)) == 2);
}

// A resource written into the block may lose its last caller reference before the commands run.
GPU_TEST_CASE("push-constant-block-handle-lifetime", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::Bindless))
        SKIP("bindless unsupported");
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(
        device,
        R"(
import push_constants;
struct Dispatch
{
    StructuredBuffer<uint>.Handle input;
    RWStructuredBuffer<uint>.Handle output;
    uint index;
};
[PushConstants] [[vk::push_constant]] ConstantBuffer<Dispatch> dispatch;
[shader("compute")][numthreads(1, 1, 1)]
void main()
{
    dispatch.output[dispatch.index] = dispatch.input[0] + dispatch.index;
}
)",
        program.writeRef()
    ));
    checkBlockLayout(device, program, 20);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    auto output = createUintBuffer(device, BufferUsage::UnorderedAccess, {0, 0});
    auto root = device->createRootShaderObject(program);
    REQUIRE(root);
    ShaderCursor block = ShaderCursor(root)["dispatch"];
    REQUIRE_CALL(block["output"]
                     .setDescriptorHandle(output, DescriptorHandleAccess::ReadWrite, Format::Undefined, 0, kEntireBuffer));

    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto* pass = encoder->beginComputePass();
    pass->bindPipeline(pipeline, root);
    for (uint32_t index = 0; index < 2; ++index)
    {
        auto input = createUintBuffer(device, BufferUsage::ShaderResource, {40 + index * 10});
        REQUIRE_CALL(block["input"]
                         .setDescriptorHandle(input, DescriptorHandleAccess::Read, Format::Undefined, 0, kEntireBuffer));
        REQUIRE_CALL(block["index"].setData(index));
        pass->dispatchCompute(1, 1, 1);
    }
    pass->end();
    root.setNull();
    auto commands = encoder->finish();
    REQUIRE(commands);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, output, std::array<uint32_t, 2>{40, 51});
}

GPU_TEST_CASE("push-constant-block-acceleration-structure", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::Bindless) || !device->hasFeature(Feature::RayQuery))
        SKIP("bindless ray queries unsupported");
    ComPtr<ICommandQueue> queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue);
    TLAS tlas(device, queue, blas.blas);

    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(
        device,
        R"(
import push_constants;
struct Query
{
    RaytracingAccelerationStructure.Handle scene;
    RWStructuredBuffer<uint>.Handle result;
};
[PushConstants] [[vk::push_constant]] ConstantBuffer<Query> query;
[shader("compute")][numthreads(1, 1, 1)]
void main()
{
    RayDesc ray;
    ray.Origin = float3(0.25, 0.25, 0.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 100.0;
    RayQuery<RAY_FLAG_NONE> rayQuery;
    rayQuery.TraceRayInline(query.scene, RAY_FLAG_NONE, 0xff, ray);
    while (rayQuery.Proceed())
    {
    }
    query.result[0] = rayQuery.CommittedStatus();
}
)",
        program.writeRef()
    ));
    checkBlockLayout(device, program, 16);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    auto result = createUintBuffer(device, BufferUsage::UnorderedAccess, {0});

    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    auto root = pass->bindPipeline(pipeline);
    REQUIRE(root);
    ShaderCursor block = ShaderCursor(root)["query"];
    REQUIRE_CALL(block["scene"].setDescriptorHandle(tlas.tlas.get()));
    REQUIRE_CALL(block["result"]
                     .setDescriptorHandle(result, DescriptorHandleAccess::ReadWrite, Format::Undefined, 0, kEntireBuffer));
    pass->dispatchCompute(1, 1, 1);
    pass->end();
    submit(device, encoder);
    // COMMITTED_TRIANGLE_HIT
    compareComputeResult(device, result, std::array<uint32_t, 1>{1});
}

GPU_TEST_CASE("push-constant-block-nvapi", D3D12)
{
    if (!device->hasFeature(Feature::Bindless) || !device->hasCapability(Capability::hlsl_nvapi) ||
        !device->hasFeature(Feature::AtomicFloat))
        SKIP("NVAPI float atomics unavailable");
    const char* source = R"(
import push_constants;
struct Root { RWByteAddressBuffer.Handle output; };
[PushConstants] [[vk::push_constant]] ConstantBuffer<Root> root;
[shader("compute")][numthreads(1, 1, 1)]
void main()
{
    RWByteAddressBuffer output = root.output;
    float original;
    output.InterlockedAddF32(0, 2.5f, original);
}
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    checkBlockLayout(device, program, 8);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    BufferDesc desc = {};
    desc.size = sizeof(float);
    desc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    desc.defaultState = ResourceState::UnorderedAccess;
    float initial = 1.f;
    auto output = device->createBuffer(desc, &initial);
    REQUIRE(output);
    auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    auto root = pass->bindPipeline(pipeline);
    REQUIRE_CALL(ShaderCursor(root)["root"]["output"]
                     .setDescriptorHandle(output, DescriptorHandleAccess::ReadWrite, Format::Undefined, 0, kEntireBuffer));
    pass->dispatchCompute(1, 1, 1);
    pass->end();
    submit(device, encoder);
    compareComputeResult(device, output, std::array<float, 1>{3.5f});
}

// The constants sit between the root descriptors and the tables, and roots that share a parameter block still
// share its table when their blocks differ.
GPU_TEST_CASE("push-constant-block-shared-parameter-block", D3D12 | Vulkan)
{
    const char* source = R"(
import push_constants;
[__AttributeUsage(_AttributeTargets.Var)] struct rootAttribute {};
struct Nested { [root] RWStructuredBuffer<uint> output; };
struct Block { uint base; ParameterBlock<Nested> nested; };
struct Dispatch { uint index; uint value; };
ParameterBlock<Block> params;
[root] StructuredBuffer<uint> input;
[PushConstants] [[vk::push_constant]] ConstantBuffer<Dispatch> dispatch;
[shader("compute")][numthreads(1,1,1)]
void main(uniform RWStructuredBuffer<uint> direct)
{
    params.nested.output[dispatch.index] += params.base + dispatch.value + input[0];
    direct[dispatch.index] += dispatch.value;
})";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    checkBlockLayout(device, program, 8);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    const BufferUsage usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess;
    auto output = createUintBuffer(device, usage, {0, 0});
    auto direct = createUintBuffer(device, usage, {0, 0});
    auto input = createUintBuffer(device, usage, {3, 0});
    auto a = device->createRootShaderObject(program);
    auto b = device->createRootShaderObject(program);
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE_CALL(ShaderCursor(a)["params"]["base"].setData(uint32_t(20)));
    REQUIRE_CALL(ShaderCursor(a)["params"]["nested"]["output"].setBinding(output));
    ComPtr<IShaderObject> block;
    REQUIRE_CALL(a->getObject(ShaderCursor(a)["params"].m_offset, block.writeRef()));
    REQUIRE_CALL(ShaderCursor(b)["params"].setObject(block));
    for (auto root : {a.get(), b.get()})
    {
        REQUIRE_CALL(ShaderCursor(root)["input"].setBinding(input));
        REQUIRE_CALL(ShaderCursor(root->getEntryPoint(0))["direct"].setBinding(direct));
    }
    REQUIRE_CALL(ShaderCursor(a)["dispatch"]["index"].setData(uint32_t(0)));
    REQUIRE_CALL(ShaderCursor(a)["dispatch"]["value"].setData(uint32_t(2)));
    REQUIRE_CALL(ShaderCursor(b)["dispatch"]["index"].setData(uint32_t(1)));
    REQUIRE_CALL(ShaderCursor(b)["dispatch"]["value"].setData(uint32_t(9)));
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    for (auto root : {a.get(), b.get(), a.get()})
    {
        pass->bindPipeline(pipeline, root);
        pass->dispatchCompute(1, 1, 1);
    }
    pass->end();
    ComPtr<ICommandBuffer> commands;
    REQUIRE_CALL(encoder->finish(commands.writeRef()));
#if SLANG_RHI_ENABLE_BINDING_COUNTERS
    CHECK(getBindingCounters(commands).cbAllocations == 1);
    CHECK(getBindingCounters(commands).tableHits >= 3);
#endif
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, output, std::array<uint32_t, 2>{50, 32});
    compareComputeResult(device, direct, std::array<uint32_t, 2>{4, 9});
}

// D3D12 checks a block where it builds the root signature. Vulkan only limits the total size of the ranges.
GPU_TEST_CASE("push-constant-block-conditions", D3D12 | Vulkan)
{
    ErrorRecorder recorder;
    DeviceExtraOptions options;
    options.debugCallback = &recorder;
    device = createTestingDevice(ctx, ctx->deviceType, false, &options);
    REQUIRE(device != nullptr);

    struct Case
    {
        const char* name;
        const char* declarations;
        const char* expression;
        // Null when the block is admitted.
        const char* error;
    };
    const Case cases[] = {
        {
            "64 bytes",
            "struct Data { uint4 a; uint4 b; uint4 c; uint4 d; };\n"
            "[PushConstants] [[vk::push_constant]] ConstantBuffer<Data> data;\n",
            "data.a.x + data.d.w",
            nullptr,
        },
        {
            "80 bytes",
            "struct Data { uint4 a; uint4 b; uint4 c; uint4 d; uint4 e; };\n"
            "[PushConstants] [[vk::push_constant]] ConstantBuffer<Data> data;\n",
            "data.a.x + data.e.w",
            "Push-constant block 'data' has 80 bytes",
        },
        {
            "array",
            "struct Data { uint4 a[2]; };\n"
            "[PushConstants] [[vk::push_constant]] ConstantBuffer<Data> data;\n",
            "data.a[0].x + data.a[1].w",
            "Push-constant block 'data' has 32 bytes",
        },
        {
            "odd size",
            "struct Data { uint16_t a; uint16_t b; uint16_t c; };\n"
            "[PushConstants] [[vk::push_constant]] ConstantBuffer<Data> data;\n",
            "data.a + data.c",
            "Push-constant block 'data' has 6 bytes",
        },
        {
            "two blocks",
            "struct Data { uint a; };\n"
            "[PushConstants] [[vk::push_constant]] ConstantBuffer<Data> first;\n"
            "[PushConstants] [[vk::push_constant]] ConstantBuffer<Data> second;\n",
            "first.a + second.a",
            "push-constant blocks 'first' and 'second'",
        },
    };
    for (const auto& test : cases)
    {
        CAPTURE(test.name);
        recorder.errors.clear();
        const std::string source = std::string("import push_constants;\nRWStructuredBuffer<uint> output;\n") +
                                   test.declarations + "[shader(\"compute\")][numthreads(1,1,1)] void main() { output[0] = " +
                                   test.expression + "; }";
        ComPtr<IShaderProgram> program;
        if (test.error)
        {
            if (device->getDeviceType() != DeviceType::D3D12)
                continue;
            CHECK(SLANG_FAILED(loadComputeProgramFromSource(device, source, program.writeRef())));
            CHECK(recorder.count(test.error) == 1);
            continue;
        }
        REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
        checkBlockLayout(device, program, 64);
        auto pipeline = device->createComputePipeline({.program = program});
        REQUIRE(pipeline);
        auto output = createUintBuffer(device, BufferUsage::UnorderedAccess, {0});
        auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        auto root = pass->bindPipeline(pipeline);
        REQUIRE_CALL(ShaderCursor(root)["output"].setBinding(output));
        REQUIRE_CALL(ShaderCursor(root)["data"]["a"].setData(std::array<uint32_t, 4>{3, 0, 0, 0}));
        REQUIRE_CALL(ShaderCursor(root)["data"]["d"].setData(std::array<uint32_t, 4>{0, 0, 0, 4}));
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        submit(device, encoder);
        compareComputeResult(device, output, std::array<uint32_t, 1>{7});
        CHECK(recorder.errors.empty());
    }
}

// Root constants share the 64 DWORDs of a root signature with every table and root descriptor.
GPU_TEST_CASE("push-constant-block-over-root-signature-budget", D3D12)
{
    ErrorRecorder recorder;
    DeviceExtraOptions options;
    options.debugCallback = &recorder;
    device = createTestingDevice(ctx, ctx->deviceType, false, &options);
    REQUIRE(device != nullptr);

    std::string source = "import push_constants;\nstruct Block { uint value; };\nRWStructuredBuffer<uint> output;\n";
    for (int i = 0; i < 49; ++i)
        source += "ParameterBlock<Block> p" + std::to_string(i) + ";\n";
    source += "struct Data { uint4 a; uint4 b; uint4 c; uint4 d; };\n"
              "[PushConstants] [[vk::push_constant]] ConstantBuffer<Data> data;\n"
              "[shader(\"compute\")][numthreads(1,1,1)] void main() { output[0] = data.a.x";
    for (int i = 0; i < 49; ++i)
        source += " + p" + std::to_string(i) + ".value";
    source += "; }";
    ComPtr<IShaderProgram> program;
    CHECK(SLANG_FAILED(loadComputeProgramFromSource(device, source, program.writeRef())));
    CHECK(recorder.count("Push-constant block 'data' does not fit") == 1);
}
