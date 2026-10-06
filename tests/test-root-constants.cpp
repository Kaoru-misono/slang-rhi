#include "testing.h"
#include "shader-object.h"
#include "debug-layer/debug-shader-object.h"
#include "debug-layer/debug-command-buffer.h"
#include "command-buffer.h"
#include "device.h"
#include "d3d12/d3d12-shader-object-layout.h"
#include "vulkan/vk-shader-object-layout.h"

using namespace rhi;
using namespace rhi::testing;

namespace {

RootShaderObject* getNativeRoot(IShaderObject* object)
{
    auto* wrapped = dynamic_cast<debug::DebugShaderObject*>(object);
    return checked_cast<RootShaderObject*>(wrapped ? wrapped->baseObject.get() : object);
}

void checkRootLayout(IDevice* device, IShaderProgram* program, uint32_t expectedSize)
{
    auto root = device->createRootShaderObject(program);
    REQUIRE(root);
    auto layout = getNativeRoot(root)->m_layout.get();
    if (device->getDeviceType() == DeviceType::D3D12)
    {
        auto native = checked_cast<d3d12::RootShaderObjectLayoutImpl*>(layout);
        for (const auto& entry : native->m_entryPoints)
        {
            CHECK(entry.layout->m_ordinaryDataIsConstants == (expectedSize != 0));
            if (expectedSize)
            {
                CHECK(entry.layout->getTotalOrdinaryDataSize() == expectedSize);
                CHECK(entry.layout->getOrdinaryDataBufferCount() == 0);
                CHECK(entry.constantIndex != UINT32_MAX);
            }
        }
    }
    else if (expectedSize)
    {
        auto native = checked_cast<vk::RootShaderObjectLayoutImpl*>(layout);
        CHECK(native->m_coalescedEntryPointConstants);
        REQUIRE(native->getAllPushConstantRanges().size() == 1);
        CHECK(native->getAllPushConstantRanges()[0].size == expectedSize);
    }
}

struct Root
{
    uint32_t dispatch[4] = {};
};
static_assert(sizeof(Root) == 16);

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

} // namespace

GPU_TEST_CASE("root-constants-compute", D3D12 | Vulkan)
{
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-root-constants", "computeMain", program.writeRef()));
    checkRootLayout(device, program, 16);
    ComputePipelineDesc desc = {};
    desc.program = program;
    auto pipeline = device->createComputePipeline(desc);
    REQUIRE(pipeline);
    for (bool explicitRoot : {false, true})
    {
        CAPTURE(explicitRoot);
        Resources resources(device);
        auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        Root root{{0, 2, 0, 0}};
        ComPtr<IShaderObject> first;
        ComPtr<IShaderObject> second;
        if (explicitRoot)
        {
            REQUIRE_CALL(device->createRootShaderObject(program, first.writeRef()));
            REQUIRE_CALL(device->createRootShaderObject(program, second.writeRef()));
            pass->bindPipeline(pipeline, first);
        }
        else
            first = pass->bindPipeline(pipeline);
        REQUIRE(first);
        resources.bind(first);
        REQUIRE_CALL(ShaderCursor(first->getEntryPoint(0))["root"].setData(&root, sizeof(root)));
        pass->dispatchCompute(1, 1, 1);
        root.dispatch[0] = 1;
        root.dispatch[1] = 9;
        if (explicitRoot)
        {
            resources.bind(second);
            REQUIRE_CALL(ShaderCursor(second->getEntryPoint(0))["root"].setData(&root, sizeof(root)));
            pass->bindPipeline(pipeline, second);
        }
        else
            REQUIRE_CALL(ShaderCursor(first->getEntryPoint(0))["root"].setData(&root, sizeof(root)));
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        submit(device, encoder);
        compareComputeResult(device, resources.output, std::array<uint32_t, 2>{22, 29});
    }
}

GPU_TEST_CASE("root-constants-render", D3D12 | Vulkan)
{
    Resources resources(device);
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-root-constants", {"vertexMain", "fragmentMain"}, program.writeRef()));
    checkRootLayout(device, program, 16);
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
    for (uint32_t writeMode : {0u, 1u, 2u, 3u})
        for (bool explicitRoot : {false, true})
        {
            CAPTURE(explicitRoot);
            CAPTURE(writeMode);
            auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
            RenderPassColorAttachment color = {};
            color.view = texture->getDefaultView();
            color.loadOp = LoadOp::Clear;
            RenderPassDesc passDesc = {};
            passDesc.colorAttachmentCount = 1;
            passDesc.colorAttachments = &color;
            auto pass = encoder->beginRenderPass(passDesc);
            Root root{{0, 2, 3, 0}};
            ComPtr<IShaderObject> object;
            if (explicitRoot)
            {
                REQUIRE_CALL(device->createRootShaderObject(program, object.writeRef()));
                pass->bindPipeline(pipeline, object);
            }
            else
                object = pass->bindPipeline(pipeline);
            REQUIRE(object);
            resources.bind(object);
            REQUIRE_CALL(ShaderCursor(object->getEntryPoint(1))["extra"]["bias"].setData(uint32_t(5)));
            for (uint32_t i = 0; i < 2; ++i)
            {
                Root value = root;
                if (writeMode == 0)
                    value = {};
                if (writeMode == 1 && i == 1)
                    value.dispatch[2]++;
                if (writeMode == 3 && i == 1)
                    value = {};
                REQUIRE_CALL(ShaderCursor(object->getEntryPoint(i))["root"].setData(&value, sizeof(value)));
            }
            RenderState state = {};
            state.viewportCount = state.scissorRectCount = 1;
            state.viewports[0] = Viewport::fromSize(4, 4);
            state.scissorRects[0] = ScissorRect::fromSize(4, 4);
            pass->setRenderState(state);
            DrawArguments draw = {};
            draw.vertexCount = 3;
            if ((writeMode == 1 || writeMode == 3) && device->getDeviceType() == DeviceType::Vulkan)
            {
                ExpectedErrorScope expected;
                struct Capture : IDebugCallback
                {
                    Device* device;
                    IDebugCallback* previous;
                    std::string messages;
                    Capture(Device* device)
                        : device(device)
                        , previous(device->m_debugCallback)
                    {
                        device->m_debugCallback = this;
                    }
                    ~Capture() { device->m_debugCallback = previous; }
                    void SLANG_MCALL handleMessage(
                        DebugMessageType type,
                        DebugMessageSource source,
                        const char* text
                    ) override
                    {
                        messages += text;
                        previous->handleMessage(type, source, text);
                    }
                } capture(getNativeRoot(object)->m_layout->getDevice());
                pass->draw(draw);
                pass->end();
                ComPtr<ICommandBuffer> commands;
                CHECK(SLANG_FAILED(encoder->finish(commands.writeRef())));
                CHECK(capture.messages.find("vertexMain") != std::string::npos);
                CHECK(capture.messages.find("fragmentMain") != std::string::npos);
                continue;
            }
            pass->draw(draw);
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
                    CHECK(row[x * 4] == (writeMode == 0 ? 20.f : 22.f));
                    CHECK(row[x * 4 + 1] == (writeMode == 0 || writeMode == 3 ? 35.f : writeMode == 1 ? 39.f : 38.f));
                }
            }
        }
}

GPU_TEST_CASE("root-constants-ray-tracing", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing unsupported");
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-root-constants-rt", "rayMain", program.writeRef()));
    checkRootLayout(device, program, 0);
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
    for (bool explicitRoot : {false, true})
    {
        CAPTURE(explicitRoot);
        Resources resources(device);
        auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
        auto pass = encoder->beginRayTracingPass();
        Root root{{0, 2, 0, 0}};
        ComPtr<IShaderObject> object;
        if (explicitRoot)
        {
            REQUIRE_CALL(device->createRootShaderObject(program, object.writeRef()));
            pass->bindPipeline(pipeline, table, object);
        }
        else
            object = pass->bindPipeline(pipeline, table);
        REQUIRE(object);
        resources.bind(object);
        REQUIRE_CALL(
            ShaderCursor(object)["root"].getDereferenced()["dispatch"].setData(root.dispatch, sizeof(root.dispatch))
        );
        pass->dispatchRays(0, 1, 1, 1);
        root.dispatch[0] = 1;
        root.dispatch[1] = 9;
        REQUIRE_CALL(
            ShaderCursor(object)["root"].getDereferenced()["dispatch"].setData(root.dispatch, sizeof(root.dispatch))
        );
        pass->dispatchRays(0, 1, 1, 1);
        pass->end();
        submit(device, encoder);
        compareComputeResult(device, resources.output, std::array<uint32_t, 2>{22, 29});
    }
}

GPU_TEST_CASE("root-constants-resource-binding", D3D12 | Vulkan)
{
    const char* source = R"(
RWStructuredBuffer<uint> output;
[shader("compute")][numthreads(1, 1, 1)]
void main(uniform uint root) { output[0] = root; }
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    checkRootLayout(device, program, 4);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    BufferDesc desc = {};
    desc.size = desc.elementSize = sizeof(uint32_t);
    desc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    desc.defaultState = ResourceState::UnorderedAccess;
    uint32_t zero = 0;
    auto output = device->createBuffer(desc, &zero);
    REQUIRE(output);
    auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    auto root = pass->bindPipeline(pipeline);
    REQUIRE_CALL(ShaderCursor(root)["output"].setBinding(output));
    REQUIRE_CALL(ShaderCursor(root)["root"].setData(uint32_t(42)));
    pass->dispatchCompute(1, 1, 1);
    pass->end();
    submit(device, encoder);
    compareComputeResult(device, output, std::array<uint32_t, 1>{42});
}

GPU_TEST_CASE("root-constants-nvapi", D3D12)
{
    if (!device->hasFeature(Feature::Bindless) || !device->hasCapability(Capability::hlsl_nvapi) ||
        !device->hasFeature(Feature::AtomicFloat))
        SKIP("NVAPI float atomics unavailable");
    const char* source = R"(
struct Root { RWByteAddressBuffer.Handle output; };
[shader("compute")][numthreads(1, 1, 1)]
void main(uniform Root root)
{
    RWByteAddressBuffer output = root.output;
    float original;
    output.InterlockedAddF32(0, 2.5f, original);
}
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    checkRootLayout(device, program, 8);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    BufferDesc desc = {};
    desc.size = sizeof(float);
    desc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    desc.defaultState = ResourceState::UnorderedAccess;
    float initial = 1.f;
    auto output = device->createBuffer(desc, &initial);
    REQUIRE(output);
    DescriptorHandle handle = {};
    REQUIRE_CALL(
        output->getDescriptorHandle(DescriptorHandleAccess::ReadWrite, Format::Undefined, kEntireBuffer, &handle)
    );
    auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    auto root = pass->bindPipeline(pipeline);
    REQUIRE_CALL(ShaderCursor(root)["root"]["output"].setDescriptorHandle(handle, output));
    pass->dispatchCompute(1, 1, 1);
    pass->end();
    submit(device, encoder);
    compareComputeResult(device, output, std::array<float, 1>{3.5f});
}

GPU_TEST_CASE("root-constants-shared-block-aba", D3D12 | Vulkan)
{
    const char* source = R"(
[__AttributeUsage(_AttributeTargets.Var)] struct rootAttribute {};
struct Nested { [root] RWStructuredBuffer<uint> output; };
struct Block { uint base; ParameterBlock<Nested> nested; };
ParameterBlock<Block> params;
[root] StructuredBuffer<uint> input;
[shader("compute")][numthreads(1,1,1)]
void main(uniform uint index, uniform uint value, uniform RWStructuredBuffer<uint> direct)
{
    params.nested.output[index] += params.base + value + input[0];
    direct[index] += value;
})";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    checkRootLayout(device, program, 8);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    BufferDesc desc = {};
    desc.size = 8;
    desc.elementSize = 4;
    desc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    desc.defaultState = ResourceState::UnorderedAccess;
    uint32_t zeros[2] = {};
    auto output = device->createBuffer(desc, zeros);
    auto direct = device->createBuffer(desc, zeros);
    uint32_t inputs[2] = {3, 0};
    auto input = device->createBuffer(desc, inputs);
    REQUIRE(output);
    REQUIRE(direct);
    REQUIRE(input);
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
    REQUIRE_CALL(ShaderCursor(a)["index"].setData(uint32_t(0)));
    REQUIRE_CALL(ShaderCursor(a)["value"].setData(uint32_t(2)));
    REQUIRE_CALL(ShaderCursor(b)["index"].setData(uint32_t(1)));
    REQUIRE_CALL(ShaderCursor(b)["value"].setData(uint32_t(9)));
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
    auto wrapped = dynamic_cast<debug::DebugCommandBuffer*>(commands.get());
    auto native = checked_cast<CommandBuffer*>(wrapped ? wrapped->baseObject.get() : commands.get());
    CHECK(native->m_bindingCounters.cbAllocations == 1);
    CHECK(native->m_bindingCounters.tableHits >= 3);
#endif
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, output, std::array<uint32_t, 2>{50, 32});
    compareComputeResult(device, direct, std::array<uint32_t, 2>{4, 9});
}

GPU_TEST_CASE("root-constants-promotion-boundaries", D3D12 | Vulkan)
{
    struct Case
    {
        const char* fields;
        const char* expression;
        uint32_t size;
        bool promoted;
    };
    for (const auto& test :
         {Case{"", "7", 0, false},
          Case{"uint4 a; uint4 b; uint4 c; uint4 d;", "root.a.x + root.d.w", 64, true},
          Case{"uint4 a; uint4 b; uint4 c; uint4 d; uint4 e;", "root.a.x + root.e.w", 80, false},
          Case{"uint4 a[2];", "root.a[0].x + root.a[1].w", 32, false}})
    {
        CAPTURE(test.size);
        std::string source = "RWStructuredBuffer<uint> output;\n";
        if (test.size)
            source += std::string("struct Data { ") + test.fields + " };\n";
        source += "[shader(\"compute\")][numthreads(1,1,1)] void main(";
        if (test.size)
            source += "uniform Data root";
        source += std::string(") { output[0] = ") + test.expression + "; }";
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(loadComputeProgramFromSource(device, source.c_str(), program.writeRef()));
        if (device->getDeviceType() == DeviceType::D3D12)
            checkRootLayout(device, program, test.promoted ? test.size : 0);
        auto pipeline = device->createComputePipeline({.program = program});
        REQUIRE(pipeline);
        BufferDesc desc = {};
        desc.size = desc.elementSize = 4;
        desc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
        auto output = device->createBuffer(desc);
        REQUIRE(output);
        auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        auto root = pass->bindPipeline(pipeline);
        REQUIRE_CALL(ShaderCursor(root)["output"].setBinding(output));
        if (test.size)
        {
            uint32_t data[20] = {};
            data[0] = 3;
            data[test.size / 4 - 1] = 4;
            REQUIRE_CALL(ShaderCursor(root)["root"].setData(data, test.size));
        }
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        submit(device, encoder);
        compareComputeResult(device, output, std::array<uint32_t, 1>{7});
    }
}

GPU_TEST_CASE("root-constants-over-budget", D3D12)
{
    std::string source = "struct Block { uint value; };\nRWStructuredBuffer<uint> output;\n";
    for (int i = 0; i < 49; ++i)
        source += "ParameterBlock<Block> p" + std::to_string(i) + ";\n";
    source +=
        "struct Data { uint4 a; uint4 b; uint4 c; uint4 d; };\n"
        "[shader(\"compute\")][numthreads(1,1,1)] void main(uniform Data root) { output[0] = root.a.x";
    for (int i = 0; i < 49; ++i)
        source += " + p" + std::to_string(i) + ".value";
    source += "; }";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source.c_str(), program.writeRef()));
    checkRootLayout(device, program, 0);
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    BufferDesc desc = {};
    desc.size = desc.elementSize = 4;
    desc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    auto output = device->createBuffer(desc);
    REQUIRE(output);
    auto encoder = device->getQueue(QueueType::Graphics)->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    auto root = pass->bindPipeline(pipeline);
    REQUIRE_CALL(ShaderCursor(root)["output"].setBinding(output));
    REQUIRE_CALL(ShaderCursor(root)["root"]["a"].setData(std::array<uint32_t, 4>{2, 0, 0, 0}));
    for (int i = 0; i < 49; ++i)
        REQUIRE_CALL(ShaderCursor(root)[("p" + std::to_string(i)).c_str()]["value"].setData(uint32_t(1)));
    pass->dispatchCompute(1, 1, 1);
    pass->end();
    submit(device, encoder);
    compareComputeResult(device, output, std::array<uint32_t, 1>{51});
}

GPU_TEST_CASE("root-constants-stage-layouts", D3D12 | Vulkan)
{
    // The vertex stage declares uint4 and the fragment stage float4: same size, different values.
    ComPtr<IShaderProgram> program;
    if (device->getDeviceType() == DeviceType::Vulkan)
    {
        ExpectedErrorScope expectedError;
        CHECK(SLANG_FAILED(
            loadProgram(device, "test-root-constants", {"vertexMain", "fragmentTint"}, program.writeRef())
        ));
        return;
    }
    REQUIRE_CALL(loadProgram(device, "test-root-constants", {"vertexMain", "fragmentTint"}, program.writeRef()));
    checkRootLayout(device, program, 16);
}
