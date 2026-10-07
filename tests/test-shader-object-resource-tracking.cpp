#include "testing.h"
#include "command-buffer.h"
#include "debug-layer/debug-command-buffer.h"
#include "debug-layer/debug-shader-object.h"
#include "binding-table-cache.h"
#include <array>
#include <cstring>
#include <string>
#include <vector>

#if SLANG_RHI_ENABLE_VULKAN
#include "vulkan/vk-api.h"
#include "vulkan/vk-shader-object-layout.h"
#endif

using namespace rhi;
using namespace rhi::testing;

GPU_TEST_CASE("shader-object-typed-descriptor-recorded-owners", D3D12 | Vulkan)
{
    const char* source = R"(
struct Scene { ByteAddressBuffer.Handle input; uint bias; uint padding; };
ParameterBlock<Scene> scene;
RWStructuredBuffer<uint> output;
[shader("compute")][numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { output[id.x] = scene.input.Load<uint>(id.x * 4) + scene.bias; }
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline != nullptr);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    ShaderCursor cursor(root);
    ComPtr<IShaderObject> child;
    REQUIRE_CALL(device->createShaderObjectFromTypeLayout(
        cursor["scene"].getTypeLayout()->getElementTypeLayout(), child.writeRef()));
    REQUIRE_CALL(cursor["scene"].setObject(child));
    ShaderCursor fields(child);
    const uint32_t values[2][4]{{3, 11, 19, 31}, {101, 107, 127, 149}};
    BufferDesc inputDesc{};
    inputDesc.size = sizeof(values[0]);
    inputDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
    inputDesc.defaultState = ResourceState::ShaderResource;
    auto input = device->createBuffer(inputDesc, values[0]);
    REQUIRE(input != nullptr);
    BufferDesc outputDesc{};
    outputDesc.size = sizeof(values[0]);
    outputDesc.elementSize = sizeof(uint32_t);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    std::array<ComPtr<IBuffer>, 2> outputs;
    std::array<ComPtr<ICommandBuffer>, 2> commands;
    auto queue = device->getQueue(QueueType::Graphics);
    for (uint32_t batch = 0; batch < 2; ++batch)
    {
        if (batch)
        {
            input.setNull();
            REQUIRE_CALL(queue->waitOnHost());
            input = device->createBuffer(inputDesc, values[1]);
            REQUIRE(input != nullptr);
        }
        const uint32_t initial[4]{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
        outputs[batch] = device->createBuffer(outputDesc, initial);
        REQUIRE(outputs[batch] != nullptr);
        REQUIRE_CALL(
            fields["input"].setDescriptorHandle(input, DescriptorHandleAccess::Read, Format::Undefined, 0, kEntireBuffer)
        );
        const uint32_t bias = batch ? 23 : 7;
        REQUIRE_CALL(fields["bias"].setData(bias));
        REQUIRE_CALL(cursor["output"].setBinding(outputs[batch]));
        auto encoder = queue->createCommandEncoder();
        auto* pass = encoder->beginComputePass();
        pass->bindPipeline(pipeline, root);
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        REQUIRE_CALL(encoder->finish({}, commands[batch].writeRef()));
    }
    root.setNull();
    child.setNull();
    REQUIRE_CALL(queue->submit(commands[1]));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expectedB{};
    std::array<uint32_t, 4> expectedA{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expectedA[i] = values[0][i] + 7;
        expectedB[i] = values[1][i] + 23;
    }
    compareComputeResult(device, outputs[1], expectedB);
    REQUIRE_CALL(queue->submit(commands[0]));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, outputs[0], expectedA);
}

GPU_TEST_CASE("shader-object-sampler-table-constants-only", D3D12 | Vulkan)
{
    const char* source = R"(
struct Params { uint index; uint bias; uint2 padding; };
ConstantBuffer<Params> params;
Texture2D<float> tex;
SamplerState samp;
RWStructuredBuffer<uint> output;
[shader("compute")][numthreads(1, 1, 1)]
void main() { output[params.index] = params.bias + uint(4 * tex.SampleLevel(samp, float2(0.5), 0)); }
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline != nullptr);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    const float texels[]{1.f, 2.f, 3.f, 4.f};
    TextureDesc textureDesc{};
    textureDesc.size = {2, 2, 1};
    textureDesc.format = Format::R32Float;
    textureDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
    const SubresourceData data{texels, 2 * sizeof(float), 0};
    auto texture = device->createTexture(textureDesc, &data);
    REQUIRE(texture != nullptr);
    SamplerDesc samplerDesc{};
    samplerDesc.minFilter = samplerDesc.magFilter = TextureFilteringMode::Linear;
    auto sampler = device->createSampler(samplerDesc);
    REQUIRE(sampler != nullptr);
    // Greater than D3D12's maximum shader-visible sampler heap capacity.
    constexpr uint32_t count = 4097;
    std::vector<uint32_t> initial(count, 0xdeadbeef);
    BufferDesc outputDesc{};
    outputDesc.size = count * sizeof(uint32_t);
    outputDesc.elementSize = sizeof(uint32_t);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    auto output = device->createBuffer(outputDesc, initial.data());
    REQUIRE(output != nullptr);
    ShaderCursor cursor(root);
    REQUIRE_CALL(cursor["tex"].setBinding(texture));
    REQUIRE_CALL(cursor["samp"].setBinding(sampler));
    REQUIRE_CALL(cursor["output"].setBinding(output));
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto* pass = encoder->beginComputePass();
    pass->bindPipeline(pipeline, root);
    std::vector<uint32_t> expected(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        const uint32_t params[]{i, i * 7 + 13, 0, 0};
        REQUIRE_CALL(cursor["params"].getDereferenced().setData(params, sizeof(params)));
        pass->dispatchCompute(1, 1, 1);
        expected[i] = i * 7 + 23;
    }
    pass->end();
    texture.setNull();
    sampler.setNull();
    root.setNull();
    pipeline.setNull();
    program.setNull();
    ComPtr<ICommandBuffer> commands;
    REQUIRE_CALL(encoder->finish({}, commands.writeRef()));
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, output, std::span<uint32_t>(expected));
}

GPU_TEST_CASE("shader-object-recorded-compute-snapshots", D3D12 | Vulkan)
{
    const char* source = R"(
struct Params { uint bias; uint base; uint count; uint padding; };
struct Resources {
    StructuredBuffer<uint> input; RWStructuredBuffer<uint> output;
    Texture2D<float> texture; SamplerState sampler;
};
ConstantBuffer<Params> params;
ParameterBlock<Resources> resources;
[shader("compute")][numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < params.count)
        resources.output[params.base + id.x] = resources.input[id.x] + params.bias
            + uint(4 * resources.texture.SampleLevel(resources.sampler, float2(0.5), 0));
}
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    ComputePipelineDesc desc{};
    desc.program = program;
    auto pipeline = device->createComputePipeline(desc);
    REQUIRE(pipeline != nullptr);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    const float texels[]{1.f, 2.f, 3.f, 4.f};
    TextureDesc textureDesc{};
    textureDesc.size = {2, 2, 1};
    textureDesc.format = Format::R32Float;
    textureDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
    const SubresourceData textureData{texels, 2 * sizeof(float), 0};
    auto texture = device->createTexture(textureDesc, &textureData);
    REQUIRE(texture != nullptr);
    std::array<ComPtr<ISampler>, 2> samplers;
    for (uint32_t i = 0; i < 2; ++i)
    {
        SamplerDesc samplerDesc{};
        samplerDesc.minFilter = samplerDesc.magFilter = i ? TextureFilteringMode::Linear : TextureFilteringMode::Point;
        REQUIRE_CALL(device->createSampler(samplerDesc, samplers[i].writeRef()));
    }
    std::array<ComPtr<IBuffer>, 3> inputs;
    for (uint32_t n = 0; n < inputs.size(); ++n)
    {
        std::array<uint32_t, 32> values;
        for (uint32_t i = 0; i < values.size(); ++i) values[i] = i * (3 + 2 * n) + 7 + 4 * n;
        BufferDesc bufferDesc{};
        bufferDesc.size = sizeof(values);
        bufferDesc.elementSize = sizeof(uint32_t);
        bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
        REQUIRE_CALL(device->createBuffer(bufferDesc, values.data(), inputs[n].writeRef()));
    }
    std::array<ComPtr<IBuffer>, 2> outputs;
    for (auto& output : outputs)
    {
        std::array<uint32_t, 64> initial;
        initial.fill(0xdeadbeef);
        BufferDesc bufferDesc{};
        bufferDesc.size = sizeof(initial);
        bufferDesc.elementSize = sizeof(uint32_t);
        bufferDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
        REQUIRE_CALL(device->createBuffer(bufferDesc, initial.data(), output.writeRef()));
    }
    auto queue = device->getQueue(QueueType::Graphics);
    std::array<ComPtr<ICommandEncoder>, 2> encoders;
    const uint32_t sources[2][2]{{0, 1}, {2, 0}};
    const uint32_t biases[2][2]{{101, 211}, {307, 401}};
    for (uint32_t batch = 0; batch < 2; ++batch)
    {
        encoders[batch] = queue->createCommandEncoder();
        auto* pass = encoders[batch]->beginComputePass();
        pass->bindPipeline(pipeline, root);
        for (uint32_t draw = 0; draw < 2; ++draw)
        {
            const uint32_t params[]{biases[batch][draw], draw * 32, 32, 0};
            ShaderCursor cursor(root);
            REQUIRE_CALL(cursor["params"].getDereferenced().setData(params, sizeof(params)));
            REQUIRE_CALL(cursor["resources"]["input"].setBinding(inputs[sources[batch][draw]]));
            REQUIRE_CALL(cursor["resources"]["output"].setBinding(outputs[batch]));
            REQUIRE_CALL(cursor["resources"]["texture"].setBinding(texture));
            REQUIRE_CALL(cursor["resources"]["sampler"].setBinding(samplers[draw]));
            pass->dispatchCompute(1, 1, 1);
        }
        pass->end();
    }
    // A has not even been finished. B updates the same root and executes first.
    auto commandsB = encoders[1]->finish();
    REQUIRE(commandsB != nullptr);
    REQUIRE_CALL(queue->submit(commandsB));
    REQUIRE_CALL(queue->waitOnHost());
    inputs = {};
    samplers = {};
    texture.setNull();
    root.setNull();
    pipeline.setNull();
    program.setNull();
    encoders[1].setNull();
    commandsB.setNull();
    auto commandsA = encoders[0]->finish();
    REQUIRE(commandsA != nullptr);
    encoders[0].setNull();
    REQUIRE_CALL(queue->submit(commandsA));
    REQUIRE_CALL(queue->waitOnHost());
    for (uint32_t batch = 0; batch < 2; ++batch)
    {
        std::array<uint32_t, 64> expected{};
        for (uint32_t draw = 0; draw < 2; ++draw)
            for (uint32_t i = 0; i < 32; ++i)
            {
                const auto n = sources[batch][draw];
                expected[draw * 32 + i] = i * (3 + 2 * n) + 7 + 4 * n + biases[batch][draw]
                    + (draw == 0 ? 16 : 10);
            }
        compareComputeResult(device, outputs[batch], expected);
    }
}

GPU_TEST_CASE("shader-object-recorded-graphics-snapshots", D3D12 | Vulkan)
{
    bool sharedImmutableBlock = false;
    SUBCASE("one mutable root") {}
    SUBCASE("two roots share an immutable child") { sharedImmutableBlock = true; }
    const char* source = R"(
struct Params { float4 tint; };
struct Resources { StructuredBuffer<float4> color; };
ConstantBuffer<Params> params;
ParameterBlock<Resources> resources;
[shader("vertex")]
float4 vertexMain(uint id : SV_VertexID) : SV_Position
{
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * 2 - 1, 0, 1);
}
[shader("fragment")]
float4 fragmentMain() : SV_Target { return resources.color[0] + params.tint; }
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadRenderProgramFromSource(device, source, "vertexMain", "fragmentMain", program.writeRef()));
    ColorTargetDesc targetDesc{};
    targetDesc.format = Format::RGBA32Float;
    RenderPipelineDesc desc{};
    desc.program = program;
    desc.targets = &targetDesc;
    desc.targetCount = 1;
    desc.rasterizer.cullMode = CullMode::None;
    desc.depthStencil.depthTestEnable = false;
    desc.depthStencil.depthWriteEnable = false;
    auto pipeline = device->createRenderPipeline(desc);
    REQUIRE(pipeline != nullptr);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    const float colors[2][4]{{0.25f, 0.5f, 0.75f, 1.f}, {0.75f, 0.25f, 0.5f, 1.f}};
    std::array<ComPtr<IBuffer>, 2> inputs;
    std::array<ComPtr<ITexture>, 2> outputs;
    for (uint32_t i = 0; i < 2; ++i)
    {
        BufferDesc bufferDesc{};
        bufferDesc.size = sizeof(colors[i]);
        bufferDesc.elementSize = sizeof(colors[i]);
        bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
        REQUIRE_CALL(device->createBuffer(bufferDesc, colors[i], inputs[i].writeRef()));
        TextureDesc textureDesc{};
        textureDesc.type = TextureType::Texture2D;
        textureDesc.size = {8, 4, 1};
        textureDesc.format = targetDesc.format;
        textureDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
        textureDesc.defaultState = ResourceState::RenderTarget;
        REQUIRE_CALL(device->createTexture(textureDesc, nullptr, outputs[i].writeRef()));
    }
    auto queue = device->getQueue(QueueType::Graphics);
    std::array<ComPtr<ICommandEncoder>, 2> encoders;
    ComPtr<IShaderObject> secondRoot;
    if (sharedImmutableBlock)
    {
        secondRoot = device->createRootShaderObject(program);
        REQUIRE(secondRoot != nullptr);
        auto childCursor = ShaderCursor(root)["resources"];
        REQUIRE_CALL(childCursor["color"].setBinding(inputs[0]));
        auto child = root->getObject(childCursor.m_offset);
        REQUIRE(child != nullptr);
        REQUIRE_CALL(ShaderCursor(secondRoot)["resources"].setObject(child));
    }
    for (uint32_t batch = 0; batch < 2; ++batch)
    {
        encoders[batch] = queue->createCommandEncoder();
        RenderPassColorAttachment color{};
        color.view = outputs[batch]->getDefaultView();
        color.loadOp = LoadOp::Clear;
        color.storeOp = StoreOp::Store;
        RenderPassDesc passDesc{};
        passDesc.colorAttachments = &color;
        passDesc.colorAttachmentCount = 1;
        auto* pass = encoders[batch]->beginRenderPass(passDesc);
        auto* activeRoot = sharedImmutableBlock && batch ? secondRoot.get() : root.get();
        pass->bindPipeline(pipeline, activeRoot);
        for (uint32_t draw = 0; draw < 2; ++draw)
        {
            const float tint[]{float(batch), float(draw) * 0.125f, 0.f, 0.f};
            ShaderCursor cursor(activeRoot);
            REQUIRE_CALL(cursor["params"].getDereferenced().setData(tint, sizeof(tint)));
            if (!sharedImmutableBlock)
                REQUIRE_CALL(cursor["resources"]["color"].setBinding(inputs[(batch + draw) % 2]));
            RenderState state{};
            state.viewportCount = state.scissorRectCount = 1;
            state.viewports[0] = Viewport::fromSize(8, 4);
            state.scissorRects[0] = {draw * 4, 0, draw * 4 + 4, 4};
            pass->setRenderState(state);
            pass->draw(DrawArguments{.vertexCount = 3});
        }
        pass->end();
    }
    auto commandsB = encoders[1]->finish();
    REQUIRE(commandsB != nullptr);
    REQUIRE_CALL(queue->submit(commandsB));
    REQUIRE_CALL(queue->waitOnHost());
    inputs = {};
    secondRoot.setNull();
    root.setNull();
    pipeline.setNull();
    program.setNull();
    commandsB.setNull();
    encoders[1].setNull();
    auto commandsA = encoders[0]->finish();
    REQUIRE(commandsA != nullptr);
    encoders[0].setNull();
    REQUIRE_CALL(queue->submit(commandsA));
    REQUIRE_CALL(queue->waitOnHost());
    for (uint32_t batch = 0; batch < 2; ++batch)
    {
        std::array<float, 8 * 4 * 4> expected{};
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 8; ++x)
                for (uint32_t c = 0; c < 4; ++c)
                    expected[(y * 8 + x) * 4 + c] = colors[sharedImmutableBlock ? 0 : (batch + x / 4) % 2][c]
                        + (c == 0 ? float(batch) : c == 1 ? float(x / 4) * 0.125f : 0.f);
        std::array<uint32_t, 8 * 4 * 4> expectedBits{};
        std::memcpy(expectedBits.data(), expected.data(), sizeof(expected));
        compareComputeResult(device, outputs[batch], 0, 0, expectedBits);
    }
}

GPU_TEST_CASE("shader-object-recorded-immutable-block-transitions", D3D12 | Vulkan)
{
    const char* source = R"(
struct Params { uint base; uint3 padding; };
struct Resources { Texture2D<uint> input; RWStructuredBuffer<uint> output; };
ConstantBuffer<Params> params;
ParameterBlock<Resources> resources;
[shader("compute")][numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    resources.output[params.base + id.x] = resources.input.Load(int3(id.x % 2, id.x / 2, 0));
}
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    ComputePipelineDesc desc{};
    desc.program = program;
    auto pipeline = device->createComputePipeline(desc);
    REQUIRE(pipeline != nullptr);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    std::array<ComPtr<ITexture>, 2> textures;
    const uint32_t values[2][4]{{11, 23, 37, 41}, {53, 67, 79, 83}};
    for (uint32_t i = 0; i < 2; ++i)
    {
        TextureDesc textureDesc{};
        textureDesc.size = {2, 2, 1};
        textureDesc.format = Format::R32Uint;
        textureDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopySource | TextureUsage::CopyDestination;
        textureDesc.defaultState = ResourceState::ShaderResource;
        const SubresourceData data{values[i], 2 * sizeof(uint32_t), 0};
        REQUIRE_CALL(device->createTexture(textureDesc, &data, textures[i].writeRef()));
    }
    const std::array<uint32_t, 8> initial{101, 101, 101, 101, 101, 101, 101, 101};
    BufferDesc outputDesc{};
    outputDesc.size = sizeof(initial);
    outputDesc.elementSize = sizeof(uint32_t);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    auto output = device->createBuffer(outputDesc, initial.data());
    REQUIRE(output != nullptr);
    BufferDesc copiedDesc{};
    copiedDesc.size = sizeof(initial);
    copiedDesc.usage = BufferUsage::CopyDestination | BufferUsage::CopySource;
    auto copied = device->createBuffer(copiedDesc, nullptr);
    REQUIRE(copied != nullptr);
    ShaderCursor cursor(root);
    REQUIRE_CALL(cursor["resources"]["input"].setBinding(textures[0]));
    REQUIRE_CALL(cursor["resources"]["output"].setBinding(output));
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    for (uint32_t step = 0; step < 2; ++step)
    {
        if (step)
        {
            encoder->copyBuffer(copied, 0, output, 0, sizeof(initial));
            encoder->copyTexture(textures[0], {}, {}, textures[1], {}, {}, Extent3D::kWholeTexture);
        }
        const uint32_t params[]{step * 4, 0, 0, 0};
        REQUIRE_CALL(cursor["params"].getDereferenced().setData(params, sizeof(params)));
        auto* pass = encoder->beginComputePass();
        pass->bindPipeline(pipeline, root);
        pass->dispatchCompute(1, 1, 1);
        pass->end();
    }
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    const std::array<uint32_t, 8> expected{11, 23, 37, 41, 53, 67, 79, 83};
    compareComputeResult(device, output, expected);
    const std::array<uint32_t, 8> copiedExpected{11, 23, 37, 41, 101, 101, 101, 101};
    compareComputeResult(device, copied, copiedExpected);
}

GPU_TEST_CASE("shader-object-recorded-explicit-sets-and-bindless", D3D12 | Vulkan)
{
    bool inlineConstants = true;
    bool nonemptyRoot = false;
    SUBCASE("inline empty root with two explicit child sets") {}
    SUBCASE("inline nonempty root with explicit child sets") { nonemptyRoot = true; }
    SUBCASE("buffered nonempty root with explicit child sets") { inlineConstants = false; nonemptyRoot = true; }
    std::string source = std::string("#define INLINE ") + (inlineConstants ? "1\n" : "0\n")
        + "#define NONEMPTY_ROOT " + (nonemptyRoot ? "1\n" : "0\n") + R"(
struct Params { uint2 extra; uint bias; uint padding; };
#if INLINE
[[vk::push_constant]] ConstantBuffer<Params> params;
#else
[[vk::binding(3, 0)]] ConstantBuffer<Params> params;
#endif
struct Scene {
    StructuredBuffer<uint> input;
#if !NONEMPTY_ROOT
    RWStructuredBuffer<uint> output;
#endif
};
struct Other { StructuredBuffer<uint> input; };
#if NONEMPTY_ROOT
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(0, 1)]] ParameterBlock<Scene> scene;
[[vk::binding(0, 2)]] ParameterBlock<Other> other;
#else
[[vk::binding(0, 0)]] ParameterBlock<Scene> scene;
[[vk::binding(0, 1)]] ParameterBlock<Other> other;
#endif
[shader("compute")][numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    ByteAddressBuffer.Handle extra = ByteAddressBuffer.Handle(params.extra);
    uint value = 3 * scene.input[id.x] + 5 * other.input[id.x] + extra.Load<uint>(id.x * 4) + params.bias;
#if NONEMPTY_ROOT
    output[id.x] = value;
#else
    scene.output[id.x] = value;
#endif
}
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline != nullptr);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    const uint32_t values[3][4]{{3, 5, 7, 11}, {13, 17, 19, 23}, {29, 31, 37, 41}};
    std::array<ComPtr<IBuffer>, 3> inputs;
    for (uint32_t i = 0; i < 3; ++i)
    {
        BufferDesc desc{};
        desc.size = sizeof(values[i]);
        desc.elementSize = i < 2 ? sizeof(uint32_t) : 0;
        desc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
        REQUIRE_CALL(device->createBuffer(desc, values[i], inputs[i].writeRef()));
    }
    const std::array<uint32_t, 4> initial{101, 101, 101, 101};
    BufferDesc outputDesc{};
    outputDesc.size = sizeof(initial);
    outputDesc.elementSize = sizeof(uint32_t);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    auto output = device->createBuffer(outputDesc, initial.data());
    REQUIRE(output != nullptr);
    DescriptorHandle handle{};
    REQUIRE_CALL(inputs[2]->getDescriptorHandle(DescriptorHandleAccess::Read, Format::Undefined, kEntireBuffer, &handle));
    ShaderCursor cursor(root);
    REQUIRE_CALL(cursor["params"]["extra"].setDescriptorHandle(handle));
    const uint32_t bias = 9;
    REQUIRE_CALL(cursor["params"]["bias"].setData(&bias, sizeof(bias)));
    REQUIRE_CALL(cursor["scene"]["input"].setBinding(inputs[0]));
    REQUIRE_CALL(cursor["other"]["input"].setBinding(inputs[1]));
    REQUIRE_CALL((nonemptyRoot ? cursor["output"] : cursor["scene"]["output"]).setBinding(output));
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    // Raw handles alone do not retain resources: the explicit access records the native buffer.
    encoder->setBufferState(inputs[2], ResourceState::ShaderResource);
    auto* pass = encoder->beginComputePass();
    pass->bindPipeline(pipeline, root);
    pass->dispatchCompute(1, 1, 1);
    pass->end();
    root.setNull();
    inputs = {};
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, output, std::array<uint32_t, 4>{112, 140, 162, 198});
}

namespace {
class UndeclaredStateRecorder : public IDebugCallback
{
public:
    std::vector<std::string> reports;

    virtual SLANG_NO_THROW void SLANG_MCALL handleMessage(
        DebugMessageType type,
        DebugMessageSource source,
        const char* message
    ) override
    {
        SLANG_UNUSED(source);
        if (type == DebugMessageType::Error && std::strstr(message, "which was not declared"))
            reports.emplace_back(message);
    }
};
} // namespace

// A resource reached only through a typed descriptor handle is checked against the state its access needs, both when
// the binding table is built and when a later bind reuses it.
GPU_TEST_CASE("shader-object-typed-descriptor-owner-states", D3D12 | Vulkan)
{
    UndeclaredStateRecorder recorder;
    DeviceExtraOptions options;
    options.resourceStateMode = ResourceStateMode::Explicit;
    options.validateResourceStates = true;
    options.debugCallback = &recorder;
    device = createTestingDevice(ctx, ctx->deviceType, false, &options);
    REQUIRE(device != nullptr);

    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(
        device,
        R"(
[shader("compute")][numthreads(1, 1, 1)]
void main(uniform RWStructuredBuffer<uint>.Handle output)
{
    output[0] = 7;
}
)",
        program.writeRef()
    ));
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline != nullptr);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    BufferDesc outputDesc{};
    outputDesc.size = sizeof(uint32_t);
    outputDesc.elementSize = sizeof(uint32_t);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    outputDesc.label = "TypedHandleOutput";
    auto output = device->createBuffer(outputDesc);
    REQUIRE(output != nullptr);
    REQUIRE_CALL(ShaderCursor(root->getEntryPoint(0))["output"]
                     .setDescriptorHandle(output, DescriptorHandleAccess::ReadWrite, Format::Undefined, 0, kEntireBuffer));

    auto queue = device->getQueue(QueueType::Graphics);
    auto dispatchTwice = [&](ResourceState declared)
    {
        auto encoder = queue->createCommandEncoder();
        encoder->setBufferState(output, declared);
        for (int dispatch = 0; dispatch < 2; ++dispatch)
        {
            auto* pass = encoder->beginComputePass();
            pass->bindPipeline(pipeline, root);
            pass->dispatchCompute(1, 1, 1);
            pass->end();
        }
        REQUIRE_CALL(queue->submit(encoder->finish()));
        REQUIRE_CALL(queue->waitOnHost());
    };

    dispatchTwice(ResourceState::UnorderedAccess);
    CHECK(recorder.reports.empty());

    dispatchTwice(ResourceState::ShaderResource);
    REQUIRE(recorder.reports.size() == 2);
    for (const auto& report : recorder.reports)
    {
        CHECK(report.find("TypedHandleOutput") != std::string::npos);
        CHECK(report.find(enumToString(ResourceState::UnorderedAccess)) != std::string::npos);
    }
}

// Legacy shader-object binding of the same lifetime regression.

GPU_TEST_CASE("shader-object-resource-tracking", ALL & ~CPU)
{
    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-shader-object-resource-tracking", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    ComPtr<IBuffer> globalBuffer;
    {
        BufferDesc bufferDesc = {};
        bufferDesc.size = sizeof(float);
        bufferDesc.usage = BufferUsage::CopyDestination | BufferUsage::ShaderResource;
        float initialData[] = {10.f};
        REQUIRE_CALL(device->createBuffer(bufferDesc, initialData, globalBuffer.writeRef()));
    }

    ComPtr<ITexture> globalTexture;
    {
        TextureDesc textureDesc = {};
        textureDesc.size = {2, 2, 1};
        textureDesc.format = Format::R32Float;
        textureDesc.usage = TextureUsage::CopyDestination | TextureUsage::ShaderResource;
        float initialData[] = {1.f, 2.f, 3.f, 4.f};
        SubresourceData subresourceData[] = {{initialData, sizeof(float) * 2, 0}};
        REQUIRE_CALL(device->createTexture(textureDesc, subresourceData, globalTexture.writeRef()));
    }

    ComPtr<ISampler> globalSampler;
    {
        SamplerDesc samplerDesc = {};
        samplerDesc.minFilter = TextureFilteringMode::Point;
        samplerDesc.magFilter = TextureFilteringMode::Point;
        REQUIRE_CALL(device->createSampler(samplerDesc, globalSampler.writeRef()));
    }

    ComPtr<IBuffer> buffer;
    {
        BufferDesc bufferDesc = {};
        bufferDesc.size = sizeof(float);
        bufferDesc.usage = BufferUsage::CopyDestination | BufferUsage::ShaderResource;
        float initialData[] = {20.f};
        REQUIRE_CALL(device->createBuffer(bufferDesc, initialData, buffer.writeRef()));
    }

    ComPtr<ITexture> texture;
    {
        TextureDesc textureDesc = {};
        textureDesc.size = {2, 2, 1};
        textureDesc.format = Format::R32Float;
        textureDesc.usage = TextureUsage::CopyDestination | TextureUsage::ShaderResource;
        float initialData[] = {1.f, 2.f, 3.f, 4.f};
        SubresourceData subresourceData[] = {{initialData, sizeof(float) * 2, 0}};
        REQUIRE_CALL(device->createTexture(textureDesc, subresourceData, texture.writeRef()));
    }

    ComPtr<ISampler> sampler;
    {
        SamplerDesc samplerDesc = {};
        samplerDesc.minFilter = TextureFilteringMode::Linear;
        samplerDesc.magFilter = TextureFilteringMode::Linear;
        REQUIRE_CALL(device->createSampler(samplerDesc, sampler.writeRef()));
    }

    ComPtr<IBuffer> resultBuffer;
    {
        BufferDesc bufferDesc = {};
        bufferDesc.size = 4 * sizeof(float);
        bufferDesc.usage = BufferUsage::CopySource | BufferUsage::UnorderedAccess;
        REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, resultBuffer.writeRef()));
    }

    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        ShaderCursor globalsCursor(rootObject);
        globalsCursor["globalBuffer"].setBinding(globalBuffer);
        globalsCursor["globalTexture"].setBinding(globalTexture);
        globalsCursor["globalSampler"].setBinding(globalSampler);
        ShaderCursor entryPointCursor(rootObject->getEntryPoint(0));
        entryPointCursor["buffer"].setBinding(buffer);
        entryPointCursor["texture"].setBinding(texture);
        entryPointCursor["sampler"].setBinding(sampler);
        entryPointCursor["resultBuffer"].setBinding(resultBuffer);

        // At this point, the shader object should keep all resources alive.
        // We can release them and they should not be destroyed until the command
        // encoder is finished and the command buffer is executed.
        globalBuffer.setNull();
        globalTexture.setNull();
        globalSampler.setNull();
        buffer.setNull();
        texture.setNull();
        sampler.setNull();

        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();

        // Finish command encoding and release the command encoder,
        // making sure it does not keep any resources alive, but the command buffer
        // still does.
        ComPtr<ICommandBuffer> commandBuffer = commandEncoder->finish();
        commandEncoder.setNull();

        queue->submit(commandBuffer);
        queue->waitOnHost();
    }

    compareComputeResult(device, resultBuffer, makeArray<float>(10.f, 1.f, 20.f, 2.5f));
}

GPU_TEST_CASE("shader-object-native-set-indices", D3D12 | Vulkan)
{
    uint32_t bufferSet = 1;
    uint32_t textureSet = 1;
    bool buffered = false;
    bool nestedBlock = false;
    SUBCASE("root resources in set 1 with child set 0") {}
    SUBCASE("multiple root sets") { textureSet = 2; }
    SUBCASE("root set 3 with holes and bindless") { bufferSet = textureSet = 3; }
    SUBCASE("buffered constants and root resources in set 3") { bufferSet = textureSet = 3; buffered = true; }

    SUBCASE("cached nested blocks with a root-set gap") { bufferSet = textureSet = 3; nestedBlock = true; }

    std::string source = "#define BUFFER_SET " + std::to_string(bufferSet)
        + "\n#define TEXTURE_SET " + std::to_string(textureSet)
        + "\n#define BUFFER_SPACE space" + std::to_string(bufferSet)
        + "\n#define TEXTURE_SPACE space" + std::to_string(textureSet)
        + "\n#define BUFFERED " + (buffered ? "1\n" : "0\n")
        + "#define NESTED_BLOCK " + (nestedBlock ? "1\n" : "0\n") + R"(
struct Params { uint base; uint bias; uint2 rawInput; };
#if BUFFERED
[[vk::binding(6, BUFFER_SET)]] ConstantBuffer<Params> params : register(b6, BUFFER_SPACE);
#else
[[vk::push_constant]] ConstantBuffer<Params> params;
#endif
[[vk::binding(3, BUFFER_SET)]] StructuredBuffer<uint> rootInput : register(t3, BUFFER_SPACE);
[[vk::binding(4, TEXTURE_SET)]] Texture2D<uint> rootTexture : register(t4, TEXTURE_SPACE);
struct Extra { StructuredBuffer<uint> input; ByteAddressBuffer rawInput; };
struct Scene {
    StructuredBuffer<uint> input;
    RWStructuredBuffer<uint> output;
#if NESTED_BLOCK
    ParameterBlock<Extra> extra;
#endif
};
[[vk::binding(0, 0)]] ParameterBlock<Scene> scene;
[shader("compute")][numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
#if NESTED_BLOCK
    uint rawValue = scene.extra.rawInput.Load<uint>(id.x * 4);
#else
    ByteAddressBuffer.Handle rawInput = ByteAddressBuffer.Handle(params.rawInput);
    uint rawValue = rawInput.Load<uint>(id.x * 4);
#endif
    scene.output[params.base + id.x] = 7 * rootInput[id.x] + 11 * scene.input[id.x]
        + 13 * rootTexture.Load(int3(id.x % 2, id.x / 2, 0))
        + rawValue + params.bias
#if NESTED_BLOCK
        + 19 * scene.extra.input[id.x]
#endif
        ;
}
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline != nullptr);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);

    const uint32_t values[4][4]{{3, 5, 7, 11}, {43, 47, 53, 59}, {13, 17, 19, 23}, {29, 31, 37, 41}};
    std::array<ComPtr<IBuffer>, 4> inputs;
    for (uint32_t i = 0; i < inputs.size(); ++i)
    {
        BufferDesc desc{};
        desc.size = sizeof(values[i]);
        desc.elementSize = i == 3 ? 0 : sizeof(uint32_t);
        desc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
        REQUIRE_CALL(device->createBuffer(desc, values[i], inputs[i].writeRef()));
    }
    const uint32_t texels[2][4]{{2, 4, 8, 16}, {32, 64, 128, 256}};
    std::array<ComPtr<ITexture>, 2> textures;
    for (uint32_t i = 0; i < textures.size(); ++i)
    {
        TextureDesc desc{};
        desc.size = {2, 2, 1};
        desc.format = Format::R32Uint;
        desc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
        SubresourceData data{texels[i], 2 * sizeof(uint32_t), 0};
        REQUIRE_CALL(device->createTexture(desc, &data, textures[i].writeRef()));
    }
    std::array<uint32_t, 12> initial;
    initial.fill(0xdeadbeef);
    BufferDesc outputDesc{};
    outputDesc.size = sizeof(initial);
    outputDesc.elementSize = sizeof(uint32_t);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    auto output = device->createBuffer(outputDesc, initial.data());
    REQUIRE(output != nullptr);

    DescriptorHandle handle{};
    REQUIRE_CALL(inputs[3]->getDescriptorHandle(DescriptorHandleAccess::Read, Format::Undefined, kEntireBuffer, &handle));
    ShaderCursor cursor(root);
    REQUIRE_CALL(cursor["params"]["rawInput"].setDescriptorHandle(handle));
    REQUIRE_CALL(cursor["scene"]["input"].setBinding(inputs[2]));
    REQUIRE_CALL(cursor["scene"]["output"].setBinding(output));
    if (nestedBlock)
    {
        REQUIRE_CALL(cursor["scene"]["extra"]["input"].setBinding(inputs[1]));
        REQUIRE_CALL(cursor["scene"]["extra"]["rawInput"].setBinding(inputs[3]));
    }

    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    encoder->setBufferState(inputs[3], ResourceState::ShaderResource);
    auto* pass = encoder->beginComputePass();
    pass->bindPipeline(pipeline, root);
    const uint32_t biases[]{3, 11, 17};
    std::array<uint32_t, 12> expected{};
    for (uint32_t dispatch = 0; dispatch < 3; ++dispatch)
    {
        uint32_t resourceIndex = dispatch == 2 ? 1 : 0;
        uint32_t base = dispatch * 4;
        REQUIRE_CALL(cursor["params"]["base"].setData(&base, sizeof(base)));
        REQUIRE_CALL(cursor["params"]["bias"].setData(&biases[dispatch], sizeof(biases[dispatch])));
        REQUIRE_CALL(cursor["rootInput"].setBinding(inputs[resourceIndex]));
        REQUIRE_CALL(cursor["rootTexture"].setBinding(textures[resourceIndex]));
        pass->dispatchCompute(1, 1, 1);
        for (uint32_t i = 0; i < 4; ++i)
            expected[base + i] = 7 * values[resourceIndex][i] + 11 * values[2][i]
                + 13 * texels[resourceIndex][i] + values[3][i] + biases[dispatch]
                + (nestedBlock ? 19 * values[1][i] : 0);
    }
    pass->end();
    root.setNull();
    inputs = {};
    textures = {};
    pipeline.setNull();
    program.setNull();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, output, expected);
}
#if SLANG_RHI_ENABLE_VULKAN
GPU_TEST_CASE("shader-object-native-set-limit-with-unused-heap", Vulkan)
{
    DeviceNativeHandles handles{};
    REQUIRE_CALL(device->getNativeDeviceHandles(&handles));
    REQUIRE(handles.handles[0].type == NativeHandleType::VkInstance);
    REQUIRE(handles.handles[1].type == NativeHandleType::VkPhysicalDevice);
    rhi::vk::VulkanModule module;
    REQUIRE_CALL(module.init());
    rhi::vk::VulkanApi api{};
    REQUIRE_CALL(api.initGlobalProcs(module));
    REQUIRE_CALL(api.initInstanceProcs(reinterpret_cast<VkInstance>(handles.handles[0].value)));
    VkPhysicalDeviceProperties properties{};
    api.vkGetPhysicalDeviceProperties(reinterpret_cast<VkPhysicalDevice>(handles.handles[1].value), &properties);
    const uint32_t setCount = std::min(uint32_t(rhi::vk::kMaxDescriptorSets), properties.limits.maxBoundDescriptorSets);
    REQUIRE(setCount > 0);
    CAPTURE(setCount);

    bool useHeap = false;
    SUBCASE("ordinary descriptors use every legal set") {}
    SUBCASE("actual heap use without a legal set is rejected") { useHeap = true; }
    std::string source;
    for (uint32_t i = 0; i < setCount; ++i)
        source += "[[vk::binding(0," + std::to_string(i) + ")]] StructuredBuffer<uint> input"
            + std::to_string(i) + ";\n";
    source += R"(
[[vk::binding(1,0)]] RWStructuredBuffer<uint> output;
struct Params { uint2 rawInput; uint2 padding; };
[[vk::push_constant]] ConstantBuffer<Params> params;
[shader("compute")][numthreads(4,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
uint value = 0;
)";
    for (uint32_t i = 0; i < setCount; ++i)
        source += "value += " + std::to_string(i + 1) + " * input" + std::to_string(i) + "[id.x];\n";
    if (useHeap)
        source += "ByteAddressBuffer.Handle raw = ByteAddressBuffer.Handle(params.rawInput); value += raw.Load<uint>(id.x * 4);\n";
    source += "output[id.x] = value; }\n";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    ComputePipelineDesc pipelineDesc{};
    pipelineDesc.program = program;
    ComPtr<IComputePipeline> pipeline;
    if (useHeap)
    {
        ExpectedErrorScope expectedError;
        CHECK(device->createComputePipeline(pipelineDesc, pipeline.writeRef()) == SLANG_E_INVALID_ARG);
        CHECK(pipeline == nullptr);
        return;
    }
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    ShaderCursor cursor(root);
    std::vector<ComPtr<IBuffer>> inputs(setCount);
    std::array<uint32_t, 4> expected{};
    BufferDesc inputDesc{};
    inputDesc.size = sizeof(expected);
    inputDesc.elementSize = sizeof(uint32_t);
    inputDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
    for (uint32_t i = 0; i < setCount; ++i)
    {
        const std::array<uint32_t, 4> values{i + 3, i * 3 + 7, i * 5 + 11, i * 7 + 13};
        REQUIRE_CALL(device->createBuffer(inputDesc, values.data(), inputs[i].writeRef()));
        const std::string name = "input" + std::to_string(i);
        REQUIRE_CALL(cursor[name.c_str()].setBinding(inputs[i]));
        for (uint32_t j = 0; j < expected.size(); ++j)
            expected[j] += (i + 1) * values[j];
    }
    const std::array<uint32_t, 4> initial{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
    BufferDesc outputDesc = inputDesc;
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    auto output = device->createBuffer(outputDesc, initial.data());
    REQUIRE(output != nullptr);
    REQUIRE_CALL(cursor["output"].setBinding(output));
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto* pass = encoder->beginComputePass();
    pass->bindPipeline(pipeline, root);
    pass->dispatchCompute(1, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, output, expected);
}
#endif

GPU_TEST_CASE("shader-object-table-cache-constant-rebind", D3D12 | Vulkan)
{
    bool pushConstantBlock = false;
    SUBCASE("global CB") {}
    SUBCASE("push-constant block") { pushConstantBlock = true; }
    std::string source = R"(
struct Constants { uint index; uint bias; uint2 padding; };
struct Resources { StructuredBuffer<uint> input[15]; RWStructuredBuffer<uint> output; };
ConstantBuffer<Constants> constants;
ParameterBlock<Resources> resources;
[shader("compute")][numthreads(1,1,1)]
void main() {
    uint sum = 0;
    for (uint i = 0; i < 15; ++i) sum += (i + 1) * resources.input[i][0];
    resources.output[constants.index] = sum + constants.bias;
}
)";
    if (pushConstantBlock)
    {
        source.insert(0, "import push_constants;");
        source.insert(source.find("ConstantBuffer<Constants> constants;"), "[PushConstants] [[vk::push_constant]] ");
    }
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    ComputePipelineDesc pipelineDesc{};
    pipelineDesc.program = program;
    auto pipeline = device->createComputePipeline(pipelineDesc);
    REQUIRE(pipeline);
    uint32_t inputValue = 19;
    BufferDesc desc{};
    desc.size = sizeof(inputValue);
    desc.elementSize = sizeof(uint32_t);
    desc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
    std::array<ComPtr<IBuffer>, 15> inputs;
    uint32_t inputSum = 0;
    for (uint32_t i = 0; i < inputs.size(); ++i)
    {
        inputValue = 19 + i * 3;
        REQUIRE_CALL(device->createBuffer(desc, &inputValue, inputs[i].writeRef()));
        inputSum += (i + 1) * inputValue;
    }
    std::array<uint32_t, 256> initial{};
    initial.fill(0xbaadf00d);
    desc.size = sizeof(initial);
    desc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    auto output = device->createBuffer(desc, initial.data());
    REQUIRE(output);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root);
    ShaderCursor cursor(root);
    for (uint32_t i = 0; i < inputs.size(); ++i)
        REQUIRE_CALL(cursor["resources"]["input"][i].setBinding(inputs[i]));
    REQUIRE_CALL(cursor["resources"]["output"].setBinding(output));
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto* pass = encoder->beginComputePass();
    pass->bindPipeline(pipeline, root);
    std::array<uint32_t, 256> expected;
    for (uint32_t i = 0; i < expected.size(); ++i)
    {
        uint32_t constants[]{i, 7 * i + 3, 0, 0};
        REQUIRE_CALL(cursor["constants"].getDereferenced().setData(constants, sizeof(constants)));
        pass->dispatchCompute(1, 1, 1);
        expected[i] = inputSum + 7 * i + 3;
    }
    pass->end();
    inputs = {};
    root.setNull();
    pipeline.setNull();
    program.setNull();
    auto commands = encoder->finish();
    REQUIRE(commands);
#if SLANG_RHI_ENABLE_BINDING_COUNTERS
    auto* wrapped = dynamic_cast<debug::DebugCommandBuffer*>(commands.get());
    auto* internal = checked_cast<CommandBuffer*>(wrapped ? wrapped->baseObject.get() : commands.get());
    // A constant buffer is part of the root table, so every rewrite rebuilds it. A push-constant block is not.
    const uint64_t rootTables = pushConstantBlock ? 1 : expected.size();
    CHECK(internal->m_bindingCounters.trackedRefs == 16);
    CHECK(internal->m_bindingCounters.tableMisses == 1 + rootTables);
    CHECK(internal->m_bindingCounters.tableHits == 2 * expected.size() - 1 - rootTables);
    const uint64_t implicitNvapiDescriptors = device->getDeviceType() == DeviceType::D3D12 &&
        device->hasCapability(Capability::hlsl_nvapi) ? rootTables : 0;
    const uint64_t ordinaryDescriptors = pushConstantBlock ? 0 : expected.size();
    CHECK(internal->m_bindingCounters.descriptorsWritten == 16 + ordinaryDescriptors + implicitNvapiDescriptors);
    CHECK(internal->m_bindingCounters.cbAllocations == ordinaryDescriptors);
#endif
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(device, output, expected);
}

GPU_TEST_CASE("shader-object-table-cache-nested-relocation", D3D12 | Vulkan)
{
    bool rootDescriptors = false;
    SUBCASE("descriptor tables") {}
    SUBCASE("root descriptors") { rootDescriptors = true; }
    std::string source = R"(
struct Data { uint bias; uint3 padding; };
struct Leaf { StructuredBuffer<uint> input; };
struct Group { ConstantBuffer<Data> data; ParameterBlock<Leaf> leaf; };
struct Params { uint index; uint3 padding; };
ConstantBuffer<Params> params;
ParameterBlock<Group> left;
ParameterBlock<Group> right;
RWStructuredBuffer<uint> output;
[shader("compute")][numthreads(1,1,1)]
void main() {
    output[params.index] += left.data.bias + left.leaf.input[0]
        + 7 * (right.data.bias + right.leaf.input[0]);
}
)";
    if (rootDescriptors)
    {
        source.insert(0, "[__AttributeUsage(_AttributeTargets.Var)] struct rootAttribute {};\n");
        auto pos = source.find("StructuredBuffer<uint> input;");
        source.insert(pos, "[root] ");
    }
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    ComputePipelineDesc desc{};
    desc.program = program;
    auto pipeline = device->createComputePipeline(desc);
    REQUIRE(pipeline);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root);
    ShaderCursor cursor(root);
    auto left = root->getObject(cursor["left"].m_offset);
    auto right = root->getObject(cursor["right"].m_offset);
    REQUIRE(left);
    REQUIRE(right);
    auto leaf = left->getObject(ShaderCursor(left)["leaf"].m_offset);
    REQUIRE(leaf);
    REQUIRE_CALL(ShaderCursor(right)["leaf"].setObject(leaf));
    std::array<ComPtr<IBuffer>, 3> inputs;
    const uint32_t values[]{11, 29, 47};
    BufferDesc bufferDesc{};
    bufferDesc.size = sizeof(uint32_t);
    bufferDesc.elementSize = sizeof(uint32_t);
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
    for (uint32_t i = 0; i < inputs.size(); ++i)
        REQUIRE_CALL(device->createBuffer(bufferDesc, &values[i], inputs[i].writeRef()));
    std::array<ComPtr<IBuffer>, 2> outputs;
    std::array<ComPtr<ICommandBuffer>, 2> commands;
    std::array<std::array<uint32_t, 5>, 2> expected{};
    auto queue = device->getQueue(QueueType::Graphics);
    for (uint32_t batch = 0; batch < 2; ++batch)
    {
        bufferDesc.size = sizeof(expected[batch]);
        bufferDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopyDestination | BufferUsage::CopySource;
        REQUIRE_CALL(device->createBuffer(bufferDesc, expected[batch].data(), outputs[batch].writeRef()));
        REQUIRE_CALL(cursor["output"].setBinding(outputs[batch]));
        REQUIRE_CALL(ShaderCursor(leaf)["input"].setBinding(inputs[batch]));
        const uint32_t rightData[]{13 + batch, 0, 0, 0};
        REQUIRE_CALL(ShaderCursor(right)["data"].getDereferenced().setData(rightData, sizeof(rightData)));
        auto encoder = queue->createCommandEncoder();
        auto* pass = encoder->beginComputePass();
        pass->bindPipeline(pipeline, root);
        uint32_t leftBias = 3 + batch;
        uint32_t value = values[batch];
        for (uint32_t step = 0; step < 5; ++step)
        {
            const uint32_t params[]{step, 0, 0, 0};
            REQUIRE_CALL(cursor["params"].getDereferenced().setData(params, sizeof(params)));
            if (step == 0 || step == 2)
            {
                if (step == 2) leftBias += 17;
                const uint32_t data[]{leftBias, 0, 0, 0};
                REQUIRE_CALL(ShaderCursor(left)["data"].getDereferenced().setData(data, sizeof(data)));
            }
            if (step == 3)
            {
                value = values[batch + 1];
                REQUIRE_CALL(ShaderCursor(leaf)["input"].setBinding(inputs[batch + 1]));
            }
            for (uint32_t repeat = 0; repeat < 3; ++repeat)
                pass->dispatchCompute(1, 1, 1);
            expected[batch][step] = 3 * (leftBias + value + 7 * (rightData[0] + value));
        }
        pass->end();
        REQUIRE_CALL(encoder->finish({}, commands[batch].writeRef()));
#if SLANG_RHI_ENABLE_BINDING_COUNTERS
        auto* wrapped = dynamic_cast<debug::DebugCommandBuffer*>(commands[batch].get());
        auto* internal = checked_cast<CommandBuffer*>(wrapped ? wrapped->baseObject.get() : commands[batch].get());
        CHECK(internal->m_bindingCounters.tableHits > 40);
        CHECK(internal->m_bindingCounters.tableMisses > 0);
#endif
    }
    REQUIRE_CALL(queue->submit(commands[1]));
    REQUIRE_CALL(queue->waitOnHost());
    inputs = {};
    leaf.setNull(); left.setNull(); right.setNull(); root.setNull();
    pipeline.setNull(); program.setNull();
    commands[1].setNull();
    REQUIRE_CALL(queue->submit(commands[0]));
    REQUIRE_CALL(queue->waitOnHost());
    for (uint32_t batch = 0; batch < 2; ++batch)
        compareComputeResult(device, outputs[batch], expected[batch]);
}

GPU_TEST_CASE("shader-object-exact-closure-key", D3D12 | Vulkan)
{
    const char* source = R"(
struct Data { uint value; uint3 pad; };
struct Resources { RWStructuredBuffer<uint> output; };
ConstantBuffer<Data> data;
ParameterBlock<Resources> params;
[shader("compute")][numthreads(1,1,1)] void main() { params.output[0] = data.value; }
)";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));
    auto root = device->createRootShaderObject(program);
    REQUIRE(root);
    auto* debugRoot = dynamic_cast<debug::DebugShaderObject*>(root.get());
    auto* native = checked_cast<RootShaderObject*>(debugRoot ? debugRoot->baseObject.get() : root.get());
    ShaderObjectLayout* specialized = nullptr;
    REQUIRE_CALL(native->getSpecializedLayout(specialized));
    BindingClosureKey key;
    std::vector<RefPtr<RefObject>> retained;
    key.capture(native, true, specialized, retained);
    CHECK(key.matches(native, true));
    const uint32_t data[]{19, 0, 0, 0};
    REQUIRE_CALL(ShaderCursor(root)["data"].getDereferenced().setData(data, sizeof(data)));
    CHECK_FALSE(key.matches(native, true));
    key.objects.clear();
    key.capture(native, true, specialized, retained);
    BufferDesc desc{};
    desc.size = 4; desc.elementSize = 4; desc.usage = BufferUsage::UnorderedAccess;
    auto output = device->createBuffer(desc);
    REQUIRE(output);
    REQUIRE_CALL(ShaderCursor(root)["params"]["output"].setBinding(output));
    CHECK(key.matches(native, true)); // PB contents are not the parent's closure.
    auto replacementRoot = device->createRootShaderObject(program);
    REQUIRE(replacementRoot);
    auto replacement = replacementRoot->getObject(ShaderCursor(replacementRoot)["params"].m_offset);
    REQUIRE_CALL(ShaderCursor(root)["params"].setObject(replacement));
    CHECK_FALSE(key.matches(native, true)); // setObject still invalidates its parent.
    key.objects.clear();
    key.capture(native, true, specialized, retained);
    REQUIRE(key.objects.size() >= 2);
    // Construct a collision in the replaced hash-combine operation.
    auto combine = [](uint32_t parent, uint32_t child)
    {
        return parent ^ (child + 0x9e3779b9u + (parent << 6) + (parent >> 2));
    };
    const uint32_t original = combine(key.objects[0].version, key.objects[1].version);
    ++key.objects[0].version;
    const uint32_t parent = key.objects[0].version;
    key.objects[1].version = (original ^ parent) - 0x9e3779b9u - (parent << 6) - (parent >> 2);
    CHECK(combine(parent, key.objects[1].version) == original);
    CHECK_FALSE(key.matches(native, true));
}

GPU_TEST_CASE("shader-object-table-cache-resource2-cancel-reset", D3D12 | Vulkan)
{
    std::string source = R"(
struct Resources { Sampler2D combined; RWStructuredBuffer<float4> output; StructuredBuffer<uint> inactive; };
ParameterBlock<Resources> resources;
[shader("compute")][numthreads(1,1,1)]
void main() { resources.output[0] += resources.combined.SampleLevel(float2(0.5), 0); }
)";
    const bool combined = device->getDeviceType() == DeviceType::Vulkan;
    if (!combined)
    {
        auto pos = source.find("Sampler2D combined;");
        source.replace(pos, strlen("Sampler2D combined;"), "Texture2D<float4> texture; SamplerState sampler;");
        pos = source.find("resources.combined.SampleLevel(float2(0.5), 0)");
        source.replace(pos, strlen("resources.combined.SampleLevel(float2(0.5), 0)"),
                       "resources.texture.SampleLevel(resources.sampler, float2(0.5), 0)");
    }
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source.c_str(), program.writeRef()));
    auto pipeline = device->createComputePipeline({.program = program});
    REQUIRE(pipeline);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root);
    const float texel[]{1, 2, 3, 4};
    TextureDesc textureDesc{};
    textureDesc.size = {1, 1, 1}; textureDesc.format = Format::RGBA32Float;
    textureDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
    const SubresourceData textureData{texel, sizeof(texel), 0};
    auto texture = device->createTexture(textureDesc, &textureData);
    auto sampler = device->createSampler({});
    REQUIRE(texture); REQUIRE(sampler);
    BufferDesc desc{};
    desc.size = sizeof(texel); desc.elementSize = sizeof(texel);
    desc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopyDestination | BufferUsage::CopySource;
    const std::array<float, 4> initial{};
    auto output = device->createBuffer(desc, initial.data());
    REQUIRE(output);
    desc.size = desc.elementSize = sizeof(uint32_t);
    desc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
    const uint32_t unused = 123;
    auto inactive = device->createBuffer(desc, &unused);
    REQUIRE(inactive);
    ShaderCursor cursor(root);
    if (combined)
        REQUIRE_CALL(cursor["resources"]["combined"].setBinding(Binding(texture, sampler)));
    else
    {
        REQUIRE_CALL(cursor["resources"]["texture"].setBinding(texture));
        REQUIRE_CALL(cursor["resources"]["sampler"].setBinding(sampler));
    }
    REQUIRE_CALL(cursor["resources"]["output"].setBinding(output));
    auto counter = device->createBuffer(desc, &unused);
    REQUIRE(counter);
    REQUIRE_CALL(cursor["resources"]["inactive"].setBinding(Binding(inactive, counter)));
    auto queue = device->getQueue(QueueType::Graphics);
    uint64_t previousId = 0;
    for (uint32_t batch = 0; batch < 3; ++batch)
    {
        auto encoder = queue->createCommandEncoder();
        auto* pass = encoder->beginComputePass();
        pass->bindPipeline(pipeline, root);
        for (uint32_t repeat = 0; repeat < 3; ++repeat) pass->dispatchCompute(1, 1, 1);
        pass->end();
        if (batch == 0) continue; // Cancel without finish/submit; no GPU writes are expected.
        auto commands = encoder->finish();
        REQUIRE(commands);
#if SLANG_RHI_ENABLE_BINDING_COUNTERS
        auto* wrapped = dynamic_cast<debug::DebugCommandBuffer*>(commands.get());
        auto* internal = checked_cast<CommandBuffer*>(wrapped ? wrapped->baseObject.get() : commands.get());
        CHECK(internal->m_bindingCounters.trackedRefs == 5);
        CHECK(internal->m_bindingCounters.tableMisses == 2);
        CHECK(internal->m_bindingCounters.tableHits == 4);
        CHECK(internal->m_bindingCounters.commandBufferId != previousId);
        previousId = internal->m_bindingCounters.commandBufferId;
#endif
        if (batch == 2)
        {
            texture.setNull(); sampler.setNull(); inactive.setNull(); counter.setNull();
            root.setNull(); pipeline.setNull(); program.setNull();
        }
        REQUIRE_CALL(queue->submit(commands));
        REQUIRE_CALL(queue->waitOnHost());
        std::array<float, 4> expected;
        for (uint32_t i = 0; i < expected.size(); ++i) expected[i] = float(batch * 3) * texel[i];
        compareComputeResult(device, output, expected);
    }
    SLANG_UNUSED(previousId);
}

GPU_TEST_CASE("shader-object-table-cache-raygen-snapshot", D3D12 | Vulkan)
{
    REQUIRE(device->hasFeature(Feature::RayTracing));
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, "test-binding-cache-sbt", {"raygenMain"}, program.writeRef()));
    RayTracingPipelineDesc desc{};
    desc.program = program;
    auto pipeline = device->createRayTracingPipeline(desc);
    REQUIRE(pipeline);
    const char* entryNames[]{"raygenMain"};
    ShaderTableDesc tableDesc{};
    tableDesc.program = program;
    tableDesc.rayGenShaderCount = 1;
    tableDesc.rayGenShaderEntryPointNames = entryNames;
    ComPtr<IShaderTable> table;
    REQUIRE_CALL(device->createShaderTable(tableDesc, table.writeRef()));
    REQUIRE(table);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root);
    std::array<ComPtr<IBuffer>, 2> outputs;
    std::array<ComPtr<ICommandEncoder>, 2> encoders;
    std::array<std::array<uint32_t, 4>, 2> expected{};
    auto queue = device->getQueue(QueueType::Graphics);
    for (uint32_t batch = 0; batch < 2; ++batch)
    {
        BufferDesc bufferDesc{};
        bufferDesc.size = sizeof(expected[batch]);
        bufferDesc.elementSize = sizeof(uint32_t);
        bufferDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
        REQUIRE_CALL(device->createBuffer(bufferDesc, expected[batch].data(), outputs[batch].writeRef()));
        REQUIRE_CALL(ShaderCursor(root)["resources"]["output"].setBinding(outputs[batch]));
        encoders[batch] = queue->createCommandEncoder();
        auto* pass = encoders[batch]->beginRayTracingPass();
        pass->bindPipeline(pipeline, table, root);
        for (uint32_t i = 0; i < 4; ++i)
        {
            const uint32_t data[]{i, 11 + batch * 17 + i * 3, 0, 0};
            REQUIRE_CALL(ShaderCursor(root->getEntryPoint(0))["params"].setData(data, sizeof(data)));
            for (uint32_t repeat = 0; repeat < 3; ++repeat)
                pass->dispatchRays(0, 1, 1, 1);
            expected[batch][i] = 3 * data[1];
        }
        pass->end();
    }
    auto commandsB = encoders[1]->finish();
    REQUIRE(commandsB);
    REQUIRE_CALL(queue->submit(commandsB));
    REQUIRE_CALL(queue->waitOnHost());
    root.setNull(); pipeline.setNull(); table.setNull(); program.setNull();
    commandsB.setNull(); encoders[1].setNull();
    auto commandsA = encoders[0]->finish();
    REQUIRE(commandsA);
#if SLANG_RHI_ENABLE_BINDING_COUNTERS
    auto* wrapped = dynamic_cast<debug::DebugCommandBuffer*>(commandsA.get());
    auto* internal = checked_cast<CommandBuffer*>(wrapped ? wrapped->baseObject.get() : commandsA.get());
    CHECK(internal->m_bindingCounters.tableMisses == 5);
    CHECK(internal->m_bindingCounters.tableHits == 19);
#endif
    encoders[0].setNull();
    REQUIRE_CALL(queue->submit(commandsA));
    REQUIRE_CALL(queue->waitOnHost());
    for (uint32_t batch = 0; batch < 2; ++batch)
        compareComputeResult(device, outputs[batch], expected[batch]);
}
