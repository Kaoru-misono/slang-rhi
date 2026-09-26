#include "testing.h"
#include <array>
#include <cstring>

using namespace rhi;
using namespace rhi::testing;

struct Vertex
{
    float position[3];
};

struct Instance
{
    float position[3];
    float color[3];
};

static const int kVertexCount = 6;
static const Vertex kVertexData[kVertexCount] = {
    // Triangle 1
    {0, 0, 0.5},
    {1, 0, 0.5},
    {0, 1, 0.5},

    // Triangle 2
    {-1, 0, 0.5},
    {0, 0, 0.5},
    {-1, 1, 0.5},
};

static const int kInstanceCount = 2;
static const Instance kInstanceData[kInstanceCount] = {
    {{0, 0, 0}, {1, 0, 0}},
    {{0, -1, 0}, {0, 0, 1}},
};

static const int kIndexCount = 6;
static const uint32_t kIndexData[kIndexCount] = {
    0,
    2,
    5,
    0,
    1,
    2,
};

const int kWidth = 256;
const int kHeight = 256;
const Format format = Format::RGBA32Float;

static ComPtr<IBuffer> createVertexBuffer(IDevice* device)
{
    BufferDesc vertexBufferDesc;
    vertexBufferDesc.size = kVertexCount * sizeof(Vertex);
    vertexBufferDesc.usage = BufferUsage::VertexBuffer;
    vertexBufferDesc.defaultState = ResourceState::VertexBuffer;
    ComPtr<IBuffer> vertexBuffer = device->createBuffer(vertexBufferDesc, &kVertexData[0]);
    REQUIRE(vertexBuffer != nullptr);
    return vertexBuffer;
}

static ComPtr<IBuffer> createInstanceBuffer(IDevice* device)
{
    BufferDesc instanceBufferDesc;
    instanceBufferDesc.size = kInstanceCount * sizeof(Instance);
    instanceBufferDesc.usage = BufferUsage::VertexBuffer;
    instanceBufferDesc.defaultState = ResourceState::VertexBuffer;
    ComPtr<IBuffer> instanceBuffer = device->createBuffer(instanceBufferDesc, &kInstanceData[0]);
    REQUIRE(instanceBuffer != nullptr);
    return instanceBuffer;
}

static ComPtr<IBuffer> createIndexBuffer(IDevice* device)
{
    BufferDesc indexBufferDesc;
    indexBufferDesc.size = kIndexCount * sizeof(uint32_t);
    indexBufferDesc.usage = BufferUsage::IndexBuffer;
    indexBufferDesc.defaultState = ResourceState::IndexBuffer;
    ComPtr<IBuffer> indexBuffer = device->createBuffer(indexBufferDesc, &kIndexData[0]);
    REQUIRE(indexBuffer != nullptr);
    return indexBuffer;
}

static ComPtr<ITexture> createColorBuffer(IDevice* device)
{
    TextureDesc colorBufferDesc;
    colorBufferDesc.type = TextureType::Texture2D;
    colorBufferDesc.size.width = kWidth;
    colorBufferDesc.size.height = kHeight;
    colorBufferDesc.size.depth = 1;
    colorBufferDesc.mipCount = 1;
    colorBufferDesc.format = format;
    colorBufferDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    colorBufferDesc.defaultState = ResourceState::RenderTarget;
    ComPtr<ITexture> colorBuffer = device->createTexture(colorBufferDesc, nullptr);
    REQUIRE(colorBuffer != nullptr);
    return colorBuffer;
}

/// Legacy binds through the root shader object; Flat binds an explicit pipeline layout.
enum class DrawBinding
{
    Legacy,
    Flat,
};

static void bindDrawPipeline(IRenderPassEncoder* pass, IRenderPipeline* pipeline, DrawBinding binding)
{
    if (binding == DrawBinding::Legacy)
    {
        REQUIRE(pass->bindPipeline(pipeline) != nullptr);
        return;
    }
    REQUIRE_CALL(pass->bindPipeline(pipeline, FlatBindingDesc{}));
}

static ComPtr<IPipelineLayout> createDrawPipelineLayout(IDevice* device, IShaderProgram* program, DrawBinding binding)
{
    if (binding == DrawBinding::Legacy)
        return nullptr;
    PipelineLayoutDesc desc{};
    desc.program = program;
    ComPtr<IPipelineLayout> layout;
    REQUIRE_CALL(device->createPipelineLayout(desc, layout.writeRef()));
    REQUIRE(layout != nullptr);
    return layout;
}

static void runCmdDrawPipelineSwitchVertexStride(IDevice* device, DrawBinding binding)
{
    // Stride 16 visits rows 0,1,2; stride 32 visits rows 0,2,4.
    // These triangles cover complementary halves of the same render target.
    const float vertices[5][4] = {
        {-1.f, -1.f, 0.5f, 0.f},
        { 1.f, -1.f, 0.5f, 0.f},
        { 1.f,  1.f, 0.5f, 0.f},
        { 0.f,  0.f, 0.5f, 0.f},
        {-1.f,  1.f, 0.5f, 0.f},
    };
    BufferDesc bufferDesc{};
    bufferDesc.size = sizeof(vertices);
    bufferDesc.usage = BufferUsage::VertexBuffer;
    bufferDesc.defaultState = ResourceState::VertexBuffer;
    auto vertexBuffer = device->createBuffer(bufferDesc, vertices);
    REQUIRE(vertexBuffer != nullptr);
    auto instanceBuffer = createInstanceBuffer(device);
    auto target = createColorBuffer(device);
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-cmd-draw", {"vertexMain", "fragmentMain"}, program.writeRef()));
    auto pipelineLayout = createDrawPipelineLayout(device, program, binding);

    ComPtr<IRenderPipeline> pipelines[2];
    for (uint32_t i = 0; i < 2; ++i)
    {
        VertexStreamDesc streams[] = {
            {i == 0 ? 16u : 32u, InputSlotClass::PerVertex, 0},
            {sizeof(Instance), InputSlotClass::PerInstance, 1},
        };
        InputElementDesc elements[] = {
            {"POSITIONA", 0, Format::RGB32Float, 0, 0},
            {"POSITIONB", 0, Format::RGB32Float, offsetof(Instance, position), 1},
            {"COLOR", 0, Format::RGB32Float, offsetof(Instance, color), 1},
        };
        InputLayoutDesc inputDesc{};
        inputDesc.inputElements = elements;
        inputDesc.inputElementCount = SLANG_COUNT_OF(elements);
        inputDesc.vertexStreams = streams;
        inputDesc.vertexStreamCount = SLANG_COUNT_OF(streams);
        auto inputLayout = device->createInputLayout(inputDesc);
        REQUIRE(inputLayout != nullptr);
        ColorTargetDesc colorTarget{};
        colorTarget.format = format;
        RenderPipelineDesc desc{};
        desc.program = program;
        desc.layout = pipelineLayout;
        desc.inputLayout = inputLayout;
        desc.targets = &colorTarget;
        desc.targetCount = 1;
        desc.rasterizer.cullMode = CullMode::None;
        desc.depthStencil.depthTestEnable = false;
        desc.depthStencil.depthWriteEnable = false;
        REQUIRE_CALL(device->createRenderPipeline(desc, pipelines[i].writeRef()));
    }

    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    RenderPassColorAttachment color{};
    color.view = target->getDefaultView();
    color.loadOp = LoadOp::Clear;
    color.storeOp = StoreOp::Store;
    RenderPassDesc passDesc{};
    passDesc.colorAttachments = &color;
    passDesc.colorAttachmentCount = 1;
    auto pass = encoder->beginRenderPass(passDesc);
    REQUIRE(pass != nullptr);
    RenderState state{};
    state.viewports[0] = Viewport::fromSize(kWidth, kHeight);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(kWidth, kHeight);
    state.scissorRectCount = 1;
    state.vertexBuffers[0] = vertexBuffer;
    state.vertexBuffers[1] = instanceBuffer;
    state.vertexBufferCount = 2;
    pass->setRenderState(state);
    DrawArguments draw{};
    draw.vertexCount = 3;
    draw.instanceCount = 1;
    bindDrawPipeline(pass, pipelines[0], binding);
    pass->draw(draw);
    // No render-state setter between draws: only the pipeline changes.
    bindDrawPipeline(pass, pipelines[1], binding);
    pass->draw(draw);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());

    ComPtr<ISlangBlob> result;
    SubresourceLayout layout{};
    REQUIRE_CALL(device->readTexture(target, 0, 0, result.writeRef(), &layout));
    const auto* bytes = static_cast<const uint8_t*>(result->getBufferPointer());
    const float expected[4] = {1.f, 0.f, 0.f, 1.f};
    for (uint32_t y = 0; y < kHeight; ++y)
    {
        for (uint32_t x = 0; x < kWidth; ++x)
        {
            CAPTURE(x);
            CAPTURE(y);
            CHECK(std::memcmp(bytes + y * layout.rowPitch + x * sizeof(expected), expected, sizeof(expected)) == 0);
        }
    }
}

GPU_TEST_CASE("cmd-draw-pipeline-switch-vertex-stride", D3D12 | Vulkan)
{
    runCmdDrawPipelineSwitchVertexStride(device, DrawBinding::Legacy);
}

GPU_TEST_CASE("cmd-draw-pipeline-switch-vertex-stride-flat", D3D12 | Vulkan)
{
    runCmdDrawPipelineSwitchVertexStride(device, DrawBinding::Flat);
}

class BaseDrawTest
{
public:
    ComPtr<IDevice> device;

    ComPtr<IRenderPipeline> pipeline;

    ComPtr<IBuffer> vertexBuffer;
    ComPtr<IBuffer> instanceBuffer;
    ComPtr<ITexture> colorBuffer;
    ComPtr<ITextureView> colorBufferView;

    DrawBinding binding = DrawBinding::Flat;

    void init(IDevice* device_, DrawBinding binding_)
    {
        this->device = device_;
        this->binding = binding_;
    }

    void createRequiredResources()
    {
        VertexStreamDesc vertexStreams[] = {
            {sizeof(Vertex), InputSlotClass::PerVertex, 0},
            {sizeof(Instance), InputSlotClass::PerInstance, 1},
        };

        InputElementDesc inputElements[] = {
            // Vertex buffer data
            {"POSITIONA", 0, Format::RGB32Float, offsetof(Vertex, position), 0},

            // Instance buffer data
            {"POSITIONB", 0, Format::RGB32Float, offsetof(Instance, position), 1},
            {"COLOR", 0, Format::RGB32Float, offsetof(Instance, color), 1},
        };
        InputLayoutDesc inputLayoutDesc = {};
        inputLayoutDesc.inputElementCount = SLANG_COUNT_OF(inputElements);
        inputLayoutDesc.inputElements = inputElements;
        inputLayoutDesc.vertexStreamCount = SLANG_COUNT_OF(vertexStreams);
        inputLayoutDesc.vertexStreams = vertexStreams;
        auto inputLayout = device->createInputLayout(inputLayoutDesc);
        REQUIRE(inputLayout != nullptr);

        vertexBuffer = createVertexBuffer(device);
        instanceBuffer = createInstanceBuffer(device);
        colorBuffer = createColorBuffer(device);

        ComPtr<IShaderProgram> shaderProgram;
        REQUIRE_CALL(loadProgram(device, "test-cmd-draw", {"vertexMain", "fragmentMain"}, shaderProgram.writeRef()));
        auto pipelineLayout = createDrawPipelineLayout(device, shaderProgram, binding);

        ColorTargetDesc colorTarget;
        colorTarget.format = format;
        RenderPipelineDesc pipelineDesc = {};
        pipelineDesc.program = shaderProgram.get();
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.inputLayout = inputLayout;
        pipelineDesc.targets = &colorTarget;
        pipelineDesc.targetCount = 1;
        pipelineDesc.depthStencil.depthTestEnable = false;
        pipelineDesc.depthStencil.depthWriteEnable = false;
        REQUIRE_CALL(device->createRenderPipeline(pipelineDesc, pipeline.writeRef()));

        TextureViewDesc colorBufferViewDesc = {};
        colorBufferViewDesc.format = format;
        REQUIRE_CALL(device->createTextureView(colorBuffer, colorBufferViewDesc, colorBufferView.writeRef()));
    }

    void checkTestResults(
        int pixelCount,
        int channelCount,
        const int* testXCoords,
        const int* testYCoords,
        float* testResults
    )
    {
        // Read texture values back from four specific pixels located within the triangles
        // and compare against expected values (because testing every single pixel will be too long and tedious
        // and requires maintaining reference images).
        ComPtr<ISlangBlob> resultBlob;
        SubresourceLayout layout;
        REQUIRE_CALL(device->readTexture(colorBuffer, 0, 0, resultBlob.writeRef(), &layout));
        auto result = (float*)resultBlob->getBufferPointer();

        int cursor = 0;
        for (int i = 0; i < pixelCount; ++i)
        {
            auto x = testXCoords[i];
            auto y = testYCoords[i];
            auto pixelPtr = result + x * channelCount + y * layout.rowPitch / sizeof(float);
            for (int j = 0; j < channelCount; ++j)
            {
                testResults[cursor] = pixelPtr[j];
                cursor++;
            }
        }

        float expectedResult[] =
            {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
        compareResultFuzzy(testResults, expectedResult, SLANG_COUNT_OF(expectedResult));
    }
};

struct DrawInstancedTest : BaseDrawTest
{
    void setUpAndDraw()
    {
        createRequiredResources();

        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        RenderPassColorAttachment colorAttachment;
        colorAttachment.view = colorBufferView;
        colorAttachment.loadOp = LoadOp::Clear;
        colorAttachment.storeOp = StoreOp::Store;
        RenderPassDesc renderPass;
        renderPass.colorAttachments = &colorAttachment;
        renderPass.colorAttachmentCount = 1;
        auto passEncoder = commandEncoder->beginRenderPass(renderPass);

        bindDrawPipeline(passEncoder, pipeline, binding);

        RenderState state;
        state.viewports[0] = Viewport::fromSize(kWidth, kHeight);
        state.viewportCount = 1;
        state.scissorRects[0] = ScissorRect::fromSize(kWidth, kHeight);
        state.scissorRectCount = 1;
        state.vertexBuffers[0] = vertexBuffer;
        state.vertexBuffers[1] = instanceBuffer;
        state.vertexBufferCount = 2;
        passEncoder->setRenderState(state);

        DrawArguments args;
        args.vertexCount = kVertexCount;
        args.instanceCount = kInstanceCount;
        passEncoder->draw(args);
        passEncoder->end();

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    void run()
    {
        setUpAndDraw();

        const int kPixelCount = 4;
        const int kChannelCount = 4;
        int testXCoords[kPixelCount] = {64, 192, 64, 192};
        int testYCoords[kPixelCount] = {100, 100, 250, 250};
        float testResults[kPixelCount * kChannelCount];

        checkTestResults(kPixelCount, kChannelCount, testXCoords, testYCoords, testResults);
    }
};

struct DrawIndexedInstancedTest : BaseDrawTest
{
    ComPtr<IBuffer> indexBuffer;

    void setUpAndDraw()
    {
        createRequiredResources();

        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        RenderPassColorAttachment colorAttachment;
        colorAttachment.view = colorBufferView;
        colorAttachment.loadOp = LoadOp::Clear;
        colorAttachment.storeOp = StoreOp::Store;
        RenderPassDesc renderPass;
        renderPass.colorAttachments = &colorAttachment;
        renderPass.colorAttachmentCount = 1;
        auto passEncoder = commandEncoder->beginRenderPass(renderPass);

        bindDrawPipeline(passEncoder, pipeline, binding);

        RenderState state;
        state.vertexBuffers[0] = vertexBuffer;
        state.vertexBuffers[1] = instanceBuffer;
        state.vertexBufferCount = 2;
        state.indexBuffer = indexBuffer;
        state.indexFormat = IndexFormat::Uint32;
        state.viewports[0] = Viewport::fromSize(kWidth, kHeight);
        state.viewportCount = 1;
        state.scissorRects[0] = ScissorRect::fromSize(kWidth, kHeight);
        state.scissorRectCount = 1;
        passEncoder->setRenderState(state);

        DrawArguments args;
        args.vertexCount = kVertexCount;
        args.instanceCount = kInstanceCount;
        passEncoder->drawIndexed(args);
        passEncoder->end();

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    void run()
    {
        indexBuffer = createIndexBuffer(device);

        setUpAndDraw();

        const int kPixelCount = 4;
        const int kChannelCount = 4;
        int testXCoords[kPixelCount] = {64, 192, 64, 192};
        int testYCoords[kPixelCount] = {32, 100, 150, 250};
        float testResults[kPixelCount * kChannelCount];

        checkTestResults(kPixelCount, kChannelCount, testXCoords, testYCoords, testResults);
    }
};

struct DrawIndirectTest : BaseDrawTest
{
    ComPtr<IBuffer> indirectBuffer;

    struct IndirectArgData
    {
        float padding; // Ensure args and count don't start at 0 offset for testing purposes
        IndirectDrawArguments args;
    };

    ComPtr<IBuffer> createIndirectBuffer()
    {
        static const IndirectArgData kIndirectData = {
            42.0f,        // padding
            {6, 2, 0, 0}, // args
        };

        BufferDesc bufferDesc;
        bufferDesc.size = sizeof(IndirectArgData);
        bufferDesc.usage = BufferUsage::IndirectArgument | BufferUsage::ShaderResource;
        bufferDesc.defaultState = ResourceState::IndirectArgument;
        ComPtr<IBuffer> buffer = device->createBuffer(bufferDesc, &kIndirectData);
        REQUIRE(buffer != nullptr);
        return buffer;
    }

    void setUpAndDraw()
    {
        createRequiredResources();

        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();
        if (device->getDeviceType() == DeviceType::D3D12)
            commandEncoder->setBufferState(indirectBuffer, ResourceState::ShaderResource);

        RenderPassColorAttachment colorAttachment;
        colorAttachment.view = colorBufferView;
        colorAttachment.loadOp = LoadOp::Clear;
        colorAttachment.storeOp = StoreOp::Store;
        RenderPassDesc renderPass;
        renderPass.colorAttachments = &colorAttachment;
        renderPass.colorAttachmentCount = 1;
        auto passEncoder = commandEncoder->beginRenderPass(renderPass);

        bindDrawPipeline(passEncoder, pipeline, binding);

        RenderState state;
        state.vertexBuffers[0] = vertexBuffer;
        state.vertexBuffers[1] = instanceBuffer;
        state.vertexBufferCount = 2;
        state.viewports[0] = Viewport::fromSize(kWidth, kHeight);
        state.viewportCount = 1;
        state.scissorRects[0] = ScissorRect::fromSize(kWidth, kHeight);
        state.scissorRectCount = 1;
        passEncoder->setRenderState(state);

        uint32_t maxDrawCount = 1;
        BufferOffsetPair argBuffer = {indirectBuffer, offsetof(IndirectArgData, args)};
        passEncoder->drawIndirect(maxDrawCount, argBuffer);
        passEncoder->end();

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    void run()
    {
        indirectBuffer = createIndirectBuffer();

        setUpAndDraw();

        const int kPixelCount = 4;
        const int kChannelCount = 4;
        int testXCoords[kPixelCount] = {64, 192, 64, 192};
        int testYCoords[kPixelCount] = {100, 100, 250, 250};
        float testResults[kPixelCount * kChannelCount];

        checkTestResults(kPixelCount, kChannelCount, testXCoords, testYCoords, testResults);
    }
};

struct DrawIndexedIndirectTest : BaseDrawTest
{
    ComPtr<IBuffer> indexBuffer;
    ComPtr<IBuffer> indirectBuffer;

    struct IndexedIndirectArgData
    {
        float padding; // Ensure args and count don't start at 0 offset for testing purposes
        IndirectDrawIndexedArguments args;
    };

    ComPtr<IBuffer> createIndirectBuffer()
    {
        static const IndexedIndirectArgData kIndexedIndirectData = {
            42.0f,           // padding
            {6, 2, 0, 0, 0}, // args
        };

        BufferDesc bufferDesc;
        bufferDesc.size = sizeof(IndexedIndirectArgData);
        bufferDesc.usage = BufferUsage::IndirectArgument | BufferUsage::ShaderResource;
        bufferDesc.defaultState = ResourceState::IndirectArgument;
        ComPtr<IBuffer> buffer = device->createBuffer(bufferDesc, &kIndexedIndirectData);
        REQUIRE(buffer != nullptr);
        return buffer;
    }

    void setUpAndDraw()
    {
        createRequiredResources();

        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();
        if (device->getDeviceType() == DeviceType::D3D12)
            commandEncoder->setBufferState(indirectBuffer, ResourceState::ShaderResource);

        RenderPassColorAttachment colorAttachment;
        colorAttachment.view = colorBufferView;
        colorAttachment.loadOp = LoadOp::Clear;
        colorAttachment.storeOp = StoreOp::Store;
        RenderPassDesc renderPass;
        renderPass.colorAttachments = &colorAttachment;
        renderPass.colorAttachmentCount = 1;
        auto passEncoder = commandEncoder->beginRenderPass(renderPass);

        bindDrawPipeline(passEncoder, pipeline, binding);

        RenderState state;
        state.vertexBuffers[0] = vertexBuffer;
        state.vertexBuffers[1] = instanceBuffer;
        state.vertexBufferCount = 2;
        state.indexBuffer = indexBuffer;
        state.indexFormat = IndexFormat::Uint32;
        state.viewports[0] = Viewport::fromSize(kWidth, kHeight);
        state.viewportCount = 1;
        state.scissorRects[0] = ScissorRect::fromSize(kWidth, kHeight);
        state.scissorRectCount = 1;
        passEncoder->setRenderState(state);

        uint32_t maxDrawCount = 1;
        BufferOffsetPair argBuffer = {indirectBuffer, offsetof(IndexedIndirectArgData, args)};
        passEncoder->drawIndexedIndirect(maxDrawCount, argBuffer);
        passEncoder->end();

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    void run()
    {
        indexBuffer = createIndexBuffer(device);
        indirectBuffer = createIndirectBuffer();

        setUpAndDraw();

        const int kPixelCount = 4;
        const int kChannelCount = 4;
        int testXCoords[kPixelCount] = {64, 192, 64, 192};
        int testYCoords[kPixelCount] = {32, 100, 150, 250};
        float testResults[kPixelCount * kChannelCount];

        checkTestResults(kPixelCount, kChannelCount, testXCoords, testYCoords, testResults);
    }
};

template<typename T>
void testDraw(IDevice* device, DrawBinding binding)
{
    T test;
    test.init(device, binding);
    test.run();
}

static void runCmdDrawInstanced(IDevice* device, DrawBinding binding)
{
    testDraw<DrawInstancedTest>(device, binding);
}

GPU_TEST_CASE("cmd-draw-instanced", D3D11 | D3D12 | Vulkan | Metal | WGPU)
{
    runCmdDrawInstanced(device, DrawBinding::Legacy);
}

GPU_TEST_CASE("cmd-draw-instanced-flat", D3D12 | Vulkan)
{
    runCmdDrawInstanced(device, DrawBinding::Flat);
}

static void runCmdDrawIndexedInstanced(IDevice* device, DrawBinding binding)
{
    testDraw<DrawIndexedInstancedTest>(device, binding);
}

GPU_TEST_CASE("cmd-draw-indexed-instanced", D3D11 | D3D12 | Vulkan | Metal | WGPU)
{
    runCmdDrawIndexedInstanced(device, DrawBinding::Legacy);
}

GPU_TEST_CASE("cmd-draw-indexed-instanced-flat", D3D12 | Vulkan)
{
    runCmdDrawIndexedInstanced(device, DrawBinding::Flat);
}

static void runCmdDrawIndirect(IDevice* device, DrawBinding binding)
{
    testDraw<DrawIndirectTest>(device, binding);
}

GPU_TEST_CASE("cmd-draw-indirect", D3D11 | D3D12 | Vulkan)
{
    runCmdDrawIndirect(device, DrawBinding::Legacy);
}

GPU_TEST_CASE("cmd-draw-indirect-flat", D3D12 | Vulkan)
{
    runCmdDrawIndirect(device, DrawBinding::Flat);
}

static void runCmdDrawIndexedIndirect(IDevice* device, DrawBinding binding)
{
    testDraw<DrawIndexedIndirectTest>(device, binding);
}

GPU_TEST_CASE("cmd-draw-indexed-indirect", D3D11 | D3D12 | Vulkan)
{
    runCmdDrawIndexedIndirect(device, DrawBinding::Legacy);
}

GPU_TEST_CASE("cmd-draw-indexed-indirect-flat", D3D12 | Vulkan)
{
    runCmdDrawIndexedIndirect(device, DrawBinding::Flat);
}

namespace dynamicDrawStateTests {
ComPtr<ITexture> makeTarget(IDevice* device, bool depth = false)
{
    TextureDesc desc{};
    desc.size = {5, 1, 1};
    desc.format = depth ? Format::D32FloatS8Uint : Format::RGBA32Float;
    desc.usage = depth ? TextureUsage::DepthStencil : TextureUsage::RenderTarget | TextureUsage::CopySource;
    desc.defaultState = depth ? ResourceState::DepthWrite : ResourceState::RenderTarget;
    auto texture = device->createTexture(desc);
    REQUIRE(texture != nullptr);
    return texture;
}

ComPtr<IRenderPipeline> makePipeline(
    IDevice* device,
    DrawBinding binding,
    const char* fragment,
    bool stencil,
    bool blend = false
)
{
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, "test-dynamic-draw-state", {"vsMain", fragment}, program.writeRef()));
    auto pipelineLayout = createDrawPipelineLayout(device, program, binding);
    ColorTargetDesc target{};
    target.format = Format::RGBA32Float;
    target.enableBlend = blend;
    if (blend)
    {
        target.color.srcFactor = BlendFactor::BlendColor;
        target.alpha.srcFactor = BlendFactor::BlendColor;
    }
    RenderPipelineDesc desc{};
    desc.program = program;
    desc.layout = pipelineLayout;
    desc.targets = &target;
    desc.targetCount = 1;
    desc.rasterizer.cullMode = CullMode::None;
    desc.depthStencil.depthTestEnable = false;
    desc.depthStencil.depthWriteEnable = false;
    desc.depthStencil.stencilEnable = stencil;
    if (stencil)
    {
        desc.depthStencil.format = Format::D32FloatS8Uint;
        desc.depthStencil.stencilWriteMask = 0;
        desc.depthStencil.frontFace.stencilFunc = ComparisonFunc::Equal;
        desc.depthStencil.backFace.stencilFunc = ComparisonFunc::Equal;
    }
    ComPtr<IRenderPipeline> pipeline;
    REQUIRE_CALL(device->createRenderPipeline(desc, pipeline.writeRef()));
    return pipeline;
}

RenderState fullState()
{
    RenderState state{};
    state.viewports[0] = Viewport::fromSize(5, 1);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(5, 1);
    state.scissorRectCount = 1;
    return state;
}

void checkPixels(IDevice* device, ITexture* texture, const std::array<float, 20>& expected)
{
    ComPtr<ISlangBlob> bytes;
    SubresourceLayout layout{};
    REQUIRE_CALL(device->readTexture(texture, 0, 0, bytes.writeRef(), &layout));
    REQUIRE(layout.size.width == 5);
    REQUIRE(layout.size.height == 1);
    REQUIRE(layout.colPitch == 4 * sizeof(float));
    REQUIRE(bytes->getBufferSize() >= sizeof(expected));
    for (size_t i = 0; i < expected.size(); ++i)
    {
        float actual = 0;
        std::memcpy(&actual, static_cast<const uint8_t*>(bytes->getBufferPointer()) + i * sizeof(float), sizeof(float));
        CAPTURE(i);
        CHECK(actual == expected[i]);
    }
}

static void runDynamicStencilReferenceDrawPipelinePass(IDevice* device, DrawBinding binding)
{
    auto first = makeTarget(device);
    auto second = makeTarget(device);
    auto depth = makeTarget(device, true);
    auto white = makePipeline(device, binding, "psWhite", true);
    auto red = makePipeline(device, binding, "psRed", true);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    RenderPassColorAttachment color{};
    color.view = first->getDefaultView();
    color.loadOp = LoadOp::Clear;
    color.storeOp = StoreOp::Store;
    RenderPassDepthStencilAttachment ds{};
    ds.view = depth->getDefaultView();
    ds.stencilClearValue = 7;
    RenderPassDesc passDesc{.colorAttachments = &color, .colorAttachmentCount = 1, .depthStencilAttachment = &ds};
    auto pass = encoder->beginRenderPass(passDesc);
    auto state = fullState();
    DrawArguments draw{};
    draw.vertexCount = 3;
    auto drawPixel = [&](uint32_t x, uint32_t ref)
    {
        state.stencilRef = ref;
        state.scissorRects[0] = {x, 0, x + 1, 1};
        pass->setRenderState(state);
        pass->draw(draw);
    };
    bindDrawPipeline(pass, white, binding);
    drawPixel(0, 7);
    bindDrawPipeline(pass, red, binding);
    pass->draw(draw);
    bindDrawPipeline(pass, white, binding);
    drawPixel(1, 3);
    bindDrawPipeline(pass, red, binding);
    drawPixel(2, 3);
    drawPixel(3, 7);
    bindDrawPipeline(pass, white, binding);
    drawPixel(4, 7);
    pass->end();
    color.view = second->getDefaultView();
    ds.stencilClearValue = 0;
    pass = encoder->beginRenderPass(passDesc);
    bindDrawPipeline(pass, white, binding);
    pass->setRenderState(fullState());
    pass->draw(draw);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    checkPixels(device, first, {1,0,0,1, 0,0,0,0, 0,0,0,0, 1,0,0,1, 1,1,1,1});
    checkPixels(device, second, {1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1});
}

GPU_TEST_CASE("dynamic-stencil-reference-draw-pipeline-pass", D3D12 | Vulkan)
{
    runDynamicStencilReferenceDrawPipelinePass(device, DrawBinding::Legacy);
}

GPU_TEST_CASE("dynamic-stencil-reference-draw-pipeline-pass-flat", D3D12 | Vulkan)
{
    runDynamicStencilReferenceDrawPipelinePass(device, DrawBinding::Flat);
}

static void runDynamicBlendFactorDefaultZero(IDevice* device, DrawBinding binding)
{
    auto target = makeTarget(device);
    auto pipeline = makePipeline(device, binding, "psWhite", false, true);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    RenderPassColorAttachment color{};
    color.view = target->getDefaultView();
    color.loadOp = LoadOp::Clear;
    for (float& channel : color.clearValue)
        channel = 0.5f;
    RenderPassDesc desc{.colorAttachments = &color, .colorAttachmentCount = 1};
    auto pass = encoder->beginRenderPass(desc);
    bindDrawPipeline(pass, pipeline, binding);
    pass->setRenderState(fullState());
    DrawArguments draw{};
    draw.vertexCount = 3;
    pass->draw(draw);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    checkPixels(device, target, {});
}

GPU_TEST_CASE("dynamic-blend-factor-default-zero", D3D12 | Vulkan)
{
    runDynamicBlendFactorDefaultZero(device, DrawBinding::Legacy);
}

GPU_TEST_CASE("dynamic-blend-factor-default-zero-flat", D3D12 | Vulkan)
{
    runDynamicBlendFactorDefaultZero(device, DrawBinding::Flat);
}

static void runDynamicBlendFactorDrawPipelinePass(IDevice* device, DrawBinding binding)
{
    auto first = makeTarget(device);
    auto second = makeTarget(device);
    auto white = makePipeline(device, binding, "psWhite", false, true);
    auto red = makePipeline(device, binding, "psRed", false, true);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    RenderPassColorAttachment color{};
    color.view = first->getDefaultView();
    for (float& channel : color.clearValue)
        channel = 0.5f;
    RenderPassDesc desc{.colorAttachments = &color, .colorAttachmentCount = 1};
    auto pass = encoder->beginRenderPass(desc);
    auto state = fullState();
    DrawArguments draw{};
    draw.vertexCount = 3;
    const float a[4] = {0.25f, 0.5f, 0.75f, 0.125f};
    const float b[4] = {0.75f, 0.25f, 0.5f, 0.875f};
    auto drawPixel = [&](uint32_t x)
    {
        state.scissorRects[0] = {x, 0, x + 1, 1};
        pass->setRenderState(state);
        pass->draw(draw);
    };
    bindDrawPipeline(pass, white, binding);
    std::memcpy(state.blendFactor, a, sizeof(a));
    drawPixel(0);
    bindDrawPipeline(pass, red, binding);
    pass->draw(draw);
    bindDrawPipeline(pass, white, binding);
    std::memcpy(state.blendFactor, b, sizeof(b));
    drawPixel(1);
    bindDrawPipeline(pass, red, binding);
    drawPixel(2);
    std::memcpy(state.blendFactor, a, sizeof(a));
    drawPixel(3);
    bindDrawPipeline(pass, white, binding);
    drawPixel(4);
    pass->end();
    color.view = second->getDefaultView();
    pass = encoder->beginRenderPass(desc);
    bindDrawPipeline(pass, white, binding);
    pass->setRenderState(fullState());
    pass->draw(draw);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    checkPixels(device, first, {
        .25f,0,0,.125f, .75f,.25f,.5f,.875f, .75f,0,0,.875f,
        .25f,0,0,.125f, .25f,.5f,.75f,.125f
    });
    checkPixels(device, second, {});
}

GPU_TEST_CASE("dynamic-blend-factor-draw-pipeline-pass", D3D12 | Vulkan)
{
    runDynamicBlendFactorDrawPipelinePass(device, DrawBinding::Legacy);
}

GPU_TEST_CASE("dynamic-blend-factor-draw-pipeline-pass-flat", D3D12 | Vulkan)
{
    runDynamicBlendFactorDrawPipelinePass(device, DrawBinding::Flat);
}

GPU_TEST_CASE("input-layout-explicit-location-validation", Vulkan)
{
    VertexStreamDesc stream{};
    stream.stride = 16;
    InputElementDesc elements[2] = {
        {"POSITION", 0, Format::RG32Float, 0, 0},
        {"TEXCOORD", 0, Format::RG32Float, 8, 0},
    };
    InputLayoutDesc desc{};
    desc.inputElements = elements;
    desc.inputElementCount = 2;
    desc.vertexStreams = &stream;
    desc.vertexStreamCount = 1;
    ComPtr<IInputLayout> layout;
    REQUIRE_CALL(device->createInputLayout(desc, layout.writeRef()));
    layout.setNull();
    elements[0].shaderLocation = 3;
    elements[1].shaderLocation = 7;
    REQUIRE_CALL(device->createInputLayout(desc, layout.writeRef()));
    layout.setNull();
    elements[1].shaderLocation = 3;
    CHECK(device->createInputLayout(desc, layout.writeRef()) == SLANG_E_INVALID_ARG);
    CHECK(layout == nullptr);
    elements[0].shaderLocation = 1;
    elements[1].shaderLocation = UINT32_MAX;
    CHECK(device->createInputLayout(desc, layout.writeRef()) == SLANG_E_INVALID_ARG);
    CHECK(layout == nullptr);
    elements[0].shaderLocation = UINT32_MAX - 1;
    CHECK(device->createInputLayout(desc, layout.writeRef()) == SLANG_E_INVALID_ARG);
    CHECK(layout == nullptr);
    elements[0].shaderLocation = device->getInfo().limits.maxVertexInputElements;
    CHECK(device->createInputLayout(desc, layout.writeRef()) == SLANG_E_INVALID_ARG);
    CHECK(layout == nullptr);
}

static void runCmdDrawPipelineSwitchUnusedVertexStreams(IDevice* device, DrawBinding binding)
{
    auto target = makeTarget(device);
    auto vertexBuffer = createVertexBuffer(device);
    auto instanceBuffer = createInstanceBuffer(device);
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, "test-dynamic-draw-state", {"vsMain", "psRed"}, program.writeRef()));
    auto pipelineLayout = createDrawPipelineLayout(device, program, binding);
    ComPtr<IRenderPipeline> pipelines[3];
    for (uint32_t i = 0; i < 3; ++i)
    {
        ComPtr<IInputLayout> inputLayout;
        if (i < 2)
        {
            VertexStreamDesc streams[] = {
                {sizeof(Vertex), InputSlotClass::PerVertex, 0},
                {sizeof(Instance), InputSlotClass::PerInstance, 1},
            };
            InputLayoutDesc inputDesc{};
            inputDesc.vertexStreams = streams;
            inputDesc.vertexStreamCount = 2 - i;
            REQUIRE_CALL(device->createInputLayout(inputDesc, inputLayout.writeRef()));
        }
        ColorTargetDesc colorTarget{};
        colorTarget.format = Format::RGBA32Float;
        colorTarget.enableBlend = true;
        colorTarget.color.srcFactor = BlendFactor::One;
        colorTarget.color.dstFactor = BlendFactor::One;
        colorTarget.alpha.srcFactor = BlendFactor::One;
        colorTarget.alpha.dstFactor = BlendFactor::One;
        RenderPipelineDesc desc{};
        desc.program = program;
        desc.layout = pipelineLayout;
        desc.inputLayout = inputLayout;
        desc.targets = &colorTarget;
        desc.targetCount = 1;
        desc.rasterizer.cullMode = CullMode::None;
        desc.depthStencil.depthTestEnable = false;
        desc.depthStencil.depthWriteEnable = false;
        REQUIRE_CALL(device->createRenderPipeline(desc, pipelines[i].writeRef()));
    }
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    RenderPassColorAttachment color{};
    color.view = target->getDefaultView();
    color.loadOp = LoadOp::Clear;
    color.storeOp = StoreOp::Store;
    RenderPassDesc desc{.colorAttachments = &color, .colorAttachmentCount = 1};
    auto pass = encoder->beginRenderPass(desc);
    auto state = fullState();
    state.vertexBuffers[0] = vertexBuffer;
    state.vertexBuffers[1] = instanceBuffer;
    state.vertexBufferCount = 2;
    pass->setRenderState(state);
    DrawArguments draw{};
    draw.vertexCount = 3;
    for (const auto& pipeline : pipelines)
    {
        bindDrawPipeline(pass, pipeline, binding);
        pass->draw(draw);
    }
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    checkPixels(device, target, {3,0,0,3, 3,0,0,3, 3,0,0,3, 3,0,0,3, 3,0,0,3});
}

GPU_TEST_CASE("cmd-draw-pipeline-switch-unused-vertex-streams", D3D12 | Vulkan)
{
    runCmdDrawPipelineSwitchUnusedVertexStreams(device, DrawBinding::Legacy);
}

GPU_TEST_CASE("cmd-draw-pipeline-switch-unused-vertex-streams-flat", D3D12 | Vulkan)
{
    runCmdDrawPipelineSwitchUnusedVertexStreams(device, DrawBinding::Flat);
}

} // namespace dynamicDrawStateTests
