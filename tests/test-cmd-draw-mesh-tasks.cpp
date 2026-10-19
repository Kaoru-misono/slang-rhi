#include "testing.h"

#include <array>

using namespace rhi;
using namespace rhi::testing;

namespace {

const uint32_t kColumnCount = 4;
const uint32_t kWidth = 64;
const uint32_t kHeight = 16;

struct DrawMeshTasksArgs
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

// The count and the records start away from offset 0. The first record lights column 0, the second all columns.
struct IndirectData
{
    uint32_t padding0;
    uint32_t count;
    uint32_t padding1[2];
    DrawMeshTasksArgs args[2];
};

const IndirectData kIndirectData = {0, 1, {0, 0}, {{1, 1, 1}, {kColumnCount, 1, 1}}};

// Draws up to `maxDrawCount` records, counted by `kIndirectData.count` when `useCount` is set, and returns which
// columns were lit.
std::array<bool, kColumnCount> drawColumns(IDevice* device, uint32_t maxDrawCount, bool useCount)
{
    BufferDesc bufferDesc;
    bufferDesc.size = sizeof(IndirectData);
    bufferDesc.usage = BufferUsage::IndirectArgument | BufferUsage::ShaderResource;
    bufferDesc.defaultState = ResourceState::ShaderResource;
    ComPtr<IBuffer> indirectBuffer = device->createBuffer(bufferDesc, &kIndirectData);
    REQUIRE(indirectBuffer != nullptr);

    TextureDesc targetDesc;
    targetDesc.type = TextureType::Texture2D;
    targetDesc.size = {kWidth, kHeight, 1};
    targetDesc.mipCount = 1;
    targetDesc.format = Format::RGBA32Float;
    targetDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    targetDesc.defaultState = ResourceState::RenderTarget;
    ComPtr<ITexture> target = device->createTexture(targetDesc, nullptr);
    REQUIRE(target != nullptr);

    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-cmd-draw-mesh-tasks", {"meshMain", "fragmentMain"}, program.writeRef()));
    ColorTargetDesc colorTarget;
    colorTarget.format = Format::RGBA32Float;
    RenderPipelineDesc pipelineDesc;
    pipelineDesc.program = program;
    pipelineDesc.targets = &colorTarget;
    pipelineDesc.targetCount = 1;
    pipelineDesc.rasterizer.cullMode = CullMode::None;
    pipelineDesc.depthStencil.depthTestEnable = false;
    pipelineDesc.depthStencil.depthWriteEnable = false;
    ComPtr<IRenderPipeline> pipeline;
    REQUIRE_CALL(device->createRenderPipeline(pipelineDesc, pipeline.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    RenderPassColorAttachment color;
    color.view = target->getDefaultView();
    color.loadOp = LoadOp::Clear;
    color.storeOp = StoreOp::Store;
    RenderPassDesc passDesc;
    passDesc.colorAttachments = &color;
    passDesc.colorAttachmentCount = 1;
    auto pass = encoder->beginRenderPass(passDesc);
    REQUIRE(pass != nullptr);
    REQUIRE(pass->bindPipeline(pipeline) != nullptr);
    RenderState state;
    state.viewports[0] = Viewport::fromSize(kWidth, kHeight);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(kWidth, kHeight);
    state.scissorRectCount = 1;
    pass->setRenderState(state);
    BufferOffsetPair argBuffer = {indirectBuffer, offsetof(IndirectData, args)};
    BufferOffsetPair countBuffer =
        useCount ? BufferOffsetPair{indirectBuffer, offsetof(IndirectData, count)} : BufferOffsetPair{};
    pass->drawMeshTasksIndirect(maxDrawCount, argBuffer, countBuffer);
    pass->end();
    REQUIRE_CALL(queue->submit(encoder->finish()));
    REQUIRE_CALL(queue->waitOnHost());

    ComPtr<ISlangBlob> pixels;
    SubresourceLayout layout;
    REQUIRE_CALL(device->readTexture(target, 0, 0, pixels.writeRef(), &layout));
    std::array<bool, kColumnCount> lit = {};
    for (uint32_t column = 0; column < kColumnCount; ++column)
    {
        uint32_t x = column * (kWidth / kColumnCount) + kWidth / kColumnCount / 2;
        const auto* pixel = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(pixels->getBufferPointer()) + (kHeight / 2) * layout.rowPitch +
            x * 4 * sizeof(float)
        );
        lit[column] = pixel[0] == 1.f;
    }
    return lit;
}

} // namespace

GPU_TEST_CASE("cmd-draw-mesh-tasks-indirect", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::MeshShader))
        SKIP("mesh shaders not supported");

    CHECK(drawColumns(device, 1, false) == std::array<bool, kColumnCount>{true, false, false, false});
    CHECK(drawColumns(device, 2, false) == std::array<bool, kColumnCount>{true, true, true, true});
}

GPU_TEST_CASE("cmd-draw-mesh-tasks-indirect-count", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::MeshShader))
        SKIP("mesh shaders not supported");

    // The count buffer holds 1, below the maximum of 2.
    CHECK(drawColumns(device, 2, true) == std::array<bool, kColumnCount>{true, false, false, false});
}
