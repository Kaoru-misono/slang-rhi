#include "testing.h"
#include "test-ray-tracing-common.h"

#include <array>
#include <cstring>
#include <vector>
#include <memory>

using namespace rhi;
using namespace rhi::testing;

namespace {

constexpr uint32_t kRtSize = 4;
constexpr float kRtHitValue = 0.75f;
constexpr float kRtMissValue = 0.25f;

struct alignas(16) RtParams
{
    uint32_t dims[2];
    float hitValue;
    float missValue;
};
static_assert(sizeof(RtParams) == 16);

auto loadRtProgram(IDevice* device, bool multipleRaygen = false) -> ComPtr<IShaderProgram>
{
    std::vector<const char*> entryPoints{"rayGenMain", "missMain", "closestHitMain"};
    if (multipleRaygen)
        entryPoints.emplace_back("rayGenAlternate");
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(
        device,
        "test-flat-binding-rt",
        entryPoints,
        program.writeRef()
    ));
    REQUIRE(program != nullptr);
    return program;
}

auto createRtOutputTexture(IDevice* device) -> ComPtr<ITexture>
{
    TextureDesc desc{};
    desc.type = TextureType::Texture2D;
    desc.size = {kRtSize, kRtSize, 1};
    desc.mipCount = 1;
    desc.format = Format::RGBA32Float;
    desc.usage = TextureUsage::UnorderedAccess | TextureUsage::CopySource;
    desc.defaultState = ResourceState::UnorderedAccess;
    ComPtr<ITexture> texture;
    REQUIRE_CALL(device->createTexture(desc, nullptr, texture.writeRef()));
    REQUIRE(texture != nullptr);
    return texture;
}

auto createRtShaderTable(IDevice* device, IShaderProgram* program, bool multipleRaygen = false, bool reverse = false)
    -> ComPtr<IShaderTable>
{
    const char* rayGenNames[]{reverse ? "rayGenAlternate" : "rayGenMain", reverse ? "rayGenMain" : "rayGenAlternate"};
    const char* missNames[]{"missMain"};
    const char* hitGroupNames[]{"hitGroup"};
    ShaderTableDesc desc{};
    desc.program = program;
    desc.rayGenShaderCount = multipleRaygen ? 2 : 1;
    desc.rayGenShaderEntryPointNames = rayGenNames;
    desc.missShaderCount = 1;
    desc.missShaderEntryPointNames = missNames;
    desc.hitGroupCount = 1;
    desc.hitGroupNames = hitGroupNames;
    ComPtr<IShaderTable> shaderTable;
    REQUIRE_CALL(device->createShaderTable(desc, shaderTable.writeRef()));
    REQUIRE(shaderTable != nullptr);
    return shaderTable;
}

auto createRtPipeline(IDevice* device, IShaderProgram* program)
    -> ComPtr<IRayTracingPipeline>
{
    HitGroupDesc hitGroup{};
    hitGroup.hitGroupName = "hitGroup";
    hitGroup.closestHitEntryPoint = "closestHitMain";
    RayTracingPipelineDesc desc{};
    desc.program = program;
    desc.hitGroupCount = 1;
    desc.hitGroups = &hitGroup;
    desc.maxRayPayloadSize = 16;
    desc.maxAttributeSizeInBytes = 8;
    desc.maxRecursion = 1;
    ComPtr<IRayTracingPipeline> pipeline;
    REQUIRE_CALL(device->createRayTracingPipeline(desc, pipeline.writeRef()));
    REQUIRE(pipeline != nullptr);
    return pipeline;
}

/// The triangle spans x + y <= 1 in the z = 1 plane, so a ray at (0.05 + 0.25 * x, 0.05 + 0.25 * y)
/// hits exactly when the two indices sum to at most three.
auto rtExpectedTexels() -> std::array<float, kRtSize * kRtSize * 4>
{
    std::array<float, kRtSize * kRtSize * 4> expected{};
    for (uint32_t y = 0; y < kRtSize; ++y)
    {
        for (uint32_t x = 0; x < kRtSize; ++x)
        {
            const float value = x + y <= 3 ? kRtHitValue : kRtMissValue;
            float* texel = expected.data() + (y * kRtSize + x) * 4;
            texel[0] = texel[1] = texel[2] = value;
            texel[3] = 1.f;
        }
    }
    return expected;
}

} // namespace

GPU_TEST_CASE("shader-object-recorded-ray-tracing-snapshots", D3D12 | Vulkan)
{
    bool inlineConstants = false;
    bool sparseSets = false;
    SUBCASE("buffered constants") {}
    SUBCASE("inline constants and parameter block") { inlineConstants = true; }
    SUBCASE("buffered constants and parameter block in set 3") { sparseSets = true; }
    REQUIRE(device->hasFeature(Feature::RayTracing));
    auto queue = device->getQueue(QueueType::Graphics);
    auto blas = std::make_unique<SingleTriangleBLAS>(device, queue);
    auto tlas = std::make_unique<TLAS>(device, queue, blas->blas, nullptr, true);
    ComPtr<IShaderProgram> program;
    const char* moduleName = sparseSets ? "test-shader-object-sparse-rt"
        : inlineConstants ? "test-flat-binding-rt" : "test-shader-object-retained-rt";
    REQUIRE_CALL(loadAndLinkProgram(device, moduleName,
        {"rayGenMain", "rayGenAlternate", "missMain", "closestHitMain"}, program.writeRef()));
    auto pipeline = createRtPipeline(device, program);
    auto root = device->createRootShaderObject(program);
    REQUIRE(root != nullptr);
    std::array<ComPtr<ITexture>, 4> outputs;
    std::array<ComPtr<ICommandEncoder>, 2> encoders;
    for (uint32_t batch = 0; batch < 2; ++batch)
    {
        // The same raygen index selects a different shader after table reordering.
        auto table = createRtShaderTable(device, program, true, batch == 1);
        encoders[batch] = queue->createCommandEncoder();
        auto* pass = encoders[batch]->beginRayTracingPass();
        pass->bindPipeline(pipeline, table, root);
        for (uint32_t draw = 0; draw < 2; ++draw)
        {
            const uint32_t index = batch * 2 + draw;
            outputs[index] = createRtOutputTexture(device);
            const RtParams params{{kRtSize, kRtSize}, 0.75f + float(index), 0.25f + float(index)};
            ShaderCursor cursor(root);
            REQUIRE_CALL(cursor["params"].getDereferenced().setData(&params, sizeof(params)));
            REQUIRE_CALL(cursor["scene"]["tlas"].setBinding(tlas->tlas));
            REQUIRE_CALL(cursor["scene"]["output"].setBinding(outputs[index]->getDefaultView()));
            pass->dispatchRays(draw, kRtSize, kRtSize, 1);
        }
        pass->end();
        table.setNull();
    }
    auto commandsB = encoders[1]->finish();
    REQUIRE(commandsB != nullptr);
    REQUIRE_CALL(queue->submit(commandsB));
    REQUIRE_CALL(queue->waitOnHost());
    encoders[1].setNull();
    commandsB.setNull();
    root.setNull();
    pipeline.setNull();
    program.setNull();
    blas.reset();
    tlas.reset();
    // Only recorded native dependencies now own the old TLAS/BLAS, pipeline and table.
    auto commandsA = encoders[0]->finish();
    REQUIRE(commandsA != nullptr);
    encoders[0].setNull();
    REQUIRE_CALL(queue->submit(commandsA));
    REQUIRE_CALL(queue->waitOnHost());
    for (uint32_t index = 0; index < outputs.size(); ++index)
    {
        auto expected = rtExpectedTexels();
        const bool alternate = ((index % 2) == 1) != ((index / 2) == 1);
        for (uint32_t pixel = 0; pixel < kRtSize * kRtSize; ++pixel)
            for (uint32_t c = 0; c < 3; ++c)
                expected[pixel * 4 + c] += float(index) + (alternate ? 0.125f : 0.f);
        std::array<uint32_t, kRtSize * kRtSize * 4> expectedBits{};
        std::memcpy(expectedBits.data(), expected.data(), sizeof(expected));
        compareComputeResult(device, outputs[index], 0, 0, expectedBits);
    }
}

GPU_TEST_CASE("shader-object-ray-tracing", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    auto queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue);
    TLAS tlas(device, queue, blas.blas);

    auto program = loadRtProgram(device);
    auto output = createRtOutputTexture(device);

    auto pipeline = createRtPipeline(device, program);
    auto shaderTable = createRtShaderTable(device, program);

    const RtParams params{.dims = {kRtSize, kRtSize}, .hitValue = kRtHitValue, .missValue = kRtMissValue};
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginRayTracingPass();
    auto root = pass->bindPipeline(pipeline, shaderTable);
    REQUIRE(root != nullptr);
    REQUIRE_CALL(ShaderCursor(root)["params"].getDereferenced().setData(&params, sizeof(params)));
    REQUIRE_CALL(ShaderCursor(root)["scene"]["tlas"].setBinding(tlas.tlas));
    REQUIRE_CALL(ShaderCursor(root)["scene"]["output"].setBinding(output));
    pass->dispatchRays(0, kRtSize, kRtSize, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());

    compareComputeResult(device, output, 0, 0, rtExpectedTexels());
}

GPU_TEST_CASE("shader-object-ray-tracing-multiple-raygen-table-retention", D3D12 | Vulkan)
{
    REQUIRE(device->hasFeature(Feature::RayTracing));
    bool releaseAfterSubmit = false;
    SUBCASE("release caller owners after recording") {}
    SUBCASE("release caller owners after submission") { releaseAfterSubmit = true; }

    auto queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue);
    TLAS tlas(device, queue, blas.blas);
    auto program = loadRtProgram(device, true);
    auto pipeline = createRtPipeline(device, program);

    constexpr std::array<uint32_t, 3> indices{0, 1, 0};
    std::array<ComPtr<ITexture>, 6> outputs;
    std::array<ComPtr<ICommandBuffer>, 2> commands;
    const RtParams params{.dims = {kRtSize, kRtSize}, .hitValue = kRtHitValue, .missValue = kRtMissValue};
    for (uint32_t tableIndex = 0; tableIndex < 2; ++tableIndex)
    {
        // B reverses the record order: index zero must now execute the alternate raygen.
        auto table = createRtShaderTable(device, program, true, tableIndex == 1);
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginRayTracingPass();
        for (uint32_t dispatch = 0; dispatch < indices.size(); ++dispatch)
        {
            auto& output = outputs[tableIndex * indices.size() + dispatch];
            output = createRtOutputTexture(device);
            auto root = pass->bindPipeline(pipeline, table);
            REQUIRE(root != nullptr);
            REQUIRE_CALL(ShaderCursor(root)["params"].getDereferenced().setData(&params, sizeof(params)));
            REQUIRE_CALL(ShaderCursor(root)["scene"]["tlas"].setBinding(tlas.tlas));
            REQUIRE_CALL(ShaderCursor(root)["scene"]["output"].setBinding(output));
            pass->dispatchRays(indices[dispatch], kRtSize, kRtSize, 1);
        }
        pass->end();
        commands[tableIndex] = encoder->finish();
        REQUIRE(commands[tableIndex] != nullptr);
        if (releaseAfterSubmit)
        {
            REQUIRE_CALL(queue->submit(commands[tableIndex]));
            commands[tableIndex].setNull();
        }
        table.setNull();
    }

    // Drop all caller-side program owners, before waiting.
    pipeline.setNull();
    program.setNull();
    if (!releaseAfterSubmit)
    {
        for (auto& command : commands)
        {
            REQUIRE_CALL(queue->submit(command));
            command.setNull();
        }
    }
    // Submission may already have completed; this does not assert a deterministic in-flight window.
    REQUIRE_CALL(queue->waitOnHost());

    for (uint32_t tableIndex = 0; tableIndex < 2; ++tableIndex)
    {
        for (uint32_t dispatch = 0; dispatch < indices.size(); ++dispatch)
        {
            CAPTURE(tableIndex);
            CAPTURE(dispatch);
            auto expected = rtExpectedTexels();
            const bool alternate = (indices[dispatch] == 1) != (tableIndex == 1);
            if (alternate)
            {
                for (size_t pixel = 0; pixel < kRtSize * kRtSize; ++pixel)
                {
                    expected[pixel * 4] += 0.125f;
                    expected[pixel * 4 + 1] += 0.125f;
                    expected[pixel * 4 + 2] += 0.125f;
                }
            }
            ComPtr<ISlangBlob> bytes;
            SubresourceLayout readback{};
            REQUIRE_CALL(device->readTexture(
                outputs[tableIndex * indices.size() + dispatch], 0, 0, bytes.writeRef(), &readback
            ));
            REQUIRE(readback.size.width == kRtSize);
            REQUIRE(readback.size.height == kRtSize);
            REQUIRE(readback.size.depth == 1);
            REQUIRE(readback.colPitch == 4 * sizeof(float));
            constexpr size_t rowBytes = kRtSize * 4 * sizeof(float);
            REQUIRE(readback.rowPitch >= rowBytes);
            REQUIRE(bytes->getBufferSize() >= (kRtSize - 1) * readback.rowPitch + rowBytes);
            for (uint32_t row = 0; row < kRtSize; ++row)
            {
                auto const* actual = static_cast<const uint8_t*>(bytes->getBufferPointer()) + row * readback.rowPitch;
                CHECK(std::memcmp(actual, expected.data() + row * kRtSize * 4, rowBytes) == 0);
            }
        }
    }
}
