#include "testing.h"
#include "test-ray-tracing-common.h"

#include <array>
#include <cstring>
#include <vector>

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

auto createRtSetLayout(IDevice* device) -> ComPtr<IBindingSetLayout>
{
    const BindingSetLayoutEntry entries[]{
        {BindingKind::RaytracingAccelerationStructure, 1, "tlas"},
        {BindingKind::RWTexture2D, 1, "output"},
    };
    const BindingSetLayoutDesc desc{.entries = entries, .entryCount = 2, .label = "scene"};
    ComPtr<IBindingSetLayout> layout;
    REQUIRE_CALL(device->createBindingSetLayout(desc, layout.writeRef()));
    REQUIRE(layout != nullptr);
    return layout;
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

auto createRtPipeline(IDevice* device, IShaderProgram* program, IPipelineLayout* layout)
    -> ComPtr<IRayTracingPipeline>
{
    HitGroupDesc hitGroup{};
    hitGroup.hitGroupName = "hitGroup";
    hitGroup.closestHitEntryPoint = "closestHitMain";
    RayTracingPipelineDesc desc{};
    desc.program = program;
    desc.layout = layout;
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

GPU_TEST_CASE("flat-binding-ray-tracing", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    auto queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue);
    TLAS tlas(device, queue, blas.blas);

    auto program = loadRtProgram(device);
    auto setLayout = createRtSetLayout(device);
    const PipelineLayoutSetDesc sets[]{{"scene", setLayout}};
    const PipelineLayoutDesc layoutDesc{program, sets, 1, sizeof(RtParams), "ray tracing"};
    ComPtr<IPipelineLayout> layout;
    REQUIRE_CALL(device->createPipelineLayout(layoutDesc, layout.writeRef()));
    CHECK(layout->getSetCount() == 1);
    CHECK(layout->getConstantsSize() == sizeof(RtParams));
    // `[[vk::push_constant]]` exists only on Vulkan; D3D12 reflects the same block as an ordinary
    // constant buffer, which the fork sources from a root CBV.
    const auto expectedProfile =
        device->getDeviceType() == DeviceType::Vulkan ? ConstantsProfile::Inline : ConstantsProfile::Buffered;
    CHECK(layout->getConstantsProfile() == expectedProfile);

    auto output = createRtOutputTexture(device);
    const BindingSetEntry entries[]{
        {.slot = 0, .resource = tlas.tlas},
        {.slot = 1, .resource = output->getDefaultView()},
    };
    const BindingSetDesc setDesc{.layout = setLayout, .entries = entries, .entryCount = 2, .label = "scene"};
    ComPtr<IBindingSet> set;
    REQUIRE_CALL(device->createBindingSet(setDesc, set.writeRef()));

    auto pipeline = createRtPipeline(device, program, layout);
    auto shaderTable = createRtShaderTable(device, program);

    const RtParams params{.dims = {kRtSize, kRtSize}, .hitValue = kRtHitValue, .missValue = kRtMissValue};
    IBindingSet* const boundSets[]{set};
    const FlatBindingDesc bindings{
        .constants = &params,
        .constantsSize = sizeof(params),
        .sets = boundSets,
        .setCount = 1,
    };

    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginRayTracingPass();
    REQUIRE_CALL(pass->bindPipeline(pipeline, shaderTable, bindings));
    pass->dispatchRays(0, kRtSize, kRtSize, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());

    compareComputeResult(device, output, 0, 0, rtExpectedTexels());
}

GPU_TEST_CASE("flat-binding-ray-tracing-multiple-raygen-table-retention", D3D12 | Vulkan)
{
    REQUIRE(device->hasFeature(Feature::RayTracing));
    bool releaseAfterSubmit = false;
    SUBCASE("release caller owners after recording") {}
    SUBCASE("release caller owners after submission") { releaseAfterSubmit = true; }

    auto queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue);
    TLAS tlas(device, queue, blas.blas);
    auto program = loadRtProgram(device, true);
    auto setLayout = createRtSetLayout(device);
    const PipelineLayoutSetDesc sets[]{{"scene", setLayout}};
    const PipelineLayoutDesc layoutDesc{program, sets, 1, sizeof(RtParams), "multiple raygen"};
    ComPtr<IPipelineLayout> layout;
    REQUIRE_CALL(device->createPipelineLayout(layoutDesc, layout.writeRef()));
    auto pipeline = createRtPipeline(device, program, layout);

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
            const BindingSetEntry entries[]{
                {.slot = 0, .resource = tlas.tlas},
                {.slot = 1, .resource = output->getDefaultView()},
            };
            const BindingSetDesc setDesc{.layout = setLayout, .entries = entries, .entryCount = 2, .label = "scene"};
            ComPtr<IBindingSet> set;
            REQUIRE_CALL(device->createBindingSet(setDesc, set.writeRef()));
            IBindingSet* const boundSets[]{set};
            const FlatBindingDesc bindings{
                .constants = &params,
                .constantsSize = sizeof(params),
                .sets = boundSets,
                .setCount = 1,
            };
            REQUIRE_CALL(pass->bindPipeline(pipeline, table, bindings));
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

    // Drop all caller-side program owners, including the pipeline layout, before waiting.
    pipeline.setNull();
    layout.setNull();
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

GPU_TEST_CASE("flat-binding-ray-tracing-layout-rejections", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    auto program = loadRtProgram(device);
    ExpectedErrorScope expectedErrors;
    {
        INFO("binding kinds in the wrong order");
        const BindingSetLayoutEntry entries[]{
            {BindingKind::RWTexture2D, 1, "output"},
            {BindingKind::RaytracingAccelerationStructure, 1, "tlas"},
        };
        const BindingSetLayoutDesc desc{.entries = entries, .entryCount = 2, .label = "swapped"};
        ComPtr<IBindingSetLayout> setLayout;
        REQUIRE_CALL(device->createBindingSetLayout(desc, setLayout.writeRef()));
        const PipelineLayoutSetDesc sets[]{{"scene", setLayout}};
        const PipelineLayoutDesc layoutDesc{program, sets, 1, sizeof(RtParams), "swapped"};
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(layoutDesc, layout.writeRef())));
    }
    {
        INFO("set layout missing the parameter block's last entry");
        const BindingSetLayoutEntry entries[]{{BindingKind::RaytracingAccelerationStructure, 1, "tlas"}};
        const BindingSetLayoutDesc desc{.entries = entries, .entryCount = 1, .label = "short"};
        ComPtr<IBindingSetLayout> setLayout;
        REQUIRE_CALL(device->createBindingSetLayout(desc, setLayout.writeRef()));
        const PipelineLayoutSetDesc sets[]{{"scene", setLayout}};
        const PipelineLayoutDesc layoutDesc{program, sets, 1, sizeof(RtParams), "short"};
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(layoutDesc, layout.writeRef())));
    }
    {
        INFO("set path the program does not have");
        auto setLayout = createRtSetLayout(device);
        const PipelineLayoutSetDesc sets[]{{"missing", setLayout}};
        const PipelineLayoutDesc layoutDesc{program, sets, 1, sizeof(RtParams), "missing path"};
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(layoutDesc, layout.writeRef())));
    }
    {
        INFO("layout omits the execution constants the program declares");
        auto setLayout = createRtSetLayout(device);
        const PipelineLayoutSetDesc sets[]{{"scene", setLayout}};
        const PipelineLayoutDesc layoutDesc{program, sets, 1, 0, "no constants"};
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(layoutDesc, layout.writeRef())));
    }
}
