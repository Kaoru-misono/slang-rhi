#include "testing.h"
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
struct alignas(16) Params
{
    float direction[3];
    float scale;
    float samples[2][4];
};
static_assert(sizeof(Params) == 48 && alignof(Params) == 16);

constexpr const char* sceneSource = R"(
struct Params
{
    float3 direction;
    float scale;
    float4 samples[2];
};
struct Scene
{
    Texture2D albedo;
    SamplerState sampler;
    StructuredBuffer<float4> items;
    RWStructuredBuffer<float4> output;
};
ParameterBlock<Scene> scene;
ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    scene.output[id.x] = scene.albedo.SampleLevel(scene.sampler, float2(0, 0), 0)
        + scene.items[id.x] + float4(params.direction * params.scale, 0) + params.samples[0];
}
)";

constexpr const char* ordinaryDataSource = R"(
struct Params
{
    float3 direction;
    float scale;
    float4 samples[2];
};
struct Scene
{
    float4 tint;
    Texture2D albedo;
    SamplerState sampler;
    StructuredBuffer<float4> items;
    RWStructuredBuffer<float4> output;
};
ParameterBlock<Scene> scene;
ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    scene.output[id.x] = scene.albedo.SampleLevel(scene.sampler, float2(0, 0), 0)
        + scene.items[id.x] + float4(params.direction * params.scale, 0) + params.samples[0];
}
)";

constexpr const char* looseResourceSource = R"(
struct Params
{
    float3 direction;
    float scale;
    float4 samples[2];
};
struct Scene
{
    SamplerState sampler;
    StructuredBuffer<float4> items;
    RWStructuredBuffer<float4> output;
};
Texture2D albedo;
ParameterBlock<Scene> scene;
ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    scene.output[id.x] = albedo.SampleLevel(scene.sampler, float2(0, 0), 0)
        + scene.items[id.x] + float4(params.direction * params.scale, 0) + params.samples[0];
}
)";

constexpr const char* noConstantsSource = R"(
struct Params
{
    float3 direction;
    float scale;
    float4 samples[2];
};
struct Scene
{
    Texture2D albedo;
    SamplerState sampler;
    StructuredBuffer<float4> items;
    RWStructuredBuffer<float4> output;
};
ParameterBlock<Scene> scene;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    scene.output[id.x] = scene.albedo.SampleLevel(scene.sampler, float2(0, 0), 0)
        + scene.items[id.x];
}
)";

auto createSceneSetLayout(IDevice* device, BindingKind albedoKind = BindingKind::Texture2D)
    -> ComPtr<IBindingSetLayout>
{
    const BindingSetLayoutEntry entries[]{
        {albedoKind, 1, "albedo"},
        {BindingKind::SamplerState, 1, "sampler"},
        {BindingKind::StructuredBuffer, 1, "items"},
        {BindingKind::RWStructuredBuffer, 1, "output"},
    };
    BindingSetLayoutDesc desc{};
    desc.entries = entries;
    desc.entryCount = 4;
    desc.label = "scene";
    ComPtr<IBindingSetLayout> layout;
    REQUIRE_CALL(device->createBindingSetLayout(desc, layout.writeRef()));
    REQUIRE(layout != nullptr);
    return layout;
}
} // namespace

GPU_TEST_CASE("flat-binding-layout-roundtrip", D3D12 | Vulkan)
{
    std::string names[]{"albedo", "sampler", "buffers"};
    std::string label = "roundtrip";
    BindingSetLayoutEntry entries[]{
        {BindingKind::Texture2D, 1, names[0].c_str()},
        {BindingKind::SamplerState, 1, names[1].c_str()},
        {BindingKind::RWStructuredBuffer, 4, names[2].c_str()},
    };
    BindingSetLayoutDesc desc{};
    desc.entries = entries;
    desc.entryCount = 3;
    desc.label = label.c_str();
    ComPtr<IBindingSetLayout> layout;
    REQUIRE_CALL(device->createBindingSetLayout(desc, layout.writeRef()));
    REQUIRE(layout != nullptr);

    for (auto& entry : entries)
        entry = {};
    for (auto& name : names)
        name.clear();
    label.clear();

    const auto& stored = layout->getDesc();
    REQUIRE(stored.entryCount == 3);
    REQUIRE(stored.entries != nullptr);
    CHECK(stored.entries[0].kind == BindingKind::Texture2D);
    CHECK(stored.entries[0].count == 1);
    REQUIRE(stored.entries[0].name != nullptr);
    CHECK(std::string_view(stored.entries[0].name) == "albedo");
    CHECK(stored.entries[1].kind == BindingKind::SamplerState);
    CHECK(stored.entries[1].count == 1);
    REQUIRE(stored.entries[1].name != nullptr);
    CHECK(std::string_view(stored.entries[1].name) == "sampler");
    CHECK(stored.entries[2].kind == BindingKind::RWStructuredBuffer);
    CHECK(stored.entries[2].count == 4);
    REQUIRE(stored.entries[2].name != nullptr);
    CHECK(std::string_view(stored.entries[2].name) == "buffers");
    REQUIRE(stored.label != nullptr);
    CHECK(std::string_view(stored.label) == "roundtrip");
}

GPU_TEST_CASE("flat-binding-layout-rejections", D3D12 | Vulkan)
{
    ExpectedErrorScope expectedErrors;
    BindingSetLayoutEntry entry{BindingKind::Texture2D, 1, "albedo"};
    BindingSetLayoutDesc desc{};
    desc.entries = &entry;
    desc.entryCount = 1;
    {
        INFO("zero entry count");
        auto invalid = desc;
        invalid.entryCount = 0;
        ComPtr<IBindingSetLayout> layout;
        CHECK(SLANG_FAILED(device->createBindingSetLayout(invalid, layout.writeRef())));
    }
    {
        INFO("null entries");
        auto invalid = desc;
        invalid.entries = nullptr;
        ComPtr<IBindingSetLayout> layout;
        CHECK(SLANG_FAILED(device->createBindingSetLayout(invalid, layout.writeRef())));
    }
    {
        INFO("zero binding count");
        auto invalidEntry = entry;
        invalidEntry.count = 0;
        auto invalid = desc;
        invalid.entries = &invalidEntry;
        ComPtr<IBindingSetLayout> layout;
        CHECK(SLANG_FAILED(device->createBindingSetLayout(invalid, layout.writeRef())));
    }
    {
        INFO("null output layout");
        CHECK(SLANG_FAILED(device->createBindingSetLayout(desc, nullptr)));
    }
}

GPU_TEST_CASE("flat-binding-pipeline-layout-accepted", D3D12 | Vulkan)
{
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, sceneSource, program.writeRef()));
    auto sceneLayout = createSceneSetLayout(device);
    const PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
    PipelineLayoutDesc desc{};
    desc.program = program;
    desc.sets = sets;
    desc.setCount = 1;
    desc.constantsSize = sizeof(Params);
    ComPtr<IPipelineLayout> layout;
    REQUIRE_CALL(device->createPipelineLayout(desc, layout.writeRef()));
    REQUIRE(layout != nullptr);
    CHECK(layout->getSetCount() == 1);
    CHECK(layout->getConstantsProfile() == ConstantsProfile::Buffered);
    CHECK(layout->getConstantsSize() == sizeof(Params));
    const auto& stored = layout->getDesc();
    REQUIRE(stored.setCount == 1);
    REQUIRE(stored.sets != nullptr);
    CHECK(stored.sets[0].layout == sceneLayout.get());
    CHECK(stored.constantsSize == sizeof(Params));
}

GPU_TEST_CASE("flat-binding-pipeline-layout-rejections", D3D12 | Vulkan)
{
    ExpectedErrorScope expectedErrors;
    {
        INFO("constants size mismatch");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, sceneSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constantsSize = sizeof(Params) + 16;
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("constants absent from program");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, noConstantsSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constantsSize = sizeof(Params);
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("constants absent from layout");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, sceneSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constantsSize = 0;
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("ordinary data inside a parameter block");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, ordinaryDataSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constantsSize = sizeof(Params);
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("loose global resource");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, looseResourceSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constantsSize = sizeof(Params);
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("missing set path");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, sceneSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constantsSize = sizeof(Params);
        desc.setCount = 0;
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("extra set path");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, sceneSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constantsSize = sizeof(Params);
        auto otherLayout = createSceneSetLayout(device);
        const PipelineLayoutSetDesc extraSets[]{{"scene", sceneLayout}, {"missing", otherLayout}};
        desc.sets = extraSets;
        desc.setCount = 2;
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("kind mismatch");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, sceneSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device, BindingKind::RWTexture2D);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constantsSize = sizeof(Params);
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("entry count mismatch");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, sceneSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constantsSize = sizeof(Params);
        const BindingSetLayoutEntry entries[]{
            {BindingKind::Texture2D, 1, "albedo"},
            {BindingKind::SamplerState, 1, "sampler"},
            {BindingKind::StructuredBuffer, 1, "items"},
        };
        BindingSetLayoutDesc setDesc{};
        setDesc.entries = entries;
        setDesc.entryCount = 3;
        setDesc.label = "scene";
        ComPtr<IBindingSetLayout> shortLayout;
        REQUIRE_CALL(device->createBindingSetLayout(setDesc, shortLayout.writeRef()));
        REQUIRE(shortLayout != nullptr);
        sets[0].layout = shortLayout;
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
}

GPU_TEST_CASE("flat-binding-set-rejections", D3D12 | Vulkan)
{
    auto sceneLayout = createSceneSetLayout(device);
    TextureDesc textureDesc{};
    textureDesc.type = TextureType::Texture2D;
    textureDesc.size = {4, 4, 1};
    textureDesc.format = Format::RGBA32Float;
    textureDesc.usage = TextureUsage::ShaderResource;
    textureDesc.defaultState = ResourceState::ShaderResource;
    auto texture = device->createTexture(textureDesc);
    REQUIRE(texture != nullptr);
    auto view = texture->getDefaultView();
    REQUIRE(view != nullptr);
    auto sampler = device->createSampler({});
    REQUIRE(sampler != nullptr);

    BufferDesc bufferDesc{};
    bufferDesc.size = 64;
    bufferDesc.elementSize = 16;
    bufferDesc.usage = BufferUsage::ShaderResource;
    bufferDesc.defaultState = ResourceState::ShaderResource;
    auto items = device->createBuffer(bufferDesc);
    REQUIRE(items != nullptr);
    bufferDesc.usage = BufferUsage::UnorderedAccess;
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    auto output = device->createBuffer(bufferDesc);
    REQUIRE(output != nullptr);

    const std::array<BindingSetEntry, 4> entries{{
        {.slot = 0, .resource = view},
        {.slot = 1, .resource = sampler},
        {.slot = 2, .resource = items},
        {.slot = 3, .resource = output},
    }};
    BindingSetDesc desc{};
    desc.layout = sceneLayout;
    desc.entries = entries.data();
    desc.entryCount = 4;
    ComPtr<IBindingSet> bindingSet;
    REQUIRE_CALL(device->createBindingSet(desc, bindingSet.writeRef()));
    REQUIRE(bindingSet != nullptr);
    CHECK(bindingSet->getLayout() == sceneLayout.get());

    ExpectedErrorScope expectedErrors;
    {
        INFO("wrong kind");
        auto invalidEntries = entries;
        auto invalid = desc;
        invalid.entries = invalidEntries.data();
        invalidEntries[0].resource = items;
        ComPtr<IBindingSet> rejectedSet;
        CHECK(SLANG_FAILED(device->createBindingSet(invalid, rejectedSet.writeRef())));
    }
    {
        INFO("out-of-range slot");
        auto invalidEntries = entries;
        auto invalid = desc;
        invalid.entries = invalidEntries.data();
        invalidEntries[0].slot = 4;
        ComPtr<IBindingSet> rejectedSet;
        CHECK(SLANG_FAILED(device->createBindingSet(invalid, rejectedSet.writeRef())));
    }
    {
        INFO("out-of-range array index");
        auto invalidEntries = entries;
        auto invalid = desc;
        invalid.entries = invalidEntries.data();
        invalidEntries[0].arrayIndex = 1;
        ComPtr<IBindingSet> rejectedSet;
        CHECK(SLANG_FAILED(device->createBindingSet(invalid, rejectedSet.writeRef())));
    }
    {
        INFO("missing slot");
        auto invalidEntries = entries;
        auto invalid = desc;
        invalid.entries = invalidEntries.data();
        invalid.entryCount = 3;
        ComPtr<IBindingSet> rejectedSet;
        CHECK(SLANG_FAILED(device->createBindingSet(invalid, rejectedSet.writeRef())));
    }
    {
        INFO("duplicate slot");
        auto invalidEntries = entries;
        auto invalid = desc;
        invalid.entries = invalidEntries.data();
        invalidEntries[3] = invalidEntries[0];
        ComPtr<IBindingSet> rejectedSet;
        CHECK(SLANG_FAILED(device->createBindingSet(invalid, rejectedSet.writeRef())));
    }
    {
        INFO("no resource in entry");
        auto invalidEntries = entries;
        auto invalid = desc;
        invalid.entries = invalidEntries.data();
        invalidEntries[0].resource = nullptr;
        ComPtr<IBindingSet> rejectedSet;
        CHECK(SLANG_FAILED(device->createBindingSet(invalid, rejectedSet.writeRef())));
    }
    {
        INFO("buffer range on a non-buffer entry");
        auto invalidEntries = entries;
        auto invalid = desc;
        invalid.entries = invalidEntries.data();
        invalidEntries[0].bufferRange = {0, 16};
        ComPtr<IBindingSet> rejectedSet;
        CHECK(SLANG_FAILED(device->createBindingSet(invalid, rejectedSet.writeRef())));
    }
}

namespace {
struct alignas(16) FlatConstants
{
    uint32_t offset;
    uint32_t multiplier;
    uint32_t bias;
    uint32_t tag;
};

struct alignas(16) FlatWideConstants
{
    uint32_t offset;
    uint32_t multiplier;
    uint32_t bias;
    uint32_t tag;
    uint32_t extra[4];
};

static_assert(std::is_trivially_copyable_v<FlatConstants> && std::is_standard_layout_v<FlatConstants>);
static_assert(sizeof(FlatConstants) == 16 && alignof(FlatConstants) == 16);
static_assert(offsetof(FlatConstants, offset) == 0 && offsetof(FlatConstants, multiplier) == 4);
static_assert(offsetof(FlatConstants, bias) == 8 && offsetof(FlatConstants, tag) == 12);
static_assert(std::is_trivially_copyable_v<FlatWideConstants> && std::is_standard_layout_v<FlatWideConstants>);
static_assert(sizeof(FlatWideConstants) == 32 && alignof(FlatWideConstants) == 16);
static_assert(offsetof(FlatWideConstants, offset) == 0 && offsetof(FlatWideConstants, multiplier) == 4);
static_assert(offsetof(FlatWideConstants, bias) == 8 && offsetof(FlatWideConstants, tag) == 12);
static_assert(offsetof(FlatWideConstants, extra) == 16 && sizeof(FlatWideConstants::extra) == 16);

struct alignas(16) FlatBindlessConstants
{
    uint32_t offset;
    uint32_t multiplier;
    uint32_t bias;
    uint32_t tag;
    uint32_t handle[4];
};

static_assert(std::is_trivially_copyable_v<FlatBindlessConstants> && std::is_standard_layout_v<FlatBindlessConstants>);
static_assert(sizeof(FlatBindlessConstants) == 32 && alignof(FlatBindlessConstants) == 16);
static_assert(offsetof(FlatBindlessConstants, offset) == 0 && offsetof(FlatBindlessConstants, multiplier) == 4);
static_assert(offsetof(FlatBindlessConstants, bias) == 8 && offsetof(FlatBindlessConstants, tag) == 12);
static_assert(offsetof(FlatBindlessConstants, handle) == 16 && sizeof(FlatBindlessConstants::handle) == 16);

struct FlatVariant
{
    bool inlineConstants = false;
    bool wideConstants = false;
    bool bindlessHandle = false;
    uint32_t samplerCount = 0;
};

auto createFlatSession(IDevice* device, const FlatVariant& variant) -> ComPtr<slang::ISession>
{
    auto deviceSession = device->getSlangSession();
    auto* globalSession = deviceSession->getGlobalSession();
    slang::TargetDesc target{};
    target.format = device->getDeviceType() == DeviceType::Vulkan ? SLANG_SPIRV : SLANG_DXIL;
    if (target.format == SLANG_DXIL)
    {
        // ResourceDescriptorHeap, which a descriptor handle lowers to, needs SM 6.6.
        target.profile = globalSession->findProfile(variant.bindlessHandle ? "sm_6_6" : "sm_6_0");
    }
    target.forceGLSLScalarBufferLayout = true;
    slang::CompilerOptionEntry options[2]{};
    options[0].name = slang::CompilerOptionName::EmitSpirvDirectly;
    options[0].value.kind = slang::CompilerOptionValueKind::Int;
    options[0].value.intValue0 = 1;
    options[1].name = slang::CompilerOptionName::MatrixLayoutRow;
    options[1].value.kind = slang::CompilerOptionValueKind::Int;
    options[1].value.intValue0 = 1;
    const std::string samplerCount = std::to_string(variant.samplerCount);
    const slang::PreprocessorMacroDesc macros[]{
        {"INLINE_CONSTANTS", variant.inlineConstants ? "1" : "0"},
        {"WIDE_CONSTANTS", variant.wideConstants ? "1" : "0"},
        {"BINDLESS_HANDLE", variant.bindlessHandle ? "1" : "0"},
        {"SAMPLER_COUNT", samplerCount.c_str()},
    };
    auto searchPaths = getSlangSearchPaths();
    slang::SessionDesc desc{};
    desc.targets = &target;
    desc.targetCount = 1;
    desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    desc.searchPaths = searchPaths.data();
    desc.searchPathCount = static_cast<SlangInt>(searchPaths.size());
    desc.preprocessorMacros = macros;
    desc.preprocessorMacroCount = 4;
    desc.compilerOptionEntries = options;
    desc.compilerOptionEntryCount = 2;
    ComPtr<slang::ISession> session;
    REQUIRE_CALL(globalSession->createSession(desc, session.writeRef()));
    return session;
}

struct FlatScene
{
    ComPtr<IBuffer> input;
    ComPtr<IBuffer> output;
    ComPtr<ITexture> texture;
    ComPtr<ITextureView> textureView;
    ComPtr<ISampler> sampler;
    ComPtr<IBindingSetLayout> setLayout;
    ComPtr<IBindingSet> set;
};

constexpr uint32_t kFlatSampledValue = 3;

auto createFlatSceneSetLayout(IDevice* device) -> ComPtr<IBindingSetLayout>
{
    const BindingSetLayoutEntry entries[]{
        {BindingKind::StructuredBuffer, 1, "input"},
        {BindingKind::RWStructuredBuffer, 1, "output"},
        {BindingKind::Texture2D, 1, "texture"},
        {BindingKind::SamplerState, 1, "sampler"},
    };
    const BindingSetLayoutDesc desc{.entries = entries, .entryCount = 4, .label = "scene"};
    ComPtr<IBindingSetLayout> layout;
    REQUIRE_CALL(device->createBindingSetLayout(desc, layout.writeRef()));
    REQUIRE(layout != nullptr);
    return layout;
}

auto createFlatSceneResources(IDevice* device, std::span<const uint32_t> input, uint32_t outputCount, FlatScene& scene)
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
    const std::array<float, 4> texel{static_cast<float>(kFlatSampledValue), 0.f, 0.f, 0.f};
    const SubresourceData data{.data = texel.data(), .rowPitch = 16, .slicePitch = 16};
    REQUIRE_CALL(device->createTexture(textureDesc, &data, scene.texture.writeRef()));
    scene.textureView = scene.texture->getDefaultView();
    REQUIRE(scene.textureView != nullptr);
    scene.sampler = device->createSampler({});
    REQUIRE(scene.sampler != nullptr);
}

auto createFlatSceneSet(
    IDevice* device,
    const FlatScene& scene,
    IBuffer* input,
    IBuffer* output,
    const char* label
) -> ComPtr<IBindingSet>
{
    // Pipeline-layout validation must first record the structured-buffer strides checked at set creation.
    const BindingSetEntry entries[]{
        {.slot = 0, .resource = input},
        {.slot = 1, .resource = output},
        {.slot = 2, .resource = scene.textureView},
        {.slot = 3, .resource = scene.sampler},
    };
    const BindingSetDesc desc{
        .layout = scene.setLayout,
        .entries = entries,
        .entryCount = 4,
        .label = label,
    };
    ComPtr<IBindingSet> set;
    REQUIRE_CALL(device->createBindingSet(desc, set.writeRef()));
    REQUIRE(set != nullptr);
    return set;
}

auto createFlatSceneBindingSet(IDevice* device, FlatScene& scene) -> void
{
    scene.set = createFlatSceneSet(device, scene, scene.input, scene.output, "scene");
}

auto createFlatPipelineLayout(
    IDevice* device,
    IShaderProgram* program,
    IBindingSetLayout* setLayout,
    size_t constantsSize,
    const char* label
) -> ComPtr<IPipelineLayout>
{
    const PipelineLayoutSetDesc sets[]{{"scene", setLayout}};
    const PipelineLayoutDesc desc{program, sets, 1, constantsSize, label};
    ComPtr<IPipelineLayout> layout;
    REQUIRE_CALL(device->createPipelineLayout(desc, layout.writeRef()));
    REQUIRE(layout != nullptr);
    return layout;
}

auto flatExpected(uint32_t previous, uint32_t input, const FlatConstants& constants) -> uint32_t
{
    return previous * constants.tag + input * constants.multiplier + constants.bias + kFlatSampledValue;
}

auto flatBindings(const void* constants, size_t size, std::span<IBindingSet* const> sets) -> FlatBindingDesc
{
    return {
        .constants = constants,
        .constantsSize = size,
        .sets = sets.data(),
        .setCount = static_cast<uint32_t>(sets.size()),
    };
}

/// The only implementable override beyond IUnknown is getLayout(), so a binding set cannot be
/// mutated after creation. This stops compiling the day a setter is added to the interface.
struct FlatImmutableBindingSet final : IBindingSet
{
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID&, void** outObject) override
    {
        *outObject = nullptr;
        return SLANG_E_NO_INTERFACE;
    }

    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 1; }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 1; }
    virtual SLANG_NO_THROW IBindingSetLayout* SLANG_MCALL getLayout() override { return nullptr; }
};

static_assert(!std::is_abstract_v<FlatImmutableBindingSet>);

/// Resources created with initial data leave an upload in flight that owns the device, so a case
/// that aborts before reaching its own submit would keep the device alive past the end of the run.
struct FlatDeviceDrain
{
    IDevice* device;

    ~FlatDeviceDrain() { device->getQueue(QueueType::Graphics)->waitOnHost(); }
};

/// Devices these cases create themselves are not wired to the suite's debug callback, so they would
/// fail with no explanation at all.
struct FlatMessageSink final : IDebugCallback
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

auto createFlatSamplerSetLayout(IDevice* device, uint32_t samplerCount) -> ComPtr<IBindingSetLayout>
{
    const BindingSetLayoutEntry entries[]{
        {BindingKind::Texture2D, 1, "texture"},
        {BindingKind::SamplerState, samplerCount, "samplers"},
        {BindingKind::RWStructuredBuffer, 1, "output"},
    };
    const BindingSetLayoutDesc desc{.entries = entries, .entryCount = 3, .label = "samplers"};
    ComPtr<IBindingSetLayout> layout;
    REQUIRE_CALL(device->createBindingSetLayout(desc, layout.writeRef()));
    REQUIRE(layout != nullptr);
    return layout;
}

struct FlatSamplerScene
{
    ComPtr<ITexture> texture;
    ComPtr<ITextureView> textureView;
    ComPtr<IBuffer> output;
    std::vector<ComPtr<ISampler>> samplers;
    ComPtr<IBindingSet> set;
};

auto createFlatSamplerScene(IDevice* device, IBindingSetLayout* setLayout, uint32_t samplerCount, uint32_t outputCount)
    -> FlatSamplerScene
{
    FlatSamplerScene scene;
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
    const std::array<float, 4> texel{static_cast<float>(kFlatSampledValue), 0.f, 0.f, 0.f};
    const SubresourceData data{.data = texel.data(), .rowPitch = 16, .slicePitch = 16};
    REQUIRE_CALL(device->createTexture(textureDesc, &data, scene.texture.writeRef()));
    scene.textureView = scene.texture->getDefaultView();
    REQUIRE(scene.textureView != nullptr);
    scene.samplers.resize(samplerCount);
    std::vector<BindingSetEntry> entries{{.slot = 0, .resource = scene.textureView}};
    for (uint32_t i = 0; i < samplerCount; ++i)
    {
        SamplerDesc samplerDesc{};
        // Distinct descriptor values prevent backend sampler interning from masking heap exhaustion.
        samplerDesc.mipLODBias = static_cast<float>(i) * 0.25f;
        REQUIRE_CALL(device->createSampler(samplerDesc, scene.samplers[i].writeRef()));
        entries.emplace_back(BindingSetEntry{.slot = 1, .arrayIndex = i, .resource = scene.samplers[i]});
    }
    entries.emplace_back(BindingSetEntry{.slot = 2, .resource = scene.output});
    const BindingSetDesc desc{
        .layout = setLayout,
        .entries = entries.data(),
        .entryCount = static_cast<uint32_t>(entries.size()),
        .label = "samplers",
    };
    REQUIRE_CALL(device->createBindingSet(desc, scene.set.writeRef()));
    REQUIRE(scene.set != nullptr);
    return scene;
}

// Both backends and both profiles use the same CPU-computed expectation table.
auto runFlatCompute(IDevice* device, bool inlineConstants) -> void
{
    const FlatDeviceDrain drain{device};
    auto session = createFlatSession(device, {.inlineConstants = inlineConstants});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, input, 4, scene);
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, sizeof(FlatConstants), "compute");
    createFlatSceneBindingSet(device, scene);
    CHECK(layout->getSetCount() == 1);
    CHECK(layout->getConstantsSize() == sizeof(FlatConstants));
    if (device->getDeviceType() == DeviceType::Vulkan)
    {
        const auto profile = inlineConstants ? ConstantsProfile::Inline : ConstantsProfile::Buffered;
        CHECK(layout->getConstantsProfile() == profile);
    }
    else
    {
        // [[vk::push_constant]] has no DXIL counterpart; both profiles become a root CBV on D3D12.
        CHECK(layout->getConstantsProfile() != ConstantsProfile::None);
    }
    const ComputePipelineDesc desc{.program = program, .layout = layout};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    const FlatConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    IBindingSet* const sets[]{scene.set};
    const auto bindings = flatBindings(&constants, sizeof(constants), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
    pass->dispatchCompute(4, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[i] = flatExpected(0, input[i], constants);
    }
    compareComputeResult(device, scene.output, expected);
}
} // namespace

GPU_TEST_CASE("flat-binding-compute-buffered", D3D12 | Vulkan)
{
    runFlatCompute(device, false);
}

GPU_TEST_CASE("flat-binding-compute-inline", D3D12 | Vulkan)
{
    runFlatCompute(device, true);
}

GPU_TEST_CASE("flat-binding-graphics-vs-ps", D3D12 | Vulkan)
{
    const FlatDeviceDrain drain{device};
    auto session = createFlatSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(
        loadAndLinkProgram(device, session, "test-flat-binding-cases", {"vsMain", "psMain"}, program.writeRef())
    );
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, input, 4, scene);
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, sizeof(FlatConstants), "graphics");
    createFlatSceneBindingSet(device, scene);
    ColorTargetDesc target{};
    target.format = Format::RGBA32Float;
    RenderPipelineDesc desc{.program = program, .layout = layout, .targets = &target, .targetCount = 1};
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
    FlatConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 11};
    IBindingSet* const sets[]{scene.set};
    const auto bindings = flatBindings(&constants, sizeof(constants), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    RenderPassColorAttachment attachment{};
    attachment.view = view;
    attachment.loadOp = LoadOp::Clear;
    attachment.storeOp = StoreOp::Store;
    RenderPassDesc passDesc{};
    passDesc.colorAttachments = &attachment;
    passDesc.colorAttachmentCount = 1;
    auto pass = encoder->beginRenderPass(passDesc);
    REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
    RenderState state{};
    state.viewports[0] = Viewport::fromSize(2, 2);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(2, 2);
    state.scissorRectCount = 1;
    pass->setRenderState(state);
    DrawArguments draw{};
    draw.vertexCount = 3;
    pass->draw(draw);
    const auto bindShaderObject = [&](bool suppliedRoot)
    {
        auto legacyDesc = desc;
        legacyDesc.layout = nullptr;
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
        }
        REQUIRE(root != nullptr);
        constants.bias = 19;
        ShaderCursor cursor(root);
        REQUIRE_CALL(cursor["constants"].getDereferenced().setData(&constants, sizeof(constants)));
        REQUIRE_CALL(cursor["scene"]["input"].setBinding(scene.input));
        REQUIRE_CALL(cursor["scene"]["output"].setBinding(scene.output));
        REQUIRE_CALL(cursor["scene"]["texture"].setBinding(scene.textureView));
        REQUIRE_CALL(cursor["scene"]["sampler"].setBinding(scene.sampler));
        pass->setRenderState(state);
        pass->draw(draw);
    };
    SUBCASE("flat only") {}
    SUBCASE("switch to encoder-created root") { bindShaderObject(false); }
    SUBCASE("switch to supplied root") { bindShaderObject(true); }
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
        static_cast<float>(input[0] * constants.multiplier + constants.bias + constants.tag + kFlatSampledValue),
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

GPU_TEST_CASE("flat-binding-set-immutable-and-early-release", D3D12 | Vulkan)
{
    const FlatDeviceDrain drain{device};
    auto session = createFlatSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, input, 4, scene);
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, sizeof(FlatConstants), "early-release");
    createFlatSceneBindingSet(device, scene);
    const ComputePipelineDesc desc{.program = program, .layout = layout};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    // Leave resource ownership to the set before recording; retain only the output for readback.
    scene.textureView.setNull();
    scene.texture.setNull();
    scene.sampler.setNull();
    scene.input.setNull();
    const FlatConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    IBindingSet* const sets[]{scene.set};
    const auto bindings = flatBindings(&constants, sizeof(constants), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
    pass->dispatchCompute(4, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[i] = flatExpected(0, input[i], constants);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("flat-binding-constants-copied-at-bind", D3D12 | Vulkan)
{
    const FlatDeviceDrain drain{device};
    auto session = createFlatSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, input, 4, scene);
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, sizeof(FlatConstants), "constants-copy");
    createFlatSceneBindingSet(device, scene);
    const ComputePipelineDesc desc{.program = program, .layout = layout};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    FlatConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const FlatConstants expectedConstants = constants;
    IBindingSet* const sets[]{scene.set};
    const auto bindings = flatBindings(&constants, sizeof(constants), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
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
        expected[i] = flatExpected(0, input[i], expectedConstants);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("flat-binding-uav-order", D3D12 | Vulkan)
{
    const FlatDeviceDrain drain{device};
    auto session = createFlatSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, input, 4, scene);
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, sizeof(FlatConstants), "uav-order");
    createFlatSceneBindingSet(device, scene);
    const ComputePipelineDesc desc{.program = program, .layout = layout};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    const FlatConstants first{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const FlatConstants second{.offset = 0, .multiplier = 5, .bias = 1, .tag = 2};
    IBindingSet* const sets[]{scene.set};
    auto bindings = flatBindings(&first, sizeof(first), sets);
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
    pass->dispatchCompute(4, 1, 1);
    // The second dispatch must observe the first: recorded accesses must produce a UAV barrier between them.
    bindings = flatBindings(&second, sizeof(second), sets);
    REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
    pass->dispatchCompute(4, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[i] = flatExpected(flatExpected(0, input[i], first), input[i], second);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("flat-binding-set-shared-by-two-command-buffers", D3D12 | Vulkan)
{
    const FlatDeviceDrain drain{device};
    auto session = createFlatSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, input, 8, scene);
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, sizeof(FlatConstants), "shared-set");
    createFlatSceneBindingSet(device, scene);
    const ComputePipelineDesc desc{.program = program, .layout = layout};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    const FlatConstants constantsA{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const FlatConstants constantsB{.offset = 4, .multiplier = 5, .bias = 1, .tag = 0};
    IBindingSet* const sets[]{scene.set};
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoderA = queue->createCommandEncoder();
    ComPtr<ICommandBuffer> commandsA;
    {
        const auto bindings = flatBindings(&constantsA, sizeof(constantsA), sets);
        auto pass = encoderA->beginComputePass();
        REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
        pass->dispatchCompute(4, 1, 1);
        pass->end();
        commandsA = encoderA->finish();
        REQUIRE(commandsA != nullptr);
    }
    auto encoderB = queue->createCommandEncoder();
    ComPtr<ICommandBuffer> commandsB;
    {
        const auto bindings = flatBindings(&constantsB, sizeof(constantsB), sets);
        auto pass = encoderB->beginComputePass();
        REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
        pass->dispatchCompute(4, 1, 1);
        pass->end();
        commandsB = encoderB->finish();
        REQUIRE(commandsB != nullptr);
    }
    ICommandBuffer* buffersA[]{commandsA};
    ICommandBuffer* buffersB[]{commandsB};
    uint64_t sequenceA = 0;
    const SubmitDesc submitA{.commandBuffers = buffersA, .commandBufferCount = 1, .outSequence = &sequenceA};
    REQUIRE_CALL(queue->submit(submitA));
    REQUIRE_CALL(queue->waitForSequence(sequenceA, kTimeoutInfinite));
    // The empty submission retires A and reclaims its transient/descriptor allocations before B uses the shared set.
    uint64_t retirementSequence = 0;
    const SubmitDesc empty{.outSequence = &retirementSequence};
    REQUIRE_CALL(queue->submit(empty));
    REQUIRE_CALL(queue->waitForSequence(retirementSequence, kTimeoutInfinite));
    uint64_t sequenceB = 0;
    const SubmitDesc submitB{.commandBuffers = buffersB, .commandBufferCount = 1, .outSequence = &sequenceB};
    REQUIRE_CALL(queue->submit(submitB));
    REQUIRE_CALL(queue->waitForSequence(sequenceB, kTimeoutInfinite));
    std::array<uint32_t, 8> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[constantsA.offset + i] = flatExpected(0, input[i], constantsA);
        expected[constantsB.offset + i] = flatExpected(0, input[i], constantsB);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("flat-binding-set-reused-across-constants-layouts", D3D12 | Vulkan)
{
    const FlatDeviceDrain drain{device};
    auto narrowSession = createFlatSession(device, {});
    auto wideSession = createFlatSession(device, {.wideConstants = true});
    ComPtr<IShaderProgram> narrowProgram;
    ComPtr<IShaderProgram> wideProgram;
    REQUIRE_CALL(
        loadAndLinkProgram(device, narrowSession, "test-flat-binding-cases", "csMain", narrowProgram.writeRef())
    );
    REQUIRE_CALL(loadAndLinkProgram(device, wideSession, "test-flat-binding-cases", "csMain", wideProgram.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, input, 8, scene);
    auto narrowLayout =
        createFlatPipelineLayout(device, narrowProgram, scene.setLayout, sizeof(FlatConstants), "narrow");
    auto wideLayout = createFlatPipelineLayout(device, wideProgram, scene.setLayout, sizeof(FlatWideConstants), "wide");
    createFlatSceneBindingSet(device, scene);
    CHECK(narrowLayout->getConstantsSize() == 16);
    CHECK(wideLayout->getConstantsSize() == 32);
    CHECK(narrowLayout->getConstantsSize() != wideLayout->getConstantsSize());
    CHECK(scene.set->getLayout() == scene.setLayout.get());
    CHECK(narrowLayout->getDesc().sets[0].layout == scene.set->getLayout());
    CHECK(wideLayout->getDesc().sets[0].layout == scene.set->getLayout());
    const ComputePipelineDesc narrowDesc{.program = narrowProgram, .layout = narrowLayout};
    const ComputePipelineDesc wideDesc{.program = wideProgram, .layout = wideLayout};
    ComPtr<IComputePipeline> narrowPipeline;
    ComPtr<IComputePipeline> widePipeline;
    REQUIRE_CALL(device->createComputePipeline(narrowDesc, narrowPipeline.writeRef()));
    REQUIRE_CALL(device->createComputePipeline(wideDesc, widePipeline.writeRef()));
    const FlatConstants narrow{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const FlatWideConstants wide{.offset = 4, .multiplier = 5, .bias = 1, .tag = 0, .extra = {2, 0, 0, 9}};
    IBindingSet* const sets[]{scene.set};
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    {
        const auto bindings = flatBindings(&narrow, sizeof(narrow), sets);
        REQUIRE_CALL(pass->bindPipeline(narrowPipeline, bindings));
        pass->dispatchCompute(4, 1, 1);
    }
    {
        const auto bindings = flatBindings(&wide, sizeof(wide), sets);
        REQUIRE_CALL(pass->bindPipeline(widePipeline, bindings));
        pass->dispatchCompute(4, 1, 1);
    }
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    const FlatConstants narrowView{
        .offset = wide.offset,
        .multiplier = wide.multiplier,
        .bias = wide.bias,
        .tag = wide.tag,
    };
    std::array<uint32_t, 8> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[narrow.offset + i] = flatExpected(0, input[i], narrow);
        expected[wide.offset + i] = flatExpected(0, input[i], narrowView) + wide.extra[0] + wide.extra[3];
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("flat-binding-reverse-order-submission", D3D12 | Vulkan)
{
    const FlatDeviceDrain drain{device};
    auto session = createFlatSession(device, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, input, 8, scene);
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, sizeof(FlatConstants), "reverse-order");
    createFlatSceneBindingSet(device, scene);
    const ComputePipelineDesc desc{.program = program, .layout = layout};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
    const FlatConstants constantsA{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const FlatConstants constantsB{.offset = 4, .multiplier = 5, .bias = 1, .tag = 0};
    IBindingSet* const sets[]{scene.set};
    auto queue = device->getQueue(QueueType::Graphics);
    // Reverse submission requires constant bytes to be copied per command buffer, not per queue.
    auto encoderA = queue->createCommandEncoder();
    ComPtr<ICommandBuffer> commandsA;
    {
        const auto bindings = flatBindings(&constantsA, sizeof(constantsA), sets);
        auto pass = encoderA->beginComputePass();
        REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
        pass->dispatchCompute(4, 1, 1);
        pass->end();
        commandsA = encoderA->finish();
        REQUIRE(commandsA != nullptr);
    }
    auto encoderB = queue->createCommandEncoder();
    ComPtr<ICommandBuffer> commandsB;
    {
        const auto bindings = flatBindings(&constantsB, sizeof(constantsB), sets);
        auto pass = encoderB->beginComputePass();
        REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
        pass->dispatchCompute(4, 1, 1);
        pass->end();
        commandsB = encoderB->finish();
        REQUIRE(commandsB != nullptr);
    }
    ICommandBuffer* buffersB[]{commandsB};
    uint64_t sequenceB = 0;
    const SubmitDesc submitB{.commandBuffers = buffersB, .commandBufferCount = 1, .outSequence = &sequenceB};
    REQUIRE_CALL(queue->submit(submitB));
    ICommandBuffer* buffersA[]{commandsA};
    uint64_t sequenceA = 0;
    const SubmitDesc submitA{.commandBuffers = buffersA, .commandBufferCount = 1, .outSequence = &sequenceA};
    REQUIRE_CALL(queue->submit(submitA));
    CHECK(sequenceA > sequenceB);
    REQUIRE_CALL(queue->waitForSequence(sequenceA, kTimeoutInfinite));
    std::array<uint32_t, 8> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[constantsA.offset + i] = flatExpected(0, input[i], constantsA);
        expected[constantsB.offset + i] = flatExpected(0, input[i], constantsB);
    }
    compareComputeResult(device, scene.output, expected);
}

GPU_TEST_CASE("flat-binding-handle-access-keepalive", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::Bindless))
    {
        SKIP("Bindless is not supported");
    }
    const FlatDeviceDrain drain{device};
    auto session = createFlatSession(device, {.bindlessHandle = true});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(device, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, input, 4, scene);
    auto layout =
        createFlatPipelineLayout(device, program, scene.setLayout, sizeof(FlatBindlessConstants), "handle-keepalive");
    createFlatSceneBindingSet(device, scene);
    const ComputePipelineDesc desc{.program = program, .layout = layout};
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
    FlatBindlessConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0, .handle = {}};
    std::memcpy(constants.handle, &handle.value, sizeof(handle.value));
    const ResourceAccess accesses[]{{.buffer = extra, .state = ResourceState::ShaderResource}};
    IBindingSet* const sets[]{scene.set};
    auto bindings = flatBindings(&constants, sizeof(constants), sets);
    bindings.accesses = accesses;
    bindings.accessCount = 1;
    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
    // Only the recorded access can keep this buffer alive; no binding set owns it.
    extra.setNull();
    pass->dispatchCompute(4, 1, 1);
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    const FlatConstants narrowView{
        .offset = constants.offset,
        .multiplier = constants.multiplier,
        .bias = constants.bias,
        .tag = constants.tag,
    };
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        expected[i] = flatExpected(0, input[i], narrowView) + extraData[i];
    }
    compareComputeResult(device, scene.output, expected);
}

namespace {
/// The buffer the two halves of the transition case hand between a set entry and a descriptor
/// handle. It carries both usages, and no element size, because a descriptor handle is always a raw
/// view while a set entry takes its stride from the reflected element type.
auto createFlatTransitionBuffer(IDevice* device, uint32_t count, const char* label) -> ComPtr<IBuffer>
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

auto flatBindlessConstants(const FlatConstants& values, const DescriptorHandle& handle) -> FlatBindlessConstants
{
    FlatBindlessConstants constants{
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

GPU_TEST_CASE("flat-binding-handle-access-transition", D3D12 | Vulkan)
{
    if (!device->hasFeature(Feature::Bindless))
    {
        SKIP("Bindless is not supported");
    }
    const FlatDeviceDrain drain{device};
    auto session = createFlatSession(device, {.bindlessHandle = true});
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
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(device);
    createFlatSceneResources(device, seed, kCount, scene);
    auto bridgeToHandle = createFlatTransitionBuffer(device, kCount, "bridge-to-handle");
    auto bridgeToSet = createFlatTransitionBuffer(device, kCount, "bridge-to-set");
    auto handleReadResult = createFlatTransitionBuffer(device, kCount, "handle-read-result");
    auto setReadResult = createFlatTransitionBuffer(device, kCount, "set-read-result");

    auto setWriteLayout =
        createFlatPipelineLayout(device, setWrite, scene.setLayout, sizeof(FlatBindlessConstants), "set-write");
    auto handleReadLayout =
        createFlatPipelineLayout(device, handleRead, scene.setLayout, sizeof(FlatBindlessConstants), "handle-read");
    auto handleWriteLayout =
        createFlatPipelineLayout(device, handleWrite, scene.setLayout, sizeof(FlatBindlessConstants), "handle-write");
    auto setReadLayout =
        createFlatPipelineLayout(device, setRead, scene.setLayout, sizeof(FlatBindlessConstants), "set-read");

    auto setWriteSet = createFlatSceneSet(device, scene, scene.input, bridgeToHandle, "set-write");
    auto handleReadSet = createFlatSceneSet(device, scene, scene.input, handleReadResult, "handle-read");
    auto handleWriteSet = createFlatSceneSet(device, scene, scene.input, setReadResult, "handle-write");
    auto setReadSet = createFlatSceneSet(device, scene, bridgeToSet, setReadResult, "set-read");

    auto createPipeline = [&](IShaderProgram* program, IPipelineLayout* layout)
    {
        ComPtr<IComputePipeline> pipeline;
        const ComputePipelineDesc desc{.program = program, .layout = layout};
        REQUIRE_CALL(device->createComputePipeline(desc, pipeline.writeRef()));
        return pipeline;
    };
    auto setWritePipeline = createPipeline(setWrite, setWriteLayout);
    auto handleReadPipeline = createPipeline(handleRead, handleReadLayout);
    auto handleWritePipeline = createPipeline(handleWrite, handleWriteLayout);
    auto setReadPipeline = createPipeline(setRead, setReadLayout);

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

    const FlatConstants produceToHandle{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const FlatConstants consumeFromHandle{.offset = 0, .multiplier = 0, .bias = 0, .tag = 2};
    const FlatConstants produceToSet{.offset = 0, .multiplier = 5, .bias = 1, .tag = 0};
    const FlatConstants consumeFromSet{.offset = 0, .multiplier = 0, .bias = 0, .tag = 3};
    const auto setWriteConstants = flatBindlessConstants(produceToHandle, readHandle);
    const auto handleReadConstants = flatBindlessConstants(consumeFromHandle, readHandle);
    const auto handleWriteConstants = flatBindlessConstants(produceToSet, writeHandle);
    const auto setReadConstants = flatBindlessConstants(consumeFromSet, writeHandle);
    const ResourceAccess readAccess{.buffer = bridgeToHandle, .state = ResourceState::ShaderResource};
    const ResourceAccess writeAccess{.buffer = bridgeToSet, .state = ResourceState::UnorderedAccess};

    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    auto dispatch = [&](IComputePipeline* pipeline,
                        IBindingSet* set,
                        const FlatBindlessConstants& constants,
                        const ResourceAccess* access)
    {
        IBindingSet* const sets[]{set};
        auto bindings = flatBindings(&constants, sizeof(constants), sets);
        bindings.accesses = access;
        bindings.accessCount = access ? 1 : 0;
        REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
        pass->dispatchCompute(kCount, 1, 1);
    };
    // Written through the set's RWStructuredBuffer entry, then read through a handle the access
    // list declares as ShaderResource.
    dispatch(setWritePipeline, setWriteSet, setWriteConstants, nullptr);
    dispatch(handleReadPipeline, handleReadSet, handleReadConstants, &readAccess);
    // And the other way around: written through a handle the access list declares as
    // UnorderedAccess, then read through the set's StructuredBuffer entry.
    dispatch(handleWritePipeline, handleWriteSet, handleWriteConstants, &writeAccess);
    dispatch(setReadPipeline, setReadSet, setReadConstants, nullptr);
    // What the readback below establishes is that a value produced through either mechanism reaches
    // a consumer using the other one. It does not on its own prove the transitions were emitted:
    // dropping the two `accesses` entries still produces the same values here, because neither
    // backend's validation looks at the state of a resource reached through the bindless heap and
    // this hardware tolerates the missing barrier. Reviewing the recorded barriers is the only way
    // to check that part.
    pass->end();
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

// Vulkan has no counterpart to the sampler deduplication below: a Vulkan binding set owns its
// descriptors for its whole lifetime, so rebinding one costs nothing and cannot exhaust a heap.
// Vulkan descriptor pool and constant arena exhaustion is likewise not covered, for want of a
// device-side limit knob that would trigger it without making device creation itself fail; see
// §19.8 of the binding framework design for the full list of what this file does not cover.
GPU_TEST_CASE("flat-binding-d3d12-sampler-dedup-and-exhaustion", D3D12 | DontCreateDevice)
{
    // createFlatSession uses getSlangSearchPaths() in its own Slang session, so devices need no test search paths.
    // The bindless set reserves its sampler descriptors up front, so only the surplus is left for binding sets.
    constexpr uint32_t bindlessSamplerCount = 8;
    {
        INFO("deduplication");
        constexpr uint32_t samplerCount = 4;
        constexpr uint32_t bindCount = 32;
        D3D12DeviceExtendedDesc extendedDesc{};
        extendedDesc.samplerHeapSize = bindlessSamplerCount + 64;
        // The device must never outlive its callback, and a leaked device would only crash at exit.
        static FlatMessageSink sink;
        sink.messages.clear();
        DeviceDesc deviceDesc{.deviceType = ctx->deviceType, .adapter = getSelectedDeviceAdapter(ctx->deviceType)};
        deviceDesc.next = &extendedDesc;
        deviceDesc.bindless.samplerCount = bindlessSamplerCount;
        deviceDesc.debugCallback = &sink;
        ComPtr<IDevice> local;
        const Result created = getRHI()->createDevice(deviceDesc, local.writeRef());
        INFO(sink.messages);
        REQUIRE_CALL(created);
        const FlatDeviceDrain drain{local};
        auto session = createFlatSession(local, {.samplerCount = samplerCount});
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(loadAndLinkProgram(local, session, "test-flat-binding-cases", "csSamplers", program.writeRef()));
        auto setLayout = createFlatSamplerSetLayout(local, samplerCount);
        auto layout = createFlatPipelineLayout(local, program, setLayout, sizeof(FlatConstants), "sampler-dedup");
        auto scene = createFlatSamplerScene(local, setLayout, samplerCount, bindCount);
        const ComputePipelineDesc desc{.program = program, .layout = layout};
        ComPtr<IComputePipeline> pipeline;
        REQUIRE_CALL(local->createComputePipeline(desc, pipeline.writeRef()));
        IBindingSet* const sets[]{scene.set};
        auto queue = local->getQueue(QueueType::Graphics);
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        std::array<uint32_t, bindCount> expected{};
        // 32 * 4 descriptors overflow a 64-entry heap unless the command buffer caches the set's sampler table.
        for (uint32_t i = 0; i < bindCount; ++i)
        {
            const FlatConstants constants{.offset = i, .multiplier = 2, .bias = 5, .tag = 0};
            const auto bindings = flatBindings(&constants, sizeof(constants), sets);
            REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
            pass->dispatchCompute(1, 1, 1);
            expected[i] = samplerCount * kFlatSampledValue * constants.multiplier + constants.bias;
        }
        pass->end();
        auto commands = encoder->finish();
        REQUIRE(commands != nullptr);
        REQUIRE_CALL(queue->submit(commands));
        REQUIRE_CALL(queue->waitOnHost());
        compareComputeResult(local, scene.output, expected);
    }
    {
        INFO("exhaustion");
        constexpr uint32_t samplerCount = 32;
        D3D12DeviceExtendedDesc extendedDesc{};
        extendedDesc.samplerHeapSize = bindlessSamplerCount + 16;
        // The device must never outlive its callback, and a leaked device would only crash at exit.
        static FlatMessageSink sink;
        sink.messages.clear();
        DeviceDesc deviceDesc{.deviceType = ctx->deviceType, .adapter = getSelectedDeviceAdapter(ctx->deviceType)};
        deviceDesc.next = &extendedDesc;
        deviceDesc.bindless.samplerCount = bindlessSamplerCount;
        deviceDesc.debugCallback = &sink;
        ComPtr<IDevice> local;
        const Result created = getRHI()->createDevice(deviceDesc, local.writeRef());
        INFO(sink.messages);
        REQUIRE_CALL(created);
        const FlatDeviceDrain drain{local};
        auto session = createFlatSession(local, {.samplerCount = samplerCount});
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(loadAndLinkProgram(local, session, "test-flat-binding-cases", "csSamplers", program.writeRef()));
        auto setLayout = createFlatSamplerSetLayout(local, samplerCount);
        auto layout = createFlatPipelineLayout(local, program, setLayout, sizeof(FlatConstants), "sampler-exhaustion");
        auto scene = createFlatSamplerScene(local, setLayout, samplerCount, 1);
        const ComputePipelineDesc desc{.program = program, .layout = layout};
        ComPtr<IComputePipeline> pipeline;
        REQUIRE_CALL(local->createComputePipeline(desc, pipeline.writeRef()));
        const FlatConstants constants{.offset = 0, .multiplier = 2, .bias = 5, .tag = 0};
        IBindingSet* const sets[]{scene.set};
        const auto bindings = flatBindings(&constants, sizeof(constants), sets);
        auto queue = local->getQueue(QueueType::Graphics);
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        // A set of 32 samplers cannot fit a 16-entry heap, and the arena must fail rather than
        // overwrite descriptors in use.
        {
            ExpectedErrorScope expectedErrors;
            CHECK(SLANG_FAILED(pass->bindPipeline(pipeline, bindings)));
        }
        pass->end();
    }
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
struct FlatTransitionMessageSink final : IDebugCallback
{
    std::mutex               mutex;
    std::vector<std::string> transitions;
    std::string              errors;
    bool                     correctCategory = true;

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
        if (std::string_view(message).starts_with("Flat binding required state transition:"))
        {
            transitions.emplace_back(message);
            correctCategory &= type == DebugMessageType::Warning && source == DebugMessageSource::Layer;
        }
    }
};
} // namespace

GPU_TEST_CASE("flat-binding-state-transition-diagnostics", D3D12 | Vulkan | DontCreateDevice)
{
    bool validation = true;
    bool pretransition = false;
    bool shaderObject = false;
    SUBCASE("flat implicit transitions warn") {}
    SUBCASE("flat explicit transitions do not warn") { pretransition = true; }
    SUBCASE("validation disabled does not warn") { validation = false; }
    SUBCASE("shader object does not warn") { shaderObject = true; }

    FlatTransitionMessageSink sink;
    DeviceDesc deviceDesc{.deviceType = ctx->deviceType, .adapter = getSelectedDeviceAdapter(ctx->deviceType)};
    deviceDesc.enableValidation = validation;
    deviceDesc.debugCallback = &sink;
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
    const FlatDeviceDrain drain{local};
    auto session = createFlatSession(local, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(local, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(local);
    createFlatSceneResources(local, input, 4, scene);
    auto layout = createFlatPipelineLayout(local, program, scene.setLayout, sizeof(FlatConstants), "diagnostics");
    createFlatSceneBindingSet(local, scene);
    const ComputePipelineDesc desc{.program = program, .layout = shaderObject ? nullptr : layout.get()};
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(local->createComputePipeline(desc, pipeline.writeRef()));

    // A labeled source proves copy/clear transitions themselves never produce flat-binding warnings.
    const BufferDesc sourceDesc{
        .size = sizeof(input),
        .elementSize = 4,
        .usage = BufferUsage::CopySource | BufferUsage::CopyDestination,
        .defaultState = ResourceState::CopySource,
        .label = "diagnostic-copy-source",
    };
    auto source = local->createBuffer(sourceDesc, input.data());
    REQUIRE(source != nullptr);
    auto queue = local->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    encoder->clearBuffer(scene.output);
    encoder->copyBuffer(scene.input, 0, source, 0, sizeof(input));
    encoder->setTextureState(scene.texture, ResourceState::CopyDestination);
    // Output must already be UAV: repeated dispatches below still require real UAV ordering.
    encoder->setBufferState(scene.output, ResourceState::UnorderedAccess);
    if (pretransition)
    {
        encoder->setBufferState(scene.input, ResourceState::ShaderResource);
        encoder->setTextureState(scene.texture, ResourceState::ShaderResource);
    }
    auto pass = encoder->beginComputePass();
    const FlatConstants first{.offset = 0, .multiplier = 3, .bias = 7, .tag = 0};
    const FlatConstants second{.offset = 0, .multiplier = 5, .bias = 11, .tag = 2};
    IBindingSet* const sets[]{scene.set};
    for (const auto& constants : {first, second})
    {
        if (shaderObject)
        {
            auto root = pass->bindPipeline(pipeline);
            REQUIRE(root != nullptr);
            ShaderCursor cursor(root);
            REQUIRE_CALL(cursor["constants"].getDereferenced().setData(&constants, sizeof(constants)));
            REQUIRE_CALL(cursor["scene"]["input"].setBinding(scene.input));
            REQUIRE_CALL(cursor["scene"]["output"].setBinding(scene.output));
            REQUIRE_CALL(cursor["scene"]["texture"].setBinding(scene.textureView));
            REQUIRE_CALL(cursor["scene"]["sampler"].setBinding(scene.sampler));
        }
        else
        {
            REQUIRE_CALL(pass->bindPipeline(pipeline, flatBindings(&constants, sizeof(constants), sets)));
        }
        pass->dispatchCompute(4, 1, 1);
    }
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
        expected[i] = flatExpected(flatExpected(0, input[i], first), input[i], second);
    compareComputeResult(local, scene.output, expected);
    const std::lock_guard lock(sink.mutex);
    INFO(sink.errors);
    CHECK(sink.errors.empty());
    CHECK(sink.correctCategory);
    if (validation && !pretransition && !shaderObject)
    {
        REQUIRE(sink.transitions.size() == 2);
        CHECK(sink.transitions[0].find("buffer 'flat-scene-input'") != std::string::npos);
        CHECK(sink.transitions[0].find("target=ShaderResource") != std::string::npos);
        CHECK(sink.transitions[1].find("texture 'flat-scene-texture'") != std::string::npos);
        CHECK(sink.transitions[1].find("target=ShaderResource") != std::string::npos);
        CHECK(sink.transitions[1].find("layer=0 layerCount=1 mip=0 mipCount=1") != std::string::npos);
    }
    else
    {
        CHECK(sink.transitions.empty());
    }
}


GPU_TEST_CASE("flat-binding-state-transition-read-union", D3D12 | Vulkan | DontCreateDevice)
{
    FlatTransitionMessageSink sink;
    DeviceDesc deviceDesc{.deviceType = ctx->deviceType, .adapter = getSelectedDeviceAdapter(ctx->deviceType)};
    deviceDesc.enableValidation = true;
    deviceDesc.debugCallback = &sink;
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
    const FlatDeviceDrain drain{local};
    auto session = createFlatSession(local, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(local, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(local);
    createFlatSceneResources(local, input, 4, scene);
    auto layout = createFlatPipelineLayout(local, program, scene.setLayout, sizeof(FlatConstants), "read-union");
    createFlatSceneBindingSet(local, scene);
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(local->createComputePipeline({.program = program, .layout = layout}, pipeline.writeRef()));
    const std::array<uint32_t, 3> dispatch{4, 1, 1};
    const BufferDesc argumentDesc{
        .size = sizeof(dispatch),
        .elementSize = 4,
        .usage = BufferUsage::ShaderResource | BufferUsage::IndirectArgument | BufferUsage::CopyDestination,
        .defaultState = ResourceState::IndirectArgument,
        .label = "flat-read-union-arguments",
    };
    auto arguments = local->createBuffer(argumentDesc, dispatch.data());
    REQUIRE(arguments != nullptr);
    // SRV extends the already-known indirect state once. Subsequent SRV and indirect requests
    // are subsets of that effective union, and must not be diagnosed as additional transitions.
    const ResourceAccess accesses[]{
        {.buffer = arguments, .state = ResourceState::ShaderResource},
        {.buffer = arguments, .state = ResourceState::IndirectArgument},
    };
    IBindingSet* const sets[]{scene.set};
    const FlatConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 2};
    auto bindings = flatBindings(&constants, sizeof(constants), sets);
    bindings.accesses = accesses;
    bindings.accessCount = 2;
    auto queue = local->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    REQUIRE_CALL(pass->bindPipeline(pipeline, bindings));
    // Reuse the very same BindingData, covering both the retained read union and UAV ordering.
    pass->dispatchComputeIndirect({arguments, 0});
    pass->dispatchComputeIndirect({arguments, 0});
    pass->end();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < 4; ++i)
        expected[i] = flatExpected(flatExpected(0, input[i], constants), input[i], constants);
    compareComputeResult(local, scene.output, expected);
    const std::lock_guard lock(sink.mutex);
    INFO(sink.errors);
    CHECK(sink.errors.empty());
    CHECK(sink.correctCategory);
    REQUIRE(sink.transitions.size() == 1);
    CHECK(sink.transitions[0].find("buffer 'flat-read-union-arguments', target=ShaderResource") != std::string::npos);
}


namespace {
constexpr uint32_t kIdentitySamplerCount = 4;

auto samplerIdentityColor(uint32_t setIndex, uint32_t samplerIndex) -> std::array<uint32_t, 4>
{
    return {setIndex + 1, samplerIndex + 1, setIndex + samplerIndex + 2, 1};
}

auto createSamplerIdentitySet(IDevice* device, IBindingSetLayout* layout, const FlatSamplerScene& scene, uint32_t setIndex)
    -> ComPtr<IBindingSet>
{
    std::array<ComPtr<ISampler>, kIdentitySamplerCount> samplers;
    std::vector<BindingSetEntry> entries{{.slot = 0, .resource = scene.textureView}};
    for (uint32_t i = 0; i < kIdentitySamplerCount; ++i)
    {
        SamplerDesc desc{};
        desc.addressU = desc.addressV = desc.addressW = TextureAddressingMode::ClampToBorder;
        const auto color = samplerIdentityColor(setIndex, i);
        for (uint32_t channel = 0; channel < 4; ++channel)
            desc.borderColor[channel] = static_cast<float>(color[channel]);
        REQUIRE_CALL(device->createSampler(desc, samplers[i].writeRef()));
        entries.emplace_back(BindingSetEntry{.slot = 1, .arrayIndex = i, .resource = samplers[i]});
    }
    entries.emplace_back(BindingSetEntry{.slot = 2, .resource = scene.output});
    const BindingSetDesc desc{
        .layout = layout,
        .entries = entries.data(),
        .entryCount = static_cast<uint32_t>(entries.size()),
        .label = "sampler-identity",
    };
    ComPtr<IBindingSet> set;
    REQUIRE_CALL(device->createBindingSet(desc, set.writeRef()));
    // Only the immutable set retains the samplers after this helper returns.
    return set;
}

auto writeSamplerIdentityExpected(std::span<uint32_t> expected, uint32_t setIndex, const FlatConstants& constants)
    -> void
{
    for (uint32_t channel = 0; channel < 4; ++channel)
    {
        uint32_t sum = 0;
        for (uint32_t i = 0; i < kIdentitySamplerCount; ++i)
            sum += samplerIdentityColor(setIndex, i)[channel] * (i + 1);
        expected[constants.offset + channel] = sum * constants.multiplier + constants.bias;
    }
}
} // namespace

GPU_TEST_CASE("flat-binding-d3d12-sampler-cache-16-17-sets", D3D12 | DontCreateDevice)
{
    uint32_t setCount = 16;
    SUBCASE("16 distinct immutable sets") {}
    SUBCASE("17 distinct immutable sets") { setCount = 17; }
    constexpr uint32_t rounds = 8;
    constexpr uint32_t bindlessSamplerCount = 8;
    FlatTransitionMessageSink sink;
    D3D12DeviceExtendedDesc extendedDesc{};
    // Exactly one sampler table per set fits; even one uncached rebind must fail.
    extendedDesc.samplerHeapSize = bindlessSamplerCount + setCount * kIdentitySamplerCount;
    DeviceDesc deviceDesc{.deviceType = ctx->deviceType, .adapter = getSelectedDeviceAdapter(ctx->deviceType)};
    deviceDesc.next = &extendedDesc;
    deviceDesc.bindless.samplerCount = bindlessSamplerCount;
    deviceDesc.debugCallback = &sink;
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
    const FlatDeviceDrain drain{local};
    auto session = createFlatSession(local, {.samplerCount = kIdentitySamplerCount});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(local, session, "test-flat-binding-cases", "csSamplerColors", program.writeRef()));
    auto setLayout = createFlatSamplerSetLayout(local, kIdentitySamplerCount);
    auto layout = createFlatPipelineLayout(local, program, setLayout, sizeof(FlatConstants), "sampler-identities");
    const uint32_t outputCount = rounds * setCount * 4;
    auto scene = createFlatSamplerScene(local, setLayout, kIdentitySamplerCount, outputCount);
    std::vector<ComPtr<IBindingSet>> sets;
    for (uint32_t i = 0; i < setCount; ++i)
        sets.emplace_back(createSamplerIdentitySet(local, setLayout, scene, i));
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(local->createComputePipeline({.program = program, .layout = layout}, pipeline.writeRef()));
    auto queue = local->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    auto pass = encoder->beginComputePass();
    std::vector<uint32_t> expected(outputCount, 0);
    for (uint32_t round = 0; round < rounds; ++round)
    {
        for (uint32_t slot = 0; slot < setCount; ++slot)
        {
            // Reverse every other round so a stale or slot-indexed cache cannot mimic identity lookup.
            const uint32_t setIndex = (round & 1) ? setCount - 1 - slot : slot;
            const FlatConstants constants{
                .offset = (round * setCount + slot) * 4,
                .multiplier = round + 2,
                .bias = slot + 5,
                .tag = 0,
            };
            IBindingSet* const boundSets[]{sets[setIndex]};
            REQUIRE_CALL(pass->bindPipeline(pipeline, flatBindings(&constants, sizeof(constants), boundSets)));
            pass->dispatchCompute(1, 1, 1);
            writeSamplerIdentityExpected(expected, setIndex, constants);
        }
    }
    pass->end();
    // The recorded command buffer must retain all sets and their sampler identities.
    sets.clear();
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(local, scene.output, std::span(expected));
    const std::lock_guard lock(sink.mutex);
    INFO(sink.errors);
    CHECK(sink.errors.empty());
}

GPU_TEST_CASE("flat-binding-d3d12-sampler-cache-exhaustion-recovery", D3D12 | DontCreateDevice)
{
    constexpr uint32_t capacity = 16;
    constexpr uint32_t bindlessSamplerCount = 8;
    FlatTransitionMessageSink sink;
    D3D12DeviceExtendedDesc extendedDesc{};
    extendedDesc.samplerHeapSize = bindlessSamplerCount + capacity * kIdentitySamplerCount;
    DeviceDesc deviceDesc{.deviceType = ctx->deviceType, .adapter = getSelectedDeviceAdapter(ctx->deviceType)};
    deviceDesc.next = &extendedDesc;
    deviceDesc.bindless.samplerCount = bindlessSamplerCount;
    deviceDesc.debugCallback = &sink;
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
    const FlatDeviceDrain drain{local};
    auto session = createFlatSession(local, {.samplerCount = kIdentitySamplerCount});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(local, session, "test-flat-binding-cases", "csSamplerColors", program.writeRef()));
    auto setLayout = createFlatSamplerSetLayout(local, kIdentitySamplerCount);
    auto layout = createFlatPipelineLayout(local, program, setLayout, sizeof(FlatConstants), "sampler-capacity");
    // One extra result after the successful rebind stays zero as a canary.
    auto scene = createFlatSamplerScene(local, setLayout, kIdentitySamplerCount, (capacity + 2) * 4);
    std::vector<ComPtr<IBindingSet>> sets;
    for (uint32_t i = 0; i <= capacity; ++i)
        sets.emplace_back(createSamplerIdentitySet(local, setLayout, scene, i));
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(local->createComputePipeline({.program = program, .layout = layout}, pipeline.writeRef()));
    auto queue = local->getQueue(QueueType::Graphics);
    std::vector<uint32_t> expected((capacity + 2) * 4, 0);
    {
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        for (uint32_t i = 0; i < capacity; ++i)
        {
            const FlatConstants constants{.offset = i * 4, .multiplier = 3, .bias = i + 7, .tag = 0};
            IBindingSet* const boundSets[]{sets[i]};
            REQUIRE_CALL(pass->bindPipeline(pipeline, flatBindings(&constants, sizeof(constants), boundSets)));
            pass->dispatchCompute(1, 1, 1);
            writeSamplerIdentityExpected(expected, i, constants);
        }
        const FlatConstants constants{.offset = capacity * 4, .multiplier = 5, .bias = 19, .tag = 0};
        IBindingSet* const rejectedSets[]{sets[capacity]};
        const auto rejected = flatBindings(&constants, sizeof(constants), rejectedSets);
        {
            const std::lock_guard lock(sink.mutex);
            INFO(sink.errors);
            CHECK(sink.errors.empty());
        }
        const Result result = pass->bindPipeline(pipeline, rejected);
        CHECK((result == SLANG_E_OUT_OF_MEMORY));
        {
            const std::lock_guard lock(sink.mutex);
            CHECK(sink.errors.find("no room for the 4 sampler descriptors") != std::string::npos);
            // Consume only the two expected allocation errors, not unrelated driver/validation failures.
            std::string_view remaining = sink.errors;
            uint32_t errorCount = 0;
            while (!remaining.empty())
            {
                const size_t end = remaining.find('\n');
                const auto line = remaining.substr(0, end);
                if (!line.empty())
                {
                    CHECK((line.starts_with("GPU descriptor heap allocation failed (requested 4 descriptors,") ||
                           line == "Flat binding: no room for the 4 sampler descriptors of binding set 0."));
                    ++errorCount;
                }
                if (end == std::string_view::npos)
                    break;
                remaining.remove_prefix(end + 1);
            }
            CHECK(errorCount == 2);
            sink.errors.clear();
        }
        // Exhaustion must not evict earlier tables: reusing set zero still succeeds at full capacity.
        IBindingSet* const reusedSets[]{sets[0]};
        REQUIRE_CALL(pass->bindPipeline(pipeline, flatBindings(&constants, sizeof(constants), reusedSets)));
        pass->dispatchCompute(1, 1, 1);
        writeSamplerIdentityExpected(expected, 0, constants);
        pass->end();
        auto commands = encoder->finish();
        REQUIRE(commands != nullptr);
        REQUIRE_CALL(queue->submit(commands));
        REQUIRE_CALL(queue->waitOnHost());
        compareComputeResult(local, scene.output, std::span(expected));
    }
    // Completion plus release allows pool reuse/reset. Allocate the previously rejected identity first,
    // then old identities: stale cache entries would reference overwritten sampler descriptors.
    {
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        for (uint32_t i = 0; i < capacity; ++i)
        {
            const uint32_t setIndex = i == 0 ? capacity : i - 1;
            const FlatConstants constants{.offset = i * 4, .multiplier = 7, .bias = i + 23, .tag = 0};
            IBindingSet* const boundSets[]{sets[setIndex]};
            REQUIRE_CALL(pass->bindPipeline(pipeline, flatBindings(&constants, sizeof(constants), boundSets)));
            pass->dispatchCompute(1, 1, 1);
            writeSamplerIdentityExpected(expected, setIndex, constants);
        }
        pass->end();
        auto commands = encoder->finish();
        REQUIRE(commands != nullptr);
        REQUIRE_CALL(queue->submit(commands));
        REQUIRE_CALL(queue->waitOnHost());
        // Includes the unchanged successful-rebind result and all four zero canary words.
        compareComputeResult(local, scene.output, std::span(expected));
    }
    const std::lock_guard lock(sink.mutex);
    INFO(sink.errors);
    CHECK(sink.errors.empty());
}


namespace {
auto tryCreateQuotaSceneSet(IDevice* device, const FlatScene& scene, IBindingSet** outSet) -> Result
{
    const BindingSetEntry entries[]{
        {.slot = 0, .resource = scene.input},
        {.slot = 1, .resource = scene.output},
        {.slot = 2, .resource = scene.textureView},
        {.slot = 3, .resource = scene.sampler},
    };
    const BindingSetDesc desc{.layout = scene.setLayout, .entries = entries, .entryCount = 4, .label = "quota-scene"};
    return device->createBindingSet(desc, outSet);
}

auto checkQuotaMessages(FlatTransitionMessageSink& sink) -> void
{
    const std::lock_guard lock(sink.mutex);
    std::string_view remaining = sink.errors;
    while (!remaining.empty())
    {
        const size_t end = remaining.find('\n');
        const auto line = remaining.substr(0, end);
        CHECK((line.empty() || line == "Vulkan maxBindingSets quota exhausted."));
        if (end == std::string_view::npos)
            break;
        remaining.remove_prefix(end + 1);
    }
    sink.errors.clear();
}
} // namespace

GPU_TEST_CASE("flat-binding-vulkan-persistent-set-quota", Vulkan | DontCreateDevice)
{
    FlatTransitionMessageSink sink;
    VulkanDeviceExtendedDesc extendedDesc{};
    extendedDesc.maxBindingSets = 1;
    DeviceDesc deviceDesc{.deviceType = ctx->deviceType, .adapter = getSelectedDeviceAdapter(ctx->deviceType)};
    deviceDesc.next = &extendedDesc;
    deviceDesc.debugCallback = &sink;
    deviceDesc.enableValidation = true;
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
    const FlatDeviceDrain drain{local};
    auto session = createFlatSession(local, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(local, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(local);
    createFlatSceneResources(local, input, 12, scene);
    auto layout = createFlatPipelineLayout(local, program, scene.setLayout, sizeof(FlatConstants), "persistent-quota");
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(local->createComputePipeline({.program = program, .layout = layout}, pipeline.writeRef()));
    auto queue = local->getQueue(QueueType::Graphics);
    std::array<uint32_t, 12> expected{};
    for (uint32_t cycle = 0; cycle < 3; ++cycle)
    {
        REQUIRE_CALL(tryCreateQuotaSceneSet(local, scene, scene.set.writeRef()));
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        const FlatConstants constants{.offset = cycle * 4, .multiplier = cycle + 2, .bias = cycle + 7, .tag = 0};
        IBindingSet* const sets[]{scene.set};
        REQUIRE_CALL(pass->bindPipeline(pipeline, flatBindings(&constants, sizeof(constants), sets)));
        pass->dispatchCompute(4, 1, 1);
        // The recording is now the only set owner; it must still occupy the persistent quota.
        scene.set.setNull();
        ComPtr<IBindingSet> rejected;
        CHECK((tryCreateQuotaSceneSet(local, scene, rejected.writeRef()) == SLANG_E_OUT_OF_MEMORY));
        CHECK(rejected == nullptr);
        pass->end();
        if (cycle != 1)
        {
            auto commands = encoder->finish();
            REQUIRE(commands != nullptr);
            REQUIRE_CALL(queue->submit(commands));
            REQUIRE_CALL(queue->waitOnHost());
            for (uint32_t i = 0; i < 4; ++i)
                expected[cycle * 4 + i] = flatExpected(0, input[i], constants);
        }
        // Cycle 1 abandons the failed recording without finish/submit; its dispatch must never execute.
        encoder.setNull();
        REQUIRE_CALL(queue->waitOnHost());
        compareComputeResult(local, scene.output, expected);
        checkQuotaMessages(sink);
    }
}

namespace {
Size getVulkanUniformAlignment(IDevice* device)
{
#if SLANG_RHI_ENABLE_VULKAN
    DeviceNativeHandles handles{};
    REQUIRE_CALL(device->getNativeDeviceHandles(&handles));
    REQUIRE(handles.handles[0].type == NativeHandleType::VkInstance);
    REQUIRE(handles.handles[1].type == NativeHandleType::VkPhysicalDevice);
    vk::VulkanModule module;
    vk::VulkanApi api{};
    REQUIRE_CALL(module.init());
    REQUIRE_CALL(api.initGlobalProcs(module));
    REQUIRE_CALL(api.initInstanceProcs(reinterpret_cast<VkInstance>(handles.handles[0].value)));
    VkPhysicalDeviceProperties properties{};
    api.vkGetPhysicalDeviceProperties(reinterpret_cast<VkPhysicalDevice>(handles.handles[1].value), &properties);
    REQUIRE(properties.limits.minUniformBufferOffsetAlignment > 0);
    return properties.limits.minUniformBufferOffsetAlignment;
#else
    FAIL("Vulkan is required for the uniform alignment query");
    return 1;
#endif
}

auto runVulkanCommandQuota(DeviceType deviceType, bool constantQuota, bool retainRecording) -> void
{
    FlatTransitionMessageSink sink;
    VulkanDeviceExtendedDesc extendedDesc{};
    if (constantQuota)
        extendedDesc.maxCommandBufferConstantBytes = 4096 + sizeof(FlatConstants) - 1;
    else
        extendedDesc.maxCommandBufferDescriptorSets = 1;
    DeviceDesc deviceDesc{.deviceType = deviceType, .adapter = getSelectedDeviceAdapter(deviceType)};
    deviceDesc.next = &extendedDesc;
    deviceDesc.debugCallback = &sink;
    deviceDesc.enableValidation = true;
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
    const FlatDeviceDrain drain{local};
    const Size alignment = getVulkanUniformAlignment(local);
    const Size chargedBytes = (sizeof(FlatConstants) + alignment - 1) / alignment * alignment;
    const Size expectedCapacity = constantQuota ? extendedDesc.maxCommandBufferConstantBytes / chargedBytes : 1;
    REQUIRE(expectedCapacity > 0);
    // Buffered constants allocate both real uniform bytes and an internal descriptor set on every bind.
    auto session = createFlatSession(local, {});
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadAndLinkProgram(local, session, "test-flat-binding-cases", "csMain", program.writeRef()));
    const std::array<uint32_t, 4> input{5, 6, 7, 8};
    constexpr uint32_t maxSuccessfulBinds = 4096 / sizeof(FlatConstants);
    constexpr uint32_t slotsPerCycle = maxSuccessfulBinds + 1;
    FlatScene scene;
    scene.setLayout = createFlatSceneSetLayout(local);
    createFlatSceneResources(local, input, 3 * slotsPerCycle * 4, scene);
    auto layout = createFlatPipelineLayout(local, program, scene.setLayout, sizeof(FlatConstants), "command-quota");
    REQUIRE(layout->getConstantsProfile() == ConstantsProfile::Buffered);
    createFlatSceneBindingSet(local, scene);
    // Persistent allocations are independent of the command descriptor quota (which is just one).
    ComPtr<IBindingSet> secondSet;
    REQUIRE_CALL(tryCreateQuotaSceneSet(local, scene, secondSet.writeRef()));
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(local->createComputePipeline({.program = program, .layout = layout}, pipeline.writeRef()));
    auto queue = local->getQueue(QueueType::Graphics);
    std::vector<uint32_t> expected(3 * slotsPerCycle * 4, 0);
    uint32_t firstCapacity = 0;
    std::array<ComPtr<ICommandBuffer>, 3> pending;
    for (uint32_t cycle = 0; cycle < 3; ++cycle)
    {
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        uint32_t successfulBinds = 0;
        bool exhausted = false;
        for (uint32_t i = 0; i <= maxSuccessfulBinds; ++i)
        {
            const FlatConstants constants{
                .offset = (cycle * slotsPerCycle + i) * 4,
                .multiplier = cycle + 2,
                .bias = i + 7,
                .tag = 0,
            };
            IBindingSet* const sets[]{(i & 1) ? secondSet.get() : scene.set.get()};
            const Result result = pass->bindPipeline(pipeline, flatBindings(&constants, sizeof(constants), sets));
            if (SLANG_FAILED(result))
            {
                CHECK((result == SLANG_E_OUT_OF_MEMORY));
                exhausted = true;
                break;
            }
            REQUIRE(i < maxSuccessfulBinds);
            ++successfulBinds;
            pass->dispatchCompute(4, 1, 1);
            if (cycle != 1)
            {
                for (uint32_t j = 0; j < 4; ++j)
                    expected[constants.offset + j] = flatExpected(0, input[j], constants);
            }
        }
        REQUIRE(exhausted);
        REQUIRE(successfulBinds > 0);
        // Every successful buffered bind consumes exactly one aligned constant allocation.
        // The next bind must fail even when the configured budget has an unallocatable remainder.
        CHECK(successfulBinds == expectedCapacity);
        if (cycle == 0)
            firstCapacity = successfulBinds;
        else
            CHECK(successfulBinds == firstCapacity);
        pass->end();
        if (cycle != 1)
        {
            auto commands = encoder->finish();
            REQUIRE(commands != nullptr);
            if (retainRecording)
                pending[cycle] = commands;
            else
            {
                REQUIRE_CALL(queue->submit(commands));
                REQUIRE_CALL(queue->waitOnHost());
            }
        }
        // The middle cycle deliberately releases an exhausted encoder without submitting any work.
        encoder.setNull();
        REQUIRE_CALL(queue->waitOnHost());
        // Full output includes every accepted dispatch, rejected-bind slots, and the abandoned cycle's canaries.
        if (!retainRecording)
            compareComputeResult(local, scene.output, std::span(expected));
        const std::lock_guard lock(sink.mutex);
        INFO(sink.errors);
        CHECK(sink.errors.empty());
    }
    if (retainRecording)
    {
        // Both successful recordings coexist while the middle, exhausted encoder is abandoned.
        // This checks old descriptor/constant bytes remain intact without claiming a fixed in-flight window.
        REQUIRE_CALL(queue->submit(pending[0]));
        REQUIRE_CALL(queue->submit(pending[2]));
        pending = {};
        REQUIRE_CALL(queue->waitOnHost());
        compareComputeResult(local, scene.output, std::span(expected));
        const std::lock_guard lock(sink.mutex);
        INFO(sink.errors);
        CHECK(sink.errors.empty());
    }
}
} // namespace

GPU_TEST_CASE("flat-binding-vulkan-command-descriptor-quota", Vulkan | DontCreateDevice)
{
    bool retainRecording = false;
    SUBCASE("completion and abandoned encoder recovery") {}
    SUBCASE("retain older recording during exhaustion") { retainRecording = true; }
    runVulkanCommandQuota(ctx->deviceType, false, retainRecording);
}

GPU_TEST_CASE("flat-binding-vulkan-command-constant-quota", Vulkan | DontCreateDevice)
{
    bool retainRecording = false;
    SUBCASE("completion and abandoned encoder recovery") {}
    SUBCASE("retain older recording during exhaustion") { retainRecording = true; }
    runVulkanCommandQuota(ctx->deviceType, true, retainRecording);
}
