#include "testing.h"

#include <array>
#include <cstddef>
#include <cstring>
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

auto paramsSchema() -> const ConstantTypeDesc&
{
    using Kind = slang::TypeReflection::Kind;
    using Scalar = slang::TypeReflection::ScalarType;
    static const ConstantTypeDesc floatType{
        .kind = Kind::Scalar,
        .size = 4,
        .alignment = 4,
        .scalar = Scalar::Float32,
    };
    static const ConstantTypeDesc float3Type{
        .kind = Kind::Vector,
        .size = 12,
        .alignment = 4,
        .scalar = Scalar::Float32,
        .elementCount = 3,
    };
    static const ConstantTypeDesc float4Type{
        .kind = Kind::Vector,
        .size = 16,
        .alignment = 4,
        .scalar = Scalar::Float32,
        .elementCount = 4,
    };
    static const ConstantTypeDesc samplesType{
        .kind = Kind::Array,
        .size = sizeof(Params::samples),
        .alignment = 16,
        .elementCount = 2,
        .elementStride = 16,
        .elementType = &float4Type,
    };
    static const ConstantFieldDesc fields[]{
        {"direction", offsetof(Params, direction), &float3Type},
        {"scale", offsetof(Params, scale), &floatType},
        {"samples", offsetof(Params, samples), &samplesType},
    };
    static const ConstantTypeDesc params{
        .kind = Kind::Struct,
        .size = sizeof(Params),
        .alignment = alignof(Params),
        .fields = fields,
        .fieldCount = 3,
    };
    return params;
}

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
    desc.constants = &paramsSchema();
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
    CHECK(stored.constants == nullptr);
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
        desc.constants = &paramsSchema();
        auto schema = paramsSchema();
        schema.size += 16;
        desc.constants = &schema;
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("constants type mismatch");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, sceneSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constants = &paramsSchema();
        auto schema = paramsSchema();
        std::vector<ConstantFieldDesc> fields(schema.fields, schema.fields + schema.fieldCount);
        auto scaleType = *fields[1].type;
        scaleType.scalar = slang::TypeReflection::ScalarType::UInt32;
        fields[1].type = &scaleType;
        schema.fields = fields.data();
        desc.constants = &schema;
        ComPtr<IPipelineLayout> layout;
        CHECK(SLANG_FAILED(device->createPipelineLayout(desc, layout.writeRef())));
    }
    {
        INFO("constants stride mismatch");
        ComPtr<IShaderProgram> program;
        REQUIRE_CALL(rhi::testing::loadComputeProgramFromSource(device, sceneSource, program.writeRef()));
        auto sceneLayout = createSceneSetLayout(device);
        PipelineLayoutSetDesc sets[]{{"scene", sceneLayout}};
        PipelineLayoutDesc desc{};
        desc.program = program;
        desc.sets = sets;
        desc.setCount = 1;
        desc.constants = &paramsSchema();
        auto schema = paramsSchema();
        std::vector<ConstantFieldDesc> fields(schema.fields, schema.fields + schema.fieldCount);
        auto samplesType = *fields[2].type;
        samplesType.elementStride = 32;
        fields[2].type = &samplesType;
        schema.fields = fields.data();
        desc.constants = &schema;
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
        desc.constants = &paramsSchema();
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
        desc.constants = &paramsSchema();
        desc.constants = nullptr;
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
        desc.constants = &paramsSchema();
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
        desc.constants = &paramsSchema();
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
        desc.constants = &paramsSchema();
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
        desc.constants = &paramsSchema();
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
        desc.constants = &paramsSchema();
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
        desc.constants = &paramsSchema();
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

auto flatConstantsSchema() -> const ConstantTypeDesc&
{
    using Kind = slang::TypeReflection::Kind;
    using Scalar = slang::TypeReflection::ScalarType;
    static const ConstantTypeDesc uintType{Kind::Scalar, 4, 4, Scalar::UInt32};
    static const ConstantFieldDesc fields[]{
        {"offset", offsetof(FlatConstants, offset), &uintType},
        {"multiplier", offsetof(FlatConstants, multiplier), &uintType},
        {"bias", offsetof(FlatConstants, bias), &uintType},
        {"tag", offsetof(FlatConstants, tag), &uintType},
    };
    static const ConstantTypeDesc constants{
        .kind = Kind::Struct,
        .size = sizeof(FlatConstants),
        .alignment = alignof(FlatConstants),
        .fields = fields,
        .fieldCount = 4,
    };
    return constants;
}

auto flatWideConstantsSchema() -> const ConstantTypeDesc&
{
    using Kind = slang::TypeReflection::Kind;
    using Scalar = slang::TypeReflection::ScalarType;
    static const ConstantTypeDesc uintType{Kind::Scalar, 4, 4, Scalar::UInt32};
    static const ConstantTypeDesc uint4Type{Kind::Vector, 16, 4, Scalar::UInt32, 4};
    static const ConstantFieldDesc fields[]{
        {"offset", offsetof(FlatWideConstants, offset), &uintType},
        {"multiplier", offsetof(FlatWideConstants, multiplier), &uintType},
        {"bias", offsetof(FlatWideConstants, bias), &uintType},
        {"tag", offsetof(FlatWideConstants, tag), &uintType},
        {"extra", offsetof(FlatWideConstants, extra), &uint4Type},
    };
    static const ConstantTypeDesc constants{
        .kind = Kind::Struct,
        .size = sizeof(FlatWideConstants),
        .alignment = alignof(FlatWideConstants),
        .fields = fields,
        .fieldCount = 5,
    };
    return constants;
}

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

auto flatBindlessConstantsSchema() -> const ConstantTypeDesc&
{
    using Kind = slang::TypeReflection::Kind;
    using Scalar = slang::TypeReflection::ScalarType;
    static const ConstantTypeDesc uintType{Kind::Scalar, 4, 4, Scalar::UInt32};
    static const ConstantTypeDesc uint4Type{Kind::Vector, 16, 4, Scalar::UInt32, 4};
    static const ConstantFieldDesc fields[]{
        {"offset", offsetof(FlatBindlessConstants, offset), &uintType},
        {"multiplier", offsetof(FlatBindlessConstants, multiplier), &uintType},
        {"bias", offsetof(FlatBindlessConstants, bias), &uintType},
        {"tag", offsetof(FlatBindlessConstants, tag), &uintType},
        {"handle", offsetof(FlatBindlessConstants, handle), &uint4Type},
    };
    static const ConstantTypeDesc constants{
        .kind = Kind::Struct,
        .size = sizeof(FlatBindlessConstants),
        .alignment = alignof(FlatBindlessConstants),
        .fields = fields,
        .fieldCount = 5,
    };
    return constants;
}

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
    const ConstantTypeDesc& constants,
    const char* label
) -> ComPtr<IPipelineLayout>
{
    const PipelineLayoutSetDesc sets[]{{"scene", setLayout}};
    const PipelineLayoutDesc desc{program, sets, 1, &constants, label};
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
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, flatConstantsSchema(), "compute");
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
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, flatConstantsSchema(), "graphics");
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
    const FlatConstants constants{.offset = 0, .multiplier = 3, .bias = 7, .tag = 11};
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
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, flatConstantsSchema(), "early-release");
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
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, flatConstantsSchema(), "constants-copy");
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
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, flatConstantsSchema(), "uav-order");
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
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, flatConstantsSchema(), "shared-set");
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
        createFlatPipelineLayout(device, narrowProgram, scene.setLayout, flatConstantsSchema(), "narrow");
    auto wideLayout = createFlatPipelineLayout(device, wideProgram, scene.setLayout, flatWideConstantsSchema(), "wide");
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
    auto layout = createFlatPipelineLayout(device, program, scene.setLayout, flatConstantsSchema(), "reverse-order");
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
        createFlatPipelineLayout(device, program, scene.setLayout, flatBindlessConstantsSchema(), "handle-keepalive");
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

    const auto& constantsSchema = flatBindlessConstantsSchema();
    auto setWriteLayout = createFlatPipelineLayout(device, setWrite, scene.setLayout, constantsSchema, "set-write");
    auto handleReadLayout =
        createFlatPipelineLayout(device, handleRead, scene.setLayout, constantsSchema, "handle-read");
    auto handleWriteLayout =
        createFlatPipelineLayout(device, handleWrite, scene.setLayout, constantsSchema, "handle-write");
    auto setReadLayout = createFlatPipelineLayout(device, setRead, scene.setLayout, constantsSchema, "set-read");

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
        auto layout = createFlatPipelineLayout(local, program, setLayout, flatConstantsSchema(), "sampler-dedup");
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
        auto layout = createFlatPipelineLayout(local, program, setLayout, flatConstantsSchema(), "sampler-exhaustion");
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
