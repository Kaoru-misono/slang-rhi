#include "testing.h"
#include "constant-layout.h"

#include <string>

using namespace rhi;
using namespace rhi::testing;

namespace {

/// Reflects one execution constant block the way `createPipelineLayout` does: a global
/// `ConstantBuffer` compiled with the target options the flat-binding path uses.
auto reflectConstants(
    ComPtr<slang::ISession>& session,
    SlangCompileTarget format,
    SlangMatrixLayoutMode matrixLayout,
    const char* declarations
) -> slang::TypeLayoutReflection*
{
    slang::TargetDesc target{};
    target.format = format;
    target.forceGLSLScalarBufferLayout = true;
    slang::SessionDesc desc{};
    desc.targets = &target;
    desc.targetCount = 1;
    desc.defaultMatrixLayoutMode = matrixLayout;
    REQUIRE_CALL(getSlangGlobalSession()->createSession(desc, session.writeRef()));

    const std::string source = std::string(declarations) + R"(
ConstantBuffer<Constants> constants;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
}
)";
    ComPtr<ISlangBlob> diagnostics;
    auto* module = session->loadModuleFromSourceString(
        "constant-layout-probe",
        "constant-layout-probe.slang",
        source.c_str(),
        diagnostics.writeRef()
    );
    diagnoseIfNeeded(diagnostics);
    REQUIRE(module != nullptr);
    auto* reflection = module->getLayout();
    REQUIRE(reflection != nullptr);
    REQUIRE(reflection->getParameterCount() == 1);
    auto* block = reflection->getParameterByIndex(0)->getTypeLayout();
    REQUIRE(block != nullptr);
    auto* constants = block->getElementTypeLayout();
    REQUIRE(constants != nullptr);
    return constants;
}

constexpr const char* kPortableDeclarations = R"(
struct Nested
{
    float3 weights;
    uint tag;
};

struct Constants
{
    int index;
    uint count;
    float scale;
    float bias;
    float2 uv;
    uint2 tile;
    float4 color;
    float4 samples[2];
    float4x4 transform;
    Nested nested;
};
)";

} // namespace

TEST_CASE("constant-layout-portable-subset")
{
    for (auto format : {SLANG_DXIL, SLANG_SPIRV})
    {
        CAPTURE(format);
        ComPtr<slang::ISession> session;
        auto* constants =
            reflectConstants(session, format, SLANG_MATRIX_LAYOUT_COLUMN_MAJOR, kPortableDeclarations);
        std::string diagnostic;
        REQUIRE_CALL(validateConstantLayout(constants, diagnostic));
        CHECK(diagnostic.empty());
        // The point of the subset: the block is byte for byte what a 4-byte-packed C struct is.
        CHECK(constants->getSize() == 160);
    }
}

TEST_CASE("constant-layout-rejections")
{
    for (auto format : {SLANG_DXIL, SLANG_SPIRV})
    {
        CAPTURE(format);
        std::string diagnostic;
        auto reject = [&](const char* declarations, const char* reason)
        {
            INFO(declarations);
            ComPtr<slang::ISession> session;
            auto* constants =
                reflectConstants(session, format, SLANG_MATRIX_LAYOUT_COLUMN_MAJOR, declarations);
            CHECK(SLANG_FAILED(validateConstantLayout(constants, diagnostic)));
            INFO(diagnostic);
            CHECK(diagnostic.find(reason) != std::string::npos);
        };

        reject("struct Constants { bool flag; };", "constants.flag: expected a 32-bit");
        reject("struct Constants { double value; };", "constants.value: expected a 32-bit");
        reject("struct Constants { half value; };", "constants.value: expected a 32-bit");
        reject("struct Constants { int64_t wide; };", "constants.wide: expected a 32-bit");
        reject("struct Constants { float2x2 m; };", "constants.m: the only portable matrix");
        reject("struct Constants { float3x4 m; };", "constants.m: the only portable matrix");
        reject("struct Constants { float values[4]; };", "constants.values: portable arrays");
        reject("struct Constants { float2 pairs[2]; };", "constants.pairs: portable arrays");
        reject("struct Constants { float4 rows[2][2]; };", "constants.rows: portable arrays");
        reject("struct Constants { float a; float3 b; };", "constants.b: a three-component vector starts");
        reject("struct Constants { float a; float2 b; float c; };", "constants.b: a two-component vector starts");
        reject("struct Constants { float2 a; float2 b; float2 c; };", "constants: struct size");
        reject("struct Constants { float4 value; Texture2D texture; };", "not an execution constant");
        reject("typedef float4 Constants;", "constants: an execution constant block must be a struct");
    }
}

/// Both backends take the same bytes, so a matrix the two read differently is not portable.
TEST_CASE("constant-layout-row-major-matrix-rejection")
{
    for (auto format : {SLANG_DXIL, SLANG_SPIRV})
    {
        CAPTURE(format);
        ComPtr<slang::ISession> session;
        auto* constants =
            reflectConstants(session, format, SLANG_MATRIX_LAYOUT_ROW_MAJOR, kPortableDeclarations);
        std::string diagnostic;
        CHECK(SLANG_FAILED(validateConstantLayout(constants, diagnostic)));
        INFO(diagnostic);
        CHECK(diagnostic.find("constants.transform: portable matrices are column-major") != std::string::npos);
    }
}

/// D3D12 constant buffers never let a field straddle a 16-byte row, so there the reflected offset
/// already disagrees with the packed C layout; under Vulkan's scalar layout the placement rules are
/// what catches these blocks.
TEST_CASE("constant-layout-placement-rejections")
{
    ComPtr<slang::ISession> session;
    std::string diagnostic;
    auto reject = [&](const char* declarations, const char* reason)
    {
        INFO(declarations);
        auto* constants =
            reflectConstants(session, SLANG_SPIRV, SLANG_MATRIX_LAYOUT_COLUMN_MAJOR, declarations);
        CHECK(SLANG_FAILED(validateConstantLayout(constants, diagnostic)));
        INFO(diagnostic);
        CHECK(diagnostic.find(reason) != std::string::npos);
    };

    reject("struct Constants { float3 a; float3 b; float2 c; };", "constants.a: a three-component vector");
    reject("struct Constants { float2 a; float4 b; float2 c; };", "constants.b: four-component vectors");
    reject("struct Constants { float2 a; float4 b[1]; float2 c; };", "constants.b: four-component vectors");
    reject("struct Constants { float2 a; float4x4 m; float2 c; };", "constants.m: four-component vectors");
}

TEST_CASE("constant-layout-missing-reflection")
{
    std::string diagnostic;
    CHECK(SLANG_FAILED(validateConstantLayout(nullptr, diagnostic)));
    CHECK(diagnostic == "constants: missing reflection");
}
