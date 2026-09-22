#include "testing.h"
#include "constant-layout.h"

using namespace rhi;
using namespace rhi::testing;

TEST_CASE("constant-layout-portable-type-rejection")
{
    using Kind = slang::TypeReflection::Kind;
    using Scalar = slang::TypeReflection::ScalarType;
    for (auto format : {SLANG_DXIL, SLANG_SPIRV})
    {
        CAPTURE(format);
        slang::TargetDesc target{};
        target.format = format;
        target.forceGLSLScalarBufferLayout = true;
        slang::SessionDesc desc{};
        desc.targets = &target;
        desc.targetCount = 1;
        desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
        ComPtr<slang::ISession> session;
        REQUIRE_CALL(getSlangGlobalSession()->createSession(desc, session.writeRef()));
        ComPtr<ISlangBlob> diagnostics;
        auto module = session->loadModuleFromSourceString(
            "constant-layout-rejection",
            "constant-layout-rejection.slang",
            R"(
                struct Types
                {
                    bool flag;
                    int64_t wide;
                    float values[4];
                    float2 pairs[2];
                    float2x2 matrix;
                };
                struct ResourceParams
                {
                    float4 value;
                    Texture2D<float4> texture;
                };
            )",
            diagnostics.writeRef()
        );
        diagnoseIfNeeded(diagnostics);
        REQUIRE(module != nullptr);
        auto reflection = module->getLayout();
        REQUIRE(reflection != nullptr);
        auto type = reflection->findTypeByName("Types");
        REQUIRE(type != nullptr);
        auto layout = session->getTypeLayout(type);
        REQUIRE(layout != nullptr);
        std::string diagnostic;
        auto field = [&](const char* name)
        {
            auto index = layout->findFieldIndexByName(name);
            REQUIRE(index >= 0);
            return layout->getFieldByIndex(unsigned(index))->getTypeLayout();
        };
        auto reject = [&](slang::TypeLayoutReflection* actual, const ConstantTypeDesc& expected, const char* reason)
        {
            CHECK(SLANG_FAILED(validateConstantLayout(actual, expected, diagnostic)));
            INFO(diagnostic);
            CHECK(diagnostic.find(reason) != std::string::npos);
        };
        ConstantTypeDesc scalar{Kind::Scalar, 4, 4, Scalar::Bool};
        reject(field("flag"), scalar, "32-bit");
        scalar.size = 8;
        scalar.alignment = 8;
        scalar.scalar = Scalar::Int64;
        reject(field("wide"), scalar, "32-bit");

        scalar = {Kind::Scalar, 4, 4, Scalar::Float32};
        ConstantTypeDesc array{
            .kind = Kind::Array,
            .size = field("values")->getSize(),
            .alignment = 4,
            .elementCount = 4,
            .elementStride = 4,
            .elementType = &scalar,
        };
        reject(field("values"), array, "four-component");
        auto vector = ConstantTypeDesc{Kind::Vector, 8, 4, Scalar::Float32, 2};
        array.size = field("pairs")->getSize();
        array.elementType = &vector;
        array.elementCount = 2;
        array.elementStride = 8;
        reject(field("pairs"), array, "four-component");
        ConstantTypeDesc matrix{
            .kind = Kind::Matrix,
            .size = field("matrix")->getSize(),
            .alignment = 4,
            .scalar = Scalar::Float32,
            .matrixLayout = SLANG_MATRIX_LAYOUT_ROW_MAJOR,
        };
        reject(field("matrix"), matrix, "float4x4");
        auto resourceType = reflection->findTypeByName("ResourceParams");
        REQUIRE(resourceType != nullptr);
        auto resource = session->getTypeLayout(resourceType);
        REQUIRE(resource != nullptr);
        ConstantTypeDesc resourceSchema{Kind::Struct, resource->getSize(), 16};
        reject(resource, resourceSchema, "not an execution constant");
        reject(nullptr, scalar, "missing reflection");
    }
}
