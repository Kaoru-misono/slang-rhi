#pragma once

#include "constant-layout.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rhi::testing::flat {

struct alignas(16) Nested
{
    uint32_t tag;
    float weights[3];
};

struct alignas(16) Params
{
    float direction[3];
    float scale;
    float samples[2][4];
    float matrix[4][4];
    Nested nested;
};

static_assert(std::is_trivially_copyable_v<Params> && std::is_standard_layout_v<Params>);
static_assert(sizeof(Params) == 128 && alignof(Params) == 16);
static_assert(offsetof(Params, direction) == 0 && offsetof(Params, scale) == 12);
static_assert(offsetof(Params, samples) == 16 && offsetof(Params, matrix) == 48);
static_assert(offsetof(Params, nested) == 112);
static_assert(sizeof(Nested) == 16 && alignof(Nested) == 16 && offsetof(Nested, weights) == 4);

inline const ConstantTypeDesc& paramsSchema()
{
    using Kind = slang::TypeReflection::Kind;
    using Scalar = slang::TypeReflection::ScalarType;
    static const ConstantTypeDesc floatType{Kind::Scalar, 4, 4, Scalar::Float32};
    static const ConstantTypeDesc uintType{Kind::Scalar, 4, 4, Scalar::UInt32};
    static const ConstantTypeDesc float3Type{Kind::Vector, 12, 4, Scalar::Float32, 3};
    static const ConstantTypeDesc float4Type{Kind::Vector, 16, 4, Scalar::Float32, 4};
    static const ConstantTypeDesc samplesType{
        .kind = Kind::Array,
        .size = sizeof(Params::samples),
        .alignment = alignof(decltype(Params::samples)),
        .elementCount = 2,
        .elementStride = sizeof(Params::samples[0]),
        .elementType = &float4Type,
    };
    static const ConstantTypeDesc matrixType{
        .kind = Kind::Matrix,
        .size = sizeof(Params::matrix),
        .alignment = alignof(decltype(Params::matrix)),
        .scalar = Scalar::Float32,
        .matrixLayout = SLANG_MATRIX_LAYOUT_ROW_MAJOR,
    };
    static const ConstantFieldDesc nestedFields[]{
        {"tag", offsetof(Nested, tag), &uintType},
        {"weights", offsetof(Nested, weights), &float3Type},
    };
    static const ConstantTypeDesc nestedType{
        .kind = Kind::Struct,
        .size = sizeof(Nested),
        .alignment = alignof(Nested),
        .fields = nestedFields,
        .fieldCount = 2,
    };
    static const ConstantFieldDesc fields[]{
        {"direction", offsetof(Params, direction), &float3Type},
        {"scale", offsetof(Params, scale), &floatType},
        {"samples", offsetof(Params, samples), &samplesType},
        {"matrix", offsetof(Params, matrix), &matrixType},
        {"nested", offsetof(Params, nested), &nestedType},
    };
    static const ConstantTypeDesc params{
        .kind = Kind::Struct,
        .size = sizeof(Params),
        .alignment = alignof(Params),
        .fields = fields,
        .fieldCount = 5,
    };
    return params;
}

} // namespace rhi::testing::flat
