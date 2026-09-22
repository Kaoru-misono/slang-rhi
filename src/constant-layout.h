#pragma once

#include <slang.h>

#include <cstddef>
#include <span>
#include <string>

namespace rhi {

struct ConstantTypeDesc;

struct ConstantFieldDesc
{
    const char* name;
    size_t offset;
    const ConstantTypeDesc* type;
};

// Describes CPU storage, independently of the reflection being checked.
// All referenced descriptors need only live for the duration of validation.
struct ConstantTypeDesc
{
    slang::TypeReflection::Kind kind = slang::TypeReflection::Kind::None;
    size_t size = 0;
    size_t alignment = 1;
    slang::TypeReflection::ScalarType scalar = slang::TypeReflection::ScalarType::None;
    size_t elementCount = 0;
    size_t elementStride = 0;
    const ConstantTypeDesc* elementType = nullptr;
    SlangMatrixLayoutMode matrixLayout = SLANG_MATRIX_LAYOUT_MODE_UNKNOWN;
    std::span<const ConstantFieldDesc> fields;
};

// Checks the portable POD subset against one fully specialized target layout.
// This validates bytes only; native slots/profile and resource-set identity are separate contracts.
SlangResult validateConstantLayout(
    slang::TypeLayoutReflection* layout,
    const ConstantTypeDesc& expected,
    std::string& diagnostic
);

} // namespace rhi
