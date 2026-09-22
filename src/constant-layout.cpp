#include "constant-layout.h"

#include <bit>
#include <string_view>

namespace rhi {
namespace {

using Kind = slang::TypeReflection::Kind;
using Scalar = slang::TypeReflection::ScalarType;

constexpr unsigned kMaxSchemaNestingDepth = 64;

SlangResult fail(std::string& diagnostic, const std::string& path, const char* reason)
{
    diagnostic = path + ": " + reason;
    return SLANG_E_INVALID_ARG;
}

bool portableScalar(Scalar scalar)
{
    return scalar == Scalar::Int32 || scalar == Scalar::UInt32 || scalar == Scalar::Float32;
}

SlangResult validate(
    slang::TypeLayoutReflection* layout,
    const ConstantTypeDesc& expected,
    const std::string& path,
    unsigned depth,
    std::string& diagnostic
)
{
    // Reject malformed/cyclic caller schemas before recursing indefinitely.
    if (!layout || depth > kMaxSchemaNestingDepth)
        return fail(diagnostic, path, "missing reflection or excessive schema nesting");
    if (layout->getKind() != expected.kind)
        return fail(diagnostic, path, "type kind mismatch");
    if (!expected.size || expected.size == SLANG_UNBOUNDED_SIZE || expected.size == SLANG_UNKNOWN_SIZE ||
        expected.size != layout->getSize())
        return fail(diagnostic, path, "byte size mismatch or unresolved size");
    if (!std::has_single_bit(expected.alignment) || expected.size % expected.alignment != 0)
        return fail(diagnostic, path, "invalid CPU alignment");

    for (unsigned i = 0; i < layout->getCategoryCount(); ++i)
    {
        if (layout->getCategoryByIndex(i) != slang::ParameterCategory::Uniform)
            return fail(diagnostic, path, "resource or specialization data is not an execution constant");
    }

    switch (expected.kind)
    {
    case Kind::Scalar:
        if (!portableScalar(expected.scalar) || expected.size != 4 || layout->getScalarType() != expected.scalar)
            return fail(diagnostic, path, "expected a matching 32-bit int, uint or float");
        break;
    case Kind::Vector:
        if (!portableScalar(expected.scalar) || layout->getScalarType() != expected.scalar)
            return fail(diagnostic, path, "vector scalar type mismatch or unsupported scalar");
        if (expected.elementCount < 2 || expected.elementCount > 4 ||
            layout->getElementCount() != expected.elementCount || expected.size != 4 * expected.elementCount)
            return fail(diagnostic, path, "vector width mismatch");
        break;
    case Kind::Matrix:
        if (expected.scalar != Scalar::Float32 || layout->getScalarType() != expected.scalar || expected.size != 64 ||
            layout->getRowCount() != 4 || layout->getColumnCount() != 4)
            return fail(diagnostic, path, "portable matrices must be float4x4");
        if ((expected.matrixLayout != SLANG_MATRIX_LAYOUT_ROW_MAJOR &&
             expected.matrixLayout != SLANG_MATRIX_LAYOUT_COLUMN_MAJOR) ||
            layout->getMatrixLayoutMode() != expected.matrixLayout)
            return fail(diagnostic, path, "matrix row/column layout mismatch");
        break;
    case Kind::Array:
        if (!expected.elementType || expected.elementType->kind != Kind::Vector ||
            expected.elementType->elementCount != 4)
            return fail(diagnostic, path, "portable arrays require four-component vector elements");
        if (!std::has_single_bit(expected.elementType->alignment) ||
            expected.alignment < expected.elementType->alignment ||
            expected.elementStride % expected.elementType->alignment != 0)
            return fail(diagnostic, path, "array alignment does not cover its CPU element type");
        if (!expected.elementCount || expected.elementCount == SLANG_UNBOUNDED_SIZE ||
            expected.elementCount == SLANG_UNKNOWN_SIZE || layout->getElementCount() != expected.elementCount)
            return fail(diagnostic, path, "array count mismatch or unresolved count");
        if (expected.elementStride != 16 ||
            layout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM) != expected.elementStride ||
            expected.size % expected.elementStride != 0 ||
            expected.size / expected.elementStride != expected.elementCount)
            return fail(diagnostic, path, "array stride mismatch");
        return validate(layout->getElementTypeLayout(), *expected.elementType, path + "[]", depth + 1, diagnostic);
    case Kind::Struct:
        if (expected.alignment < 16 || expected.size % 16 != 0)
            return fail(diagnostic, path, "portable structs require 16-byte alignment and padded size");
        if (layout->getFieldCount() != expected.fieldCount)
            return fail(diagnostic, path, "field count mismatch");
        if (expected.fieldCount && !expected.fields)
            return fail(diagnostic, path, "invalid field schema");
        for (uint32_t i = 0; i < expected.fieldCount; ++i)
        {
            const auto& field = expected.fields[i];
            if (!field.name || !*field.name || !field.type)
                return fail(diagnostic, path, "invalid field schema");
            auto fieldPath = path + "." + field.name;
            if (field.offset > expected.size || field.type->size > expected.size - field.offset)
                return fail(diagnostic, fieldPath, "field extends beyond CPU struct");
            for (uint32_t j = 0; j < i; ++j)
            {
                const auto& previous = expected.fields[j];
                if (std::string_view(previous.name) == field.name)
                    return fail(diagnostic, fieldPath, "duplicate field schema");
                if (field.offset < previous.offset + previous.type->size &&
                    previous.offset < field.offset + field.type->size)
                    return fail(diagnostic, fieldPath, "overlapping CPU fields");
            }
            if (!std::has_single_bit(field.type->alignment) || expected.alignment < field.type->alignment ||
                field.offset % field.type->alignment != 0)
                return fail(diagnostic, fieldPath, "misaligned CPU field");
            auto index = layout->findFieldIndexByName(field.name);
            if (index < 0)
                return fail(diagnostic, fieldPath, "field not found in target layout");
            auto reflected = layout->getFieldByIndex(unsigned(index));
            if (reflected->getOffset() != field.offset)
                return fail(diagnostic, fieldPath, "field offset mismatch");
            auto targetAlignment = reflected->getTypeLayout()->getAlignment();
            if (targetAlignment <= 0 || expected.alignment < size_t(targetAlignment) ||
                field.offset % size_t(targetAlignment) != 0)
                return fail(diagnostic, fieldPath, "field placement does not satisfy target alignment");
            auto result = validate(reflected->getTypeLayout(), *field.type, fieldPath, depth + 1, diagnostic);
            if (SLANG_FAILED(result))
                return result;
        }
        break;
    default:
        return fail(diagnostic, path, "unsupported execution-constant type");
    }
    return SLANG_OK;
}

} // namespace

SlangResult validateConstantLayout(
    slang::TypeLayoutReflection* layout,
    const ConstantTypeDesc& expected,
    std::string& diagnostic
)
{
    diagnostic.clear();
    return validate(layout, expected, "constants", 0, diagnostic);
}

} // namespace rhi
