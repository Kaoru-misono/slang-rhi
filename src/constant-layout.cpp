#include "constant-layout.h"

namespace rhi {
namespace {

using Kind = slang::TypeReflection::Kind;
using Scalar = slang::TypeReflection::ScalarType;

constexpr unsigned kMaxNestingDepth = 64;
constexpr size_t kRowSize = 16;

SlangResult fail(std::string& diagnostic, const std::string& path, const char* reason)
{
    diagnostic = path + ": " + reason;
    return SLANG_E_INVALID_ARG;
}

bool portableScalar(Scalar scalar)
{
    return scalar == Scalar::Int32 || scalar == Scalar::UInt32 || scalar == Scalar::Float32;
}

/// Anything the target layouts give a 16-byte alignment to; the C struct only agrees when the
/// shader author already placed it on a row boundary.
bool startsOnRow(slang::TypeLayoutReflection* layout)
{
    switch (layout->getKind())
    {
    case Kind::Vector:
        return layout->getElementCount() == 4;
    case Kind::Matrix:
    case Kind::Array:
    case Kind::Struct:
        return true;
    default:
        return false;
    }
}

bool isVector(slang::TypeLayoutReflection* layout, size_t componentCount)
{
    return layout->getKind() == Kind::Vector && layout->getElementCount() == componentCount;
}

/// A float3 leaves four bytes of its row free, and only a scalar may be packed into them.
bool scalarFollows(slang::TypeLayoutReflection* structLayout, unsigned fieldIndex)
{
    if (fieldIndex + 1 == structLayout->getFieldCount())
        return true;
    auto* next = structLayout->getFieldByIndex(fieldIndex + 1)->getTypeLayout();
    return next && next->getKind() == Kind::Scalar;
}

struct Validator
{
    std::string& diagnostic;

    SlangResult requireUniform(slang::TypeLayoutReflection* layout, const std::string& path)
    {
        for (unsigned i = 0; i < layout->getCategoryCount(); ++i)
        {
            if (layout->getCategoryByIndex(i) != slang::ParameterCategory::Uniform)
                return fail(diagnostic, path, "resource or specialization data is not an execution constant");
        }
        return SLANG_OK;
    }

    /// `outSize` is what the matching 4-byte-packed C type occupies. A reflected size or offset that
    /// differs from it means the target inserted padding the CPU struct does not have.
    SlangResult validateType(
        slang::TypeLayoutReflection* layout,
        const std::string& path,
        unsigned depth,
        size_t& outSize
    )
    {
        if (!layout || depth > kMaxNestingDepth)
            return fail(diagnostic, path, "missing reflection or excessive nesting");
        SLANG_RETURN_ON_FAIL(requireUniform(layout, path));
        switch (layout->getKind())
        {
        case Kind::Scalar:
            if (!portableScalar(layout->getScalarType()))
                return fail(diagnostic, path, "expected a 32-bit int, uint or float");
            outSize = 4;
            break;
        case Kind::Vector:
        {
            if (!portableScalar(layout->getScalarType()))
                return fail(diagnostic, path, "expected a vector of 32-bit int, uint or float");
            size_t componentCount = layout->getElementCount();
            if (componentCount < 2 || componentCount > 4)
                return fail(diagnostic, path, "portable vectors have two to four components");
            outSize = 4 * componentCount;
            break;
        }
        case Kind::Matrix:
            if (layout->getScalarType() != Scalar::Float32 || layout->getRowCount() != 4 ||
                layout->getColumnCount() != 4)
                return fail(diagnostic, path, "the only portable matrix is float4x4");
            if (layout->getMatrixLayoutMode() != SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
                return fail(diagnostic, path, "portable matrices are column-major");
            outSize = 4 * kRowSize;
            break;
        case Kind::Array:
        {
            auto* elementLayout = layout->getElementTypeLayout();
            if (!elementLayout || !isVector(elementLayout, 4))
                return fail(diagnostic, path, "portable arrays have four-component vector elements");
            size_t elementCount = layout->getElementCount();
            if (!elementCount || elementCount == SLANG_UNBOUNDED_SIZE || elementCount == SLANG_UNKNOWN_SIZE)
                return fail(diagnostic, path, "array element count must be fixed and known");
            if (layout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM) != kRowSize)
                return fail(diagnostic, path, "portable array elements are 16 bytes apart");
            size_t elementSize = 0;
            SLANG_RETURN_ON_FAIL(validateType(elementLayout, path + "[]", depth + 1, elementSize));
            outSize = kRowSize * elementCount;
            break;
        }
        case Kind::Struct:
            return validateStruct(layout, path, depth, outSize);
        default:
            return fail(diagnostic, path, "unsupported execution-constant type");
        }
        if (layout->getSize() != outSize)
            return fail(diagnostic, path, "reflected size does not match the packed C layout");
        return SLANG_OK;
    }

    SlangResult validateStruct(
        slang::TypeLayoutReflection* layout,
        const std::string& path,
        unsigned depth,
        size_t& outSize
    )
    {
        size_t offset = 0;
        for (unsigned i = 0; i < layout->getFieldCount(); ++i)
        {
            auto* field = layout->getFieldByIndex(i);
            const char* name = field->getName();
            std::string fieldPath = path + "." + (name ? name : "<unnamed>");
            auto* fieldLayout = field->getTypeLayout();
            size_t fieldSize = 0;
            SLANG_RETURN_ON_FAIL(validateType(fieldLayout, fieldPath, depth + 1, fieldSize));
            if (field->getOffset() != offset)
                return fail(diagnostic, fieldPath, "reflected offset does not match the packed C layout");
            if (startsOnRow(fieldLayout) && offset % kRowSize != 0)
                return fail(
                    diagnostic,
                    fieldPath,
                    "four-component vectors, arrays, matrices and nested structs start on a 16-byte boundary"
                );
            if (isVector(fieldLayout, 3) && !scalarFollows(layout, i))
                return fail(
                    diagnostic,
                    fieldPath,
                    "a three-component vector is followed by a 4-byte scalar or ends its struct"
                );
            offset += fieldSize;
        }
        if (offset % kRowSize != 0)
            return fail(diagnostic, path, "struct size must be a multiple of 16 bytes");
        if (layout->getSize() != offset)
            return fail(diagnostic, path, "reflected size does not match the packed C layout");
        outSize = offset;
        return SLANG_OK;
    }
};

} // namespace

SlangResult validateConstantLayout(slang::TypeLayoutReflection* layout, std::string& diagnostic)
{
    diagnostic.clear();
    const std::string path = "constants";
    if (!layout)
        return fail(diagnostic, path, "missing reflection");
    if (layout->getKind() != Kind::Struct)
        return fail(diagnostic, path, "an execution constant block must be a struct");
    Validator validator{diagnostic};
    size_t size = 0;
    return validator.validateType(layout, path, 0, size);
}

} // namespace rhi
