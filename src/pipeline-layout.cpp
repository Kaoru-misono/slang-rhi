#include "pipeline-layout.h"

#include "binding-set.h"
#include "constant-layout.h"
#include "rhi-shared.h"
#include "device.h"
#include "shader.h"

#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>

namespace rhi {
namespace {

using Kind = slang::TypeReflection::Kind;
using Scalar = slang::TypeReflection::ScalarType;
using Category = slang::ParameterCategory;

bool hasCategory(slang::VariableLayoutReflection* variable, Category category)
{
    if (variable->getCategory() == category)
        return true;
    for (unsigned i = 0; i < variable->getCategoryCount(); ++i)
    {
        if (variable->getCategoryByIndex(i) == category)
            return true;
    }
    return false;
}

bool isVaryingOnly(slang::VariableLayoutReflection* variable)
{
    if (variable->getCategoryCount() == 0)
        return true;
    for (unsigned i = 0; i < variable->getCategoryCount(); ++i)
    {
        auto category = variable->getCategoryByIndex(i);
        if (category != Category::VaryingInput && category != Category::VaryingOutput)
            return false;
    }
    return true;
}

bool bindingKindFromReflection(slang::TypeLayoutReflection* typeLayout, BindingKind& outKind, const char*& outReason)
{
    outReason = "type is not a supported binding";
    if (!typeLayout || !typeLayout->getType())
        return false;
    if (typeLayout->getKind() == Kind::Interface || typeLayout->getKind() == Kind::Specialized)
    {
        outReason = "program must be fully specialized";
        return false;
    }
    if (typeLayout->getKind() == Kind::SamplerState)
    {
        const char* name = typeLayout->getType()->getName();
        outKind = name && std::string_view(name) == "SamplerComparisonState" ? BindingKind::SamplerComparisonState
                                                                             : BindingKind::SamplerState;
        return true;
    }
    if (typeLayout->getKind() != Kind::Resource)
        return false;

    SlangResourceShape shape = typeLayout->getType()->getResourceShape();
    SlangResourceAccess access = typeLayout->getType()->getResourceAccess();
    auto baseShape = shape & SLANG_RESOURCE_BASE_SHAPE_MASK;
    if (shape & (SLANG_TEXTURE_MULTISAMPLE_FLAG | SLANG_TEXTURE_FEEDBACK_FLAG | SLANG_TEXTURE_COMBINED_FLAG |
                 SLANG_TEXTURE_SHADOW_FLAG))
    {
        outReason = "unsupported texture flavour";
        return false;
    }
    if (baseShape == SLANG_ACCELERATION_STRUCTURE)
    {
        outKind = BindingKind::RaytracingAccelerationStructure;
        return true;
    }
    if (access != SLANG_RESOURCE_ACCESS_READ && access != SLANG_RESOURCE_ACCESS_READ_WRITE)
    {
        outReason = "unsupported resource access";
        return false;
    }
    bool writable = access == SLANG_RESOURCE_ACCESS_READ_WRITE;
    bool array = (shape & SLANG_TEXTURE_ARRAY_FLAG) != 0;
    switch (baseShape)
    {
    case SLANG_TEXTURE_1D:
        if (array)
        {
            outReason = "Texture1DArray bindings are not supported";
            return false;
        }
        outKind = writable ? BindingKind::RWTexture1D : BindingKind::Texture1D;
        return true;
    case SLANG_TEXTURE_2D:
        outKind = array ? (writable ? BindingKind::RWTexture2DArray : BindingKind::Texture2DArray)
                        : (writable ? BindingKind::RWTexture2D : BindingKind::Texture2D);
        return true;
    case SLANG_TEXTURE_3D:
        if (array)
        {
            outReason = "Texture3D array bindings are not supported";
            return false;
        }
        outKind = writable ? BindingKind::RWTexture3D : BindingKind::Texture3D;
        return true;
    case SLANG_TEXTURE_CUBE:
        if (writable)
        {
            outReason = "read-write cube texture bindings are not supported";
            return false;
        }
        outKind = array ? BindingKind::TextureCubeArray : BindingKind::TextureCube;
        return true;
    case SLANG_TEXTURE_BUFFER:
        outKind = writable ? BindingKind::RWBuffer : BindingKind::Buffer;
        return true;
    case SLANG_STRUCTURED_BUFFER:
        outKind = writable ? BindingKind::RWStructuredBuffer : BindingKind::StructuredBuffer;
        return true;
    case SLANG_BYTE_ADDRESS_BUFFER:
        outKind = writable ? BindingKind::RWByteAddressBuffer : BindingKind::ByteAddressBuffer;
        return true;
    default:
        outReason = "unsupported resource shape";
        return false;
    }
}

bool bufferFormatKindFromReflection(slang::TypeLayoutReflection* elementTypeLayout, FormatKind& outKind)
{
    if (!elementTypeLayout)
        return false;
    switch (elementTypeLayout->getScalarType())
    {
    case Scalar::Float32:
        outKind = FormatKind::Float;
        return true;
    case Scalar::Int8:
    case Scalar::UInt8:
    case Scalar::Int16:
    case Scalar::UInt16:
    case Scalar::Int32:
    case Scalar::UInt32:
    case Scalar::Int64:
    case Scalar::UInt64:
        outKind = FormatKind::Integer;
        return true;
    default:
        return false;
    }
}

struct ReflectedSet
{
    std::string path;
    slang::TypeLayoutReflection* elementTypeLayout;
};

struct PipelineLayoutReflectionValidator
{
    explicit PipelineLayoutReflectionValidator(PipelineLayout* layout)
        : layout(layout)
    {
    }

    PipelineLayout* layout;
    std::vector<ReflectedSet> sets;
    std::string constantsPath;
    slang::TypeLayoutReflection* constantsTypeLayout = nullptr;
    ConstantsProfile constantsProfile = ConstantsProfile::None;

    Result fail(const std::string& path, const std::string& reason)
    {
        layout->getDevice()->printError(
            "Pipeline layout '%s', path '%s': %s",
            layout->m_desc.label ? layout->m_desc.label : "<unnamed>",
            path.c_str(),
            reason.c_str()
        );
        return SLANG_E_INVALID_ARG;
    }

    Result collectSet(const std::string& path, slang::TypeLayoutReflection* elementTypeLayout, unsigned depth)
    {
        if (depth > 64)
            return fail(path, "parameter block nesting exceeds 64");
        if (!elementTypeLayout || elementTypeLayout->getKind() != Kind::Struct)
            return fail(path, "parameter block element type must be a struct");
        sets.emplace_back(ReflectedSet{path, elementTypeLayout});
        for (unsigned i = 0; i < elementTypeLayout->getFieldCount(); ++i)
        {
            auto* field = elementTypeLayout->getFieldByIndex(i);
            auto* fieldTypeLayout = field->getTypeLayout();
            std::string fieldPath = path + "." + (field->getName() ? field->getName() : "<unnamed>");
            if (!fieldTypeLayout)
                return fail(fieldPath, "field type layout is missing");
            if (fieldTypeLayout->getKind() == Kind::ParameterBlock)
                SLANG_RETURN_ON_FAIL(collectSet(fieldPath, fieldTypeLayout->getElementTypeLayout(), depth + 1));
        }
        return SLANG_OK;
    }

    Result collectConstants(const std::string& path, slang::TypeLayoutReflection* typeLayout, ConstantsProfile profile)
    {
        if (constantsProfile != ConstantsProfile::None)
            return fail(path, "a program may declare at most one execution constant block");
        constantsPath = path;
        constantsTypeLayout = typeLayout;
        constantsProfile = profile;
        return SLANG_OK;
    }

    Result collect(slang::ProgramLayout* programLayout)
    {
        for (unsigned i = 0; i < programLayout->getParameterCount(); ++i)
        {
            auto* param = programLayout->getParameterByIndex(i);
            const char* name = param->getName() ? param->getName() : "<unnamed>";
            auto* typeLayout = param->getTypeLayout();
            if (!typeLayout)
                return fail(name, "parameter type layout is missing");
            if (typeLayout->getKind() == Kind::ParameterBlock)
            {
                SLANG_RETURN_ON_FAIL(collectSet(name, typeLayout->getElementTypeLayout(), 1));
            }
            else if (typeLayout->getKind() == Kind::ConstantBuffer)
            {
                auto profile = hasCategory(param, Category::PushConstantBuffer) ? ConstantsProfile::Inline
                                                                                : ConstantsProfile::Buffered;
                SLANG_RETURN_ON_FAIL(collectConstants(name, typeLayout->getElementTypeLayout(), profile));
            }
            else
            {
                return fail(name, "loose global parameters are not allowed; resources must live in a ParameterBlock");
            }
        }
        for (unsigned i = 0; i < programLayout->getEntryPointCount(); ++i)
        {
            auto* entryPoint = programLayout->getEntryPointByIndex(i);
            std::string entryPointPath = entryPoint->getName() ? entryPoint->getName() : "<unnamed>";
            bool hasUniformBlock = false;
            for (unsigned j = 0; j < entryPoint->getParameterCount(); ++j)
            {
                auto* param = entryPoint->getParameterByIndex(j);
                const char* name = param->getName() ? param->getName() : "<unnamed>";
                auto* typeLayout = param->getTypeLayout();
                if (!typeLayout)
                    return fail(entryPointPath + "." + name, "parameter type layout is missing");
                if (typeLayout->getKind() == Kind::ParameterBlock)
                {
                    SLANG_RETURN_ON_FAIL(collectSet(name, typeLayout->getElementTypeLayout(), 1));
                }
                else if (isVaryingOnly(param))
                {
                    // Varying and system-value inputs occupy no binding, so they are not parameters
                    // the layout has to describe.
                }
                else if (param->getCategoryCount() == 1 && param->getCategoryByIndex(0) == Category::Uniform)
                {
                    if (!hasUniformBlock)
                    {
                        auto* varLayout = entryPoint->getVarLayout();
                        SLANG_RETURN_ON_FAIL(collectConstants(
                            entryPointPath,
                            varLayout ? varLayout->getTypeLayout() : nullptr,
                            ConstantsProfile::Inline
                        ));
                        hasUniformBlock = true;
                    }
                }
                else
                {
                    return fail(
                        entryPointPath + "." + name,
                        "entry-point parameter must be a ParameterBlock or uniform"
                    );
                }
            }
        }
        return SLANG_OK;
    }

    Result validateSet(const ReflectedSet& set, BindingSetLayout* setLayout)
    {
        uint32_t entryIndex = 0;
        for (unsigned i = 0; i < set.elementTypeLayout->getFieldCount(); ++i)
        {
            auto* field = set.elementTypeLayout->getFieldByIndex(i);
            auto* typeLayout = field->getTypeLayout();
            const char* name = field->getName();
            std::string path = set.path + "." + (name ? name : "<unnamed>");
            if (typeLayout->getKind() == Kind::ParameterBlock)
                continue;
            if (typeLayout->getKind() == Kind::Interface || typeLayout->getKind() == Kind::Specialized ||
                hasCategory(field, Category::ExistentialTypeParam) ||
                hasCategory(field, Category::ExistentialObjectParam))
                return fail(path, "program must be fully specialized");
            if (hasCategory(field, Category::Uniform))
                return fail(path, "ordinary data is not allowed inside a parameter block");
            size_t count = 1;
            if (typeLayout->getKind() == Kind::Array)
            {
                count = typeLayout->getElementCount();
                if (!count || count == SLANG_UNBOUNDED_SIZE || count == SLANG_UNKNOWN_SIZE)
                    return fail(path, "binding array count must be fixed and known");
                typeLayout = typeLayout->getElementTypeLayout();
            }
            BindingKind kind;
            const char* reason;
            if (!bindingKindFromReflection(typeLayout, kind, reason))
                return fail(path, reason);
            if (entryIndex >= setLayout->m_desc.entryCount)
                return fail(path, "binding entry is missing from the set layout");
            const auto& entry = setLayout->m_desc.entries[entryIndex];
            if (entry.kind != kind)
                return fail(path, "binding kind does not match reflection");
            if (entry.count != count)
                return fail(path, "binding count does not match reflection");
            if (name && entry.name && std::string_view(name) != entry.name)
            {
                layout->getDevice()->printWarning(
                    "Pipeline layout '%s', path '%s': binding name '%s' does not match reflected name '%s'",
                    layout->m_desc.label ? layout->m_desc.label : "<unnamed>",
                    path.c_str(),
                    entry.name,
                    name
                );
            }
            auto& reflection = setLayout->m_reflection[entryIndex];
            if (kind == BindingKind::StructuredBuffer || kind == BindingKind::RWStructuredBuffer)
            {
                auto* elementTypeLayout = typeLayout->getElementTypeLayout();
                size_t stride = elementTypeLayout ? elementTypeLayout->getSize() : 0;
                if (!stride || stride == SLANG_UNBOUNDED_SIZE || stride == SLANG_UNKNOWN_SIZE ||
                    stride > std::numeric_limits<uint32_t>::max())
                    return fail(path, "structured buffer stride must be fixed, known and fit in uint32_t");
                if (reflection.structuredBufferStride && reflection.structuredBufferStride != stride)
                    return fail(path, "structured buffer stride conflicts with previously recorded reflection");
                reflection.structuredBufferStride = uint32_t(stride);
            }
            else if (kind == BindingKind::Buffer || kind == BindingKind::RWBuffer)
            {
                FormatKind formatKind;
                if (bufferFormatKindFromReflection(typeLayout->getElementTypeLayout(), formatKind))
                {
                    if (reflection.formatKindKnown && reflection.formatKind != formatKind)
                        return fail(path, "buffer format kind conflicts with previously recorded reflection");
                    reflection.formatKind = formatKind;
                    reflection.formatKindKnown = true;
                }
            }
            ++entryIndex;
        }
        if (entryIndex != setLayout->m_desc.entryCount)
        {
            const char* name = setLayout->m_desc.entries[entryIndex].name;
            std::string path = set.path + "." + (name ? name : "<extra entry>");
            return fail(path, "binding entry is not a field of the parameter block");
        }
        return SLANG_OK;
    }

    Result validateSets()
    {
        const auto& desc = layout->m_desc;
        std::unordered_set<std::string_view> reflectedPaths;
        for (const auto& set : sets)
            reflectedPaths.emplace(set.path);
        for (uint32_t i = 0; i < desc.setCount; ++i)
        {
            if (!reflectedPaths.contains(desc.sets[i].path))
                return fail(
                    desc.sets[i].path,
                    "set '" + std::string(desc.sets[i].path) + "' is not a parameter block of the program"
                );
        }
        for (size_t i = 0; i < sets.size(); ++i)
        {
            const auto& set = sets[i];
            if (i >= desc.setCount)
                return fail(set.path, "set '" + set.path + "' is missing from the layout");
            if (set.path != desc.sets[i].path)
            {
                return fail(
                    set.path,
                    "set '" + std::string(desc.sets[i].path) + "' is out of order; expected set '" + set.path + "'"
                );
            }
        }
        for (size_t i = 0; i < sets.size(); ++i)
            SLANG_RETURN_ON_FAIL(validateSet(sets[i], layout->m_setLayouts[i]));
        return SLANG_OK;
    }

    Result validateConstants()
    {
        const auto* constants = layout->m_desc.constants;
        if (!constants && constantsProfile != ConstantsProfile::None)
            return fail(constantsPath, "execution constant block is missing from the layout");
        if (constants && constantsProfile == ConstantsProfile::None)
            return fail(
                "constants",
                "layout declares execution constants but the program has no execution constant block"
            );
        if (!constants)
            return SLANG_OK;
        std::string diagnostic;
        Result result = validateConstantLayout(constantsTypeLayout, *constants, diagnostic);
        if (SLANG_FAILED(result))
        {
            layout->getDevice()->printError("%s", diagnostic.c_str());
            return result;
        }
        if (constantsProfile == ConstantsProfile::Inline &&
            constants->size > layout->getDevice()->getInlineConstantsSizeLimit())
            return fail(constantsPath, "inline execution constant block exceeds the device size limit");
        if (constants->size > std::numeric_limits<uint32_t>::max())
            return fail(constantsPath, "execution constant block size does not fit in uint32_t");
        layout->m_constantsProfile = constantsProfile;
        layout->m_constantsSize = uint32_t(constants->size);
        return SLANG_OK;
    }
};

} // namespace

// ----------------------------------------------------------------------------
// PipelineLayout
// ----------------------------------------------------------------------------

IPipelineLayout* PipelineLayout::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == IPipelineLayout::getTypeGuid())
        return static_cast<IPipelineLayout*>(this);
    return nullptr;
}

PipelineLayout::PipelineLayout(Device* device, const PipelineLayoutDesc& desc)
    : DeviceChild(device)
    , m_desc(desc)
{
    auto* sets = const_cast<PipelineLayoutSetDesc*>(m_desc.sets);
    m_descHolder.holdList(sets, m_desc.setCount);
    if (sets)
    {
        for (uint32_t i = 0; i < m_desc.setCount; ++i)
            m_descHolder.holdString(sets[i].path);
    }
    m_desc.sets = sets;
    m_descHolder.holdString(m_desc.label);
}

PipelineLayout::~PipelineLayout()
{
}

Result PipelineLayout::init()
{
    auto fail = [&](const char* path, const char* reason) -> Result
    {
        getDevice()->printError(
            "Pipeline layout '%s', path '%s': %s", m_desc.label ? m_desc.label : "<unnamed>", path, reason
        );
        return SLANG_E_INVALID_ARG;
    };
    if (!m_desc.program)
        return fail("program", "program must be non-null");
    auto* program = checked_cast<ShaderProgram*>(m_desc.program);
    if (program->getDevice() != getDevice())
        return fail("program", "program belongs to another device");
    if (program->isSpecializable())
        return fail("program", "program must be fully specialized");
    if (m_desc.setCount && !m_desc.sets)
        return fail("sets", "sets must be non-null when setCount is nonzero");

    std::unordered_set<std::string_view> paths;
    for (uint32_t i = 0; i < m_desc.setCount; ++i)
    {
        const auto& set = m_desc.sets[i];
        if (!set.path || !*set.path)
            return fail(set.path ? set.path : "<null>", "set path must be non-empty");
        if (!set.layout)
            return fail(set.path, "set layout must be non-null");
        auto* layout = checked_cast<BindingSetLayout*>(set.layout);
        if (layout->getDevice() != getDevice())
            return fail(set.path, "set layout belongs to another device");
        if (!paths.emplace(set.path).second)
            return fail(set.path, "duplicate set path");
    }

    m_program = program;
    for (uint32_t i = 0; i < m_desc.setCount; ++i)
        m_setLayouts.emplace_back(checked_cast<BindingSetLayout*>(m_desc.sets[i].layout));
    if (!m_program->linkedProgram)
        return fail("program", "linked program must be non-null");
    slang::ProgramLayout* programLayout = m_program->linkedProgram->getLayout();
    if (!programLayout)
        return fail("program", "program reflection layout must be non-null");

    PipelineLayoutReflectionValidator validator{this};
    SLANG_RETURN_ON_FAIL(validator.collect(programLayout));
    SLANG_RETURN_ON_FAIL(validator.validateSets());
    SLANG_RETURN_ON_FAIL(validator.validateConstants());
    m_bindless = getDevice()->hasFeature(Feature::Bindless);
    // The constant descriptors are a caller-owned tree StructHolder cannot copy, and validation
    // is the only thing that reads them, so getDesc() reports none rather than a dangling one.
    m_desc.constants = nullptr;
    return SLANG_OK;
}

} // namespace rhi
