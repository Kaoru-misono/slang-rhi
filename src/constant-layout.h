#pragma once

#include <slang-rhi.h>

#include <string>

namespace rhi {

// Checks one reflected execution constant block against the portable subset. Passing means a C
// struct sharing the shader's declarations has the same byte layout, so no CPU type description is
// needed: the caller only supplies its size.
SlangResult validateConstantLayout(slang::TypeLayoutReflection* layout, std::string& diagnostic);

} // namespace rhi
