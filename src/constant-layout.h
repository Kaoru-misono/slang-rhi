#pragma once

#include <slang-rhi.h>

#include <string>

namespace rhi {

// Checks the portable POD subset against one fully specialized target layout.
// This validates bytes only; native slots/profile and resource-set identity are separate contracts.
SlangResult validateConstantLayout(
    slang::TypeLayoutReflection* layout,
    const ConstantTypeDesc& expected,
    std::string& diagnostic
);

} // namespace rhi
