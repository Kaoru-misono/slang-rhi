#pragma once

#include <slang-rhi.h>

#include "core/common.h"
#include "core/struct-holder.h"

#include "reference.h"
#include "device-child.h"
#include "rhi-shared-fwd.h"

#include <vector>

namespace rhi {

class PipelineLayout : public IPipelineLayout, public DeviceChild
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    IPipelineLayout* getInterface(const Guid& guid);

public:
    PipelineLayout(Device* device, const PipelineLayoutDesc& desc);
    virtual ~PipelineLayout() override;

    /// Checks every parameter of the program against `m_desc` and records the profile and size of
    /// the execution constants. Reports the first problem, with its parameter path, through the
    /// device callback.
    Result init();

    /// Builds the native layout from what `init` validated and recorded.
    virtual Result initNative() = 0;

    BindingSetLayout* getSetLayout(uint32_t index) const { return m_setLayouts[index]; }

    // IPipelineLayout implementation
    virtual SLANG_NO_THROW const PipelineLayoutDesc& SLANG_MCALL getDesc() override { return m_desc; }
    virtual SLANG_NO_THROW ConstantsProfile SLANG_MCALL getConstantsProfile() override
    {
        return m_constantsProfile;
    }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL getConstantsSize() override { return m_constantsSize; }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL getSetCount() override { return uint32_t(m_setLayouts.size()); }

public:
    PipelineLayoutDesc m_desc;
    StructHolder m_descHolder;
    RefPtr<ShaderProgram> m_program;
    /// Set layouts in reflection space order, which is the order of `m_desc.sets`.
    std::vector<RefPtr<BindingSetLayout>> m_setLayouts;
    ConstantsProfile m_constantsProfile = ConstantsProfile::None;
    uint32_t m_constantsSize = 0;
    /// True when the program reaches resources through descriptor handles, so the native layout
    /// needs the bindless set.
    bool m_bindless = false;
};

} // namespace rhi
