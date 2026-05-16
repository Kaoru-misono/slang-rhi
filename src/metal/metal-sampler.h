#pragma once

#include "metal-base.h"

namespace rhi::metal {

class SamplerImpl : public Sampler
{
public:
    NS::SharedPtr<MTL::SamplerState> m_samplerState;
    DescriptorHandle m_descriptorHandle = {};

    SamplerImpl(Device* device, const SamplerDesc& desc);
    ~SamplerImpl();

    virtual void deleteThis() override;

    Result init();

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getDescriptorHandle(DescriptorHandle* outHandle) override;
};

} // namespace rhi::metal
