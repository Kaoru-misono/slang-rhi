#pragma once

#include "debug-base.h"

namespace rhi::debug {

class DebugCommandQueue : public DebugObject<ICommandQueue>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;
    ICommandQueue* getInterface(const Guid& guid);

    SLANG_RHI_DEBUG_OBJECT_CONSTRUCTOR(DebugCommandQueue);

public:
    // ICommandQueue implementation
    virtual SLANG_NO_THROW QueueType SLANG_MCALL getType() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createCommandEncoder(
        const CommandEncoderDesc& desc,
        ICommandEncoder** outEncoder
    ) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL submit(const SubmitDesc& desc) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL waitOnHost() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getCompletedSequence(uint64_t* outSequence) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL waitForSequence(uint64_t sequence, uint64_t timeoutNs) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getTimestampCalibration(TimestampCalibration* outCalibration) override;
};

} // namespace rhi::debug
