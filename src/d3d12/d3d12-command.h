#pragma once

#include "d3d12-base.h"
#include "d3d12-shader-object.h"
#include "../transient-buffer-heap.h"

namespace rhi::d3d12 {

class CommandQueueImpl : public CommandQueue
{

public:
    ComPtr<ID3D12Device> m_d3dDevice;
    ComPtr<ID3D12CommandQueue> m_d3dQueue;
    ComPtr<ID3D12Fence> m_trackingFence;
    uint32_t m_queueIndex = 0;
    D3D12_COMMAND_LIST_TYPE m_commandListType = D3D12_COMMAND_LIST_TYPE_DIRECT;

    /// Covers sequence assignment through publication; the native queue itself is free-threaded.
    std::mutex m_submitMutex;
    std::mutex m_mutex;
    std::list<RefPtr<CommandBufferImpl>> m_commandBuffersPool;
    std::list<RefPtr<CommandBufferImpl>> m_commandBuffersInFlight;

#if SLANG_RHI_ENABLE_AFTERMATH
    GFSDK_Aftermath_ContextHandle m_aftermathContext;
#endif

    CommandQueueImpl(Device* device, QueueType type);
    ~CommandQueueImpl();

    Result init(uint32_t queueIndex);

    Result createCommandBuffer(CommandBufferImpl** outCommandBuffer);
    Result getOrCreateCommandBuffer(CommandBufferImpl** outCommandBuffer);
    void retireCommandBuffer(CommandBufferImpl* commandBuffer);
    /// Marks the submitted recordings as consumed by `sequence` and retains them until it completes.
    void consumeCommandBuffers(const SubmitDesc& desc, uint64_t sequence);
    virtual void shutdown() override;
    virtual void retireCompletedCommandBuffers(uint64_t completed) override;
    virtual void abandonCommandBuffersAfterDeviceLoss() override;
    virtual uint64_t updateLastFinishedID() override;
    virtual Result waitForSequenceNative(uint64_t sequence, uint64_t timeoutNs) override;

    // ICommandQueue implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL createCommandEncoder(
        const CommandEncoderDesc& desc,
        ICommandEncoder** outEncoder
    ) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL submit(const SubmitDesc& desc) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL waitOnHost() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getTimestampCalibration(TimestampCalibration* outCalibration) override;
};

class CommandEncoderImpl : public CommandEncoder
{
public:
    CommandQueueImpl* m_queue;
    RefPtr<CommandBufferImpl> m_commandBuffer;

    CommandEncoderImpl(Device* device, CommandQueueImpl* queue, const CommandEncoderDesc& desc);
    ~CommandEncoderImpl();

    Result init();

    virtual Result getBindingData(RootShaderObject* rootObject, BindingData*& outBindingData) override;
    virtual Result getComputeBindingData(
        ShaderProgram* program, const void* data, size_t size,
        const ComputeBufferAccess* buffers, uint32_t bufferCount, BindingData*& outBindingData
    ) override;

    // ICommandEncoder implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL finish(
        const CommandBufferDesc& desc,
        ICommandBuffer** outCommandBuffer
    ) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
};

class CommandBufferImpl : public CommandBuffer
{
public:
    CommandQueueImpl* m_queue;
    ComPtr<ID3D12CommandAllocator> m_d3dCommandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_d3dCommandList;
    GPUDescriptorArena m_cbvSrvUavArena;
    GPUDescriptorArena m_samplerArena;
    TransientBufferArena m_constantBufferArena;
    BindingCache m_bindingCache;
    // Nonzero consumes this recording, including a submission that lost the device.
    uint64_t m_submissionID = 0;

#if SLANG_RHI_ENABLE_AFTERMATH
    GFSDK_Aftermath_ContextHandle m_aftermathContext;
    AftermathMarkerTracker m_aftermathMarkerTracker;
#endif

    CommandBufferImpl(Device* device, CommandQueueImpl* queue);
    ~CommandBufferImpl();

    Result init();
    virtual Result reset() override;

    // ICommandBuffer implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
};

} // namespace rhi::d3d12
