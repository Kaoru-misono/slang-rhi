#pragma once

#include "vk-base.h"
#include "vk-shader-object.h"
#include "../transient-buffer-heap.h"

namespace rhi::vk {

class CommandQueueImpl : public CommandQueue
{
public:
    VulkanApi& m_api;
    VkQueue m_queue;
    std::mutex* m_nativeQueueMutex = nullptr;
    uint32_t m_queueFamilyIndex;

    // Set by the surface for synchronization.
    struct
    {
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
        VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
    } m_surfaceSync;

    VkSemaphore m_trackingSemaphore = VK_NULL_HANDLE;

    std::mutex m_mutex;
    std::list<RefPtr<CommandBufferImpl>> m_commandBuffersPool;
    std::list<RefPtr<CommandBufferImpl>> m_commandBuffersInFlight;

    CommandQueueImpl(Device* device, QueueType type);
    ~CommandQueueImpl();

    void init(VkQueue queue, uint32_t queueFamilyIndex);
    void shutdown();

    Result createCommandBuffer(CommandBufferImpl** outCommandBuffer);
    Result getOrCreateCommandBuffer(CommandBufferImpl** outCommandBuffer);
    void retireCommandBuffer(CommandBufferImpl* commandBuffer);
    void retireCommandBuffers();
    /// Marks the submitted recordings as consumed by `sequence` and retains them until it completes.
    void consumeCommandBuffers(const SubmitDesc& desc, uint64_t sequence);
    virtual void retireCompletedCommandBuffers(uint64_t completed) override;
    virtual void abandonCommandBuffersAfterDeviceLoss() override;
    virtual uint64_t updateLastFinishedID() override;

    // ICommandQueue implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL createCommandEncoder(
        const CommandEncoderDesc& desc,
        ICommandEncoder** outEncoder
    ) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL submit(const SubmitDesc& desc) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL waitOnHost() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL waitForSequence(uint64_t sequence, uint64_t timeoutNs) override;
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
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    TransientBufferArena m_constantBufferArena;
    DescriptorSetAllocator m_descriptorSetAllocator;
    BindingCache m_bindingCache;
    // Nonzero consumes this recording, including a submission that lost the device.
    uint64_t m_submissionID = 0;

#if SLANG_RHI_ENABLE_AFTERMATH
    AftermathMarkerTracker m_aftermathMarkerTracker;
#endif

    CommandBufferImpl(Device* device, CommandQueueImpl* queue);
    ~CommandBufferImpl();

    Result init();
    virtual Result reset() override;

    // ICommandBuffer implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
};

} // namespace rhi::vk
