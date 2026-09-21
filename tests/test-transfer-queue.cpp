#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

// Order submissions across queues; ownership barriers are recorded by the caller.
static void syncQueues(IDevice* device, ICommandQueue* sourceQueue, ICommandQueue* destinationQueue)
{
    ComPtr<IFence> syncFence;
    REQUIRE_CALL(device->createFence({}, syncFence.writeRef()));

    // Signal on the producer queue.
    IFence* sf = syncFence.get();
    uint64_t sv = 1;
    SubmitDesc signalDesc = {};
    signalDesc.signalFences = &sf;
    signalDesc.signalFenceValues = &sv;
    signalDesc.signalFenceCount = 1;
    REQUIRE_CALL(sourceQueue->submit(signalDesc));

    // Wait on the consumer queue.
    SubmitDesc waitDesc = {};
    waitDesc.waitFences = &sf;
    waitDesc.waitFenceValues = &sv;
    waitDesc.waitFenceCount = 1;
    REQUIRE_CALL(destinationQueue->submit(waitDesc));

    REQUIRE_CALL(destinationQueue->waitOnHost());
}

// Test that getQueue(QueueType::Transfer) succeeds on D3D12 and Vulkan.
GPU_TEST_CASE("transfer-queue-create", D3D12 | Vulkan)
{
    auto queue = device->getQueue(QueueType::Transfer);
    REQUIRE(queue != nullptr);
    CHECK(queue->getType() == QueueType::Transfer);
}

// Test that getQueue(QueueType::Transfer) returns null on backends that don't support it.
GPU_TEST_CASE("transfer-queue-unsupported", CPU | D3D11)
{
    auto queue = device->getQueue(QueueType::Transfer);
    CHECK(queue == nullptr);
}

// Test copying a buffer using the transfer queue.
GPU_TEST_CASE("transfer-queue-copy", D3D12 | Vulkan)
{
    auto transferQueue = device->getQueue(QueueType::Transfer);
    REQUIRE(transferQueue != nullptr);

    const int numberCount = 4;
    float srcData[] = {1.0f, 2.0f, 3.0f, 4.0f};

    BufferDesc bufferDesc = {};
    bufferDesc.size = numberCount * sizeof(float);
    bufferDesc.usage = BufferUsage::CopySource | BufferUsage::CopyDestination;
    bufferDesc.defaultState = ResourceState::CopySource;
    bufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> srcBuffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)srcData, srcBuffer.writeRef()));

    bufferDesc.defaultState = ResourceState::CopyDestination;
    ComPtr<IBuffer> dstBuffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, dstBuffer.writeRef()));

    // Ensure the graphics queue's internal device queue is fully flushed before using
    // the transfer queue. createBuffer with initial data uses the internal device queue
    // for uploads, and its pending command buffer state can interfere with cross-queue
    // synchronization during readBuffer.
    device->getQueue(QueueType::Graphics)->waitOnHost();

    {
        auto encoder = transferQueue->createCommandEncoder();
        encoder->setBufferState(srcBuffer, ResourceState::CopySource);
        encoder->setBufferState(dstBuffer, ResourceState::CopyDestination);
        encoder->copyBuffer(dstBuffer, 0, srcBuffer, 0, numberCount * sizeof(float));
        transferQueue->submit(encoder->finish());
        transferQueue->waitOnHost();
    }

    // readBuffer uses the graphics queue internally — need GPU-GPU sync from transfer queue.
    syncQueues(device, transferQueue, device->getQueue(QueueType::Graphics));

    compareComputeResult(device, dstBuffer, makeArray<float>(1.0f, 2.0f, 3.0f, 4.0f));
}

// Test using all three queues simultaneously.
GPU_TEST_CASE("transfer-queue-all-queues", D3D12 | Vulkan)
{
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    auto computeQueue = device->getQueue(QueueType::Compute);
    auto transferQueue = device->getQueue(QueueType::Transfer);
    REQUIRE(graphicsQueue != nullptr);
    REQUIRE(computeQueue != nullptr);
    REQUIRE(transferQueue != nullptr);

    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    const int numberCount = 4;
    float initialDataA[] = {0.0f, 1.0f, 2.0f, 3.0f};
    float initialDataB[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float copyData[] = {100.0f, 200.0f, 300.0f, 400.0f};

    BufferDesc computeBufferDesc = {};
    computeBufferDesc.size = numberCount * sizeof(float);
    computeBufferDesc.format = Format::Undefined;
    computeBufferDesc.elementSize = sizeof(float);
    computeBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                              BufferUsage::CopyDestination | BufferUsage::CopySource;
    computeBufferDesc.defaultState = ResourceState::UnorderedAccess;
    computeBufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> bufferA;
    REQUIRE_CALL(device->createBuffer(computeBufferDesc, (void*)initialDataA, bufferA.writeRef()));
    ComPtr<IBuffer> bufferB;
    REQUIRE_CALL(device->createBuffer(computeBufferDesc, (void*)initialDataB, bufferB.writeRef()));

    BufferDesc copyBufferDesc = {};
    copyBufferDesc.size = numberCount * sizeof(float);
    copyBufferDesc.usage = BufferUsage::CopySource | BufferUsage::CopyDestination;
    copyBufferDesc.defaultState = ResourceState::CopySource;
    copyBufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> copySrc;
    REQUIRE_CALL(device->createBuffer(copyBufferDesc, (void*)copyData, copySrc.writeRef()));
    copyBufferDesc.defaultState = ResourceState::CopyDestination;
    ComPtr<IBuffer> copyDst;
    REQUIRE_CALL(device->createBuffer(copyBufferDesc, nullptr, copyDst.writeRef()));

    // Initial uploads run on graphics. Host waits after dispatch cannot order
    // those uploads before consumers on the other queues or transfer ownership.
    {
        auto encoder = graphicsQueue->createCommandEncoder();
        encoder->releaseBufferForQueue(bufferB, ResourceState::UnorderedAccess, QueueType::Compute);
        encoder->releaseBufferForQueue(copySrc, ResourceState::CopySource, QueueType::Transfer);
        encoder->releaseBufferForQueue(copyDst, ResourceState::CopyDestination, QueueType::Transfer);
        REQUIRE_CALL(graphicsQueue->submit(encoder->finish()));
    }
    syncQueues(device, graphicsQueue, computeQueue);
    syncQueues(device, graphicsQueue, transferQueue);

    // Dispatch on graphics queue
    {
        auto encoder = graphicsQueue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        auto rootObject = pass->bindPipeline(pipeline);
        ShaderCursor shaderCursor(rootObject);
        shaderCursor["buffer"].setBinding(bufferA);
        float value = 10.f;
        shaderCursor["value"].setData(value);
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        graphicsQueue->submit(encoder->finish());
    }

    // Dispatch on compute queue
    {
        auto encoder = computeQueue->createCommandEncoder();
        encoder->acquireBufferFromQueue(bufferB, ResourceState::UnorderedAccess, QueueType::Graphics);
        auto pass = encoder->beginComputePass();
        auto rootObject = pass->bindPipeline(pipeline);
        ShaderCursor shaderCursor(rootObject);
        shaderCursor["buffer"].setBinding(bufferB);
        float value = 5.f;
        shaderCursor["value"].setData(value);
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        encoder->releaseBufferForQueue(bufferB, ResourceState::UnorderedAccess, QueueType::Graphics);
        REQUIRE_CALL(computeQueue->submit(encoder->finish()));
    }

    // Copy on transfer queue
    {
        auto encoder = transferQueue->createCommandEncoder();
        encoder->acquireBufferFromQueue(copySrc, ResourceState::CopySource, QueueType::Graphics);
        encoder->acquireBufferFromQueue(copyDst, ResourceState::CopyDestination, QueueType::Graphics);
        encoder->copyBuffer(copyDst, 0, copySrc, 0, numberCount * sizeof(float));
        encoder->releaseBufferForQueue(copySrc, ResourceState::CopySource, QueueType::Graphics);
        encoder->releaseBufferForQueue(copyDst, ResourceState::CopyDestination, QueueType::Graphics);
        REQUIRE_CALL(transferQueue->submit(encoder->finish()));
    }

    graphicsQueue->waitOnHost();
    computeQueue->waitOnHost();
    transferQueue->waitOnHost();

    syncQueues(device, computeQueue, graphicsQueue);
    syncQueues(device, transferQueue, graphicsQueue);
    {
        auto encoder = graphicsQueue->createCommandEncoder();
        encoder->acquireBufferFromQueue(bufferB, ResourceState::UnorderedAccess, QueueType::Compute);
        encoder->acquireBufferFromQueue(copySrc, ResourceState::CopySource, QueueType::Transfer);
        encoder->acquireBufferFromQueue(copyDst, ResourceState::CopyDestination, QueueType::Transfer);
        REQUIRE_CALL(graphicsQueue->submit(encoder->finish()));
        REQUIRE_CALL(graphicsQueue->waitOnHost());
    }

    compareComputeResult(device, bufferA, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f));
    compareComputeResult(device, bufferB, makeArray<float>(16.0f, 26.0f, 36.0f, 46.0f));
    compareComputeResult(device, copyDst, makeArray<float>(100.0f, 200.0f, 300.0f, 400.0f));
}

// Test queue family ownership transfer workflow.
GPU_TEST_CASE("transfer-queue-ownership-transfer", D3D12 | Vulkan)
{
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    auto transferQueue = device->getQueue(QueueType::Transfer);
    REQUIRE(graphicsQueue != nullptr);
    REQUIRE(transferQueue != nullptr);

    const int numberCount = 4;
    float srcData[] = {1.0f, 2.0f, 3.0f, 4.0f};

    BufferDesc bufferDesc = {};
    bufferDesc.size = numberCount * sizeof(float);
    bufferDesc.usage = BufferUsage::CopySource | BufferUsage::CopyDestination | BufferUsage::UnorderedAccess;
    bufferDesc.defaultState = ResourceState::CopySource;
    bufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> srcBuffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)srcData, srcBuffer.writeRef()));
    bufferDesc.defaultState = ResourceState::CopyDestination;
    ComPtr<IBuffer> dstBuffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, dstBuffer.writeRef()));

    ComPtr<IFence> fence;
    REQUIRE_CALL(device->createFence({}, fence.writeRef()));

    // Graphics queue releases buffers to transfer queue
    {
        auto encoder = graphicsQueue->createCommandEncoder();
        encoder->releaseBufferForQueue(srcBuffer, ResourceState::CopySource, QueueType::Transfer);
        encoder->releaseBufferForQueue(dstBuffer, ResourceState::CopyDestination, QueueType::Transfer);

        ComPtr<ICommandBuffer> commandBuffer;
        REQUIRE_CALL(encoder->finish(commandBuffer.writeRef()));
        ICommandBuffer* commandBufferPtr = commandBuffer.get();

        SubmitDesc submitDesc = {};
        submitDesc.commandBuffers = &commandBufferPtr;
        submitDesc.commandBufferCount = 1;
        IFence* signalFence = fence.get();
        uint64_t signalValue = 1;
        submitDesc.signalFences = &signalFence;
        submitDesc.signalFenceValues = &signalValue;
        submitDesc.signalFenceCount = 1;
        REQUIRE_CALL(graphicsQueue->submit(submitDesc));
    }

    // Transfer queue acquires buffers and copies
    {
        auto encoder = transferQueue->createCommandEncoder();
        encoder->acquireBufferFromQueue(srcBuffer, ResourceState::CopySource, QueueType::Graphics);
        encoder->acquireBufferFromQueue(dstBuffer, ResourceState::CopyDestination, QueueType::Graphics);
        encoder->copyBuffer(dstBuffer, 0, srcBuffer, 0, numberCount * sizeof(float));

        ComPtr<ICommandBuffer> commandBuffer;
        REQUIRE_CALL(encoder->finish(commandBuffer.writeRef()));
        ICommandBuffer* commandBufferPtr = commandBuffer.get();

        SubmitDesc submitDesc = {};
        submitDesc.commandBuffers = &commandBufferPtr;
        submitDesc.commandBufferCount = 1;
        IFence* waitFence = fence.get();
        uint64_t waitValue = 1;
        submitDesc.waitFences = &waitFence;
        submitDesc.waitFenceValues = &waitValue;
        submitDesc.waitFenceCount = 1;
        REQUIRE_CALL(transferQueue->submit(submitDesc));
    }

    transferQueue->waitOnHost();

    // Sync transfer queue writes to graphics queue before readBuffer.
    syncQueues(device, transferQueue, device->getQueue(QueueType::Graphics));

    compareComputeResult(device, dstBuffer, makeArray<float>(1.0f, 2.0f, 3.0f, 4.0f));
}
