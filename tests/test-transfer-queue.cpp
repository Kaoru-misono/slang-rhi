#include "test-compute-trivial.h"

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

    // readBuffer uses the graphics queue internally - need GPU-GPU sync from transfer queue.
    syncQueues(device, transferQueue, device->getQueue(QueueType::Graphics));

    compareComputeResult(device, dstBuffer, makeArray<float>(1.0f, 2.0f, 3.0f, 4.0f));
}

// Test using all three queues simultaneously.
static void runTransferQueueAllQueues(IDevice* device)
{
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    auto computeQueue = device->getQueue(QueueType::Compute);
    auto transferQueue = device->getQueue(QueueType::Transfer);
    REQUIRE(graphicsQueue != nullptr);
    REQUIRE(computeQueue != nullptr);
    REQUIRE(transferQueue != nullptr);

    auto pipeline = trivialCompute::createPipeline(device);

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
        float value = 10.f;
        trivialCompute::bindBuffer(device, pass, pipeline, bufferA, value);
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        graphicsQueue->submit(encoder->finish());
    }

    // Dispatch on compute queue
    {
        auto encoder = computeQueue->createCommandEncoder();
        encoder->acquireBufferFromQueue(bufferB, ResourceState::UnorderedAccess, QueueType::Graphics);
        auto pass = encoder->beginComputePass();
        float value = 5.f;
        trivialCompute::bindBuffer(device, pass, pipeline, bufferB, value);
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

GPU_TEST_CASE("transfer-queue-all-queues", D3D12 | Vulkan)
{
    runTransferQueueAllQueues(device);
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

GPU_TEST_CASE("concurrent-resources-initial-data-three-queues", D3D12 | Vulkan)
{
    auto expected = makeArray<float>(3.f, 7.f, 11.f, 19.f);
    BufferDesc bufferDesc = {};
    bufferDesc.size = sizeof(expected);
    bufferDesc.usage = BufferUsage::CopySource | BufferUsage::CopyDestination;
    bufferDesc.defaultState = ResourceState::CopySource;
    bufferDesc.sharingMode = SharingMode::Concurrent;
    ComPtr<IBuffer> sourceBuffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, expected.data(), sourceBuffer.writeRef()));
    CHECK(sourceBuffer->getDesc().sharingMode == SharingMode::Concurrent);

    TextureDesc textureDesc = {};
    textureDesc.size = {4, 1, 1};
    textureDesc.format = Format::R32Float;
    textureDesc.usage = TextureUsage::CopySource | TextureUsage::CopyDestination | TextureUsage::ShaderResource;
    // D3D12 COPY queues require COMMON at queue boundaries; Vulkan readers
    // retain one layout so concurrent copies do not race layout transitions.
    textureDesc.defaultState = device->getDeviceType() == DeviceType::D3D12
        ? ResourceState::General : ResourceState::CopySource;
    textureDesc.sharingMode = SharingMode::Concurrent;
    SubresourceData initial = {expected.data(), sizeof(expected), sizeof(expected)};
    ComPtr<ITexture> sourceTexture;
    REQUIRE_CALL(device->createTexture(textureDesc, &initial, sourceTexture.writeRef()));
    CHECK(sourceTexture->getDesc().sharingMode == SharingMode::Concurrent);

    ComPtr<IFence> uploaded;
    REQUIRE_CALL(device->createFence({}, uploaded.writeRef()));
    IFence* uploadFence = uploaded;
    uint64_t uploadValue = 1;
    SubmitDesc signal = {};
    signal.signalFences = &uploadFence;
    signal.signalFenceValues = &uploadValue;
    signal.signalFenceCount = 1;
    REQUIRE_CALL(device->getQueue(QueueType::Graphics)->submit(signal));

    const QueueType types[] = {QueueType::Graphics, QueueType::Compute, QueueType::Transfer};
    ComPtr<ICommandQueue> queues[3];
    ComPtr<IBuffer> outputBuffers[3];
    ComPtr<ITexture> outputTextures[3];
    for (uint32_t i = 0; i < 3; ++i)
    {
        REQUIRE_CALL(device->getQueue(types[i], queues[i].writeRef()));
        bufferDesc.defaultState = ResourceState::CopyDestination;
        textureDesc.defaultState = device->getDeviceType() == DeviceType::D3D12
            ? ResourceState::General : ResourceState::CopyDestination;
        REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, outputBuffers[i].writeRef()));
        REQUIRE_CALL(device->createTexture(textureDesc, nullptr, outputTextures[i].writeRef()));
        auto encoder = queues[i]->createCommandEncoder();
        REQUIRE(encoder);
        // No QFOT: the resources are shared across families, but initialization still needs a wait.
        encoder->copyBuffer(outputBuffers[i], 0, sourceBuffer, 0, sizeof(expected));
        encoder->copyTexture(outputTextures[i], {}, {}, sourceTexture, {}, {}, Extent3D::kWholeTexture);
        ComPtr<ICommandBuffer> commands;
        REQUIRE_CALL(encoder->finish(commands.writeRef()));
        ICommandBuffer* command = commands;
        SubmitDesc submit = {};
        submit.commandBuffers = &command;
        submit.commandBufferCount = 1;
        submit.waitFences = &uploadFence;
        submit.waitFenceValues = &uploadValue;
        submit.waitFenceCount = 1;
        REQUIRE_CALL(queues[i]->submit(submit));
    }
    for (uint32_t i = 0; i < 3; ++i)
    {
        REQUIRE_CALL(queues[i]->waitOnHost());
        compareComputeResult(device, outputBuffers[i], expected);
        compareComputeResult(device, outputTextures[i], 0, 0, std::span<float>(expected));
    }

    // A non-default texture read layout must finish before a copy on another queue.
    auto pipeline = trivialCompute::createPipeline(device, "readTexture");
    BufferDesc resultDesc = {};
    resultDesc.size = sizeof(float);
    resultDesc.elementSize = sizeof(float);
    resultDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    resultDesc.defaultState = ResourceState::UnorderedAccess;
    resultDesc.sharingMode = SharingMode::Concurrent;
    ComPtr<IBuffer> result;
    REQUIRE_CALL(device->createBuffer(resultDesc, nullptr, result.writeRef()));
    auto encoder = queues[1]->createCommandEncoder();
    REQUIRE(encoder);
    auto pass = encoder->beginComputePass();
    REQUIRE(pass);
    trivialCompute::bind(device, pass, pipeline, {
        {.slot = 0, .resource = sourceTexture->getDefaultView()},
        {.slot = 1, .resource = result},
    });
    pass->dispatchCompute(1, 1, 1);
    pass->end();
    REQUIRE_CALL(queues[1]->submit(encoder->finish()));
    syncQueues(device, queues[1], queues[2]);
    auto copy = queues[2]->createCommandEncoder();
    REQUIRE(copy);
    copy->copyTexture(outputTextures[2], {}, {}, sourceTexture, {}, {}, Extent3D::kWholeTexture);
    REQUIRE_CALL(queues[2]->submit(copy->finish()));
    syncQueues(device, queues[2], queues[0]);
    compareComputeResult(device, result, makeArray<float>(3.f));
    compareComputeResult(device, outputTextures[2], 0, 0, std::span<float>(expected));
}

GPU_TEST_CASE("concurrent-texture-rejects-depth-and-multisample", D3D12)
{
    TextureDesc depth = {};
    depth.size = {4, 4, 1};
    depth.format = Format::D32Float;
    depth.usage = TextureUsage::DepthStencil;
    TextureDesc multisampled = {};
    multisampled.type = TextureType::Texture2DMS;
    multisampled.size = {4, 4, 1};
    multisampled.sampleCount = 4;
    multisampled.format = Format::RGBA8Unorm;
    multisampled.usage = TextureUsage::RenderTarget;
    for (TextureDesc desc : {depth, multisampled})
    {
        ComPtr<ITexture> exclusive;
        REQUIRE_CALL(device->createTexture(desc, nullptr, exclusive.writeRef()));
        desc.sharingMode = SharingMode::Concurrent;
        ComPtr<ITexture> concurrent;
        CHECK(device->createTexture(desc, nullptr, concurrent.writeRef()) == SLANG_E_INVALID_ARG);
        CHECK(concurrent == nullptr);
    }
}

GPU_TEST_CASE("shader-resource-texture-default-compute-roundtrip", D3D12)
{
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    auto computeQueue = device->getQueue(QueueType::Compute);
    REQUIRE(graphicsQueue);
    REQUIRE(computeQueue);

    auto initialPixels = makeArray<float>(3.f, 7.f, 11.f, 19.f);
    TextureDesc desc = {};
    desc.size = {4, 1, 1};
    desc.format = Format::R32Float;
    desc.usage = TextureUsage::ShaderResource | TextureUsage::UnorderedAccess |
                 TextureUsage::CopySource | TextureUsage::CopyDestination;
    desc.defaultState = ResourceState::ShaderResource;
    desc.sharingMode = SharingMode::Concurrent;
    SubresourceData initial = {initialPixels.data(), sizeof(initialPixels), sizeof(initialPixels)};
    ComPtr<ITexture> seed;
    ComPtr<ITexture> texture;
    REQUIRE_CALL(device->createTexture(desc, &initial, seed.writeRef()));
    REQUIRE_CALL(device->createTexture(desc, nullptr, texture.writeRef()));
    CHECK(seed->getDesc().defaultState == ResourceState::General);
    CHECK(texture->getDesc().defaultState == ResourceState::General);

    // Establish the default state through a real graphics copy and the encoder's
    // normal restoration, rather than only trusting the initial texture state.
    {
        auto encoder = graphicsQueue->createCommandEncoder();
        REQUIRE(encoder);
        encoder->copyTexture(texture, {}, {}, seed, {}, {}, Extent3D::kWholeTexture);
        REQUIRE_CALL(graphicsQueue->submit(encoder->finish()));
    }
    syncQueues(device, graphicsQueue, computeQueue);

    auto pipeline = trivialCompute::createPipeline(device, "writeTexture");
    {
        auto encoder = computeQueue->createCommandEncoder();
        REQUIRE(encoder);
        auto pass = encoder->beginComputePass();
        REQUIRE(pass);
        trivialCompute::bind(device, pass, pipeline, {
            {.slot = 0, .resource = texture->getDefaultView()},
        });
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        // Restore the actual default state, not the requested ShaderResource state.
        REQUIRE_CALL(computeQueue->submit(encoder->finish()));
    }
    syncQueues(device, computeQueue, graphicsQueue);

    ComPtr<ITexture> output;
    desc.defaultState = ResourceState::General;
    REQUIRE_CALL(device->createTexture(desc, nullptr, output.writeRef()));
    {
        auto encoder = graphicsQueue->createCommandEncoder();
        REQUIRE(encoder);
        encoder->copyTexture(output, {}, {}, texture, {}, {}, Extent3D::kWholeTexture);
        REQUIRE_CALL(graphicsQueue->submit(encoder->finish()));
        REQUIRE_CALL(graphicsQueue->waitOnHost());
    }
    auto expected = makeArray<float>(42.f, 7.f, 11.f, 19.f);
    auto expectedWords = makeArray<uint32_t>(0u, 0u, 0u, 0u);
    std::memcpy(expectedWords.data(), expected.data(), sizeof(expected));
    compareComputeResult(device, output, 0, 0, std::span<uint32_t>(expectedWords));
    compareComputeResult(device, texture, 0, 0, std::span<uint32_t>(expectedWords));
}
