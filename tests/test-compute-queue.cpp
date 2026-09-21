#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

// Test that getQueue(QueueType::Compute) succeeds on D3D12 and Vulkan.
GPU_TEST_CASE("compute-queue-create", D3D12 | Vulkan)
{
    auto queue = device->getQueue(QueueType::Compute);
    REQUIRE(queue != nullptr);
}

// Test that getQueue(QueueType::Compute) returns null on backends that don't support it.
GPU_TEST_CASE("compute-queue-unsupported", CPU | D3D11)
{
    auto queue = device->getQueue(QueueType::Compute);
    CHECK(queue == nullptr);
}

// Test dispatching compute work on a compute queue.
GPU_TEST_CASE("compute-queue-dispatch", D3D12 | Vulkan)
{
    auto queue = device->getQueue(QueueType::Compute);
    REQUIRE(queue != nullptr);

    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    const int numberCount = 4;
    float initialData[] = {0.0f, 1.0f, 2.0f, 3.0f};
    BufferDesc bufferDesc = {};
    bufferDesc.size = numberCount * sizeof(float);
    bufferDesc.format = Format::Undefined;
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                       BufferUsage::CopySource;
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    bufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> buffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, buffer.writeRef()));

    BufferDesc uploadDesc = {};
    uploadDesc.size = bufferDesc.size;
    uploadDesc.usage = BufferUsage::CopySource;
    uploadDesc.defaultState = ResourceState::CopySource;
    uploadDesc.memoryType = MemoryType::Upload;
    ComPtr<IBuffer> uploadBuffer;
    REQUIRE_CALL(device->createBuffer(uploadDesc, initialData, uploadBuffer.writeRef()));

    BufferDesc readbackDesc = {};
    readbackDesc.size = bufferDesc.size;
    readbackDesc.usage = BufferUsage::CopyDestination;
    readbackDesc.defaultState = ResourceState::CopyDestination;
    readbackDesc.memoryType = MemoryType::ReadBack;
    ComPtr<IBuffer> readbackBuffer;
    REQUIRE_CALL(device->createBuffer(readbackDesc, nullptr, readbackBuffer.writeRef()));

    {
        auto commandEncoder = queue->createCommandEncoder();
        commandEncoder->copyBuffer(buffer, 0, uploadBuffer, 0, bufferDesc.size);

        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        ShaderCursor shaderCursor(rootObject);
        shaderCursor["buffer"].setBinding(buffer);
        float value = 10.f;
        shaderCursor["value"].setData(value);
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();

        commandEncoder->copyBuffer(readbackBuffer, 0, buffer, 0, bufferDesc.size);

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    void* mappedData = nullptr;
    REQUIRE_CALL(device->mapBuffer(readbackBuffer, CpuAccessMode::Read, &mappedData));
    auto const expected = makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f);
    compareResultFuzzy(static_cast<float const*>(mappedData), expected.data(), expected.size());
    REQUIRE_CALL(device->unmapBuffer(readbackBuffer));
}

// Test that both graphics and compute queues can work independently.
GPU_TEST_CASE("compute-queue-both-queues", D3D12 | Vulkan)
{
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    auto computeQueue = device->getQueue(QueueType::Compute);
    REQUIRE(graphicsQueue != nullptr);
    REQUIRE(computeQueue != nullptr);

    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    const int numberCount = 4;
    float initialDataA[] = {0.0f, 1.0f, 2.0f, 3.0f};
    float initialDataB[] = {10.0f, 20.0f, 30.0f, 40.0f};

    BufferDesc bufferDesc = {};
    bufferDesc.size = numberCount * sizeof(float);
    bufferDesc.format = Format::Undefined;
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                       BufferUsage::CopySource;
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    bufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> bufferA;
    REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)initialDataA, bufferA.writeRef()));
    ComPtr<IBuffer> bufferB;
    REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, bufferB.writeRef()));

    BufferDesc uploadDesc = {};
    uploadDesc.size = bufferDesc.size;
    uploadDesc.usage = BufferUsage::CopySource;
    uploadDesc.defaultState = ResourceState::CopySource;
    uploadDesc.memoryType = MemoryType::Upload;
    ComPtr<IBuffer> bufferBUpload;
    REQUIRE_CALL(device->createBuffer(uploadDesc, initialDataB, bufferBUpload.writeRef()));

    BufferDesc readbackDesc = {};
    readbackDesc.size = bufferDesc.size;
    readbackDesc.usage = BufferUsage::CopyDestination;
    readbackDesc.defaultState = ResourceState::CopyDestination;
    readbackDesc.memoryType = MemoryType::ReadBack;
    ComPtr<IBuffer> bufferBReadback;
    REQUIRE_CALL(device->createBuffer(readbackDesc, nullptr, bufferBReadback.writeRef()));

    // Dispatch on graphics queue
    {
        auto commandEncoder = graphicsQueue->createCommandEncoder();
        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        ShaderCursor shaderCursor(rootObject);
        shaderCursor["buffer"].setBinding(bufferA);
        float value = 10.f;
        shaderCursor["value"].setData(value);
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();
        graphicsQueue->submit(commandEncoder->finish());
    }

    // Dispatch on compute queue
    {
        auto commandEncoder = computeQueue->createCommandEncoder();
        commandEncoder->copyBuffer(bufferB, 0, bufferBUpload, 0, bufferDesc.size);
        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        ShaderCursor shaderCursor(rootObject);
        shaderCursor["buffer"].setBinding(bufferB);
        float value = 5.f;
        shaderCursor["value"].setData(value);
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();
        commandEncoder->copyBuffer(bufferBReadback, 0, bufferB, 0, bufferDesc.size);
        computeQueue->submit(commandEncoder->finish());
    }

    graphicsQueue->waitOnHost();
    computeQueue->waitOnHost();

    compareComputeResult(device, bufferA, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f));
    void* mappedData = nullptr;
    REQUIRE_CALL(device->mapBuffer(bufferBReadback, CpuAccessMode::Read, &mappedData));
    auto const expectedB = makeArray<float>(16.0f, 26.0f, 36.0f, 46.0f);
    compareResultFuzzy(static_cast<float const*>(mappedData), expectedB.data(), expectedB.size());
    REQUIRE_CALL(device->unmapBuffer(bufferBReadback));
}

GPU_TEST_CASE("compute-queue-texture-read", Vulkan)
{
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    auto queue = device->getQueue(QueueType::Compute);
    REQUIRE(graphicsQueue != nullptr);
    REQUIRE(queue != nullptr);

    ComPtr<IShaderProgram> writeProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "writeTexture", writeProgram.writeRef()));
    ComPtr<IShaderProgram> readProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "readTexture", readProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = writeProgram.get();
    ComPtr<IComputePipeline> writePipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, writePipeline.writeRef()));
    pipelineDesc.program = readProgram.get();
    ComPtr<IComputePipeline> readPipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, readPipeline.writeRef()));

    constexpr float textureData = 42.0f;
    TextureDesc textureDesc = {};
    textureDesc.type = TextureType::Texture2D;
    textureDesc.size = {1, 1, 1};
    textureDesc.format = Format::R32Float;
    textureDesc.usage = TextureUsage::ShaderResource | TextureUsage::UnorderedAccess;
    textureDesc.defaultState = ResourceState::UnorderedAccess;
    ComPtr<ITexture> inputTexture;
    REQUIRE_CALL(device->createTexture(textureDesc, nullptr, inputTexture.writeRef()));

    BufferDesc outputDesc = {};
    outputDesc.size = sizeof(float);
    outputDesc.elementSize = sizeof(float);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    outputDesc.defaultState = ResourceState::UnorderedAccess;
    ComPtr<IBuffer> outputBuffer;
    REQUIRE_CALL(device->createBuffer(outputDesc, nullptr, outputBuffer.writeRef()));

    BufferDesc readbackDesc = {};
    readbackDesc.size = sizeof(float);
    readbackDesc.usage = BufferUsage::CopyDestination;
    readbackDesc.defaultState = ResourceState::CopyDestination;
    readbackDesc.memoryType = MemoryType::ReadBack;
    ComPtr<IBuffer> outputReadback;
    REQUIRE_CALL(device->createBuffer(readbackDesc, nullptr, outputReadback.writeRef()));

    ComPtr<IFence> ownershipFence;
    REQUIRE_CALL(device->createFence({}, ownershipFence.writeRef()));
    {
        auto releaseEncoder = graphicsQueue->createCommandEncoder();
        releaseEncoder->releaseTextureForQueue(
            inputTexture,
            kEntireTexture,
            ResourceState::UnorderedAccess,
            QueueType::Compute
        );

        ComPtr<ICommandBuffer> releaseCommandBuffer = releaseEncoder->finish();
        ICommandBuffer* releaseCommandBufferPtr = releaseCommandBuffer.get();
        IFence* signalFence = ownershipFence.get();
        uint64_t signalValue = 1;
        SubmitDesc releaseSubmit = {};
        releaseSubmit.commandBuffers = &releaseCommandBufferPtr;
        releaseSubmit.commandBufferCount = 1;
        releaseSubmit.signalFences = &signalFence;
        releaseSubmit.signalFenceValues = &signalValue;
        releaseSubmit.signalFenceCount = 1;
        REQUIRE_CALL(graphicsQueue->submit(releaseSubmit));
    }

    auto commandEncoder = queue->createCommandEncoder();
    commandEncoder->acquireTextureFromQueue(
        inputTexture,
        kEntireTexture,
        ResourceState::UnorderedAccess,
        QueueType::Graphics
    );
    auto writePass = commandEncoder->beginComputePass();
    auto writeRoot = writePass->bindPipeline(writePipeline);
    ShaderCursor(writeRoot)["outputTexture"].setBinding(inputTexture);
    writePass->dispatchCompute(1, 1, 1);
    writePass->end();

    auto readPass = commandEncoder->beginComputePass();
    auto readRoot = readPass->bindPipeline(readPipeline);
    ShaderCursor readCursor(readRoot);
    readCursor["inputTexture"].setBinding(inputTexture);
    readCursor["outputBuffer"].setBinding(outputBuffer);
    readPass->dispatchCompute(1, 1, 1);
    readPass->end();
    commandEncoder->copyBuffer(outputReadback, 0, outputBuffer, 0, sizeof(float));
    ComPtr<ICommandBuffer> commandBuffer = commandEncoder->finish();
    ICommandBuffer* commandBufferPtr = commandBuffer.get();
    IFence* waitFence = ownershipFence.get();
    uint64_t waitValue = 1;
    SubmitDesc submit = {};
    submit.commandBuffers = &commandBufferPtr;
    submit.commandBufferCount = 1;
    submit.waitFences = &waitFence;
    submit.waitFenceValues = &waitValue;
    submit.waitFenceCount = 1;
    REQUIRE_CALL(queue->submit(submit));
    REQUIRE_CALL(queue->waitOnHost());

    void* mappedData = nullptr;
    REQUIRE_CALL(device->mapBuffer(outputReadback, CpuAccessMode::Read, &mappedData));
    compareResultFuzzy(static_cast<float const*>(mappedData), &textureData, 1);
    REQUIRE_CALL(device->unmapBuffer(outputReadback));
}
