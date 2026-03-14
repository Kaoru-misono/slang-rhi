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
    REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)initialData, buffer.writeRef()));

    {
        auto commandEncoder = queue->createCommandEncoder();

        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        ShaderCursor shaderCursor(rootObject);
        shaderCursor["buffer"].setBinding(buffer);
        float value = 10.f;
        shaderCursor["value"].setData(value);
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    compareComputeResult(device, buffer, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f));
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
    REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)initialDataB, bufferB.writeRef()));

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
        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        ShaderCursor shaderCursor(rootObject);
        shaderCursor["buffer"].setBinding(bufferB);
        float value = 5.f;
        shaderCursor["value"].setData(value);
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();
        computeQueue->submit(commandEncoder->finish());
    }

    graphicsQueue->waitOnHost();
    computeQueue->waitOnHost();

    compareComputeResult(device, bufferA, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f));
    compareComputeResult(device, bufferB, makeArray<float>(16.0f, 26.0f, 36.0f, 46.0f));
}
