#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

// ------------------------------------------------------------------
// Test: Multi-pass compute — exercises barrier merging and split barriers.
// Two compute passes in one command buffer: pass 1 writes buffer A,
// pass 2 reads A and writes B. The state transitions between passes
// exercise both barrier merging (redundant transitions collapsed)
// and split barriers (D3D12 BEGIN_ONLY/END_ONLY at pass boundaries).
// ------------------------------------------------------------------
GPU_TEST_CASE("state-tracking-multi-pass", D3D12 | Vulkan)
{
    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    const int numberCount = 4;
    float initialA[] = {0.0f, 1.0f, 2.0f, 3.0f};
    float initialB[] = {100.0f, 200.0f, 300.0f, 400.0f};

    BufferDesc bufferDesc = {};
    bufferDesc.size = numberCount * sizeof(float);
    bufferDesc.format = Format::Undefined;
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                       BufferUsage::CopyDestination | BufferUsage::CopySource;
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    bufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> bufferA;
    REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)initialA, bufferA.writeRef()));
    ComPtr<IBuffer> bufferB;
    REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)initialB, bufferB.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);

    // Single command buffer with two compute passes.
    // This exercises split barriers between pass 1 end and pass 2 start.
    {
        auto encoder = queue->createCommandEncoder();

        // Pass 1: buffer[i] = buffer[i] + 1 + value(10) → A becomes {11, 12, 13, 14}
        {
            auto pass = encoder->beginComputePass();
            auto rootObject = pass->bindPipeline(pipeline);
            ShaderCursor cursor(rootObject);
            cursor["buffer"].setBinding(bufferA);
            float value = 10.0f;
            cursor["value"].setData(value);
            pass->dispatchCompute(1, 1, 1);
            pass->end();
        }

        // Pass 2: buffer[i] = buffer[i] + 1 + value(5) → B becomes {106, 206, 306, 406}
        {
            auto pass = encoder->beginComputePass();
            auto rootObject = pass->bindPipeline(pipeline);
            ShaderCursor cursor(rootObject);
            cursor["buffer"].setBinding(bufferB);
            float value = 5.0f;
            cursor["value"].setData(value);
            pass->dispatchCompute(1, 1, 1);
            pass->end();
        }

        queue->submit(encoder->finish());
        queue->waitOnHost();
    }

    compareComputeResult(device, bufferA, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f));
    compareComputeResult(device, bufferB, makeArray<float>(106.0f, 206.0f, 306.0f, 406.0f));
}

// ------------------------------------------------------------------
// Test: Copy chain — buffer goes through multiple state transitions
// in one command buffer: CopySrc → CopyDst → CopySrc → CopyDst.
// This exercises barrier merging when the same buffer transitions
// multiple times within one commit batch.
// ------------------------------------------------------------------
GPU_TEST_CASE("state-tracking-copy-chain", D3D12 | Vulkan)
{
    const int count = 4;
    Size dataSize = count * sizeof(float);
    float srcData[] = {42.0f, 43.0f, 44.0f, 45.0f};

    BufferDesc desc = {};
    desc.size = dataSize;
    desc.usage = BufferUsage::CopySource | BufferUsage::CopyDestination;
    desc.memoryType = MemoryType::DeviceLocal;

    // Chain: src → mid → dst
    desc.defaultState = ResourceState::CopySource;
    ComPtr<IBuffer> src;
    REQUIRE_CALL(device->createBuffer(desc, (void*)srcData, src.writeRef()));

    desc.defaultState = ResourceState::CopyDestination;
    ComPtr<IBuffer> mid;
    REQUIRE_CALL(device->createBuffer(desc, nullptr, mid.writeRef()));

    desc.defaultState = ResourceState::CopyDestination;
    ComPtr<IBuffer> dst;
    REQUIRE_CALL(device->createBuffer(desc, nullptr, dst.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);

    {
        auto encoder = queue->createCommandEncoder();

        // Copy 1: src → mid
        encoder->setBufferState(src, ResourceState::CopySource);
        encoder->setBufferState(mid, ResourceState::CopyDestination);
        encoder->copyBuffer(mid, 0, src, 0, dataSize);

        // Copy 2: mid → dst (mid transitions CopyDst → CopySrc)
        encoder->setBufferState(mid, ResourceState::CopySource);
        encoder->setBufferState(dst, ResourceState::CopyDestination);
        encoder->copyBuffer(dst, 0, mid, 0, dataSize);

        queue->submit(encoder->finish());
        queue->waitOnHost();
    }

    compareComputeResult(device, dst, makeArray<float>(42.0f, 43.0f, 44.0f, 45.0f));
}

// ------------------------------------------------------------------
// Test: Compute then copy in one command buffer.
// Compute writes to buffer, then copy reads it — exercises the
// UAV → CopySrc transition with barrier merging and split barriers.
// ------------------------------------------------------------------
GPU_TEST_CASE("state-tracking-compute-then-copy", D3D12 | Vulkan)
{
    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    const int count = 4;
    Size dataSize = count * sizeof(float);
    float initial[] = {1.0f, 2.0f, 3.0f, 4.0f};

    BufferDesc bufDesc = {};
    bufDesc.size = dataSize;
    bufDesc.format = Format::Undefined;
    bufDesc.elementSize = sizeof(float);
    bufDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                    BufferUsage::CopySource | BufferUsage::CopyDestination;
    bufDesc.defaultState = ResourceState::UnorderedAccess;
    bufDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> computeBuffer;
    REQUIRE_CALL(device->createBuffer(bufDesc, (void*)initial, computeBuffer.writeRef()));

    BufferDesc copyDstDesc = {};
    copyDstDesc.size = dataSize;
    copyDstDesc.usage = BufferUsage::CopySource | BufferUsage::CopyDestination;
    copyDstDesc.defaultState = ResourceState::CopyDestination;
    copyDstDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> copyDst;
    REQUIRE_CALL(device->createBuffer(copyDstDesc, nullptr, copyDst.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);

    {
        auto encoder = queue->createCommandEncoder();

        // Compute pass: buffer[i] += 1 + value(0) → {2, 3, 4, 5}
        {
            auto pass = encoder->beginComputePass();
            auto rootObject = pass->bindPipeline(pipeline);
            ShaderCursor cursor(rootObject);
            cursor["buffer"].setBinding(computeBuffer);
            float value = 0.0f;
            cursor["value"].setData(value);
            pass->dispatchCompute(1, 1, 1);
            pass->end();
        }
        // EndComputePass triggers split barrier (D3D12)

        // Copy: computeBuffer (UAV→CopySrc transition) → copyDst
        encoder->setBufferState(computeBuffer, ResourceState::CopySource);
        encoder->setBufferState(copyDst, ResourceState::CopyDestination);
        encoder->copyBuffer(copyDst, 0, computeBuffer, 0, dataSize);

        queue->submit(encoder->finish());
        queue->waitOnHost();
    }

    compareComputeResult(device, copyDst, makeArray<float>(2.0f, 3.0f, 4.0f, 5.0f));
}

// ------------------------------------------------------------------
// Test: Repeated UAV dispatches — same buffer written multiple times.
// Exercises UAV→UAV barrier handling (must not be merged away).
// ------------------------------------------------------------------
GPU_TEST_CASE("state-tracking-repeated-uav", D3D12 | Vulkan)
{
    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    const int count = 4;
    float initial[] = {0.0f, 0.0f, 0.0f, 0.0f};

    BufferDesc bufDesc = {};
    bufDesc.size = count * sizeof(float);
    bufDesc.format = Format::Undefined;
    bufDesc.elementSize = sizeof(float);
    bufDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                    BufferUsage::CopySource | BufferUsage::CopyDestination;
    bufDesc.defaultState = ResourceState::UnorderedAccess;
    bufDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> buffer;
    REQUIRE_CALL(device->createBuffer(bufDesc, (void*)initial, buffer.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);

    // Three dispatches on the same buffer in one command buffer.
    // Each dispatch adds (1 + value). With value=0:
    // dispatch 1: {0,0,0,0} + 1 = {1,1,1,1}
    // dispatch 2: {1,1,1,1} + 1 = {2,2,2,2}
    // dispatch 3: {2,2,2,2} + 1 = {3,3,3,3}
    // This requires UAV barriers between dispatches to ensure ordering.
    {
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();

        for (int i = 0; i < 3; ++i)
        {
            auto rootObject = pass->bindPipeline(pipeline);
            ShaderCursor cursor(rootObject);
            cursor["buffer"].setBinding(buffer);
            float value = 0.0f;
            cursor["value"].setData(value);
            pass->dispatchCompute(1, 1, 1);
        }

        pass->end();
        queue->submit(encoder->finish());
        queue->waitOnHost();
    }

    compareComputeResult(device, buffer, makeArray<float>(3.0f, 3.0f, 3.0f, 3.0f));
}

// ------------------------------------------------------------------
// Test: Descriptor heap stress — allocate many buffers to exercise
// the descriptor heap near its limits and verify diagnostics work.
// ------------------------------------------------------------------
GPU_TEST_CASE("state-tracking-many-buffers", D3D12 | Vulkan)
{
    const int bufferCount = 64;
    const int count = 4;
    Size dataSize = count * sizeof(float);

    BufferDesc desc = {};
    desc.size = dataSize;
    desc.usage = BufferUsage::CopySource | BufferUsage::CopyDestination;
    desc.defaultState = ResourceState::CopySource;
    desc.memoryType = MemoryType::DeviceLocal;

    // Create many buffers
    std::vector<ComPtr<IBuffer>> buffers(bufferCount);
    for (int i = 0; i < bufferCount; ++i)
    {
        float data[] = {float(i), float(i + 1), float(i + 2), float(i + 3)};
        REQUIRE_CALL(device->createBuffer(desc, (void*)data, buffers[i].writeRef()));
    }

    // Copy from first to last through a chain of a few
    desc.defaultState = ResourceState::CopyDestination;
    ComPtr<IBuffer> finalDst;
    REQUIRE_CALL(device->createBuffer(desc, nullptr, finalDst.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);

    {
        auto encoder = queue->createCommandEncoder();
        // Copy buffer[0] → finalDst
        encoder->setBufferState(buffers[0], ResourceState::CopySource);
        encoder->setBufferState(finalDst, ResourceState::CopyDestination);
        encoder->copyBuffer(finalDst, 0, buffers[0], 0, dataSize);
        queue->submit(encoder->finish());
        queue->waitOnHost();
    }

    // buffer[0] has data {0, 1, 2, 3}
    compareComputeResult(device, finalDst, makeArray<float>(0.0f, 1.0f, 2.0f, 3.0f));
}
