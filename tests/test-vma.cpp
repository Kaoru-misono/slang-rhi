#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

// Test that multiple small buffers can be created (exercises VMA sub-allocation).
GPU_TEST_CASE("vma-buffer-suballocation", Vulkan)
{
    const int bufferCount = 16;
    const int elementCount = 4;
    std::vector<ComPtr<IBuffer>> buffers(bufferCount);

    for (int i = 0; i < bufferCount; i++)
    {
        float initialData[] = {float(i), float(i + 1), float(i + 2), float(i + 3)};
        BufferDesc bufferDesc = {};
        bufferDesc.size = elementCount * sizeof(float);
        bufferDesc.format = Format::Undefined;
        bufferDesc.elementSize = sizeof(float);
        bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination | BufferUsage::CopySource;
        bufferDesc.defaultState = ResourceState::ShaderResource;
        bufferDesc.memoryType = MemoryType::DeviceLocal;
        REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)initialData, buffers[i].writeRef()));
    }

    // Verify each buffer has correct data.
    for (int i = 0; i < bufferCount; i++)
    {
        compareComputeResult(
            device,
            buffers[i],
            makeArray<float>(float(i), float(i + 1), float(i + 2), float(i + 3))
        );
    }
}

// Test host-visible buffer mapping (exercises VMA persistent mapping).
GPU_TEST_CASE("vma-buffer-mapping", Vulkan)
{
    const int elementCount = 4;

    BufferDesc bufferDesc = {};
    bufferDesc.size = elementCount * sizeof(float);
    bufferDesc.format = Format::Undefined;
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.usage = BufferUsage::CopyDestination | BufferUsage::CopySource;
    bufferDesc.memoryType = MemoryType::Upload;

    ComPtr<IBuffer> uploadBuffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, uploadBuffer.writeRef()));

    // Map and write data.
    void* mappedPtr = nullptr;
    REQUIRE_CALL(device->mapBuffer(uploadBuffer, CpuAccessMode::Write, &mappedPtr));
    REQUIRE(mappedPtr != nullptr);
    auto data = static_cast<float*>(mappedPtr);
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    REQUIRE_CALL(device->unmapBuffer(uploadBuffer));

    // Copy to a device-local buffer and read back to verify.
    BufferDesc deviceBufDesc = {};
    deviceBufDesc.size = elementCount * sizeof(float);
    deviceBufDesc.elementSize = sizeof(float);
    deviceBufDesc.usage = BufferUsage::CopyDestination | BufferUsage::CopySource;
    deviceBufDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> deviceBuffer;
    REQUIRE_CALL(device->createBuffer(deviceBufDesc, nullptr, deviceBuffer.writeRef()));

    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();
        commandEncoder->copyBuffer(deviceBuffer, 0, uploadBuffer, 0, elementCount * sizeof(float));
        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    compareComputeResult(device, deviceBuffer, makeArray<float>(1.0f, 2.0f, 3.0f, 4.0f));
}

// Test device-local buffer with initial data upload (exercises VMA staging path).
GPU_TEST_CASE("vma-buffer-initial-data", Vulkan)
{
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
        auto queue = device->getQueue(QueueType::Graphics);
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

// Test texture creation (exercises VMA image memory allocation).
GPU_TEST_CASE("vma-texture-create", Vulkan)
{
    TextureDesc texDesc = {};
    texDesc.type = TextureType::Texture2D;
    texDesc.size = {4, 4, 1};
    texDesc.mipCount = 1;
    texDesc.format = Format::RGBA8Unorm;
    texDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination | TextureUsage::CopySource;
    texDesc.defaultState = ResourceState::ShaderResource;
    texDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<ITexture> texture;
    REQUIRE_CALL(device->createTexture(texDesc, nullptr, texture.writeRef()));
    REQUIRE(texture != nullptr);

    const TextureDesc& desc = texture->getDesc();
    CHECK_EQ(desc.size.width, 4);
    CHECK_EQ(desc.size.height, 4);
    CHECK_EQ(desc.format, Format::RGBA8Unorm);
}

// Test that buffer cleanup works correctly (create and destroy many buffers).
GPU_TEST_CASE("vma-buffer-cleanup", Vulkan)
{
    const int iterations = 64;
    const int elementCount = 256;

    for (int i = 0; i < iterations; i++)
    {
        BufferDesc bufferDesc = {};
        bufferDesc.size = elementCount * sizeof(float);
        bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
        bufferDesc.memoryType = MemoryType::DeviceLocal;

        ComPtr<IBuffer> buffer;
        REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, buffer.writeRef()));
        // Buffer goes out of scope and is destroyed here.
    }

    // If we get here without crashing or leaking, the test passes.
    auto queue = device->getQueue(QueueType::Graphics);
    queue->waitOnHost();
}
