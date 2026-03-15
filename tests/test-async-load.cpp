#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

// Helper: GPU-GPU fence sync from transfer queue to graphics queue.
// Required because readBuffer/readTexture use the graphics queue internally.
static void syncToGraphicsQueue(IDevice* device, ICommandQueue* srcQueue)
{
    ComPtr<IFence> syncFence;
    REQUIRE_CALL(device->createFence({}, syncFence.writeRef()));

    IFence* sf = syncFence.get();
    uint64_t sv = 1;

    SubmitDesc signalDesc = {};
    signalDesc.signalFences = &sf;
    signalDesc.signalFenceValues = &sv;
    signalDesc.signalFenceCount = 1;
    REQUIRE_CALL(srcQueue->submit(signalDesc));

    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    SubmitDesc waitDesc = {};
    waitDesc.waitFences = &sf;
    waitDesc.waitFenceValues = &sv;
    waitDesc.waitFenceCount = 1;
    REQUIRE_CALL(graphicsQueue->submit(waitDesc));
    graphicsQueue->waitOnHost();
}

// ------------------------------------------------------------------
// Test: async buffer load using transfer queue
// Simulates the pattern: staging buffer → transfer queue copy → fence poll → acquire → use
// ------------------------------------------------------------------
GPU_TEST_CASE("async-load-buffer", D3D12 | Vulkan | Metal)
{
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    auto transferQueue = device->getQueue(QueueType::Transfer);
    REQUIRE(graphicsQueue != nullptr);
    REQUIRE(transferQueue != nullptr);

    const int count = 8;
    float srcData[count] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    Size dataSize = count * sizeof(float);

    // 1. Create GPU-local destination buffer (no initial data)
    // Use CopyDestination as default state since the first use is on the transfer queue.
    // D3D12 copy command lists can only handle Common/CopySource/CopyDest states.
    // The QFOT acquire on the graphics queue will transition to ShaderResource.
    BufferDesc gpuDesc = {};
    gpuDesc.size = dataSize;
    gpuDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination | BufferUsage::CopySource;
    gpuDesc.defaultState = ResourceState::CopyDestination;
    gpuDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> gpuBuffer;
    REQUIRE_CALL(device->createBuffer(gpuDesc, nullptr, gpuBuffer.writeRef()));

    // 2. Create staging buffer with CPU data
    BufferDesc stagingDesc = {};
    stagingDesc.size = dataSize;
    stagingDesc.usage = BufferUsage::CopySource;
    stagingDesc.defaultState = ResourceState::CopySource;
    stagingDesc.memoryType = MemoryType::Upload;

    ComPtr<IBuffer> stagingBuffer;
    REQUIRE_CALL(device->createBuffer(stagingDesc, (void*)srcData, stagingBuffer.writeRef()));

    // Flush internal device queue before cross-queue work
    graphicsQueue->waitOnHost();

    // 3. Submit async copy on transfer queue with fence
    ComPtr<IFence> fence;
    REQUIRE_CALL(device->createFence({}, fence.writeRef()));

    {
        auto enc = transferQueue->createCommandEncoder();
        enc->setBufferState(stagingBuffer, ResourceState::CopySource);
        enc->setBufferState(gpuBuffer, ResourceState::CopyDestination);
        enc->copyBuffer(gpuBuffer, 0, stagingBuffer, 0, dataSize);

        // Release ownership to graphics queue (buffer is in CopyDestination state after copy)
        enc->releaseBufferForQueue(gpuBuffer, ResourceState::CopyDestination, QueueType::Graphics);

        ICommandBuffer* cmdBuf = nullptr;
        enc->finish(&cmdBuf);

        uint64_t fenceValue = 1;
        IFence* sf = fence.get();
        SubmitDesc desc = {};
        desc.commandBuffers = &cmdBuf;
        desc.commandBufferCount = 1;
        desc.signalFences = &sf;
        desc.signalFenceValues = &fenceValue;
        desc.signalFenceCount = 1;
        REQUIRE_CALL(transferQueue->submit(desc));
    }

    // 4. Poll fence until transfer completes (non-blocking check)
    {
        uint64_t currentValue = 0;
        // In a real app this would be checked once per frame.
        // Here we spin until done (with a timeout).
        for (int i = 0; i < 10000; ++i)
        {
            REQUIRE_CALL(fence->getCurrentValue(&currentValue));
            if (currentValue >= 1)
                break;
        }
        CHECK(currentValue >= 1);
    }

    // 5. Acquire ownership on graphics queue and sync for readBuffer
    syncToGraphicsQueue(device, transferQueue);

    {
        auto enc = graphicsQueue->createCommandEncoder();
        enc->acquireBufferFromQueue(gpuBuffer, ResourceState::ShaderResource, QueueType::Transfer);
        graphicsQueue->submit(enc->finish());
        graphicsQueue->waitOnHost();
    }

    // 6. Staging buffer can now be safely released
    stagingBuffer = nullptr;

    // 7. Verify data
    compareComputeResult(device, gpuBuffer, makeArray<float>(10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f));
}

// ------------------------------------------------------------------
// Test: async texture load using transfer queue
// Staging buffer → copyBufferToTexture on transfer queue → fence → acquire → readTexture
// ------------------------------------------------------------------
GPU_TEST_CASE("async-load-texture", D3D12 | Vulkan | Metal)
{
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    auto transferQueue = device->getQueue(QueueType::Transfer);
    REQUIRE(graphicsQueue != nullptr);
    REQUIRE(transferQueue != nullptr);

    const uint32_t width = 4;
    const uint32_t height = 4;
    const Size texelSize = 4;  // RGBA8 = 4 bytes per pixel
    const Size rowPitch = width * texelSize;
    const Size totalSize = rowPitch * height;

    // Generate test pattern: each pixel = (x*16, y*16, 128, 255)
    uint8_t texelData[width * height * texelSize];
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            uint8_t* pixel = texelData + (y * rowPitch + x * texelSize);
            pixel[0] = (uint8_t)(x * 16);    // R
            pixel[1] = (uint8_t)(y * 16);    // G
            pixel[2] = 128;                   // B
            pixel[3] = 255;                   // A
        }
    }

    // 1. Create GPU texture (no initial data)
    // Use CopyDestination as default state — first use is on the transfer queue.
    TextureDesc texDesc = {};
    texDesc.type = TextureType::Texture2D;
    texDesc.size = {width, height, 1};
    texDesc.format = Format::RGBA8Unorm;
    texDesc.mipCount = 1;
    texDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination | TextureUsage::CopySource;
    texDesc.defaultState = ResourceState::CopyDestination;
    texDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<ITexture> gpuTexture;
    REQUIRE_CALL(device->createTexture(texDesc, nullptr, gpuTexture.writeRef()));

    // 2. Create staging buffer with texel data
    BufferDesc stagingDesc = {};
    stagingDesc.size = totalSize;
    stagingDesc.usage = BufferUsage::CopySource;
    stagingDesc.defaultState = ResourceState::CopySource;
    stagingDesc.memoryType = MemoryType::Upload;

    ComPtr<IBuffer> stagingBuffer;
    REQUIRE_CALL(device->createBuffer(stagingDesc, (void*)texelData, stagingBuffer.writeRef()));

    // Flush internal device queue
    graphicsQueue->waitOnHost();

    // 3. Submit texture upload on transfer queue
    ComPtr<IFence> fence;
    REQUIRE_CALL(device->createFence({}, fence.writeRef()));

    {
        auto enc = transferQueue->createCommandEncoder();
        enc->setBufferState(stagingBuffer, ResourceState::CopySource);
        enc->setTextureState(gpuTexture, ResourceState::CopyDestination);

        enc->copyBufferToTexture(
            gpuTexture,
            0,                           // dstLayer
            0,                           // dstMip
            Offset3D{0, 0, 0},
            stagingBuffer,
            0,                           // srcOffset
            totalSize,                   // srcSize
            rowPitch,                    // srcRowPitch
            Extent3D{width, height, 1}
        );

        // Release texture to graphics queue (texture is in CopyDestination state after copy)
        enc->releaseTextureForQueue(
            gpuTexture,
            kEntireTexture,
            ResourceState::CopyDestination,
            QueueType::Graphics
        );

        ICommandBuffer* cmdBuf = nullptr;
        enc->finish(&cmdBuf);

        uint64_t fenceValue = 1;
        IFence* sf = fence.get();
        SubmitDesc desc = {};
        desc.commandBuffers = &cmdBuf;
        desc.commandBufferCount = 1;
        desc.signalFences = &sf;
        desc.signalFenceValues = &fenceValue;
        desc.signalFenceCount = 1;
        REQUIRE_CALL(transferQueue->submit(desc));
    }

    // 4. Wait for transfer to complete
    transferQueue->waitOnHost();

    // 5. Acquire on graphics queue
    syncToGraphicsQueue(device, transferQueue);

    {
        auto enc = graphicsQueue->createCommandEncoder();
        enc->acquireTextureFromQueue(
            gpuTexture,
            kEntireTexture,
            ResourceState::ShaderResource,
            QueueType::Transfer
        );
        graphicsQueue->submit(enc->finish());
        graphicsQueue->waitOnHost();
    }

    stagingBuffer = nullptr;

    // 6. Read back and verify
    ComPtr<ISlangBlob> resultBlob;
    SubresourceLayout layout;
    REQUIRE_CALL(device->readTexture(gpuTexture, 0, 0, resultBlob.writeRef(), &layout));

    const uint8_t* result = (const uint8_t*)resultBlob->getBufferPointer();
    bool allCorrect = true;
    for (uint32_t y = 0; y < height && allCorrect; ++y)
    {
        for (uint32_t x = 0; x < width && allCorrect; ++x)
        {
            const uint8_t* pixel = result + y * layout.rowPitch + x * texelSize;
            uint8_t expectedR = (uint8_t)(x * 16);
            uint8_t expectedG = (uint8_t)(y * 16);
            if (pixel[0] != expectedR || pixel[1] != expectedG || pixel[2] != 128 || pixel[3] != 255)
            {
                allCorrect = false;
                MESSAGE(
                    "Pixel (",
                    x,
                    ",",
                    y,
                    ") mismatch: got (",
                    (int)pixel[0],
                    ",",
                    (int)pixel[1],
                    ",",
                    (int)pixel[2],
                    ",",
                    (int)pixel[3],
                    ") expected (",
                    (int)expectedR,
                    ",",
                    (int)expectedG,
                    ",128,255)"
                );
            }
        }
    }
    CHECK(allCorrect);
}

// ------------------------------------------------------------------
// Test: batch async load — multiple buffers in one submission
// ------------------------------------------------------------------
GPU_TEST_CASE("async-load-batch", D3D12 | Vulkan | Metal)
{
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    auto transferQueue = device->getQueue(QueueType::Transfer);
    REQUIRE(graphicsQueue != nullptr);
    REQUIRE(transferQueue != nullptr);

    const int count = 4;
    Size dataSize = count * sizeof(float);
    float dataA[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dataB[] = {100.0f, 200.0f, 300.0f, 400.0f};
    float dataC[] = {-1.0f, -2.0f, -3.0f, -4.0f};

    // Create 3 GPU buffers — use CopyDestination as default state for transfer queue.
    BufferDesc gpuDesc = {};
    gpuDesc.size = dataSize;
    gpuDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination | BufferUsage::CopySource;
    gpuDesc.defaultState = ResourceState::CopyDestination;
    gpuDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> gpuA, gpuB, gpuC;
    REQUIRE_CALL(device->createBuffer(gpuDesc, nullptr, gpuA.writeRef()));
    REQUIRE_CALL(device->createBuffer(gpuDesc, nullptr, gpuB.writeRef()));
    REQUIRE_CALL(device->createBuffer(gpuDesc, nullptr, gpuC.writeRef()));

    // Create 3 staging buffers
    BufferDesc stagingDesc = {};
    stagingDesc.size = dataSize;
    stagingDesc.usage = BufferUsage::CopySource;
    stagingDesc.defaultState = ResourceState::CopySource;
    stagingDesc.memoryType = MemoryType::Upload;

    ComPtr<IBuffer> stagingA, stagingB, stagingC;
    REQUIRE_CALL(device->createBuffer(stagingDesc, (void*)dataA, stagingA.writeRef()));
    REQUIRE_CALL(device->createBuffer(stagingDesc, (void*)dataB, stagingB.writeRef()));
    REQUIRE_CALL(device->createBuffer(stagingDesc, (void*)dataC, stagingC.writeRef()));

    graphicsQueue->waitOnHost();

    // Submit all 3 copies in one batch
    ComPtr<IFence> fence;
    REQUIRE_CALL(device->createFence({}, fence.writeRef()));

    {
        auto enc = transferQueue->createCommandEncoder();

        // Copy all 3 buffers
        enc->copyBuffer(gpuA, 0, stagingA, 0, dataSize);
        enc->copyBuffer(gpuB, 0, stagingB, 0, dataSize);
        enc->copyBuffer(gpuC, 0, stagingC, 0, dataSize);

        // Release all to graphics queue (buffers are in CopyDestination state after copy)
        enc->releaseBufferForQueue(gpuA, ResourceState::CopyDestination, QueueType::Graphics);
        enc->releaseBufferForQueue(gpuB, ResourceState::CopyDestination, QueueType::Graphics);
        enc->releaseBufferForQueue(gpuC, ResourceState::CopyDestination, QueueType::Graphics);

        ICommandBuffer* cmdBuf = nullptr;
        enc->finish(&cmdBuf);

        uint64_t fenceValue = 1;
        IFence* sf = fence.get();
        SubmitDesc desc = {};
        desc.commandBuffers = &cmdBuf;
        desc.commandBufferCount = 1;
        desc.signalFences = &sf;
        desc.signalFenceValues = &fenceValue;
        desc.signalFenceCount = 1;
        REQUIRE_CALL(transferQueue->submit(desc));
    }

    transferQueue->waitOnHost();

    // Sync and acquire all
    syncToGraphicsQueue(device, transferQueue);

    {
        auto enc = graphicsQueue->createCommandEncoder();
        enc->acquireBufferFromQueue(gpuA, ResourceState::ShaderResource, QueueType::Transfer);
        enc->acquireBufferFromQueue(gpuB, ResourceState::ShaderResource, QueueType::Transfer);
        enc->acquireBufferFromQueue(gpuC, ResourceState::ShaderResource, QueueType::Transfer);
        graphicsQueue->submit(enc->finish());
        graphicsQueue->waitOnHost();
    }

    // Release staging
    stagingA = nullptr;
    stagingB = nullptr;
    stagingC = nullptr;

    // Verify all 3 buffers
    compareComputeResult(device, gpuA, makeArray<float>(1.0f, 2.0f, 3.0f, 4.0f));
    compareComputeResult(device, gpuB, makeArray<float>(100.0f, 200.0f, 300.0f, 400.0f));
    compareComputeResult(device, gpuC, makeArray<float>(-1.0f, -2.0f, -3.0f, -4.0f));
}

// ------------------------------------------------------------------
// Test: fence polling — verify non-blocking getCurrentValue works
// ------------------------------------------------------------------
GPU_TEST_CASE("async-load-fence-polling", D3D12 | Vulkan | Metal)
{
    auto transferQueue = device->getQueue(QueueType::Transfer);
    REQUIRE(transferQueue != nullptr);

    ComPtr<IFence> fence;
    REQUIRE_CALL(device->createFence({}, fence.writeRef()));

    // Initial value should be 0
    uint64_t value = 999;
    REQUIRE_CALL(fence->getCurrentValue(&value));
    CHECK(value == 0);

    // Submit empty work that signals fence to 1
    {
        uint64_t fenceValue = 1;
        IFence* sf = fence.get();
        SubmitDesc desc = {};
        desc.signalFences = &sf;
        desc.signalFenceValues = &fenceValue;
        desc.signalFenceCount = 1;
        REQUIRE_CALL(transferQueue->submit(desc));
    }

    // Wait for completion
    transferQueue->waitOnHost();

    // Fence should now be >= 1
    REQUIRE_CALL(fence->getCurrentValue(&value));
    CHECK(value >= 1);

    // Signal to 2
    {
        uint64_t fenceValue = 2;
        IFence* sf = fence.get();
        SubmitDesc desc = {};
        desc.signalFences = &sf;
        desc.signalFenceValues = &fenceValue;
        desc.signalFenceCount = 1;
        REQUIRE_CALL(transferQueue->submit(desc));
    }

    transferQueue->waitOnHost();

    REQUIRE_CALL(fence->getCurrentValue(&value));
    CHECK(value >= 2);
}
