# Async Resource Loading with Transfer Queues

This guide explains how to use the transfer queue for non-blocking resource uploads. The transfer queue runs on a dedicated DMA engine (on GPUs that support it), allowing data transfers to happen concurrently with rendering and compute work.

**Backend support**: D3D12 and Vulkan. Other backends return `nullptr` from `getQueue(QueueType::Transfer)`.

## Overview

Typical frame without async loading:

```
Graphics Queue: [upload texture][wait...][render frame]
                 ^^^^^^^^^^^^^ GPU stalled during upload
```

With async loading via transfer queue:

```
Graphics Queue: [render frame N][render frame N+1][render frame N+2]...
Transfer Queue: [upload texture A    ][upload texture B         ]
                                      ^ texture A ready to use
```

## Concepts

### Queue Types

```cpp
auto graphicsQueue = device->getQueue(QueueType::Graphics);  // rendering + compute + copy
auto computeQueue  = device->getQueue(QueueType::Compute);   // compute + copy
auto transferQueue = device->getQueue(QueueType::Transfer);  // copy only (DMA engine)
```

### Fence (Timeline Semaphore)

A fence is a GPU-GPU synchronization primitive with a monotonically increasing integer value. When the GPU signals a fence to value N, any queue waiting for value N can proceed.

```cpp
ComPtr<IFence> fence;
device->createFence({}, fence.writeRef());

// GPU signals fence to value 1 after completing work
// Another GPU queue waits for fence to reach value 1 before starting
// CPU can poll: fence->getCurrentValue(&value) — non-blocking
```

### Queue Family Ownership Transfer (QFOT)

On Vulkan, resources created with `VK_SHARING_MODE_EXCLUSIVE` (the default) belong to one queue family at a time. To use a resource on a different queue family, you must explicitly transfer ownership:

1. **Release** on the source queue (before signaling fence)
2. **Acquire** on the destination queue (after waiting on fence)

On D3D12, QFOT is not needed — the API calls are automatically no-ops.

**NVIDIA queue families** (typical):

| Family | Capabilities | Queues |
|--------|-------------|--------|
| 0 | Graphics + Compute + Transfer | 16 |
| 1 | Transfer only (DMA engine) | 2 |
| 2 | Compute only | 8 |

Since transfer is on family 1 and graphics is on family 0, **QFOT is required on NVIDIA GPUs**.

## Buffer Loading

### Step 1: Create Resources

```cpp
// GPU-local destination buffer
BufferDesc gpuDesc = {};
gpuDesc.size = dataSize;
gpuDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
gpuDesc.defaultState = ResourceState::ShaderResource;
gpuDesc.memoryType = MemoryType::DeviceLocal;

ComPtr<IBuffer> gpuBuffer;
device->createBuffer(gpuDesc, nullptr, gpuBuffer.writeRef());

// CPU-writable staging buffer
BufferDesc stagingDesc = {};
stagingDesc.size = dataSize;
stagingDesc.usage = BufferUsage::CopySource;
stagingDesc.defaultState = ResourceState::CopySource;
stagingDesc.memoryType = MemoryType::Upload;

ComPtr<IBuffer> stagingBuffer;
device->createBuffer(stagingDesc, cpuData, stagingBuffer.writeRef());
```

### Step 2: Submit Copy on Transfer Queue

```cpp
ComPtr<IFence> fence;
device->createFence({}, fence.writeRef());

auto enc = transferQueue->createCommandEncoder();

// State transitions
enc->setBufferState(stagingBuffer, ResourceState::CopySource);
enc->setBufferState(gpuBuffer, ResourceState::CopyDestination);

// Copy
enc->copyBuffer(gpuBuffer, 0, stagingBuffer, 0, dataSize);

// Release ownership to graphics queue
enc->releaseBufferForQueue(gpuBuffer, ResourceState::ShaderResource, QueueType::Graphics);

// Submit and signal fence (non-blocking)
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
transferQueue->submit(desc);

// Do NOT call waitOnHost() — the whole point is to not block.
```

### Step 3: Poll for Completion (Each Frame)

```cpp
uint64_t currentValue = 0;
fence->getCurrentValue(&currentValue);
bool transferDone = (currentValue >= fenceValue);
```

`getCurrentValue` is a non-blocking CPU query. Call it once per frame.

### Step 4: Acquire and Use (Once Ready)

When the transfer completes, acquire ownership on the graphics queue before first use:

```cpp
auto enc = graphicsQueue->createCommandEncoder();

// Acquire ownership (only needed once, after transfer completes)
enc->acquireBufferFromQueue(gpuBuffer, ResourceState::ShaderResource, QueueType::Transfer);

// Now use the buffer normally
auto pass = enc->beginComputePass();
// ...bind gpuBuffer as shader resource...
pass->end();

graphicsQueue->submit(enc->finish());
```

After acquiring, the buffer belongs to the graphics queue — no further QFOT is needed for subsequent frames.

### Step 5: Release Staging Buffer

Only release the staging buffer after the fence signals — the GPU may still be reading from it:

```cpp
if (transferDone)
    stagingBuffer = nullptr;  // safe to free
```

## Texture Loading

Texture loading follows the same pattern but uses `copyBufferToTexture` and texture QFOT.

### Prepare Staging Buffer with Texel Data

Pack texel data into a staging buffer with appropriate row pitch:

```cpp
// Calculate layout
uint32_t width = 1024, height = 1024;
Format format = Format::RGBA8Unorm;
Size texelSize = 4;  // 4 bytes per pixel for RGBA8
Size rowPitch = width * texelSize;
Size totalSize = rowPitch * height;

// Create staging buffer with texel data
BufferDesc stagingDesc = {};
stagingDesc.size = totalSize;
stagingDesc.usage = BufferUsage::CopySource;
stagingDesc.defaultState = ResourceState::CopySource;
stagingDesc.memoryType = MemoryType::Upload;

ComPtr<IBuffer> stagingBuffer;
device->createBuffer(stagingDesc, texelData, stagingBuffer.writeRef());

// Create GPU texture
TextureDesc texDesc = {};
texDesc.type = TextureType::Texture2D;
texDesc.size = {width, height, 1};
texDesc.format = format;
texDesc.mipLevelCount = 1;
texDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
texDesc.defaultState = ResourceState::ShaderResource;
texDesc.memoryType = MemoryType::DeviceLocal;

ComPtr<ITexture> gpuTexture;
device->createTexture(texDesc, nullptr, gpuTexture.writeRef());
```

### Submit Texture Copy on Transfer Queue

```cpp
auto enc = transferQueue->createCommandEncoder();

enc->setBufferState(stagingBuffer, ResourceState::CopySource);
enc->setTextureState(gpuTexture, ResourceState::CopyDestination);

// Copy from staging buffer to texture
enc->copyBufferToTexture(
    gpuTexture,
    0,                      // dstLayer
    0,                      // dstMip
    Offset3D{0, 0, 0},     // dstOffset
    stagingBuffer,
    0,                      // srcOffset in buffer
    totalSize,              // srcSize
    rowPitch,               // srcRowPitch
    Extent3D{width, height, 1}
);

// Release texture to graphics queue
enc->releaseTextureForQueue(
    gpuTexture,
    kEntireTexture,  // all subresources
    ResourceState::ShaderResource,
    QueueType::Graphics
);

// Submit with fence
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
transferQueue->submit(desc);
```

### Acquire Texture on Graphics Queue

```cpp
auto enc = graphicsQueue->createCommandEncoder();
enc->acquireTextureFromQueue(
    gpuTexture,
    kEntireTexture,
    ResourceState::ShaderResource,
    QueueType::Transfer
);
// ...use texture in render pass...
```

## Batch Loading

Batching multiple uploads into a single command buffer reduces submission overhead.

```cpp
class AsyncLoadManager
{
public:
    struct PendingResource
    {
        ComPtr<IBuffer> gpuBuffer;      // or ITexture*
        ComPtr<IBuffer> stagingBuffer;
        uint64_t fenceValue;
        bool acquired = false;
    };

private:
    IDevice* m_device;
    ICommandQueue* m_transferQueue;
    ComPtr<IFence> m_fence;
    uint64_t m_nextFenceValue = 0;
    std::vector<PendingResource> m_pending;

public:
    void init(IDevice* device)
    {
        m_device = device;
        m_transferQueue = device->getQueue(QueueType::Transfer);
        m_device->createFence({}, m_fence.writeRef());
    }

    /// Queue a buffer for async upload. Returns index into pending list.
    size_t queueBufferUpload(const BufferDesc& gpuDesc, const void* data, Size dataSize)
    {
        PendingResource res;

        // Create GPU buffer
        m_device->createBuffer(gpuDesc, nullptr, res.gpuBuffer.writeRef());

        // Create staging buffer
        BufferDesc stagingDesc = {};
        stagingDesc.size = dataSize;
        stagingDesc.usage = BufferUsage::CopySource;
        stagingDesc.defaultState = ResourceState::CopySource;
        stagingDesc.memoryType = MemoryType::Upload;
        m_device->createBuffer(stagingDesc, data, res.stagingBuffer.writeRef());

        res.fenceValue = 0;  // assigned on flush
        m_pending.push_back(std::move(res));
        return m_pending.size() - 1;
    }

    /// Submit all queued uploads in one batch.
    void flush()
    {
        if (m_pending.empty())
            return;

        auto enc = m_transferQueue->createCommandEncoder();

        for (auto& res : m_pending)
        {
            if (res.fenceValue > 0)
                continue;  // already submitted

            enc->setBufferState(res.stagingBuffer, ResourceState::CopySource);
            enc->setBufferState(res.gpuBuffer, ResourceState::CopyDestination);
            enc->copyBuffer(res.gpuBuffer, 0, res.stagingBuffer, 0, res.stagingBuffer->getDesc()->size);
            enc->releaseBufferForQueue(res.gpuBuffer, ResourceState::ShaderResource, QueueType::Graphics);
        }

        ICommandBuffer* cmdBuf = nullptr;
        enc->finish(&cmdBuf);

        ++m_nextFenceValue;
        IFence* sf = m_fence.get();
        SubmitDesc desc = {};
        desc.commandBuffers = &cmdBuf;
        desc.commandBufferCount = 1;
        desc.signalFences = &sf;
        desc.signalFenceValues = &m_nextFenceValue;
        desc.signalFenceCount = 1;
        m_transferQueue->submit(desc);

        // Tag all pending resources with this fence value
        for (auto& res : m_pending)
        {
            if (res.fenceValue == 0)
                res.fenceValue = m_nextFenceValue;
        }
    }

    /// Poll for completed uploads. Returns newly ready resources.
    /// Call once per frame.
    std::vector<IBuffer*> poll()
    {
        uint64_t currentValue = 0;
        m_fence->getCurrentValue(&currentValue);

        std::vector<IBuffer*> ready;
        for (auto& res : m_pending)
        {
            if (!res.acquired && res.fenceValue > 0 && currentValue >= res.fenceValue)
            {
                ready.push_back(res.gpuBuffer.get());
                res.stagingBuffer = nullptr;  // release staging memory
            }
        }
        return ready;
    }

    /// Call on graphics queue encoder to acquire newly ready resources.
    void acquireReady(ICommandEncoder* graphicsEncoder, std::span<IBuffer*> readyBuffers)
    {
        for (auto* buf : readyBuffers)
        {
            graphicsEncoder->acquireBufferFromQueue(buf, ResourceState::ShaderResource, QueueType::Transfer);

            // Mark as acquired so we don't acquire again
            for (auto& res : m_pending)
            {
                if (res.gpuBuffer.get() == buf)
                    res.acquired = true;
            }
        }
    }
};
```

### Usage

```cpp
AsyncLoadManager loader;
loader.init(device);

// Queue uploads (non-blocking, just prepares data)
BufferDesc meshDesc = {};
meshDesc.size = vertexDataSize;
meshDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
meshDesc.defaultState = ResourceState::ShaderResource;
meshDesc.memoryType = MemoryType::DeviceLocal;

loader.queueBufferUpload(meshDesc, vertexData, vertexDataSize);
loader.queueBufferUpload(meshDesc, indexData, indexDataSize);
loader.flush();  // submits all uploads in one batch

// Render loop
while (running)
{
    auto readyBuffers = loader.poll();

    auto enc = graphicsQueue->createCommandEncoder();

    if (!readyBuffers.empty())
        loader.acquireReady(enc, readyBuffers);

    // ...render using acquired resources...

    graphicsQueue->submit(enc->finish());
}
```

## Common Pitfalls

### 1. Forgetting GPU-GPU Sync

`waitOnHost()` only blocks the CPU. It does NOT make writes visible to other GPU queues.

```cpp
// WRONG — graphics queue may not see transfer queue's writes
transferQueue->submit(enc->finish());
transferQueue->waitOnHost();
device->readBuffer(buffer, ...);  // readBuffer uses graphics queue internally

// CORRECT — use fence for GPU-GPU sync
transferQueue->submit({..., signalFence});
graphicsQueue->submit({..., waitFence});
device->readBuffer(buffer, ...);
```

### 2. Releasing Staging Buffer Too Early

The staging buffer must outlive the GPU's read from it. Only release after the fence signals:

```cpp
// WRONG
transferQueue->submit(enc->finish());
stagingBuffer = nullptr;  // GPU may still be reading!

// CORRECT
transferQueue->submit(enc->finish());
// ...later, when fence->getCurrentValue() >= signaled value...
stagingBuffer = nullptr;  // now safe
```

### 3. Skipping QFOT

On NVIDIA and AMD, the transfer queue is on a dedicated queue family. Accessing a resource on the graphics queue without QFOT after writing it on the transfer queue is undefined behavior on Vulkan:

```cpp
// WRONG on Vulkan with dedicated transfer family
enc->copyBuffer(gpuBuffer, ...);
transferQueue->submit(enc->finish());
// ...fence sync...
// graphics queue uses gpuBuffer without acquire — UNDEFINED BEHAVIOR

// CORRECT
enc->copyBuffer(gpuBuffer, ...);
enc->releaseBufferForQueue(gpuBuffer, ResourceState::ShaderResource, QueueType::Graphics);
transferQueue->submit(enc->finish());
// ...fence sync...
graphicsEncoder->acquireBufferFromQueue(gpuBuffer, ResourceState::ShaderResource, QueueType::Transfer);
```

On D3D12, the QFOT calls are no-ops, so this code is safe and zero-cost on all backends.

### 4. Using Unsupported Operations on Transfer Queue

The transfer queue only supports copy operations. Attempting other operations triggers a debug layer error:

```cpp
// WRONG — compute passes not supported on transfer queue
auto pass = transferEncoder->beginComputePass();  // debug layer error

// Transfer queue supports:
//   copyBuffer, copyTexture, copyTextureToBuffer, copyBufferToTexture, uploadBufferData
```

### 5. Blocking on Every Upload

If you call `waitOnHost()` after every upload, you lose all async benefit:

```cpp
// WRONG — defeats the purpose of async loading
transferQueue->submit(enc->finish());
transferQueue->waitOnHost();  // blocks until copy finishes

// CORRECT — submit and poll later
transferQueue->submit({..., signalFence});
// ...continue rendering...
// ...next frame: poll fence->getCurrentValue()...
```

## Graceful Fallback

Not all backends support transfer queues. Handle this gracefully:

```cpp
auto transferQueue = device->getQueue(QueueType::Transfer);
auto graphicsQueue = device->getQueue(QueueType::Graphics);

// Fall back to graphics queue if transfer queue unavailable
ICommandQueue* uploadQueue = transferQueue ? transferQueue.get() : graphicsQueue.get();
bool needQFOT = (transferQueue != nullptr);

auto enc = uploadQueue->createCommandEncoder();
enc->copyBuffer(gpuBuffer, 0, stagingBuffer, 0, dataSize);

if (needQFOT)
    enc->releaseBufferForQueue(gpuBuffer, ResourceState::ShaderResource, QueueType::Graphics);

uploadQueue->submit(enc->finish());

// Sync and acquire only if using transfer queue
if (needQFOT)
{
    // ...fence signal/wait...
    graphicsEncoder->acquireBufferFromQueue(gpuBuffer, ResourceState::ShaderResource, QueueType::Transfer);
}
```
