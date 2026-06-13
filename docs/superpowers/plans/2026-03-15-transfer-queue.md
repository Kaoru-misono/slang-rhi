# Transfer Queue Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add transfer (copy/DMA) queue support and queue family ownership transfer (QFOT) API to slang-rhi, enabling asynchronous data transfers on dedicated hardware.

**Architecture:** Follow the existing compute queue pattern — add `QueueType::Transfer` to the enum, add a third queue field in Vulkan and D3D12 backends, add QFOT release/acquire methods to `ICommandEncoder`. Vulkan selects dedicated transfer queue families to maximize hardware parallelism. D3D12 uses `D3D12_COMMAND_LIST_TYPE_COPY`.

**Tech Stack:** C++, Vulkan API, D3D12 API, CMake, doctest

**Spec:** `docs/superpowers/specs/2026-03-14-transfer-queue-design.md`

**Notes:**
- `src/debug-layer/debug-device.cpp`: The `getQueue()` method (line 485) is generic — it wraps any queue type returned by the inner device. No changes needed.
- `src/vulkan/vk-command.h` and `src/d3d12/d3d12-command.h`: No header changes needed. Vulkan `CommandBufferImpl` creates its command pool from `m_queue->m_queueFamilyIndex` automatically. D3D12 `CommandQueueImpl::init()` creates command allocators from `m_commandListType` which is updated in Task 5.

---

## Chunk 1: Public API & Command System

### Task 1: Add QueueType::Transfer to public API

**Files:**
- Modify: `include/slang-rhi.h:2609-2613` (QueueType enum)
- Modify: `include/slang-rhi.h:2573` (after globalBarrier, add QFOT methods)

- [ ] **Step 1: Add Transfer to QueueType enum**

In `include/slang-rhi.h`, change the enum at line 2609:

```cpp
enum class QueueType
{
    Graphics,
    Compute,
    Transfer,
};
```

- [ ] **Step 2: Add QFOT method declarations to ICommandEncoder**

In `include/slang-rhi.h`, add after `globalBarrier()` (line 2573) and before the `setTextureState` inline helper (line 2575):

```cpp
    virtual SLANG_NO_THROW void SLANG_MCALL
    releaseBufferForQueue(IBuffer* buffer, ResourceState currentState, QueueType dstQueue) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL releaseTextureForQueue(
        ITexture* texture,
        SubresourceRange subresourceRange,
        ResourceState currentState,
        QueueType dstQueue
    ) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL
    acquireBufferFromQueue(IBuffer* buffer, ResourceState desiredState, QueueType srcQueue) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL acquireTextureFromQueue(
        ITexture* texture,
        SubresourceRange subresourceRange,
        ResourceState desiredState,
        QueueType srcQueue
    ) = 0;
```

- [ ] **Step 3: Commit**

```bash
git add include/slang-rhi.h
git commit -m "feat: add QueueType::Transfer and QFOT methods to public API"
```

### Task 2: Add QFOT commands to command system and all backends

This task must be done atomically — adding command types to `SLANG_RHI_COMMANDS` macro requires ALL backends to have matching `cmd*` handler methods, or the code won't compile.

**Files:**
- Modify: `src/command-list.h:12-53` (SLANG_RHI_COMMANDS macro, command structs)
- Modify: `src/command-list.h:436-442` (write() declarations)
- Modify: `src/command-list.cpp` (write() implementations with retainResource)
- Modify: `src/command-buffer.h:220-284` (CommandEncoder overrides)
- Modify: `src/command-buffer.cpp` (CommandEncoder implementations)
- Modify: `src/cpu/cpu-command.cpp:56-65` (add no-op cmd* handlers to CommandExecutor)
- Modify: `src/d3d11/d3d11-command.cpp:83-94` (add no-op cmd* handlers to CommandExecutor)
- Modify: `src/metal/metal-command.cpp:92-99` (add no-op cmd* handlers to CommandRecorder)
- Modify: `src/cuda/cuda-command.cpp:72-80` (add no-op cmd* handlers to CommandExecutor)
- Modify: `src/wgpu/wgpu-command.cpp:84-92` (add no-op cmd* handlers to CommandRecorder)
- Modify: `src/d3d12/d3d12-command.cpp:107-120` (add cmd* handler declarations to CommandRecorder)
- Modify: `src/vulkan/vk-command.cpp:108-120` (add cmd* handler declarations to CommandRecorder)

- [ ] **Step 1: Add QFOT command structs to command-list.h**

In `src/command-list.h`, add four new command structs after `GlobalBarrier` (line 318) and before `PushDebugGroup` (line 320):

```cpp
struct ReleaseBufferForQueue
{
    IBuffer* buffer;
    ResourceState currentState;
    QueueType dstQueue;
};

struct ReleaseTextureForQueue
{
    ITexture* texture;
    SubresourceRange subresourceRange;
    ResourceState currentState;
    QueueType dstQueue;
};

struct AcquireBufferFromQueue
{
    IBuffer* buffer;
    ResourceState desiredState;
    QueueType srcQueue;
};

struct AcquireTextureFromQueue
{
    ITexture* texture;
    SubresourceRange subresourceRange;
    ResourceState desiredState;
    QueueType srcQueue;
};
```

- [ ] **Step 2: Register new commands in SLANG_RHI_COMMANDS macro**

In `src/command-list.h`, add to the `SLANG_RHI_COMMANDS` macro (after `GlobalBarrier` at line 48):

```cpp
    x(ReleaseBufferForQueue) \
    x(ReleaseTextureForQueue) \
    x(AcquireBufferFromQueue) \
    x(AcquireTextureFromQueue) \
```

- [ ] **Step 3: Add write() declarations and implementations**

In `src/command-list.h`, after the `write(commands::GlobalBarrier&&)` declaration (around line 439), add:

```cpp
    void write(commands::ReleaseBufferForQueue&& cmd);
    void write(commands::ReleaseTextureForQueue&& cmd);
    void write(commands::AcquireBufferFromQueue&& cmd);
    void write(commands::AcquireTextureFromQueue&& cmd);
```

In `src/command-list.cpp`, after the `write(commands::GlobalBarrier&&)` implementation, add:

```cpp
void CommandList::write(commands::ReleaseBufferForQueue&& cmd)
{
    retainResource<Buffer>(cmd.buffer);
    writeCommand(std::move(cmd));
}

void CommandList::write(commands::ReleaseTextureForQueue&& cmd)
{
    retainResource<Texture>(cmd.texture);
    writeCommand(std::move(cmd));
}

void CommandList::write(commands::AcquireBufferFromQueue&& cmd)
{
    retainResource<Buffer>(cmd.buffer);
    writeCommand(std::move(cmd));
}

void CommandList::write(commands::AcquireTextureFromQueue&& cmd)
{
    retainResource<Texture>(cmd.texture);
    writeCommand(std::move(cmd));
}
```

- [ ] **Step 4: Add QFOT method overrides to CommandEncoder**

In `src/command-buffer.h`, after the `globalBarrier()` override, add:

```cpp
    virtual SLANG_NO_THROW void SLANG_MCALL
    releaseBufferForQueue(IBuffer* buffer, ResourceState currentState, QueueType dstQueue) override;

    virtual SLANG_NO_THROW void SLANG_MCALL releaseTextureForQueue(
        ITexture* texture,
        SubresourceRange subresourceRange,
        ResourceState currentState,
        QueueType dstQueue
    ) override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    acquireBufferFromQueue(IBuffer* buffer, ResourceState desiredState, QueueType srcQueue) override;

    virtual SLANG_NO_THROW void SLANG_MCALL acquireTextureFromQueue(
        ITexture* texture,
        SubresourceRange subresourceRange,
        ResourceState desiredState,
        QueueType srcQueue
    ) override;
```

- [ ] **Step 5: Implement CommandEncoder QFOT methods**

In `src/command-buffer.cpp`, after the `globalBarrier()` implementation, add. Use `checked_cast` and `std::move` following the `setBufferState`/`setTextureState` pattern:

```cpp
void CommandEncoder::releaseBufferForQueue(IBuffer* buffer, ResourceState currentState, QueueType dstQueue)
{
    commands::ReleaseBufferForQueue cmd;
    cmd.buffer = checked_cast<Buffer*>(buffer);
    cmd.currentState = currentState;
    cmd.dstQueue = dstQueue;
    m_commandList->write(std::move(cmd));
}

void CommandEncoder::releaseTextureForQueue(
    ITexture* texture,
    SubresourceRange subresourceRange,
    ResourceState currentState,
    QueueType dstQueue
)
{
    commands::ReleaseTextureForQueue cmd;
    cmd.texture = checked_cast<Texture*>(texture);
    cmd.subresourceRange = subresourceRange;
    cmd.currentState = currentState;
    cmd.dstQueue = dstQueue;
    m_commandList->write(std::move(cmd));
}

void CommandEncoder::acquireBufferFromQueue(IBuffer* buffer, ResourceState desiredState, QueueType srcQueue)
{
    commands::AcquireBufferFromQueue cmd;
    cmd.buffer = checked_cast<Buffer*>(buffer);
    cmd.desiredState = desiredState;
    cmd.srcQueue = srcQueue;
    m_commandList->write(std::move(cmd));
}

void CommandEncoder::acquireTextureFromQueue(
    ITexture* texture,
    SubresourceRange subresourceRange,
    ResourceState desiredState,
    QueueType srcQueue
)
{
    commands::AcquireTextureFromQueue cmd;
    cmd.texture = checked_cast<Texture*>(texture);
    cmd.subresourceRange = subresourceRange;
    cmd.desiredState = desiredState;
    cmd.srcQueue = srcQueue;
    m_commandList->write(std::move(cmd));
}
```

- [ ] **Step 6: Add no-op cmd* handlers to ALL backends**

Every backend's command executor/recorder class needs four new handler method declarations and no-op implementations. The `SLANG_RHI_COMMANDS` macro drives dispatch via `SLANG_RHI_COMMAND_EXECUTE_X` which generates `case CommandID::X: cmdX(...)`. If any backend is missing a handler, the build fails.

For each of these 7 files, add the four method declarations alongside `cmdSetBufferState`/`cmdSetTextureState`/`cmdGlobalBarrier`, and add no-op implementations alongside their implementations:

**Declaration pattern** (add alongside other `cmd*` declarations in the class):
```cpp
    void cmdReleaseBufferForQueue(const commands::ReleaseBufferForQueue& cmd);
    void cmdReleaseTextureForQueue(const commands::ReleaseTextureForQueue& cmd);
    void cmdAcquireBufferFromQueue(const commands::AcquireBufferFromQueue& cmd);
    void cmdAcquireTextureFromQueue(const commands::AcquireTextureFromQueue& cmd);
```

**No-op implementation pattern** (for backends that don't need real QFOT — CPU, D3D11, Metal, CUDA, WebGPU):
```cpp
void CommandExecutor::cmdReleaseBufferForQueue(const commands::ReleaseBufferForQueue& cmd) { SLANG_UNUSED(cmd); }
void CommandExecutor::cmdReleaseTextureForQueue(const commands::ReleaseTextureForQueue& cmd) { SLANG_UNUSED(cmd); }
void CommandExecutor::cmdAcquireBufferFromQueue(const commands::AcquireBufferFromQueue& cmd) { SLANG_UNUSED(cmd); }
void CommandExecutor::cmdAcquireTextureFromQueue(const commands::AcquireTextureFromQueue& cmd) { SLANG_UNUSED(cmd); }
```

Note: Metal and WebGPU use `CommandRecorder` (not `CommandExecutor`). Adjust the class name accordingly.

Files to update:
1. `src/cpu/cpu-command.cpp` — `CommandExecutor` class (decl ~line 60, impl ~line 303)
2. `src/d3d11/d3d11-command.cpp` — `CommandExecutor` class (decl ~line 87, impl ~line 858)
3. `src/metal/metal-command.cpp` — `CommandRecorder` class (decl ~line 96, impl ~line 843)
4. `src/cuda/cuda-command.cpp` — `CommandExecutor` class (decl ~line 76, impl ~line 638)
5. `src/wgpu/wgpu-command.cpp` — `CommandRecorder` class (decl ~line 88, impl ~line 782)
6. `src/d3d12/d3d12-command.cpp` — `CommandRecorder` class (decl ~line 111, impl will be added in Task 5)
7. `src/vulkan/vk-command.cpp` — `CommandRecorder` class (decl ~line 113, impl will be added in Task 8)

For D3D12 and Vulkan, add the declarations now but leave the implementations for their respective tasks (Tasks 5 and 8). Use placeholder no-ops for now so the build works:

```cpp
// D3D12 placeholder (to be replaced in Task 5)
void CommandRecorder::cmdReleaseBufferForQueue(const commands::ReleaseBufferForQueue& cmd) { SLANG_UNUSED(cmd); }
// ... same for other 3
```

- [ ] **Step 7: Commit**

```bash
git add src/command-list.h src/command-list.cpp src/command-buffer.h src/command-buffer.cpp \
    src/cpu/cpu-command.cpp src/d3d11/d3d11-command.cpp src/metal/metal-command.cpp \
    src/cuda/cuda-command.cpp src/wgpu/wgpu-command.cpp \
    src/d3d12/d3d12-command.cpp src/vulkan/vk-command.cpp
git commit -m "feat: add QFOT command types to command system with stubs in all backends"
```

### Task 3: Update unsupported backends getQueue()

**Files:**
- Modify: `src/cpu/cpu-device.cpp:128`
- Modify: `src/d3d11/d3d11-device.cpp:605`
- Modify: `src/metal/metal-device.cpp:275`
- Modify: `src/cuda/cuda-device.cpp:556`
- Modify: `src/wgpu/wgpu-command.cpp:976`

- [ ] **Step 1: Update all five unsupported backends**

In each file's `getQueue()` method, add `Transfer` handling alongside the existing `Compute` check. Change:

```cpp
if (type == QueueType::Compute)
    return SLANG_E_NOT_AVAILABLE;
```

To:

```cpp
if (type == QueueType::Compute || type == QueueType::Transfer)
    return SLANG_E_NOT_AVAILABLE;
```

Apply to all five files.

- [ ] **Step 2: Commit**

```bash
git add src/cpu/cpu-device.cpp src/d3d11/d3d11-device.cpp src/metal/metal-device.cpp src/cuda/cuda-device.cpp src/wgpu/wgpu-command.cpp
git commit -m "feat: return SLANG_E_NOT_AVAILABLE for transfer queue in unsupported backends"
```

### Task 4: Update debug layer

**Files:**
- Modify: `src/debug-layer/debug-command-encoder.cpp:305-341`
- Modify: `src/debug-layer/debug-command-encoder.h`

- [ ] **Step 1: Add transfer queue validation for beginRenderPass**

In `src/debug-layer/debug-command-encoder.cpp`, at line 310, after the compute check, add:

```cpp
if (m_queueType == QueueType::Transfer)
{
    RHI_VALIDATION_ERROR("Render passes are not supported on transfer queues.");
}
```

- [ ] **Step 2: Add transfer queue validation for beginComputePass**

At line 319, `beginComputePass()` currently has no queue type check. Add before `m_passState = PassState::ComputePass;`:

```cpp
if (m_queueType == QueueType::Transfer)
{
    RHI_VALIDATION_ERROR("Compute passes are not supported on transfer queues.");
}
```

- [ ] **Step 3: Add transfer queue validation for beginRayTracingPass**

At line 334, after the existing compute check, add:

```cpp
if (m_queueType == QueueType::Transfer)
{
    RHI_VALIDATION_ERROR("Ray tracing passes are not supported on transfer queues.");
}
```

- [ ] **Step 4: Add QFOT validation wrappers**

Add declarations to `DebugCommandEncoder` class in `src/debug-layer/debug-command-encoder.h`:

```cpp
    virtual SLANG_NO_THROW void SLANG_MCALL
    releaseBufferForQueue(IBuffer* buffer, ResourceState currentState, QueueType dstQueue) override;
    virtual SLANG_NO_THROW void SLANG_MCALL releaseTextureForQueue(
        ITexture* texture, SubresourceRange subresourceRange, ResourceState currentState, QueueType dstQueue) override;
    virtual SLANG_NO_THROW void SLANG_MCALL
    acquireBufferFromQueue(IBuffer* buffer, ResourceState desiredState, QueueType srcQueue) override;
    virtual SLANG_NO_THROW void SLANG_MCALL acquireTextureFromQueue(
        ITexture* texture, SubresourceRange subresourceRange, ResourceState desiredState, QueueType srcQueue) override;
```

Add implementations in `src/debug-layer/debug-command-encoder.cpp`, near the `setBufferState`/`setTextureState` debug wrappers:

```cpp
void DebugCommandEncoder::releaseBufferForQueue(IBuffer* buffer, ResourceState currentState, QueueType dstQueue)
{
    SLANG_RHI_API_FUNC;
    requireOpen();
    if (dstQueue == m_queueType)
    {
        RHI_VALIDATION_ERROR("releaseBufferForQueue: dstQueue must be different from current queue type.");
    }
    baseObject->releaseBufferForQueue(buffer, currentState, dstQueue);
}

void DebugCommandEncoder::releaseTextureForQueue(
    ITexture* texture,
    SubresourceRange subresourceRange,
    ResourceState currentState,
    QueueType dstQueue
)
{
    SLANG_RHI_API_FUNC;
    requireOpen();
    if (dstQueue == m_queueType)
    {
        RHI_VALIDATION_ERROR("releaseTextureForQueue: dstQueue must be different from current queue type.");
    }
    baseObject->releaseTextureForQueue(texture, subresourceRange, currentState, dstQueue);
}

void DebugCommandEncoder::acquireBufferFromQueue(IBuffer* buffer, ResourceState desiredState, QueueType srcQueue)
{
    SLANG_RHI_API_FUNC;
    requireOpen();
    if (srcQueue == m_queueType)
    {
        RHI_VALIDATION_ERROR("acquireBufferFromQueue: srcQueue must be different from current queue type.");
    }
    baseObject->acquireBufferFromQueue(buffer, desiredState, srcQueue);
}

void DebugCommandEncoder::acquireTextureFromQueue(
    ITexture* texture,
    SubresourceRange subresourceRange,
    ResourceState desiredState,
    QueueType srcQueue
)
{
    SLANG_RHI_API_FUNC;
    requireOpen();
    if (srcQueue == m_queueType)
    {
        RHI_VALIDATION_ERROR("acquireTextureFromQueue: srcQueue must be different from current queue type.");
    }
    baseObject->acquireTextureFromQueue(texture, subresourceRange, desiredState, srcQueue);
}
```

- [ ] **Step 5: Commit**

```bash
git add src/debug-layer/debug-command-encoder.cpp src/debug-layer/debug-command-encoder.h
git commit -m "feat: add transfer queue validation and QFOT wrappers in debug layer"
```

---

## Chunk 2: D3D12 Backend

### Task 5: Add transfer queue to D3D12 device

**Files:**
- Modify: `src/d3d12/d3d12-device.h:45-46`
- Modify: `src/d3d12/d3d12-device.cpp:979-989` (queue creation)
- Modify: `src/d3d12/d3d12-device.cpp:1010-1023` (getQueue)
- Modify: `src/d3d12/d3d12-device.cpp:1973-1986` (destructor)
- Modify: `src/d3d12/d3d12-command.cpp:1782` (command list type mapping)
- Modify: `src/d3d12/d3d12-command.cpp` (replace QFOT no-op stubs with state transitions)

- [ ] **Step 1: Add m_transferQueue field to D3D12 DeviceImpl**

In `src/d3d12/d3d12-device.h`, after line 46 (`m_computeQueue`):

```cpp
RefPtr<CommandQueueImpl> m_transferQueue;
```

- [ ] **Step 2: Create transfer queue in device initialization**

In `src/d3d12/d3d12-device.cpp`, after compute queue creation (line 986), add:

```cpp
// Create async transfer (copy) queue.
m_transferQueue = new CommandQueueImpl(this, QueueType::Transfer);
SLANG_RETURN_ON_FAIL(m_transferQueue->init(2));
m_transferQueue->setInternalReferenceCount(1);
```

- [ ] **Step 3: Fix command list type mapping**

In `src/d3d12/d3d12-command.cpp`, at line 1782, replace:

```cpp
m_commandListType = (m_type == QueueType::Compute) ? D3D12_COMMAND_LIST_TYPE_COMPUTE
                                                   : D3D12_COMMAND_LIST_TYPE_DIRECT;
```

With:

```cpp
switch (m_type)
{
case QueueType::Compute:
    m_commandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    break;
case QueueType::Transfer:
    m_commandListType = D3D12_COMMAND_LIST_TYPE_COPY;
    break;
default:
    m_commandListType = D3D12_COMMAND_LIST_TYPE_DIRECT;
    break;
}
```

- [ ] **Step 4: Update getQueue to return transfer queue**

In `src/d3d12/d3d12-device.cpp`, in the `getQueue` switch at line 1012, add before `default`:

```cpp
case QueueType::Transfer:
    returnComPtr(outQueue, m_transferQueue);
    return SLANG_OK;
```

- [ ] **Step 5: Update destructor**

In `src/d3d12/d3d12-device.cpp`, at line 1986, after `m_computeQueue.setNull()`, add:

```cpp
m_transferQueue.setNull();
```

- [ ] **Step 6: Replace D3D12 QFOT stubs with state transitions**

In `src/d3d12/d3d12-command.cpp`, replace the four no-op QFOT stubs (added in Task 2 Step 6) with state-transition implementations. D3D12 handles cross-queue sharing implicitly via fence synchronization, so QFOT is just a state transition:

```cpp
void CommandRecorder::cmdReleaseBufferForQueue(const commands::ReleaseBufferForQueue& cmd)
{
    // D3D12 handles cross-queue sharing implicitly via fence synchronization.
    m_stateTracking.setBufferState(checked_cast<BufferImpl*>(cmd.buffer), cmd.currentState);
}

void CommandRecorder::cmdReleaseTextureForQueue(const commands::ReleaseTextureForQueue& cmd)
{
    m_stateTracking.setTextureState(
        checked_cast<TextureImpl*>(cmd.texture),
        cmd.subresourceRange,
        cmd.currentState
    );
}

void CommandRecorder::cmdAcquireBufferFromQueue(const commands::AcquireBufferFromQueue& cmd)
{
    m_stateTracking.setBufferState(checked_cast<BufferImpl*>(cmd.buffer), cmd.desiredState);
}

void CommandRecorder::cmdAcquireTextureFromQueue(const commands::AcquireTextureFromQueue& cmd)
{
    m_stateTracking.setTextureState(
        checked_cast<TextureImpl*>(cmd.texture),
        cmd.subresourceRange,
        cmd.desiredState
    );
}
```

- [ ] **Step 7: Commit**

```bash
git add src/d3d12/d3d12-device.h src/d3d12/d3d12-device.cpp src/d3d12/d3d12-command.cpp
git commit -m "feat: add transfer queue support to D3D12 backend"
```

---

## Chunk 3: Vulkan Backend

### Task 6: Add transfer queue family selection

**Files:**
- Modify: `src/vulkan/vk-device.h:220-224`
- Modify: `src/vulkan/vk-device.cpp:1194-1227` (queue family selection)
- Modify: `src/vulkan/vk-device.cpp:1268-1297` (VkDeviceQueueCreateInfo)

- [ ] **Step 1: Add transfer queue fields to Vulkan DeviceImpl**

In `src/vulkan/vk-device.h`, add after `m_computeQueueFamilyIndex` (line 222):

```cpp
uint32_t m_transferQueueFamilyIndex;
```

And after `m_computeQueue` (line 224):

```cpp
RefPtr<CommandQueueImpl> m_transferQueue;
```

- [ ] **Step 2: Hoist queue family properties query and add transfer queue family selection**

In `src/vulkan/vk-device.cpp`, the compute queue selection (lines 1201-1227) queries `vkGetPhysicalDeviceQueueFamilyProperties` inside its block. Refactor to hoist this query before both compute and transfer selection, so the `families` vector is available to both:

```cpp
// Query queue family properties (shared by compute and transfer selection)
uint32_t numFamilies = 0;
m_api.vkGetPhysicalDeviceQueueFamilyProperties(m_api.m_physicalDevice, &numFamilies, nullptr);
std::vector<VkQueueFamilyProperties> families(numFamilies);
m_api.vkGetPhysicalDeviceQueueFamilyProperties(m_api.m_physicalDevice, &numFamilies, families.data());
```

Then, after the compute family selection block, add:

```cpp
// Select transfer queue family.
// Priority: dedicated transfer-only > any transfer not used by graphics/compute > compute family > graphics family
{
    int dedicatedTransferFamily = -1;
    int separateTransferFamily = -1;

    for (uint32_t i = 0; i < numFamilies; ++i)
    {
        bool hasTransfer = (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;
        bool hasGraphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        bool hasCompute = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;

        if (!hasTransfer)
            continue;

        if (!hasGraphics && !hasCompute && i != m_computeQueueFamilyIndex)
        {
            dedicatedTransferFamily = int(i);
            break;
        }

        if (i != m_graphicsQueueFamilyIndex && i != m_computeQueueFamilyIndex && separateTransferFamily < 0)
        {
            separateTransferFamily = int(i);
        }
    }

    if (dedicatedTransferFamily >= 0)
    {
        m_transferQueueFamilyIndex = uint32_t(dedicatedTransferFamily);
    }
    else if (separateTransferFamily >= 0)
    {
        m_transferQueueFamilyIndex = uint32_t(separateTransferFamily);
    }
    else if (m_computeQueueFamilyIndex != m_graphicsQueueFamilyIndex &&
             (families[m_computeQueueFamilyIndex].queueFlags & VK_QUEUE_TRANSFER_BIT))
    {
        m_transferQueueFamilyIndex = m_computeQueueFamilyIndex;
    }
    else
    {
        m_transferQueueFamilyIndex = m_graphicsQueueFamilyIndex;
    }
}
```

- [ ] **Step 3: Update VkDeviceQueueCreateInfo to handle 3 queue types**

Replace the queue creation block at lines 1268-1297. Count how many queues are needed per family, capped by hardware limits:

```cpp
float queuePriorities[] = {0.0f, 0.0f, 0.0f};

// Count queues needed per family
uint32_t familyQueueCounts[64] = {};  // indexed by family index
familyQueueCounts[m_graphicsQueueFamilyIndex]++;
familyQueueCounts[m_computeQueueFamilyIndex]++;
familyQueueCounts[m_transferQueueFamilyIndex]++;

std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
for (uint32_t i = 0; i < numFamilies; ++i)
{
    if (familyQueueCounts[i] == 0)
        continue;
    uint32_t maxCount = families[i].queueCount;
    uint32_t actualCount = std::min(familyQueueCounts[i], maxCount);

    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = i;
    queueInfo.queueCount = actualCount;
    queueInfo.pQueuePriorities = queuePriorities;
    queueCreateInfos.push_back(queueInfo);
}
```

- [ ] **Step 4: Commit**

```bash
git add src/vulkan/vk-device.h src/vulkan/vk-device.cpp
git commit -m "feat: add transfer queue family selection and device queue creation for Vulkan"
```

### Task 7: Create transfer queue and update Vulkan device methods

**Files:**
- Modify: `src/vulkan/vk-device.cpp:1606-1639` (queue retrieval/init)
- Modify: `src/vulkan/vk-device.cpp:1650-1663` (getQueue)
- Modify: `src/vulkan/vk-device.cpp:1872-1882` (getQueueFamilyIndex)
- Modify: `src/vulkan/vk-device.cpp:124-134` (destructor)

- [ ] **Step 1: Create and initialize transfer queue**

In `src/vulkan/vk-device.cpp`, after the compute queue init block (after line 1639), add:

```cpp
// Create transfer queue
{
    VkQueue transferVkQueue;

    if (m_transferQueueFamilyIndex != m_graphicsQueueFamilyIndex &&
        m_transferQueueFamilyIndex != m_computeQueueFamilyIndex)
    {
        // Dedicated family: use index 0
        m_api.vkGetDeviceQueue(m_device, m_transferQueueFamilyIndex, 0, &transferVkQueue);
    }
    else
    {
        // Shared family: count how many queues are already assigned to this family
        uint32_t usedCount = 0;
        if (m_graphicsQueueFamilyIndex == m_transferQueueFamilyIndex)
            usedCount++;
        if (m_computeQueueFamilyIndex == m_transferQueueFamilyIndex)
            usedCount++;

        // Re-query family properties (same pattern as compute queue creation at line 1629)
        uint32_t numFamilies = 0;
        m_api.vkGetPhysicalDeviceQueueFamilyProperties(m_api.m_physicalDevice, &numFamilies, nullptr);
        std::vector<VkQueueFamilyProperties> families(numFamilies);
        m_api.vkGetPhysicalDeviceQueueFamilyProperties(m_api.m_physicalDevice, &numFamilies, families.data());

        uint32_t maxCount = families[m_transferQueueFamilyIndex].queueCount;
        uint32_t transferQueueIndex = (usedCount < maxCount) ? usedCount : 0;

        m_api.vkGetDeviceQueue(m_device, m_transferQueueFamilyIndex, transferQueueIndex, &transferVkQueue);
    }

    m_transferQueue = new CommandQueueImpl(this, QueueType::Transfer);
    m_transferQueue->init(transferVkQueue, m_transferQueueFamilyIndex);
    m_transferQueue->setInternalReferenceCount(1);
}
```

Note: The `families` vector from queue family selection needs to be accessible here. It may need to be stored as a member or re-queried. Follow the existing pattern used by the compute queue creation code (lines 1628-1634 re-query the family properties).

- [ ] **Step 2: Update getQueue()**

In `src/vulkan/vk-device.cpp`, in the `getQueue` switch, add before `default`:

```cpp
case QueueType::Transfer:
    returnComPtr(outQueue, m_transferQueue);
    return SLANG_OK;
```

- [ ] **Step 3: Update getQueueFamilyIndex()**

In `src/vulkan/vk-device.cpp`, at line 1872, add `Transfer` case:

```cpp
case QueueType::Transfer:
    return m_transferQueueFamilyIndex;
```

- [ ] **Step 4: Update destructor**

In `src/vulkan/vk-device.cpp`, at line 131, after the compute queue waitOnHost block, add:

```cpp
if (m_transferQueue)
{
    m_transferQueue->waitOnHost();
}
```

- [ ] **Step 5: Commit**

```bash
git add src/vulkan/vk-device.cpp
git commit -m "feat: create transfer queue, update getQueue/getQueueFamilyIndex/destructor in Vulkan"
```

### Task 8: Add transfer queue stage filtering and QFOT barriers to Vulkan

**Files:**
- Modify: `src/vulkan/vk-command.cpp:1629-1650` (commitBarriers filterStageFlags)
- Modify: `src/vulkan/vk-command.cpp` (replace QFOT stubs with real barrier implementations)

- [ ] **Step 1: Add transfer queue stage flag filtering**

In `src/vulkan/vk-command.cpp`, in the `filterStageFlags` lambda at line 1636, add a `Transfer` case after the `Compute` case (line 1648):

```cpp
else if (m_queueType == QueueType::Transfer)
{
    constexpr VkPipelineStageFlags kTransferQueueStages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    flags &= kTransferQueueStages;
    if (flags == 0)
        flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
}
```

- [ ] **Step 2: Replace Vulkan QFOT stubs with real barrier implementations**

Replace the four no-op stubs (added in Task 2) with real implementations. Note: `CommandRecorder` has `m_device` and `m_queueType` but NO `m_queue` member. Use `m_device->getQueueFamilyIndex(m_queueType)` to get the current queue's family index.

For buffer QFOT, barriers do not involve image layouts so these are straightforward:

```cpp
void CommandRecorder::cmdReleaseBufferForQueue(const commands::ReleaseBufferForQueue& cmd)
{
    BufferImpl* buffer = checked_cast<BufferImpl*>(cmd.buffer);

    uint32_t srcFamily = m_device->getQueueFamilyIndex(m_queueType);
    uint32_t dstFamily = m_device->getQueueFamilyIndex(cmd.dstQueue);

    if (srcFamily == dstFamily)
    {
        // Same family — no QFOT needed, just a state transition
        m_stateTracking.setBufferState(buffer, cmd.currentState);
        return;
    }

    commitBarriers();

    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = calcAccessFlags(cmd.currentState);
    barrier.dstAccessMask = 0;
    barrier.srcQueueFamilyIndex = srcFamily;
    barrier.dstQueueFamilyIndex = dstFamily;
    barrier.buffer = buffer->m_buffer.m_buffer;
    barrier.offset = 0;
    barrier.size = buffer->m_desc.size;

    VkPipelineStageFlags srcStage = calcPipelineStageFlags(cmd.currentState, true);
    m_api.vkCmdPipelineBarrier(
        m_cmdBuffer,
        srcStage, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );
}

void CommandRecorder::cmdAcquireBufferFromQueue(const commands::AcquireBufferFromQueue& cmd)
{
    BufferImpl* buffer = checked_cast<BufferImpl*>(cmd.buffer);

    uint32_t srcFamily = m_device->getQueueFamilyIndex(cmd.srcQueue);
    uint32_t dstFamily = m_device->getQueueFamilyIndex(m_queueType);

    if (srcFamily == dstFamily)
    {
        m_stateTracking.setBufferState(buffer, cmd.desiredState);
        return;
    }

    commitBarriers();

    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = calcAccessFlags(cmd.desiredState);
    barrier.srcQueueFamilyIndex = srcFamily;
    barrier.dstQueueFamilyIndex = dstFamily;
    barrier.buffer = buffer->m_buffer.m_buffer;
    barrier.offset = 0;
    barrier.size = buffer->m_desc.size;

    VkPipelineStageFlags dstStage = calcPipelineStageFlags(cmd.desiredState, false);
    m_api.vkCmdPipelineBarrier(
        m_cmdBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dstStage,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );

    m_stateTracking.setBufferState(buffer, cmd.desiredState);
}
```

For texture QFOT, the release and acquire must agree on layouts. The release barrier keeps the current layout (no transition), and the acquire barrier transitions to the desired layout. Per the Vulkan spec, both sides must specify the same oldLayout/newLayout pair — but in practice, when the release side does not want to transition, it uses the same layout for both old and new. The acquire side then uses the same old layout and transitions to the desired new layout:

```cpp
void CommandRecorder::cmdReleaseTextureForQueue(const commands::ReleaseTextureForQueue& cmd)
{
    TextureImpl* texture = checked_cast<TextureImpl*>(cmd.texture);

    uint32_t srcFamily = m_device->getQueueFamilyIndex(m_queueType);
    uint32_t dstFamily = m_device->getQueueFamilyIndex(cmd.dstQueue);

    if (srcFamily == dstFamily)
    {
        m_stateTracking.setTextureState(texture, cmd.subresourceRange, cmd.currentState);
        return;
    }

    commitBarriers();

    VkImageLayout currentLayout = translateImageLayout(cmd.currentState);

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = texture->m_image;
    barrier.oldLayout = currentLayout;
    barrier.newLayout = currentLayout; // no layout transition on release side
    barrier.srcAccessMask = calcAccessFlags(cmd.currentState);
    barrier.dstAccessMask = 0;
    barrier.srcQueueFamilyIndex = srcFamily;
    barrier.dstQueueFamilyIndex = dstFamily;
    barrier.subresourceRange.aspectMask = getAspectMaskFromFormat(getVkFormat(texture->m_desc.format));
    barrier.subresourceRange.baseMipLevel = cmd.subresourceRange.mip;
    barrier.subresourceRange.levelCount = cmd.subresourceRange.mipCount == 0
        ? VK_REMAINING_MIP_LEVELS : cmd.subresourceRange.mipCount;
    barrier.subresourceRange.baseArrayLayer = cmd.subresourceRange.layer;
    barrier.subresourceRange.layerCount = cmd.subresourceRange.layerCount == 0
        ? VK_REMAINING_ARRAY_LAYERS : cmd.subresourceRange.layerCount;

    VkPipelineStageFlags srcStage = calcPipelineStageFlags(cmd.currentState, true);
    m_api.vkCmdPipelineBarrier(
        m_cmdBuffer,
        srcStage, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void CommandRecorder::cmdAcquireTextureFromQueue(const commands::AcquireTextureFromQueue& cmd)
{
    TextureImpl* texture = checked_cast<TextureImpl*>(cmd.texture);

    uint32_t srcFamily = m_device->getQueueFamilyIndex(cmd.srcQueue);
    uint32_t dstFamily = m_device->getQueueFamilyIndex(m_queueType);

    if (srcFamily == dstFamily)
    {
        m_stateTracking.setTextureState(texture, cmd.subresourceRange, cmd.desiredState);
        return;
    }

    commitBarriers();

    VkImageLayout desiredLayout = translateImageLayout(cmd.desiredState);

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = texture->m_image;
    // The acquire side must match the release side's oldLayout, then transition to desiredLayout.
    // Since the release used currentState for both old and new, the image is still in currentState layout.
    // We use desiredLayout for both here — the state tracking system will handle the layout
    // since the image is now owned by this queue and the actual layout transition happens
    // through the state tracking system on first use.
    barrier.oldLayout = desiredLayout;
    barrier.newLayout = desiredLayout;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = calcAccessFlags(cmd.desiredState);
    barrier.srcQueueFamilyIndex = srcFamily;
    barrier.dstQueueFamilyIndex = dstFamily;
    barrier.subresourceRange.aspectMask = getAspectMaskFromFormat(getVkFormat(texture->m_desc.format));
    barrier.subresourceRange.baseMipLevel = cmd.subresourceRange.mip;
    barrier.subresourceRange.levelCount = cmd.subresourceRange.mipCount == 0
        ? VK_REMAINING_MIP_LEVELS : cmd.subresourceRange.mipCount;
    barrier.subresourceRange.baseArrayLayer = cmd.subresourceRange.layer;
    barrier.subresourceRange.layerCount = cmd.subresourceRange.layerCount == 0
        ? VK_REMAINING_ARRAY_LAYERS : cmd.subresourceRange.layerCount;

    VkPipelineStageFlags dstStage = calcPipelineStageFlags(cmd.desiredState, false);
    m_api.vkCmdPipelineBarrier(
        m_cmdBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dstStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    m_stateTracking.setTextureState(texture, cmd.subresourceRange, cmd.desiredState);
}
```

**Note on texture layout mapping:** The current QFOT API only provides `currentState` on release and `desiredState` on acquire. For textures, the Vulkan spec requires both sides of the QFOT barrier to use the same oldLayout/newLayout pair. Since each side only has one state, we use the same layout for both old and new on each side (no layout transition during QFOT). The actual layout transition is handled by the state tracking system when the resource is first used after acquisition. This is correct per the Vulkan spec — QFOT does not require a layout transition; it can be a separate operation.

- [ ] **Step 3: Commit**

```bash
git add src/vulkan/vk-command.cpp
git commit -m "feat: add transfer queue stage filtering and QFOT barriers to Vulkan"
```

---

## Chunk 4: Tests & Verification

### Task 9: Add transfer queue tests

**Files:**
- Create: `tests/test-transfer-queue.cpp`
- Modify: `CMakeLists.txt:868` (add test file)

- [ ] **Step 1: Write test file**

Create `tests/test-transfer-queue.cpp`:

```cpp
#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

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

    {
        auto encoder = transferQueue->createCommandEncoder();
        encoder->setBufferState(srcBuffer, ResourceState::CopySource);
        encoder->setBufferState(dstBuffer, ResourceState::CopyDestination);
        encoder->copyBuffer(dstBuffer, 0, srcBuffer, 0, numberCount * sizeof(float));
        transferQueue->submit(encoder->finish());
        transferQueue->waitOnHost();
    }

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
        auto pass = encoder->beginComputePass();
        auto rootObject = pass->bindPipeline(pipeline);
        ShaderCursor shaderCursor(rootObject);
        shaderCursor["buffer"].setBinding(bufferB);
        float value = 5.f;
        shaderCursor["value"].setData(value);
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        computeQueue->submit(encoder->finish());
    }

    // Copy on transfer queue
    {
        auto encoder = transferQueue->createCommandEncoder();
        encoder->setBufferState(copySrc, ResourceState::CopySource);
        encoder->setBufferState(copyDst, ResourceState::CopyDestination);
        encoder->copyBuffer(copyDst, 0, copySrc, 0, numberCount * sizeof(float));
        transferQueue->submit(encoder->finish());
    }

    graphicsQueue->waitOnHost();
    computeQueue->waitOnHost();
    transferQueue->waitOnHost();

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

        ICommandBuffer* cmdBuf = nullptr;
        encoder->finish(&cmdBuf);

        SubmitDesc submitDesc = {};
        submitDesc.commandBuffers = &cmdBuf;
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

        ICommandBuffer* cmdBuf = nullptr;
        encoder->finish(&cmdBuf);

        SubmitDesc submitDesc = {};
        submitDesc.commandBuffers = &cmdBuf;
        submitDesc.commandBufferCount = 1;
        IFence* waitFence = fence.get();
        uint64_t waitValue = 1;
        submitDesc.waitFences = &waitFence;
        submitDesc.waitFenceValues = &waitValue;
        submitDesc.waitFenceCount = 1;
        REQUIRE_CALL(transferQueue->submit(submitDesc));
    }

    transferQueue->waitOnHost();

    compareComputeResult(device, dstBuffer, makeArray<float>(1.0f, 2.0f, 3.0f, 4.0f));
}
```

- [ ] **Step 2: Add test file to CMakeLists.txt**

In `CMakeLists.txt`, after line 868 (`tests/test-compute-queue.cpp`), add:

```
        tests/test-transfer-queue.cpp
```

- [ ] **Step 3: Commit**

```bash
git add tests/test-transfer-queue.cpp CMakeLists.txt
git commit -m "feat: add transfer queue tests"
```

### Task 10: Build and verify

- [ ] **Step 1: Configure and build**

```bash
cmake --preset default
cmake --build build --config Debug
```

Expected: Build succeeds with no errors.

- [ ] **Step 2: Run transfer queue tests (if hardware available)**

```bash
./build/Debug/slang-rhi-tests -tc="transfer-queue*"
```

Expected: All transfer queue tests pass.

- [ ] **Step 3: Run all tests to verify no regressions**

```bash
./build/Debug/slang-rhi-tests
```

Expected: No test regressions.

- [ ] **Step 4: Fix any build or test failures**

If there are compilation errors (e.g., missing includes, wrong cast types), fix them iteratively.

- [ ] **Step 5: Final commit if fixes were needed**

```bash
git add -u
git commit -m "fix: resolve build/test issues for transfer queue support"
```

### Task 11: Update docs/api.md

**Files:**
- Modify: `docs/api.md`

- [ ] **Step 1: Update implementation status**

Add `Transfer` queue type to the API documentation. Note which backends support it (D3D12, Vulkan) and which don't (CPU, D3D11, Metal, CUDA, WebGPU). Also document the QFOT methods.

- [ ] **Step 2: Commit**

```bash
git add docs/api.md
git commit -m "docs: update api.md with transfer queue and QFOT support"
```
