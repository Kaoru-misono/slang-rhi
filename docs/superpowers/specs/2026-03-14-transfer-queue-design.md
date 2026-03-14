# Transfer Queue Support

## Summary

Add transfer (copy/DMA) queue support to slang-rhi, following the existing async compute queue pattern. Transfer queues enable asynchronous data transfers on dedicated DMA hardware, allowing copies to run concurrently with graphics and compute work.

Additionally, add explicit queue family ownership transfer (QFOT) API to support correct cross-queue resource sharing in Vulkan.

Supported backends: Vulkan and D3D12. All other backends return `SLANG_E_NOT_AVAILABLE`.

## Public API Changes

### QueueType Enum

```cpp
enum class QueueType
{
    Graphics,
    Compute,
    Transfer,
};
```

### Transfer Queue Operations

`IDevice::getQueue(QueueType::Transfer)` returns an `ICommandQueue`. The returned encoder supports only copy/transfer operations:

- `copyBuffer`
- `copyTexture`
- `copyTextureToBuffer`
- `uploadBufferData`

The following operations are **not** supported on transfer queues:

- `beginRenderPass()` — rejected by debug layer
- `beginComputePass()` — rejected by debug layer
- `beginRayTracingPass()` — rejected by debug layer
- `clearBuffer` — uses UAV clear on D3D12, not available on copy command lists
- `resolveQuery` — requires graphics/compute queue on both Vulkan and D3D12
- `uploadTextureData` — may require image layout transitions needing graphics/compute capabilities

### Queue Family Ownership Transfer API

New methods on `ICommandEncoder` for explicit cross-queue resource ownership transfer:

```cpp
// Release ownership — call on source queue's encoder before submitting
virtual void releaseBufferForQueue(IBuffer* buffer, ResourceState currentState, QueueType dstQueue) = 0;
virtual void releaseTextureForQueue(ITexture* texture, SubresourceRange range,
                                    ResourceState currentState, QueueType dstQueue) = 0;

// Acquire ownership — call on destination queue's encoder after waiting on fence
virtual void acquireBufferFromQueue(IBuffer* buffer, ResourceState desiredState, QueueType srcQueue) = 0;
virtual void acquireTextureFromQueue(ITexture* texture, SubresourceRange range,
                                     ResourceState desiredState, QueueType srcQueue) = 0;
```

### Usage Pattern

```cpp
// 1. Source queue: release resource
auto srcEncoder = graphicsQueue->createCommandEncoder();
srcEncoder->releaseBufferForQueue(buffer, ResourceState::UnorderedAccess, QueueType::Transfer);
auto srcCmdBuf = srcEncoder->finish();
graphicsQueue->submit({.commandBuffers = {srcCmdBuf},
                       .signalFences = {fence}, .signalFenceValues = {1}});

// 2. Destination queue: acquire resource and use it
auto dstEncoder = transferQueue->createCommandEncoder();
dstEncoder->acquireBufferFromQueue(buffer, ResourceState::CopySource, QueueType::Graphics);
// ... copy operations ...
auto dstCmdBuf = dstEncoder->finish();
transferQueue->submit({.commandBuffers = {dstCmdBuf},
                       .waitFences = {fence}, .waitFenceValues = {1}});
```

## Vulkan Backend

### Queue Family Selection

Add `m_transferQueueFamilyIndex` with the following priority chain, designed to maximize hardware parallelism by avoiding families already used for graphics and compute:

1. **Dedicated transfer-only family**: `VK_QUEUE_TRANSFER_BIT` without `VK_QUEUE_GRAPHICS_BIT` or `VK_QUEUE_COMPUTE_BIT`, and not the same family as compute
2. **Any transfer family not used by graphics or compute**: has `VK_QUEUE_TRANSFER_BIT`, different from both graphics and compute families
3. **Compute family with available second queue**: if compute family supports transfer and has a spare queue index
4. **Graphics family (fallback)**: always supports transfer

### Queue Creation Algorithm

When building `VkDeviceQueueCreateInfo` structures, count how many queues are needed per family:

| Scenario | Family Usage | Queues to Request |
|---|---|---|
| All 3 types on separate families | 1 per family | 1 each |
| Transfer shares with compute (different from graphics) | graphics: 1, compute+transfer: 2 | Request min(2, maxCount) from shared family |
| Transfer shares with graphics (compute separate) | graphics+transfer: 2, compute: 1 | Request min(2, maxCount) from shared family |
| All 3 share one family | 1 family | Request min(3, maxCount) queues |
| Compute and transfer both share with graphics | 1 family | Request min(3, maxCount) queues |

Queue index assignment within a shared family: graphics always gets index 0, compute gets the next available index, transfer gets the next after that. If fewer queues are available than needed, later queue types share index 0.

### Device Storage

- Add `m_transferQueue` (`RefPtr<CommandQueueImpl>`) alongside existing `m_graphicsQueue` and `m_computeQueue`
- Transfer queue gets its own `VkSemaphore` (timeline) for submission tracking
- Separate command pool created with the transfer queue family index
- Own in-flight/available command buffer lists

### Queue Family Ownership Transfer Implementation

The QFOT methods map to `VkBufferMemoryBarrier`/`VkImageMemoryBarrier` with explicit queue family indices:

- `releaseBufferForQueue`: emits a buffer barrier with `srcQueueFamilyIndex = m_queueFamilyIndex` (current queue's family) and `dstQueueFamilyIndex` resolved from the destination `QueueType`
- `acquireBufferFromQueue`: emits a buffer barrier with `srcQueueFamilyIndex` resolved from source `QueueType` and `dstQueueFamilyIndex = m_queueFamilyIndex`
- Same pattern for texture variants, with image layout transitions included in the barrier
- When source and destination queue types share the same family, emit a regular state transition barrier instead (no QFOT needed, `srcQueueFamilyIndex = dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED`)

The `getQueueFamilyIndex(QueueType)` helper must be updated with a `Transfer` case returning `m_transferQueueFamilyIndex`.

### Transfer Queue Stage Flags

`commitBarriers()` already filters pipeline stage flags for compute queues. Add equivalent filtering for transfer queues:

```cpp
constexpr VkPipelineStageFlags kTransferQueueStages =
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT |
    VK_PIPELINE_STAGE_TRANSFER_BIT |
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
```

### Device Teardown

The destructor and `waitForGpu()` must wait on and destroy `m_transferQueue`, matching the pattern for graphics and compute queues.

## D3D12 Backend

### Queue Creation

- Create a third `ID3D12CommandQueue` with type `D3D12_COMMAND_LIST_TYPE_COPY`
- Store as `m_transferQueue` with queue index 2 (graphics=0, compute=1, transfer=2)
- Own `ID3D12Fence` and submission tracking state (`m_lastSubmittedID`, `m_lastFinishedID`)

### Command List Type Mapping

The existing binary ternary in `CommandQueueImpl::init` (`d3d12-command.cpp`):

```cpp
m_commandListType = (m_type == QueueType::Compute) ? D3D12_COMMAND_LIST_TYPE_COMPUTE
                                                   : D3D12_COMMAND_LIST_TYPE_DIRECT;
```

Must become a three-way mapping:

```cpp
switch (m_type) {
    case QueueType::Compute:  m_commandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE; break;
    case QueueType::Transfer: m_commandListType = D3D12_COMMAND_LIST_TYPE_COPY; break;
    default:                  m_commandListType = D3D12_COMMAND_LIST_TYPE_DIRECT; break;
}
```

### Command Allocators

- Command allocators of type `D3D12_COMMAND_LIST_TYPE_COPY`
- Follows existing command allocator pool pattern per queue

### Submission

- `ExecuteCommandLists()` on the copy queue
- Fence signal/wait for cross-queue synchronization
- Buffers on copy queues automatically decay to `COMMON` state at `ExecuteCommandLists` boundaries

### Queue Family Ownership Transfer Implementation

D3D12 does not require explicit ownership transfers. The QFOT methods (`releaseBufferForQueue`, `acquireBufferFromQueue`, etc.) are implemented as simple state transitions or no-ops, since D3D12 handles cross-queue sharing implicitly via fence synchronization and automatic state decay.

### Device Teardown

The destructor must wait on and destroy `m_transferQueue`, matching the pattern for graphics and compute queues.

## Debug Layer

- Wrap transfer queue in `DebugCommandQueue` like graphics/compute
- Reject on transfer queue encoders with descriptive errors:
  - `beginRenderPass()`: "Render passes not supported on transfer queues"
  - `beginComputePass()`: "Compute passes not supported on transfer queues"
  - `beginRayTracingPass()`: "Ray tracing passes not supported on transfer queues"
- QFOT validation:
  - Validate that `releaseBufferForQueue`/`releaseTextureForQueue` specifies a `dstQueue` different from the current queue type
  - Validate that `acquireBufferFromQueue`/`acquireTextureFromQueue` is called on the queue matching the intended destination
  - Warn if QFOT is used but source and destination queues share the same family (unnecessary overhead)
- Copy operations pass through without additional restrictions

## Unsupported Backends

CPU, D3D11, Metal, CUDA, WebGPU:

- Explicitly add `if (type == QueueType::Transfer) return SLANG_E_NOT_AVAILABLE;` for consistency with compute queue handling
- QFOT methods are no-ops

## Tests

File: `tests/test-transfer-queue.cpp`

1. **transfer-queue-create** (D3D12 | Vulkan): verify `getQueue(QueueType::Transfer)` returns valid queue with correct type
2. **transfer-queue-unsupported** (CPU | D3D11): verify unsupported backends return error
3. **transfer-queue-copy** (D3D12 | Vulkan): create source buffer with known data, copy to destination via transfer queue, verify data matches
4. **transfer-queue-all-queues** (D3D12 | Vulkan): use all three queues simultaneously (graphics, compute, transfer), verify all complete correctly
5. **transfer-queue-ownership-transfer** (D3D12 | Vulkan): test QFOT workflow — graphics queue writes to buffer, releases ownership, transfer queue acquires and copies, verify correct results

## Implementation Approach

Follow the existing compute queue pattern — add a third queue field to each backend's DeviceImpl rather than refactoring to array-based storage. Only three meaningful hardware queue types exist (graphics, compute, transfer), so YAGNI applies.

## Files to Modify

- `include/slang-rhi.h` — add `Transfer` to `QueueType` enum; add QFOT methods to `ICommandEncoder`
- `src/vulkan/vk-device.h` — add transfer queue family index and queue storage
- `src/vulkan/vk-device.cpp` — queue family selection, queue creation, `getQueue()` handling, `getQueueFamilyIndex()` update, destructor update
- `src/vulkan/vk-command.h` — transfer queue command pool (if separate from existing)
- `src/vulkan/vk-command.cpp` — QFOT barrier emission, transfer queue stage flag filtering, command buffer creation
- `src/d3d12/d3d12-device.h` — add transfer queue storage
- `src/d3d12/d3d12-device.cpp` — copy queue creation, `getQueue()` handling, destructor update
- `src/d3d12/d3d12-command.h` — transfer queue command allocator type
- `src/d3d12/d3d12-command.cpp` — command list type mapping fix, QFOT no-op implementation
- `src/debug-layer/debug-device.cpp` — wrap transfer queue
- `src/debug-layer/debug-command-encoder.cpp` — validate encoder restrictions (render, compute, ray tracing passes), QFOT validation
- `src/command-encoder.h` — base class QFOT method declarations
- `src/cpu/cpu-device.cpp` — return `SLANG_E_NOT_AVAILABLE` for transfer queue
- `src/d3d11/d3d11-device.cpp` — return `SLANG_E_NOT_AVAILABLE` for transfer queue
- `src/metal/metal-device.cpp` — return `SLANG_E_NOT_AVAILABLE` for transfer queue
- `src/cuda/cuda-device.cpp` — return `SLANG_E_NOT_AVAILABLE` for transfer queue
- `src/wgpu/wgpu-command.cpp` — return `SLANG_E_NOT_AVAILABLE` for transfer queue
- `tests/test-transfer-queue.cpp` — new test file
- `CMakeLists.txt` — add test file
- `docs/api.md` — update implementation status
