# Transfer Queue Support

## Summary

Add transfer (copy/DMA) queue support to slang-rhi, following the existing async compute queue pattern. Transfer queues enable asynchronous data transfers on dedicated DMA hardware, allowing copies to run concurrently with graphics and compute work.

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

### Usage

`IDevice::getQueue(QueueType::Transfer)` returns an `ICommandQueue`. The returned encoder supports only copy/transfer operations:

- `copyBuffer`
- `copyTexture`
- `copyTextureToBuffer`
- `uploadBufferData`
- `uploadTextureData`
- `clearBuffer`
- `resolveQuery`

Calling `beginRenderPass()` or `beginComputePass()` on a transfer queue encoder is invalid and rejected by the debug layer.

## Vulkan Backend

### Queue Family Selection

Add `m_transferQueueFamilyIndex` with the following priority chain, designed to maximize hardware parallelism by avoiding families already used for graphics and compute:

1. **Dedicated transfer-only family**: `VK_QUEUE_TRANSFER_BIT` without `VK_QUEUE_GRAPHICS_BIT` or `VK_QUEUE_COMPUTE_BIT`, and not the same family as compute
2. **Any transfer family not used by graphics or compute**: has `VK_QUEUE_TRANSFER_BIT`, different from both graphics and compute families
3. **Compute family with available second queue**: if compute family supports transfer and has a spare queue index
4. **Graphics family (fallback)**: always supports transfer

Queue index within the chosen family: if sharing a family with another queue type, request index 1 (or next available), else use index 0.

### Device Storage

- Add `m_transferQueue` (`RefPtr<CommandQueueImpl>`) alongside existing `m_graphicsQueue` and `m_computeQueue`
- Transfer queue gets its own `VkSemaphore` (timeline) for submission tracking
- Separate command pool created with the transfer queue family index
- Own in-flight/available command buffer lists

## D3D12 Backend

### Queue Creation

- Create a third `ID3D12CommandQueue` with type `D3D12_COMMAND_LIST_TYPE_COPY`
- Store as `m_transferQueue` with queue index 2 (graphics=0, compute=1, transfer=2)
- Own `ID3D12Fence` and submission tracking state (`m_lastSubmittedID`, `m_lastFinishedID`)

### Command Allocators

- Command allocators of type `D3D12_COMMAND_LIST_TYPE_COPY`
- Follows existing command allocator pool pattern per queue

### Submission

- `ExecuteCommandLists()` on the copy queue
- Fence signal/wait for cross-queue synchronization

## Debug Layer

- Wrap transfer queue in `DebugCommandQueue` like graphics/compute
- Reject `beginRenderPass()` on transfer queue encoders with error: "Render passes not supported on transfer queues"
- Reject `beginComputePass()` on transfer queue encoders with error: "Compute passes not supported on transfer queues"
- Copy operations pass through without additional restrictions

## Unsupported Backends

CPU, D3D11, Metal, CUDA, WebGPU:

- `getQueue(QueueType::Transfer)` returns `SLANG_E_NOT_AVAILABLE`

## Tests

File: `tests/test-transfer-queue.cpp`

1. **transfer-queue-create** (D3D12 | Vulkan): verify `getQueue(QueueType::Transfer)` returns valid queue with correct type
2. **transfer-queue-unsupported** (CPU | D3D11): verify unsupported backends return error
3. **transfer-queue-copy** (D3D12 | Vulkan): create source buffer with known data, copy to destination via transfer queue, verify data matches
4. **transfer-queue-all-queues** (D3D12 | Vulkan): use all three queues simultaneously (graphics, compute, transfer), verify all complete correctly

## Implementation Approach

Follow the existing compute queue pattern — add a third queue field to each backend's DeviceImpl rather than refactoring to array-based storage. Only three meaningful hardware queue types exist (graphics, compute, transfer), so YAGNI applies.

## Files to Modify

- `include/slang-rhi.h` — add `Transfer` to `QueueType` enum
- `src/vulkan/vk-device.h` — add transfer queue family index and queue storage
- `src/vulkan/vk-device.cpp` — queue family selection, queue creation, `getQueue()` handling
- `src/vulkan/vk-command.h` — transfer queue command pool (if separate from existing)
- `src/vulkan/vk-command.cpp` — command buffer creation for transfer queue
- `src/d3d12/d3d12-device.h` — add transfer queue storage
- `src/d3d12/d3d12-device.cpp` — copy queue creation, `getQueue()` handling
- `src/d3d12/d3d12-command.h` — transfer queue command allocator type
- `src/d3d12/d3d12-command.cpp` — command list creation for copy type
- `src/debug-layer/debug-device.cpp` — wrap transfer queue
- `src/debug-layer/debug-command-encoder.cpp` — validate encoder restrictions
- `src/cpu/cpu-device.cpp` — return not available
- `src/d3d11/d3d11-device.cpp` — return not available
- `src/metal/metal-device.cpp` — return not available
- `src/cuda/cuda-device.cpp` — return not available
- `src/wgpu/wgpu-device.cpp` — return not available
- `tests/test-transfer-queue.cpp` — new test file
- `CMakeLists.txt` — add test file
