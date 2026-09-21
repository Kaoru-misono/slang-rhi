# Async Resource Loading with Transfer Queues

D3D12 and Vulkan expose `QueueType::Graphics`, `QueueType::Compute` and
`QueueType::Transfer` through `IDevice::getQueue`.
Graphics supports rendering, compute and copy; Compute supports compute and copy;
Transfer supports copy only. Vulkan may share native queues or queue families.

Uploads must complete before another queue reads their resources.
Use `SubmitDesc::signalFences` / `signalFenceValues` on the producer and
`waitFences` / `waitFenceValues` on the consumer. Keep fence values monotonic.
`ICommandQueue::waitOnHost` waits for submitted work when a CPU wait is needed.

Resources default to `SharingMode::Exclusive`. For different Vulkan queue families,
record matching `releaseBufferForQueue` / `acquireBufferFromQueue` or
`releaseTextureForQueue` / `acquireTextureFromQueue` commands and order the queues
with a fence. Texture transfers must agree on the subresource range and state.

`SharingMode::Concurrent` includes the device's graphics, compute and transfer
families. It removes ownership transfers, while memory and texture-layout
dependencies still require synchronization. The resource's `defaultState` and
automatic command-end restoration participate in those dependencies.
D3D12 concurrent textures default to `ResourceState::General` for copy-queue use.

Each command buffer is submitted once. Successful submissions retain their
recorded dependencies until the submitting queue completes. Device-wide deferred
deletion accounts for all three queues; a CPU handle release is not a completion
proof. Device loss invalidates successful-work assumptions.

See [the API contract](api.md) for queue support, resource sharing and diagnostics.
