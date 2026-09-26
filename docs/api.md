# API implementation status

## `IDevice` interface

| API                                | D3D12 | Vulkan |
|------------------------------------|-------|--------|
| `getNativeDeviceHandles`           | yes   | yes    |
| `getInfo`                          | yes   | yes    |
| `hasFeature`                       | yes   | yes    |
| `getFeatures`                      | yes   | yes    |
| `getCapabilities`                  | yes   | yes    |
| `hasCapability`                    | yes   | yes    |
| `getFormatSupport`                 | yes   | yes    |
| `getSlangSession`                  | yes   | yes    |
| `getQueue`                         | yes   | yes    |
| `createTexture`                    | yes   | yes    |
| `createTextureFromNativeHandle`    | yes   | yes    |
| `createTextureFromSharedHandle`    | :x:   | :x:    |
| `createBuffer`                     | yes   | yes    |
| `createBufferFromNativeHandle`     | yes   | yes    |
| `createBufferFromSharedHandle`     | :x:   | :x:    |
| `mapBuffer`                        | yes   | yes    |
| `unmapBuffer`                      | yes   | yes    |
| `createSampler`                    | yes   | yes    |
| `createTextureView`                | yes   | yes    |
| `createSurface`                    | yes   | yes    |
| `createInputLayout`                | yes   | yes    |
| `createShaderObject`               | yes   | yes    |
| `createShaderObjectFromTypeLayout` | yes   | yes    |
| `createRootShaderObject`           | yes   | yes    |
| `createShaderTable`                | yes   | yes    |
| `createShaderProgram`              | yes   | yes    |
| `createBindingSetLayout`           | yes   | yes    |
| `createBindingSet`                 | yes   | yes    |
| `createPipelineLayout`             | yes   | yes    |
| `createRenderPipeline`             | yes   | yes    |
| `createComputePipeline`            | yes   | yes    |
| `createRayTracingPipeline`         | yes   | yes    |
| `readTexture`                      | yes   | yes    |
| `readBuffer`                       | yes   | yes    |
| `createQueryPool`                  | yes   | yes    |
| `getAccelerationStructureSizes`    | yes   | yes    |
| `getMicromapSizes`                 | yes   | yes    |
| `getClusterOperationSizes`         | yes   | yes    |
| `createAccelerationStructure`      | yes   | yes    |
| `createMicromap`                   | yes   | yes    |
| `createFence`                      | yes   | yes    |
| `waitForFences`                    | yes   | yes    |
| `createHeap`                       | yes   | yes    |
| `getTextureAllocationInfo`         | yes   | yes    |
| `getTextureRowAlignment`           | yes   | yes    |
| `getCooperativeVectorProperties`   | yes   | yes    |
| `getCooperativeVectorMatrixSize`   | yes   | yes    |
| `convertCooperativeVectorMatrix`   | yes   | yes    |
| `isCooperativeMatrixSupported` (3) | yes   | yes    |
| `reportHeaps`                      | yes   | yes    |

(1) dummy implementation only
(2) returns nullptr but succeeds
(3) returns false when not supported

## `IBuffer` interface

| API                   | D3D12 | Vulkan |
|-----------------------|-------|--------|
| `getDesc`             | yes   | yes    |
| `getSharedHandle`     | yes   | yes    |
| `getDeviceAddress`    | yes   | yes    |
| `getDescriptorHandle` | yes   | yes    |

(1) returns host address

## `ITexture` interface

| API                    | D3D12 | Vulkan |
|------------------------|-------|--------|
| `getDesc`              | yes   | yes    |
| `getSharedHandle`      | yes   | yes    |
| `createView`           | yes   | yes    |
| `getDefaultView`       | yes   | yes    |
| `getSubresourceLayout` | yes   | yes    |

## `ITextureView` interface

| API                                        | D3D12 | Vulkan |
|--------------------------------------------|-------|--------|
| `getDesc`                                  | yes   | yes    |
| `getTexture`                               | yes   | yes    |
| `getDescriptorHandle`                      | yes   | yes    |
| `getCombinedTextureSamplerDescriptorHandle` | yes   | yes    |

## `ISampler` interface

| API                   | D3D12 | Vulkan |
|-----------------------|-------|--------|
| `getDesc`             | yes   | yes    |
| `getDescriptorHandle` | yes   | yes    |

## `IFence` interface

| API               | D3D12 | Vulkan |
|-------------------|-------|--------|
| `getCurrentValue` | yes   | yes    |
| `setCurrentValue` | yes   | yes    |
| `getNativeHandle` | yes   | yes    |
| `getSharedHandle` | yes   | yes    |

## `IShaderObject` interface

| API                         | D3D12 | Vulkan |
|-----------------------------|-------|--------|
| `getElementTypeLayout`      | yes   | yes    |
| `getContainerType`          | yes   | yes    |
| `getEntryPointCount`        | yes   | yes    |
| `getEntryPoint`             | yes   | yes    |
| `setData`                   | yes   | yes    |
| `getObject`                 | yes   | yes    |
| `setObject`                 | yes   | yes    |
| `setBinding`                | yes   | yes    |
| `setDescriptorHandle`       | yes   | yes    |
| `reserveData`               | yes   | yes    |
| `setSpecializationArgs`     | yes   | yes    |
| `getRawData`                | yes   | yes    |
| `getSize`                   | yes   | yes    |
| `setConstantBufferOverride` | :x:   | :x:    |
| `finalize`                  | yes   | yes    |
| `isFinalized`               | yes   | yes    |

## `IShaderTable` interface

## `IBindingSetLayout` interface

| API       | D3D12 | Vulkan |
|-----------|-------|--------|
| `getDesc` | yes   | yes    |

## `IBindingSet` interface

| API         | D3D12 | Vulkan |
|-------------|-------|--------|
| `getLayout` | yes   | yes    |

## `IPipelineLayout` interface

| API                   | D3D12 | Vulkan |
|-----------------------|-------|--------|
| `getDesc`             | yes   | yes    |
| `getConstantsProfile` | yes   | yes    |
| `getConstantsSize`    | yes   | yes    |
| `getSetCount`         | yes   | yes    |

## `IPipeline` interface

| API               | D3D12 | Vulkan |
|-------------------|-------|--------|
| `getProgram`      | yes   | yes    |
| `getNativeHandle` | yes   | yes    |

## `IRenderPipeline` interface

| API               | D3D12 | Vulkan |
|-------------------|-------|--------|
| `getDesc`         | yes   | yes    |
| `getNativeHandle` | yes   | yes    |

## `IComputePipeline` interface

| API               | D3D12 | Vulkan |
|-------------------|-------|--------|
| `getDesc`         | yes   | yes    |
| `getNativeHandle` | yes   | yes    |

## `IRayTracingPipeline` interface

| API               | D3D12 | Vulkan |
|-------------------|-------|--------|
| `getDesc`         | yes   | yes    |
| `getNativeHandle` | yes   | yes    |

## `IQueryPool` interface

| API                         | D3D12 | Vulkan |
|-----------------------------|-------|--------|
| `getDesc`                   | yes   | yes    |
| `getResultState`            | yes   | yes    |
| `getResult`                 | yes   | yes    |
| `reset`                     | yes   | yes    |
| `reset(queryIndex, count)`  | yes   | yes    |

## `ICommandEncoder` interface

| API                                    | D3D12 | Vulkan |
|----------------------------------------|-------|--------|
| `getDesc`                              | yes   | yes    |
| `beginRenderPass`                      | yes   | yes    |
| `beginComputePass`                     | yes   | yes    |
| `beginRayTracingPass`                  | yes   | yes    |
| `copyBuffer`                           | yes   | yes    |
| `copyTexture`                          | yes   | yes    |
| `copyTextureToBuffer`                  | yes   | yes    |
| `copyBufferToTexture`                  | yes   | yes    |
| `uploadTextureData`                    | yes   | yes    |
| `uploadBufferData`                     | yes   | yes    |
| `clearBuffer`                          | yes   | yes    |
| `clearTextureFloat`                    | yes   | yes    |
| `clearTextureUint`                     | yes   | yes    |
| `clearTextureSint`                     | yes   | yes    |
| `clearTextureDepthStencil`             | yes   | yes    |
| `resolveQuery`                         | yes   | yes    |
| `buildAccelerationStructure`           | yes   | yes    |
| `buildMicromap`                        | yes   | yes    |
| `copyAccelerationStructure`            | yes   | yes    |
| `queryAccelerationStructureProperties` | yes   | yes    |
| `executeClusterOperation`              | yes   | yes    |
| `convertCooperativeVectorMatrix`       | yes   | yes    |
| `setBufferState`                       | yes   | yes    |
| `setTextureState`                      | yes   | yes    |
| `globalBarrier`                        | yes   | yes    |
| `releaseBufferForQueue`                | yes   | yes    |
| `releaseTextureForQueue`               | yes   | yes    |
| `acquireBufferFromQueue`               | yes   | yes    |
| `acquireTextureFromQueue`              | yes   | yes    |
| `pushDebugGroup`                       | yes   | yes    |
| `popDebugGroup`                        | yes   | yes    |
| `insertDebugMarker`                    | yes   | yes    |
| `writeTimestamp`                       | yes   | yes    |
| `finish`                               | yes   | yes    |
| `getNativeHandle`                      | :x:   | :x:    |

## `IPassEncoder` interface

| API                 | D3D12 | Vulkan |
|---------------------|-------|--------|
| `pushDebugGroup`    | yes   | yes    |
| `popDebugGroup`     | yes   | yes    |
| `insertDebugMarker` | yes   | yes    |
| `writeTimestamp`    | yes   | yes    |
| `end`               | yes   | yes    |

## `IRenderPassEncoder` interface

| API                   | D3D12 | Vulkan |
|-----------------------|-------|--------|
| `bindPipeline`        | yes   | yes    |
| `bindPipeline` (flat) | yes   | yes    |
| `setRenderState`      | yes   | yes    |
| `draw`                | yes   | yes    |
| `drawIndexed`         | yes   | yes    |
| `drawIndirect`        | yes   | yes    |
| `drawIndexedIndirect` | yes   | yes    |
| `drawMeshTasks`       | yes   | yes    |

`RenderState::blendFactor` contains four dynamic blend constants, initially zero;
`stencilRef` is also dynamic and initially zero. Both backends apply these values
at draw time, independently of pipeline changes. Supply a default `RenderState`
for a new pass to start with zero constants/reference.

`InputElementDesc::shaderLocation` selects the Vulkan vertex shader location.
The default `UINT32_MAX` selects the element's index. Vulkan rejects duplicate
resolved locations and locations outside `maxVertexInputElements`, including
large values that would otherwise overflow location calculations. D3D12 continues
to match inputs using `semanticName` and `semanticIndex`.

## `IComputePassEncoder` interface

| API                       | D3D12 | Vulkan |
|---------------------------|-------|--------|
| `bindPipeline`            | yes   | yes    |
| `bindPipeline` (flat)     | yes   | yes    |
| `dispatchCompute`         | yes   | yes    |
| `dispatchComputeIndirect` | yes   | yes    |

Compute binding supports the general shader-object overloads and the `FlatBindingDesc` overload.
Flat bindings use an `IPipelineLayout`, copy constant bytes immediately, and retain and transition
resources declared by binding sets and explicit accesses. The earlier entry-point-data-only
compute binding API has been removed.

With `DeviceDesc::enableValidation`, replay of a flat binding reports `Warning` / `Layer` when
its declared accesses require an effective resource-state change. Messages identify the resource
label and target state, and include resolved layer/mip ranges for textures. Read-state unions are
compared against the tracked effective state; repeated accesses already covered by that union and
same-state UAV ordering barriers do not warn. The diagnostic does not suppress barriers or alter
failure handling. Without the debug wrapper it is disabled, and general shader-object bindings,
copy/clear commands, attachments, and default-state restoration do not produce this diagnostic.
This applies to flat render, compute, and ray-tracing bindings on D3D12 and Vulkan.

D3D12 caches flat sampler tables by immutable binding-set identity for the command buffer's
lifetime, including workloads using more than 16 distinct sets. Rebinding an existing set does
not allocate another sampler table. Command-buffer reset clears the cache along with its
descriptor arenas. Distinct sets still consume the configured shader-visible sampler heap;
exhaustion returns an allocation failure and does not overwrite tables used by earlier commands.

### Vulkan allocation failures

Native descriptor and constant-buffer allocation failures propagate without allocation-failure assertions.
Failed allocations do not evict descriptors or overwrite existing constant data. Partially successful
allocations from a failed bind remain owned until normal reset/release. Command buffers whose native
reset fails are isolated instead of being returned to the reusable pool.

## `IRayTracingPassEncoder` interface

| API                   | D3D12 | Vulkan |
|-----------------------|-------|--------|
| `bindPipeline`        | yes   | yes    |
| `bindPipeline` (flat) | yes   | yes    |
| `dispatchRays`        | yes   | yes    |

## `ICommandBuffer` interface

| API               | D3D12 | Vulkan |
|-------------------|-------|--------|
| `getDesc`         | yes   | yes    |
| `getNativeHandle` | yes   | yes    |

## `ICommandQueue` interface

| API                       | D3D12 | Vulkan |
|---------------------------|-------|--------|
| `getType`                 | yes   | yes    |
| `createCommandEncoder`    | yes   | yes    |
| `submit`                  | yes   | yes    |
| `getNativeHandle`         | yes   | yes    |
| `waitOnHost`              | yes   | yes    |
| `getCompletedSequence`    | yes   | yes    |
| `waitForSequence`         | yes   | yes    |
| `getTimestampCalibration` | yes   | yes    |
| `SubmitDesc::outSequence` | yes   | yes    |

### Queue type support

| Queue Type             | D3D12 | Vulkan |
|------------------------|-------|--------|
| `QueueType::Graphics`  | yes   | yes    |
| `QueueType::Compute`   | yes   | yes    |
| `QueueType::Transfer`  | yes   | yes    |

The queue family ownership transfer (QFOT) methods (`releaseBufferForQueue`, `releaseTextureForQueue`,
`acquireBufferFromQueue`, `acquireTextureFromQueue`) are used to transfer resource ownership between
queues of different types. These are supported on D3D12 and Vulkan. On D3D12, they are implemented
as resource state barriers. On Vulkan, they use queue family ownership transfer barriers.

## `ISurface` interface

| API                 | D3D12 | Vulkan |
|---------------------|-------|--------|
| `getInfo`           | yes   | yes    |
| `getConfig`         | yes   | yes    |
| `configure`         | yes   | yes    |
| `unconfigure`       | yes   | yes    |
| `acquireNextImage`  | yes   | yes    |
| `present`           | yes   | yes    |

Frame contract:

- Calls on one surface are externally serialized. At most one image is held: a second `acquireNextImage`
  before `present` fails with `SLANG_E_INVALID_ARG` instead of blocking.
- `acquireNextImage` returns once the image is free, waiting on the host if necessary. The image is in
  `ResourceState::Present`; recordings that write it transition from and back to that state.
- `present` orders presentation after every submission made earlier on the device's graphics queue, however many
  there are and whichever thread made them. Recordings that write the image must be submitted before `present`.
- A failed `present` keeps the image held. `configure`, `unconfigure`, and releasing the surface cancel a held
  image; discard recordings that reference it first.
- Vulkan: after an out-of-memory `present` nothing was enqueued and `present` may be retried; any other failure
  leaves the surface unconfigured (`getConfig()` returns null) until `configure` succeeds. Cancellation waits for
  the queue to go idle and fails without releasing native objects if completion cannot be established; releasing
  the surface terminates the process in that case.

## `IAccelerationStructure` interface

| API                   | D3D12 | Vulkan |
|-----------------------|-------|--------|
| `getHandle`           | yes   | yes    |
| `getDeviceAddress`    | yes   | yes    |
| `getDescriptorHandle` | yes   | yes    |

## `IHeap` interface

| API                | D3D12 | Vulkan |
|--------------------|-------|--------|
| `allocate`         | yes   | yes    |
| `free`             | yes   | yes    |
| `report`           | yes   | yes    |
| `flush`            | yes   | yes    |
| `removeEmptyPages` | yes   | yes    |
