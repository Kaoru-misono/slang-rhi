# API implementation status

## `IDevice` interface

| API                                | CPU     | CUDA | D3D11 | D3D12 | Vulkan | Metal   | WGPU |
|------------------------------------|---------|------|-------|-------|--------|---------|------|
| `getNativeDeviceHandles`           | :x:     | yes  | :x:   | yes   | yes    | yes     | :x:  |
| `getInfo`                          | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `hasFeature`                       | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `getFeatures`                      | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `getCapabilities`                  | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `hasCapability`                    | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `getFormatSupport`                 | yes     | yes  | yes   | yes   | yes    | yes (1) | yes  |
| `getSlangSession`                  | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `getQueue`                         | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createTexture`                    | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createTextureFromNativeHandle`    | :x:     | :x:  | :x:   | yes   | yes    | yes     | :x:  |
| `createTextureFromSharedHandle`    | :x:     | yes  | :x:   | :x:   | :x:    | :x:     | :x:  |
| `createBuffer`                     | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createBufferFromNativeHandle`     | :x:     | yes  | :x:   | yes   | yes    | yes     | yes  |
| `createBufferFromSharedHandle`     | :x:     | yes  | :x:   | :x:   | :x:    | :x:     | :x:  |
| `mapBuffer`                        | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `unmapBuffer`                      | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createSampler`                    | yes (2) | yes  | yes   | yes   | yes    | yes     | yes  |
| `createTextureView`                | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createSurface`                    | :x:     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createInputLayout`                | :x:     | :x:  | yes   | yes   | yes    | yes     | yes  |
| `createShaderObject`               | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createShaderObjectFromTypeLayout` | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createRootShaderObject`           | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createShaderTable`                | :x:     | yes  | :x:   | yes   | yes    | :x:     | :x:  |
| `createShaderProgram`              | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createBindingSetLayout`           | :x:     | :x:  | :x:   | yes   | yes    | :x:     | :x:  |
| `createBindingSet`                 | :x:     | :x:  | :x:   | yes   | yes    | :x:     | :x:  |
| `createPipelineLayout`             | :x:     | :x:  | :x:   | yes   | yes    | :x:     | :x:  |
| `createRenderPipeline`             | :x:     | :x:  | yes   | yes   | yes    | yes     | yes  |
| `createComputePipeline`            | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createRayTracingPipeline`         | :x:     | yes  | :x:   | yes   | yes    | :x:     | :x:  |
| `readTexture`                      | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `readBuffer`                       | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `createQueryPool`                  | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `getAccelerationStructureSizes`    | :x:     | yes  | :x:   | yes   | yes    | yes     | :x:  |
| `getMicromapSizes`                 | :x:     | yes  | :x:   | yes   | yes    | :x:     | :x:  |
| `getClusterOperationSizes`         | :x:     | yes  | :x:   | yes   | yes    | :x:     | :x:  |
| `createAccelerationStructure`      | :x:     | yes  | :x:   | yes   | yes    | yes     | :x:  |
| `createMicromap`                   | :x:     | yes  | :x:   | yes   | yes    | :x:     | :x:  |
| `createFence`                      | yes     | yes  | :x:   | yes   | yes    | yes     | yes  |
| `waitForFences`                    | yes     | yes  | :x:   | yes   | yes    | yes     | yes  |
| `createHeap`                       | :x:     | yes  | :x:   | yes   | yes    | :x:     | :x:  |
| `getTextureAllocationInfo`         | yes     | yes  | :x:   | yes   | yes    | yes     | :x:  |
| `getTextureRowAlignment`           | yes     | yes  | :x:   | yes   | yes    | yes     | yes  |
| `getCooperativeVectorProperties`   | :x:     | yes  | :x:   | yes   | yes    | :x:     | :x:  |
| `getCooperativeVectorMatrixSize`   | :x:     | yes  | :x:   | yes   | yes    | :x:     | :x:  |
| `convertCooperativeVectorMatrix`   | :x:     | yes  | :x:   | yes   | yes    | :x:     | :x:  |
| `isCooperativeMatrixSupported` (3) | yes     | yes  | yes   | yes   | yes    | yes     | yes  |
| `reportHeaps`                      | yes     | yes  | yes   | yes   | yes    | yes     | yes  |

(1) dummy implementation only
(2) returns nullptr but succeeds
(3) returns false when not supported

## `IBuffer` interface

| API                   | CPU     | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-----------------------|---------|------|-------|-------|--------|-------|------|
| `getDesc`             | yes     | yes  | yes   | yes   | yes    | yes   | yes  |
| `getSharedHandle`     | :x:     | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `getDeviceAddress`    | yes (1) | yes  | :x:   | yes   | yes    | yes   | :x:  |
| `getDescriptorHandle` | :x:     | :x:  | :x:   | yes   | yes    | :x:   | :x:  |

(1) returns host address

## `ITexture` interface

| API                    | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|------------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`              | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getSharedHandle`      | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `createView`           | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getDefaultView`       | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getSubresourceLayout` | yes | yes  | yes   | yes   | yes    | yes   | yes  |

## `ITextureView` interface

| API                                        | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|--------------------------------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`                                  | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getTexture`                               | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getDescriptorHandle`                      | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `getCombinedTextureSamplerDescriptorHandle`| :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |

## `ISampler` interface

| API                   | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-----------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`             | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getDescriptorHandle` | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |

## `IFence` interface

| API               | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-------------------|-----|------|-------|-------|--------|-------|------|
| `getCurrentValue` | yes | yes  | :x:   | yes   | yes    | yes   | yes  |
| `setCurrentValue` | yes | yes  | :x:   | yes   | yes    | yes   | yes  |
| `getNativeHandle` | :x: | :x:  | :x:   | yes   | yes    | yes   | :x:  |
| `getSharedHandle` | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |

## `IShaderObject` interface

| API                         | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-----------------------------|-----|------|-------|-------|--------|-------|------|
| `getElementTypeLayout`      | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getContainerType`          | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getEntryPointCount`        | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getEntryPoint`             | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `setData`                   | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getObject`                 | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `setObject`                 | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `setBinding`                | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `setDescriptorHandle`       | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `reserveData`               | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `setSpecializationArgs`     | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getRawData`                | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getSize`                   | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `setConstantBufferOverride` | :x: | :x:  | :x:   | :x:   | :x:    | :x:   | :x:  |
| `finalize`                  | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `isFinalized`               | yes | yes  | yes   | yes   | yes    | yes   | yes  |

## `IShaderTable` interface

## `IBindingSetLayout` interface

| API       | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-----------|-----|------|-------|-------|--------|-------|------|
| `getDesc` | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |

## `IBindingSet` interface

| API         | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-------------|-----|------|-------|-------|--------|-------|------|
| `getLayout` | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |

## `IPipelineLayout` interface

| API                   | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-----------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`             | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `getConstantsProfile` | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `getConstantsSize`    | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `getSetCount`         | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |

## `IPipeline` interface

| API               | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-------------------|-----|------|-------|-------|--------|-------|------|
| `getProgram`      | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getNativeHandle` | :x: | yes  | yes   | yes   | yes    | yes   | yes  |

## `IRenderPipeline` interface

| API               | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`         | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `getNativeHandle` | :x: | :x:  | :x:   | yes   | yes    | yes   | yes  |

## `IComputePipeline` interface

| API               | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`         | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getNativeHandle` | :x: | yes  | :x:   | yes   | yes    | yes   | yes  |

## `IRayTracingPipeline` interface

| API               | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`         | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `getNativeHandle` | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |

## `IQueryPool` interface

| API                         | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-----------------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`                   | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getResultState`            | yes | yes  | yes   | yes   | yes    | :x:   | :x:  |
| `getResult`                 | yes | yes  | yes   | yes   | yes    | :x:   | :x:  |
| `reset`                     | yes | yes  | yes   | yes   | yes    | :x:   | :x:  |
| `reset(queryIndex, count)`  | yes | yes  | yes   | yes   | yes    | :x:   | :x:  |

## `ICommandEncoder` interface

| API                                    | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|----------------------------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`                              | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `beginRenderPass`                      | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `beginComputePass`                     | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `beginRayTracingPass`                  | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `copyBuffer`                           | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `copyTexture`                          | :x: | yes  | yes   | yes   | yes    | yes   | yes  |
| `copyTextureToBuffer`                  | :x: | yes  | :x:   | yes   | yes    | yes   | yes  |
| `copyBufferToTexture`                  | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `uploadTextureData`                    | :x: | yes  | :x:   | yes   | yes    | yes   | yes  |
| `uploadBufferData`                     | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `clearBuffer`                          | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `clearTextureFloat`                    | :x: | yes  | yes   | yes   | yes    | yes   | :x:  |
| `clearTextureUint`                     | :x: | yes  | yes   | yes   | yes    | yes   | :x:  |
| `clearTextureSint`                     | :x: | yes  | yes   | yes   | yes    | yes   | :x:  |
| `clearTextureDepthStencil`             | :x: | :x:  | yes   | yes   | yes    | yes   | :x:  |
| `resolveQuery`                         | yes | :x:  | :x:   | yes   | yes    | yes   | :x:  |
| `buildAccelerationStructure`           | :x: | yes  | :x:   | yes   | yes    | yes   | :x:  |
| `buildMicromap`                        | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `copyAccelerationStructure`            | :x: | yes  | :x:   | yes   | yes    | yes   | :x:  |
| `queryAccelerationStructureProperties` | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `executeClusterOperation`              | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `convertCooperativeVectorMatrix`       | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `setBufferState`                       | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `setTextureState`                      | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `globalBarrier`                        | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `releaseBufferForQueue`                | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `releaseTextureForQueue`               | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `acquireBufferFromQueue`               | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `acquireTextureFromQueue`              | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `pushDebugGroup`                       | :x: | :x:  | :x:   | yes   | yes    | yes   | yes  |
| `popDebugGroup`                        | :x: | :x:  | :x:   | yes   | yes    | yes   | yes  |
| `insertDebugMarker`                    | :x: | :x:  | :x:   | yes   | yes    | yes   | yes  |
| `writeTimestamp`                       | yes | yes  | yes   | yes   | yes    | :x:   | :x:  |
| `finish`                               | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getNativeHandle`                      | :x: | :x:  | :x:   | :x:   | :x:    | :x:   | :x:  |

## `IPassEncoder` interface

| API                 | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|---------------------|-----|------|-------|-------|--------|-------|------|
| `pushDebugGroup`    | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `popDebugGroup`     | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `insertDebugMarker` | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `writeTimestamp`    | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `end`               | yes | yes  | yes   | yes   | yes    | yes   | yes  |

## `IRenderPassEncoder` interface

| API                   | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-----------------------|-----|------|-------|-------|--------|-------|------|
| `bindPipeline`        | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `bindPipeline` (flat) | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `setRenderState`      | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `draw`                | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `drawIndexed`         | :x: | :x:  | yes   | yes   | yes    | yes   | yes  |
| `drawIndirect`        | :x: | :x:  | yes   | yes   | yes    | :x:   | yes  |
| `drawIndexedIndirect` | :x: | :x:  | yes   | yes   | yes    | :x:   | yes  |
| `drawMeshTasks`       | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |

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

| API                       | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|---------------------------|-----|------|-------|-------|--------|-------|------|
| `bindPipeline`            | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `bindPipeline` (flat)     | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `dispatchCompute`         | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `dispatchComputeIndirect` | :x: | yes  | yes   | yes   | yes    | yes   | yes  |

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

| API                   | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-----------------------|-----|------|-------|-------|--------|-------|------|
| `bindPipeline`        | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `bindPipeline` (flat) | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `dispatchRays`        | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |

## `ICommandBuffer` interface

| API               | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-------------------|-----|------|-------|-------|--------|-------|------|
| `getDesc`         | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getNativeHandle` | :x: | :x:  | :x:   | yes   | yes    | yes   | yes  |

## `ICommandQueue` interface

| API                       | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|---------------------------|-----|------|-------|-------|--------|-------|------|
| `getType`                 | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `createCommandEncoder`    | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `submit`                  | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getNativeHandle`         | :x: | yes  | :x:   | yes   | yes    | yes   | yes  |
| `waitOnHost`              | yes | yes  | yes   | yes   | yes    | yes   | yes  |
| `getCompletedSequence`    | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `waitForSequence`         | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |
| `getTimestampCalibration` | yes | yes  | yes   | yes   | yes    | :x:   | :x:  |
| `SubmitDesc::outSequence` | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |

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

| API                 | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|---------------------|-----|------|-------|-------|--------|-------|------|
| `getInfo`           | :x: | yes  | yes   | yes   | yes    | yes   | yes  |
| `getConfig`         | :x: | yes  | yes   | yes   | yes    | yes   | yes  |
| `configure`         | :x: | yes  | yes   | yes   | yes    | yes   | yes  |
| `unconfigure`       | :x: | yes  | yes   | yes   | yes    | yes   | yes  |
| `acquireNextImage`  | :x: | yes  | yes   | yes   | yes    | yes   | yes  |
| `present`           | :x: | yes  | yes   | yes   | yes    | yes   | yes  |

Note: CUDA's surface is implemented using a Vulkan swapchain.

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

| API                   | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|-----------------------|-----|------|-------|-------|--------|-------|------|
| `getHandle`           | :x: | yes  | :x:   | yes   | yes    | yes   | :x:  |
| `getDeviceAddress`    | :x: | yes  | :x:   | yes   | yes    | yes   | :x:  |
| `getDescriptorHandle` | :x: | :x:  | :x:   | yes   | yes    | :x:   | :x:  |

## `IHeap` interface

| API                | CPU | CUDA | D3D11 | D3D12 | Vulkan | Metal | WGPU |
|--------------------|-----|------|-------|-------|--------|-------|------|
| `allocate`         | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `free`             | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `report`           | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `flush`            | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |
| `removeEmptyPages` | :x: | yes  | :x:   | yes   | yes    | :x:   | :x:  |

### Resource queue sharing

`BufferDesc::sharingMode` and `TextureDesc::sharingMode` default to `SharingMode::Exclusive`.
`Concurrent` permits access by all graphics, compute and transfer queue families selected by the device.
Vulkan deduplicates the family indices and uses exclusive native sharing when only one family exists.
D3D12 retains the descriptor policy without a native queue-family ownership operation.
For newly created Concurrent textures, D3D12 normalizes `defaultState` to `General` (native COMMON),
including the descriptor returned by `getDesc()`. This keeps creation and command-end restoration
compatible with graphics, compute and copy queues. Exclusive textures and native-handle imports
retain their existing default-state contracts; Vulkan does not apply this normalization.
Native-handle imports must describe the sharing mode used to create the native resource.

Concurrent sharing removes ownership transfers, not memory or layout dependencies. Callers must
order initial uploads and writes before readers. Independent read submissions may overlap only
while their layouts are compatible; automatic end-of-command restoration to `defaultState` also
participates in that requirement. Existing release/acquire calls on Concurrent resources retain
state handling but do not issue queue-family ownership transfers.
