# API implementation status

## `IDevice` interface

| API | D3D12 | Vulkan |
| ------------------------------------ | ------- | -------- |
| `getNativeDeviceHandles` | yes | yes |
| `getInfo` | yes | yes |
| `hasFeature` | yes | yes |
| `getFeatures` | yes | yes |
| `getCapabilities` | yes | yes |
| `hasCapability` | yes | yes |
| `getFormatSupport` | yes | yes |
| `getSlangSession` | yes | yes |
| `getQueue` | yes | yes |
| `createTexture` | yes | yes |
| `createTextureFromNativeHandle` | yes | yes |
| `createTextureFromSharedHandle` | :x: | :x: |
| `createBuffer` | yes | yes |
| `createBufferFromNativeHandle` | yes | yes |
| `createBufferFromSharedHandle` | :x: | :x: |
| `mapBuffer` | yes | yes |
| `unmapBuffer` | yes | yes |
| `createSampler` | yes | yes |
| `createTextureView` | yes | yes |
| `createSurface` | yes | yes |
| `createInputLayout` | yes | yes |
| `createShaderObject` | yes | yes |
| `createShaderObjectFromTypeLayout` | yes | yes |
| `createRootShaderObject` | yes | yes |
| `createShaderTable` | yes | yes |
| `createShaderProgram` | yes | yes |
| `createRenderPipeline` | yes | yes |
| `createComputePipeline` | yes | yes |
| `createRayTracingPipeline` | yes | yes |
| `readTexture` | yes | yes |
| `readBuffer` | yes | yes |
| `createQueryPool` | yes | yes |
| `getAccelerationStructureSizes` | yes | yes |
| `getMicromapSizes` | yes | yes |
| `getClusterOperationSizes` | yes | yes |
| `createAccelerationStructure` | yes | yes |
| `createMicromap` | yes | yes |
| `createFence` | yes | yes |
| `waitForFences` | yes | yes |
| `createHeap` | yes | yes |
| `getTextureAllocationInfo` | yes | yes |
| `getTextureRowAlignment` | yes | yes |
| `getCooperativeVectorProperties` | yes | yes |
| `getCooperativeVectorMatrixSize` | yes | yes |
| `convertCooperativeVectorMatrix` | yes | yes |
| `isCooperativeMatrixSupported` (3) | yes | yes |
| `reportHeaps` | yes | yes |

(1) dummy implementation only
(2) returns nullptr but succeeds
(3) returns false when not supported

## `IBuffer` interface

| API | D3D12 | Vulkan |
| ----------------------- | ------- | -------- |
| `getDesc` | yes | yes |
| `getSharedHandle` | yes | yes |
| `getDeviceAddress` | yes | yes |
| `getDescriptorHandle` | yes | yes |

(1) returns host address

## `ITexture` interface

| API | D3D12 | Vulkan |
| ------------------------ | ------- | -------- |
| `getDesc` | yes | yes |
| `getSharedHandle` | yes | yes |
| `createView` | yes | yes |
| `getDefaultView` | yes | yes |
| `getSubresourceLayout` | yes | yes |

## `ITextureView` interface

| API | D3D12 | Vulkan |
| -------------------------------------------- | ------- | -------- |
| `getDesc` | yes | yes |
| `getTexture` | yes | yes |
| `getDescriptorHandle` | yes | yes |
| `getCombinedTextureSamplerDescriptorHandle` | yes | yes |

## `ISampler` interface

| API | D3D12 | Vulkan |
| ----------------------- | ------- | -------- |
| `getDesc` | yes | yes |
| `getDescriptorHandle` | yes | yes |

## `IFence` interface

| API | D3D12 | Vulkan |
| ------------------- | ------- | -------- |
| `getCurrentValue` | yes | yes |
| `setCurrentValue` | yes | yes |
| `getNativeHandle` | yes | yes |
| `getSharedHandle` | yes | yes |

## `IShaderObject` interface

| API | D3D12 | Vulkan |
| ----------------------------- | ------- | -------- |
| `getElementTypeLayout` | yes | yes |
| `getContainerType` | yes | yes |
| `getEntryPointCount` | yes | yes |
| `getEntryPoint` | yes | yes |
| `setData` | yes | yes |
| `getObject` | yes | yes |
| `setObject` | yes | yes |
| `setBinding` | yes | yes |
| `setDescriptorHandle` | yes | yes |
| `reserveData` | yes | yes |
| `setSpecializationArgs` | yes | yes |
| `getRawData` | yes | yes |
| `getSize` | yes | yes |
| `setConstantBufferOverride` | :x: | :x: |
| `finalize` | yes | yes |
| `isFinalized` | yes | yes |

## `IShaderTable` interface

## `IPipeline` interface

| API | D3D12 | Vulkan |
| ------------------- | ------- | -------- |
| `getProgram` | yes | yes |
| `getNativeHandle` | yes | yes |

## `IRenderPipeline` interface

| API | D3D12 | Vulkan |
| ------------------- | ------- | -------- |
| `getDesc` | yes | yes |
| `getNativeHandle` | yes | yes |

## `IComputePipeline` interface

| API | D3D12 | Vulkan |
| ------------------- | ------- | -------- |
| `getDesc` | yes | yes |
| `getNativeHandle` | yes | yes |

## `IRayTracingPipeline` interface

| API | D3D12 | Vulkan |
| ------------------- | ------- | -------- |
| `getDesc` | yes | yes |
| `getNativeHandle` | yes | yes |

## `IQueryPool` interface

| API | D3D12 | Vulkan |
| ----------------------------- | ------- | -------- |
| `getDesc` | yes | yes |
| `getResultState` | yes | yes |
| `getResult` | yes | yes |
| `reset` | yes | yes |
| `reset(queryIndex, count)` | yes | yes |

## `ICommandEncoder` interface

| API | D3D12 | Vulkan |
| ---------------------------------------- | ------- | -------- |
| `getDesc` | yes | yes |
| `beginRenderPass` | yes | yes |
| `beginComputePass` | yes | yes |
| `beginRayTracingPass` | yes | yes |
| `copyBuffer` | yes | yes |
| `copyTexture` | yes | yes |
| `copyTextureToBuffer` | yes | yes |
| `copyBufferToTexture` | yes | yes |
| `uploadTextureData` | yes | yes |
| `uploadBufferData` | yes | yes |
| `clearBuffer` | yes | yes |
| `clearTextureFloat` | yes | yes |
| `clearTextureUint` | yes | yes |
| `clearTextureSint` | yes | yes |
| `clearTextureDepthStencil` | yes | yes |
| `resolveQuery` | yes | yes |
| `buildAccelerationStructure` | yes | yes |
| `buildMicromap` | yes | yes |
| `copyAccelerationStructure` | yes | yes |
| `queryAccelerationStructureProperties` | yes | yes |
| `executeClusterOperation` | yes | yes |
| `convertCooperativeVectorMatrix` | yes | yes |
| `setBufferState` | yes | yes |
| `setTextureState` | yes | yes |
| `globalBarrier` | yes | yes |
| `releaseBufferForQueue` | yes | yes |
| `releaseTextureForQueue` | yes | yes |
| `acquireBufferFromQueue` | yes | yes |
| `acquireTextureFromQueue` | yes | yes |
| `pushDebugGroup` | yes | yes |
| `popDebugGroup` | yes | yes |
| `insertDebugMarker` | yes | yes |
| `writeTimestamp` | yes | yes |
| `finish` | yes | yes |
| `getNativeHandle` | :x: | :x: |

## `IPassEncoder` interface

| API | D3D12 | Vulkan |
| --------------------- | ------- | -------- |
| `pushDebugGroup` | yes | yes |
| `popDebugGroup` | yes | yes |
| `insertDebugMarker` | yes | yes |
| `writeTimestamp` | yes | yes |
| `end` | yes | yes |

## `IRenderPassEncoder` interface

| API | D3D12 | Vulkan |
| ----------------------- | ------- | -------- |
| `bindPipeline` | yes | yes |
| `setRenderState` | yes | yes |
| `draw` | yes | yes |
| `drawIndexed` | yes | yes |
| `drawIndirect` | yes | yes |
| `drawIndexedIndirect` | yes | yes |
| `drawMeshTasks` | yes | yes |

## `IComputePassEncoder` interface

| API | D3D12 | Vulkan |
| --------------------------- | ------- | -------- |
| `bindPipeline` | yes | yes |
| `dispatchCompute` | yes | yes |
| `dispatchComputeIndirect` | yes | yes |

## `IRayTracingPassEncoder` interface

| API | D3D12 | Vulkan |
| ---------------- | ------- | -------- |
| `bindPipeline` | yes | yes |
| `dispatchRays` | yes | yes |

## `ICommandBuffer` interface

| API | D3D12 | Vulkan |
| ------------------- | ------- | -------- |
| `getDesc` | yes | yes |
| `getNativeHandle` | yes | yes |

## `ICommandQueue` interface

| API | D3D12 | Vulkan |
| --------------------------- | ------- | -------- |
| `getType` | yes | yes |
| `createCommandEncoder` | yes | yes |
| `submit` | yes | yes |
| `getNativeHandle` | yes | yes |
| `waitOnHost` | yes | yes |
| `getTimestampCalibration` | yes | yes |

### Queue type support

| Queue Type | D3D12 | Vulkan |
| ------------------------ | ------- | -------- |
| `QueueType::Graphics` | yes | yes |
| `QueueType::Compute` | yes | yes |
| `QueueType::Transfer` | yes | yes |

The queue family ownership transfer (QFOT) methods (`releaseBufferForQueue`, `releaseTextureForQueue`,
`acquireBufferFromQueue`, `acquireTextureFromQueue`) are used to transfer resource ownership between
queues of different types. These are supported on D3D12 and Vulkan. On D3D12, they are implemented
as resource state barriers. On Vulkan, they use queue family ownership transfer barriers.

## `ISurface` interface

| API | D3D12 | Vulkan |
| --------------------- | ------- | -------- |
| `getInfo` | yes | yes |
| `getConfig` | yes | yes |
| `configure` | yes | yes |
| `unconfigure` | yes | yes |
| `acquireNextImage` | yes | yes |
| `present` | yes | yes |

## `IAccelerationStructure` interface

| API | D3D12 | Vulkan |
| ----------------------- | ------- | -------- |
| `getHandle` | yes | yes |
| `getDeviceAddress` | yes | yes |
| `getDescriptorHandle` | yes | yes |

## `IHeap` interface

| API | D3D12 | Vulkan |
| -------------------- | ------- | -------- |
| `allocate` | yes | yes |
| `free` | yes | yes |
| `report` | yes | yes |
| `flush` | yes | yes |
| `removeEmptyPages` | yes | yes |
