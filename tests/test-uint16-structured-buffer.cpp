#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

// skip D3D11: fxc doesn't support uint16_t
// skip WGPU: crashes
static void runUint16StructuredBuffer(IDevice* device, bool parameterBlock)
{
    ComPtr<IShaderProgram> shaderProgram;
    if (parameterBlock)
        REQUIRE_CALL(loadComputeProgramFromSource(
            device,
            "#define UINT16_PARAMETER_BLOCK 1\n#include \"test-uint16-buffer.slang\"\n",
            shaderProgram.writeRef()
        ));
    else
        REQUIRE_CALL(loadProgram(device, "test-uint16-buffer", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    const int numberCount = 4;
    uint16_t initialData[] = {0, 1, 2, 3};
    BufferDesc bufferDesc = {};
    bufferDesc.size = numberCount * sizeof(uint16_t);
    bufferDesc.format = Format::Undefined;
    // Note: we don't specify any element size here, and rhi should be able to derive the
    // correct element size from the reflection infomation.
    bufferDesc.elementSize = 0;
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                       BufferUsage::CopySource;
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    bufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> buffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)initialData, buffer.writeRef()));
    REQUIRE(buffer->getDesc().elementSize == 0);
    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        REQUIRE(rootObject != nullptr);
        ShaderCursor cursor(rootObject);
        REQUIRE_CALL((parameterBlock ? cursor["resources"]["buffer"] : cursor["buffer"]).setBinding(buffer));
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();

        REQUIRE_CALL(queue->submit(commandEncoder->finish()));
        REQUIRE_CALL(queue->waitOnHost());
    }

    compareComputeResult(device, buffer, makeArray<uint16_t>(1, 2, 3, 4));
}

GPU_TEST_CASE("uint16-structured-buffer", D3D12 | Vulkan | Metal | CPU | CUDA)
{
    runUint16StructuredBuffer(device, false);
}

GPU_TEST_CASE("uint16-structured-buffer-parameterBlock", D3D12 | Vulkan)
{
    runUint16StructuredBuffer(device, true);
}
