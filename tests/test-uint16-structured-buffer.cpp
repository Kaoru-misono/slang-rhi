#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

// skip D3D11: fxc doesn't support uint16_t
// skip WGPU: crashes
GPU_TEST_CASE("uint16-structured-buffer", D3D12 | Vulkan | Metal | CPU | CUDA)
{
    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-uint16-buffer", "computeMain", shaderProgram.writeRef()));

    const BindingSetLayoutEntry entries[] = {{BindingKind::RWStructuredBuffer, 1, "buffer"}};
    BindingSetLayoutDesc setLayoutDesc = {};
    setLayoutDesc.entries = entries;
    setLayoutDesc.entryCount = SLANG_COUNT_OF(entries);
    ComPtr<IBindingSetLayout> setLayout;
    REQUIRE_CALL(device->createBindingSetLayout(setLayoutDesc, setLayout.writeRef()));
    const PipelineLayoutSetDesc layoutSets[] = {{"resources", setLayout}};
    PipelineLayoutDesc layoutDesc = {};
    layoutDesc.program = shaderProgram;
    layoutDesc.sets = layoutSets;
    layoutDesc.setCount = SLANG_COUNT_OF(layoutSets);
    ComPtr<IPipelineLayout> layout;
    REQUIRE_CALL(device->createPipelineLayout(layoutDesc, layout.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    pipelineDesc.layout = layout;
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
    const BindingSetEntry bindings[] = {{.slot = 0, .resource = buffer}};
    BindingSetDesc setDesc = {};
    setDesc.layout = setLayout;
    setDesc.entries = bindings;
    setDesc.entryCount = SLANG_COUNT_OF(bindings);
    ComPtr<IBindingSet> set;
    REQUIRE_CALL(device->createBindingSet(setDesc, set.writeRef()));

    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        auto passEncoder = commandEncoder->beginComputePass();
        IBindingSet* sets[] = {set};
        FlatBindingDesc binding = {};
        binding.sets = sets;
        binding.setCount = SLANG_COUNT_OF(sets);
        REQUIRE_CALL(passEncoder->bindPipeline(pipeline, binding));
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();

        REQUIRE_CALL(queue->submit(commandEncoder->finish()));
        REQUIRE_CALL(queue->waitOnHost());
    }

    compareComputeResult(device, buffer, makeArray<uint16_t>(1, 2, 3, 4));
}
