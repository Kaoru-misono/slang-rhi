#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

// skip D3D11: fxc doesn't support uint16_t
// skip WGPU: crashes
/// flat compiles the parameter-block variant and binds a binding set; otherwise the root shader object binds.
static void runUint16StructuredBuffer(IDevice* device, bool flat)
{
    ComPtr<IShaderProgram> shaderProgram;
    if (flat)
        REQUIRE_CALL(loadComputeProgramFromSource(
            device,
            "#define UINT16_FLAT 1\n#include \"test-uint16-buffer.slang\"\n",
            shaderProgram.writeRef()
        ));
    else
        REQUIRE_CALL(loadProgram(device, "test-uint16-buffer", "computeMain", shaderProgram.writeRef()));

    ComPtr<IBindingSetLayout> setLayout;
    ComPtr<IPipelineLayout> layout;
    if (flat)
    {
        const BindingSetLayoutEntry entries[] = {{BindingKind::RWStructuredBuffer, 1, "buffer"}};
        BindingSetLayoutDesc setLayoutDesc = {};
        setLayoutDesc.entries = entries;
        setLayoutDesc.entryCount = SLANG_COUNT_OF(entries);
        REQUIRE_CALL(device->createBindingSetLayout(setLayoutDesc, setLayout.writeRef()));
        const PipelineLayoutSetDesc layoutSets[] = {{"resources", setLayout}};
        PipelineLayoutDesc layoutDesc = {};
        layoutDesc.program = shaderProgram;
        layoutDesc.sets = layoutSets;
        layoutDesc.setCount = SLANG_COUNT_OF(layoutSets);
        REQUIRE_CALL(device->createPipelineLayout(layoutDesc, layout.writeRef()));
    }

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
    ComPtr<IBindingSet> set;
    if (flat)
    {
        const BindingSetEntry bindings[] = {{.slot = 0, .resource = buffer}};
        BindingSetDesc setDesc = {};
        setDesc.layout = setLayout;
        setDesc.entries = bindings;
        setDesc.entryCount = SLANG_COUNT_OF(bindings);
        REQUIRE_CALL(device->createBindingSet(setDesc, set.writeRef()));
    }

    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        auto passEncoder = commandEncoder->beginComputePass();
        if (flat)
        {
            IBindingSet* sets[] = {set};
            FlatBindingDesc binding = {};
            binding.sets = sets;
            binding.setCount = SLANG_COUNT_OF(sets);
            REQUIRE_CALL(passEncoder->bindPipeline(pipeline, binding));
        }
        else
        {
            auto rootObject = passEncoder->bindPipeline(pipeline);
            REQUIRE(rootObject != nullptr);
            REQUIRE_CALL(ShaderCursor(rootObject)["buffer"].setBinding(buffer));
        }
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

GPU_TEST_CASE("uint16-structured-buffer-flat", D3D12 | Vulkan)
{
    runUint16StructuredBuffer(device, true);
}
