#include "testing.h"

#include <initializer_list>

using namespace rhi;
using namespace rhi::testing;

struct IndirectComputePipeline
{
    std::vector<const char*> names;
    size_t constantsSize;
    ComPtr<IComputePipeline> pipeline;
};

static ComPtr<slang::ISession> createIndirectSession(IDevice* device, const char* entryPoint)
{
    auto deviceSession = device->getSlangSession();
    auto* globalSession = deviceSession->getGlobalSession();
    slang::TargetDesc target{};
    target.format = device->getDeviceType() == DeviceType::Vulkan ? SLANG_SPIRV : SLANG_DXIL;
    if (target.format == SLANG_DXIL)
        target.profile = globalSession->findProfile("sm_6_0");
    target.forceGLSLScalarBufferLayout = true;
    slang::CompilerOptionEntry option{};
    option.name = slang::CompilerOptionName::EmitSpirvDirectly;
    option.value.kind = slang::CompilerOptionValueKind::Int;
    option.value.intValue0 = 1;
    std::string macroName = std::string("INDIRECT_ENTRY_") + entryPoint;
    slang::PreprocessorMacroDesc macro{macroName.c_str(), "1"};
    auto searchPaths = getSlangSearchPaths();
    slang::SessionDesc desc{};
    desc.targets = &target;
    desc.targetCount = 1;
    desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    desc.searchPaths = searchPaths.data();
    desc.searchPathCount = SlangInt(searchPaths.size());
    desc.preprocessorMacros = &macro;
    desc.preprocessorMacroCount = 1;
    desc.compilerOptionEntries = &option;
    desc.compilerOptionEntryCount = 1;
    ComPtr<slang::ISession> session;
    REQUIRE_CALL(globalSession->createSession(desc, session.writeRef()));
    return session;
}

static IndirectComputePipeline createIndirectPipeline(
    IDevice* device,
    const char* entryPoint,
    std::initializer_list<const char*> names,
    size_t constantsSize
)
{
    auto session = createIndirectSession(device, entryPoint);
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, session, "test-compute-indirect", entryPoint, program.writeRef()));
    IndirectComputePipeline result;
    result.names = names;
    result.constantsSize = (constantsSize + 15) & ~size_t(15);
    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = program;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, result.pipeline.writeRef()));
    return result;
}

static ComPtr<IShaderObject> createIndirectSet(
    IDevice* device, const IndirectComputePipeline& pipeline, std::initializer_list<IBuffer*> buffers)
{
    auto root = device->createRootShaderObject(pipeline.pipeline->getProgram());
    REQUIRE(root != nullptr);
    ComPtr<IShaderObject> entry;
    REQUIRE_CALL(root->getEntryPoint(0, entry.writeRef()));
    auto resources = ShaderCursor(entry)["resources"];
    uint32_t index = 0;
    for (auto* buffer : buffers)
        REQUIRE_CALL(resources[pipeline.names[index++]].setBinding(buffer));
    return root;
}

static void bindIndirectPipeline(
    IComputePassEncoder* encoder, const IndirectComputePipeline& pipeline, IShaderObject* root,
    const void* constants = nullptr, size_t constantsSize = 0)
{
    REQUIRE(pipeline.constantsSize == ((constantsSize + 15) & ~size_t(15)));
    if (constantsSize)
    {
        std::vector<uint8_t> padded(pipeline.constantsSize, 0);
        std::memcpy(padded.data(), constants, constantsSize);
        REQUIRE_CALL(ShaderCursor(root)["constants"].getDereferenced().setData(padded.data(), padded.size()));
    }
    encoder->bindPipeline(pipeline.pipeline, root);
}

// Test dispatchComputeIndirect with a simple compute shader.
// The test sets up an indirect argument buffer with dispatch dimensions written by the GPU,
// then verifies the compute shader ran with the correct number of threads.
GPU_TEST_CASE("compute-indirect-shader-object", D3D12 | Vulkan)
{
    auto pipeline = createIndirectPipeline(
        device, "computeMain", {"outputBuffer"}, sizeof(uint32_t));
    auto writeArgsPipeline = createIndirectPipeline(
        device, "writeDispatchArgs",
        {"dispatchArgsBuffer"}, 4 * sizeof(uint32_t));

    // Create output buffer to count how many threads ran for each dispatch
    const int dispatchCount = 4;
    BufferDesc outputBufferDesc = {};
    outputBufferDesc.size = dispatchCount * sizeof(uint32_t);
    outputBufferDesc.format = Format::Undefined;
    outputBufferDesc.elementSize = sizeof(uint32_t);
    outputBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                             BufferUsage::CopySource;
    outputBufferDesc.defaultState = ResourceState::UnorderedAccess;
    outputBufferDesc.memoryType = MemoryType::DeviceLocal;

    uint32_t initialData[dispatchCount] = {0, 0, 0, 0};
    ComPtr<IBuffer> outputBuffer;
    REQUIRE_CALL(device->createBuffer(outputBufferDesc, initialData, outputBuffer.writeRef()));

    // Create indirect argument buffer (uninitialized - will be written by GPU)
    // Each IndirectDispatchArguments contains (threadGroupCountX, threadGroupCountY, threadGroupCountZ)
    // With numthreads(4, 4, 1), each group has 16 threads
    BufferDesc indirectBufferDesc = {};
    indirectBufferDesc.size = dispatchCount * sizeof(IndirectDispatchArguments);
    indirectBufferDesc.elementSize = sizeof(IndirectDispatchArguments);
    indirectBufferDesc.usage =
        BufferUsage::IndirectArgument | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination;
    indirectBufferDesc.defaultState = ResourceState::UnorderedAccess;
    indirectBufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> indirectBuffer;
    REQUIRE_CALL(device->createBuffer(indirectBufferDesc, nullptr, indirectBuffer.writeRef()));
    auto outputSet = createIndirectSet(device, pipeline, {outputBuffer});
    auto writeArgsSet = createIndirectSet(device, writeArgsPipeline, {indirectBuffer});

    // Dispatch arguments to write: (x, y, z) thread group counts
    // dispatch0: 1x1x1 = 1 group = 16 threads
    // dispatch1: 2x1x1 = 2 groups = 32 threads
    // dispatch2: 1x2x1 = 2 groups = 32 threads
    // dispatch3: 2x2x1 = 4 groups = 64 threads
    uint32_t threadGroupCounts[dispatchCount][3] = {
        {1, 1, 1},
        {2, 1, 1},
        {1, 2, 1},
        {2, 2, 1},
    };

    // Execute: first write dispatch args on GPU, then run indirect dispatches
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        // First pass: write dispatch arguments using GPU
        {
            auto passEncoder = commandEncoder->beginComputePass();
            for (int i = 0; i < dispatchCount; i++)
            {
                uint32_t constants[4] = {
                    uint32_t(i), threadGroupCounts[i][0], threadGroupCounts[i][1], threadGroupCounts[i][2]};
                bindIndirectPipeline(passEncoder, writeArgsPipeline, writeArgsSet, constants, sizeof(constants));
                passEncoder->dispatchCompute(1, 1, 1);
            }
            passEncoder->end();
        }

        // Second pass: execute indirect dispatches
        {
            auto passEncoder = commandEncoder->beginComputePass();
            for (int i = 0; i < dispatchCount; i++)
            {
                uint32_t dispatchIndex = i;
                bindIndirectPipeline(passEncoder, pipeline, outputSet, &dispatchIndex, sizeof(dispatchIndex));
                passEncoder->dispatchComputeIndirect({indirectBuffer, i * sizeof(IndirectDispatchArguments)});
            }
            passEncoder->end();
        }

        REQUIRE_CALL(queue->submit(commandEncoder->finish()));
        REQUIRE_CALL(queue->waitOnHost());
    }

    // Verify results
    // With numthreads(4, 4, 1) = 16 threads per group:
    // dispatch0: 1x1x1 = 1 group = 16 threads
    // dispatch1: 2x1x1 = 2 groups = 32 threads
    // dispatch2: 1x2x1 = 2 groups = 32 threads
    // dispatch3: 2x2x1 = 4 groups = 64 threads
    compareComputeResult(device, outputBuffer, makeArray<uint32_t>(16, 32, 32, 64));
}

// Test dispatchComputeIndirect with zero dispatch dimensions.
// This verifies the implementation handles edge cases correctly.
GPU_TEST_CASE("compute-indirect-zero-shader-object", D3D12 | Vulkan)
{
    auto pipeline = createIndirectPipeline(
        device, "computeMain", {"outputBuffer"}, sizeof(uint32_t));
    auto writeArgsPipeline = createIndirectPipeline(
        device, "writeDispatchArgs",
        {"dispatchArgsBuffer"}, 4 * sizeof(uint32_t));

    // Create output buffer
    BufferDesc outputBufferDesc = {};
    outputBufferDesc.size = sizeof(uint32_t);
    outputBufferDesc.format = Format::Undefined;
    outputBufferDesc.elementSize = sizeof(uint32_t);
    outputBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                             BufferUsage::CopySource;
    outputBufferDesc.defaultState = ResourceState::UnorderedAccess;
    outputBufferDesc.memoryType = MemoryType::DeviceLocal;

    uint32_t initialData = 0;
    ComPtr<IBuffer> outputBuffer;
    REQUIRE_CALL(device->createBuffer(outputBufferDesc, &initialData, outputBuffer.writeRef()));

    // Create indirect argument buffer (will be written by GPU with zero dispatch)
    BufferDesc indirectBufferDesc = {};
    indirectBufferDesc.size = sizeof(IndirectDispatchArguments);
    indirectBufferDesc.elementSize = sizeof(IndirectDispatchArguments);
    indirectBufferDesc.usage =
        BufferUsage::IndirectArgument | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination;
    indirectBufferDesc.defaultState = ResourceState::UnorderedAccess;
    indirectBufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> indirectBuffer;
    REQUIRE_CALL(device->createBuffer(indirectBufferDesc, nullptr, indirectBuffer.writeRef()));
    auto outputSet = createIndirectSet(device, pipeline, {outputBuffer});
    auto writeArgsSet = createIndirectSet(device, writeArgsPipeline, {indirectBuffer});

    // Execute: write zero dispatch args, then run indirect dispatch
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        // First pass: write zero dispatch arguments using GPU
        {
            auto passEncoder = commandEncoder->beginComputePass();
            uint32_t constants[4] = {0, 0, 0, 0};
            bindIndirectPipeline(passEncoder, writeArgsPipeline, writeArgsSet, constants, sizeof(constants));
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        // Second pass: execute indirect dispatch with zero dimensions
        {
            auto passEncoder = commandEncoder->beginComputePass();
            uint32_t dispatchIndex = 0;
            bindIndirectPipeline(passEncoder, pipeline, outputSet, &dispatchIndex, sizeof(dispatchIndex));
            passEncoder->dispatchComputeIndirect({indirectBuffer, 0});
            passEncoder->end();
        }

        REQUIRE_CALL(queue->submit(commandEncoder->finish()));
        REQUIRE_CALL(queue->waitOnHost());
    }

    // Verify no threads ran (output should still be 0)
    compareComputeResult(device, outputBuffer, makeArray<uint32_t>(0));
}

// Test dispatchComputeIndirect with non-zero buffer offset.
// This verifies the offset parameter is handled correctly.
GPU_TEST_CASE("compute-indirect-offset-shader-object", D3D12 | Vulkan)
{
    auto pipeline = createIndirectPipeline(
        device, "computeMain", {"outputBuffer"}, sizeof(uint32_t));
    auto writeArgsPipeline = createIndirectPipeline(
        device, "writeDispatchArgs",
        {"dispatchArgsBuffer"}, 4 * sizeof(uint32_t));

    // Create output buffer
    BufferDesc outputBufferDesc = {};
    outputBufferDesc.size = sizeof(uint32_t);
    outputBufferDesc.format = Format::Undefined;
    outputBufferDesc.elementSize = sizeof(uint32_t);
    outputBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                             BufferUsage::CopySource;
    outputBufferDesc.defaultState = ResourceState::UnorderedAccess;
    outputBufferDesc.memoryType = MemoryType::DeviceLocal;

    uint32_t initialData = 0;
    ComPtr<IBuffer> outputBuffer;
    REQUIRE_CALL(device->createBuffer(outputBufferDesc, &initialData, outputBuffer.writeRef()));

    // Create indirect argument buffer with space for 2 dispatch args (index 0 unused, index 1 used)
    // This tests that the offset parameter works correctly
    BufferDesc indirectBufferDesc = {};
    indirectBufferDesc.size = 2 * sizeof(IndirectDispatchArguments);
    indirectBufferDesc.elementSize = sizeof(IndirectDispatchArguments);
    indirectBufferDesc.usage =
        BufferUsage::IndirectArgument | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination;
    indirectBufferDesc.defaultState = ResourceState::UnorderedAccess;
    indirectBufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> indirectBuffer;
    REQUIRE_CALL(device->createBuffer(indirectBufferDesc, nullptr, indirectBuffer.writeRef()));
    auto outputSet = createIndirectSet(device, pipeline, {outputBuffer});
    auto writeArgsSet = createIndirectSet(device, writeArgsPipeline, {indirectBuffer});

    // Execute: write dispatch args at index 1, then dispatch using offset
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        // First pass: write dispatch arguments at index 1 using GPU
        {
            auto passEncoder = commandEncoder->beginComputePass();
            // Write the second slot: 3x1x1 = 3 groups = 48 threads.
            uint32_t constants[4] = {1, 3, 1, 1};
            bindIndirectPipeline(passEncoder, writeArgsPipeline, writeArgsSet, constants, sizeof(constants));
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        // Second pass: execute indirect dispatch with offset to second slot
        {
            auto passEncoder = commandEncoder->beginComputePass();
            uint32_t dispatchIndex = 0;
            bindIndirectPipeline(passEncoder, pipeline, outputSet, &dispatchIndex, sizeof(dispatchIndex));
            // Use offset to point to second dispatch args slot
            passEncoder->dispatchComputeIndirect({indirectBuffer, sizeof(IndirectDispatchArguments)});
            passEncoder->end();
        }

        REQUIRE_CALL(queue->submit(commandEncoder->finish()));
        REQUIRE_CALL(queue->waitOnHost());
    }

    // Verify correct number of threads ran (3 groups * 16 threads = 48)
    compareComputeResult(device, outputBuffer, makeArray<uint32_t>(48));
}

static void runComputeIndirectShaderReadFlat(IDevice* device, QueueType queueType)
{
    auto writePipeline = createIndirectPipeline(
        device, "writeDispatchRecord",
        {"dispatchRecords"}, 2 * sizeof(uint32_t));
    auto consumePipeline = createIndirectPipeline(
        device, "consumeDispatchRecord",
        {"dispatchRecords",
         "outputBuffer"}, 0);

    BufferDesc recordDesc = {};
    recordDesc.size = 8 * sizeof(uint32_t);
    recordDesc.elementSize = recordDesc.size;
    recordDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::ShaderResource | BufferUsage::IndirectArgument |
                       BufferUsage::CopyDestination;
    recordDesc.defaultState = ResourceState::UnorderedAccess;
    recordDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> recordBuffer;
    REQUIRE_CALL(device->createBuffer(recordDesc, nullptr, recordBuffer.writeRef()));

    BufferDesc outputDesc = {};
    outputDesc.size = sizeof(uint32_t);
    outputDesc.elementSize = sizeof(uint32_t);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    outputDesc.defaultState = ResourceState::UnorderedAccess;
    outputDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> outputBuffer;
    REQUIRE_CALL(device->createBuffer(outputDesc, nullptr, outputBuffer.writeRef()));
    auto writeSet = createIndirectSet(device, writePipeline, {recordBuffer});
    auto consumeSet = createIndirectSet(device, consumePipeline, {recordBuffer, outputBuffer});

    ComPtr<IBuffer> outputReadback;
    if (queueType == QueueType::Compute)
    {
        BufferDesc readbackDesc = {};
        readbackDesc.size = sizeof(uint32_t);
        readbackDesc.usage = BufferUsage::CopyDestination;
        readbackDesc.defaultState = ResourceState::CopyDestination;
        readbackDesc.memoryType = MemoryType::ReadBack;
        REQUIRE_CALL(device->createBuffer(readbackDesc, nullptr, outputReadback.writeRef()));
    }

    auto queue = device->getQueue(queueType);
    auto commandEncoder = queue->createCommandEncoder();

    {
        auto passEncoder = commandEncoder->beginComputePass();
        uint32_t constants[2] = {41, 32};
        bindIndirectPipeline(passEncoder, writePipeline, writeSet, constants, sizeof(constants));
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();
    }

    {
        auto passEncoder = commandEncoder->beginComputePass();
        bindIndirectPipeline(passEncoder, consumePipeline, consumeSet);
        passEncoder->dispatchComputeIndirect({recordBuffer, 0});
        passEncoder->end();
    }

    if (outputReadback)
        commandEncoder->copyBuffer(outputReadback, 0, outputBuffer, 0, sizeof(uint32_t));

    REQUIRE_CALL(queue->submit(commandEncoder->finish()));
    REQUIRE_CALL(queue->waitOnHost());

    if (outputReadback)
    {
        void* mappedData = nullptr;
        REQUIRE_CALL(device->mapBuffer(outputReadback, CpuAccessMode::Read, &mappedData));
        CHECK(*static_cast<uint32_t const*>(mappedData) == 73);
        REQUIRE_CALL(device->unmapBuffer(outputReadback));
    }
    else
    {
        compareComputeResult(device, outputBuffer, makeArray<uint32_t>(73));
    }
}

GPU_TEST_CASE("compute-indirect-shader-read-shader-object", D3D12 | Vulkan)
{
    runComputeIndirectShaderReadFlat(device, QueueType::Graphics);
}

GPU_TEST_CASE("compute-indirect-shader-read-compute-queue-shader-object", D3D12 | Vulkan)
{
    runComputeIndirectShaderReadFlat(device, QueueType::Compute);
}

// Legacy shader-object binding of the same regressions.

// Test dispatchComputeIndirect with a simple compute shader.
// The test sets up an indirect argument buffer with dispatch dimensions written by the GPU,
// then verifies the compute shader ran with the correct number of threads.
// WebGPU: the shader uses RWByteAddressBuffer atomics unavailable in WGSL.
GPU_TEST_CASE("compute-indirect", D3D12 | Vulkan | Metal | CUDA)
{
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-compute-indirect", "computeMain", program.writeRef()));

    ComPtr<IShaderProgram> writeArgsProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-indirect", "writeDispatchArgs", writeArgsProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = program.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    ComputePipelineDesc writeArgsPipelineDesc = {};
    writeArgsPipelineDesc.program = writeArgsProgram.get();
    ComPtr<IComputePipeline> writeArgsPipeline;
    REQUIRE_CALL(device->createComputePipeline(writeArgsPipelineDesc, writeArgsPipeline.writeRef()));

    // Create output buffer to count how many threads ran for each dispatch
    const int dispatchCount = 4;
    BufferDesc outputBufferDesc = {};
    outputBufferDesc.size = dispatchCount * sizeof(uint32_t);
    outputBufferDesc.format = Format::Undefined;
    outputBufferDesc.elementSize = sizeof(uint32_t);
    outputBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                             BufferUsage::CopySource;
    outputBufferDesc.defaultState = ResourceState::UnorderedAccess;
    outputBufferDesc.memoryType = MemoryType::DeviceLocal;

    uint32_t initialData[dispatchCount] = {0, 0, 0, 0};
    ComPtr<IBuffer> outputBuffer;
    REQUIRE_CALL(device->createBuffer(outputBufferDesc, initialData, outputBuffer.writeRef()));

    // Create indirect argument buffer (uninitialized - will be written by GPU)
    // Each IndirectDispatchArguments contains (threadGroupCountX, threadGroupCountY, threadGroupCountZ)
    // With numthreads(4, 4, 1), each group has 16 threads
    BufferDesc indirectBufferDesc = {};
    indirectBufferDesc.size = dispatchCount * sizeof(IndirectDispatchArguments);
    indirectBufferDesc.elementSize = sizeof(IndirectDispatchArguments);
    indirectBufferDesc.usage =
        BufferUsage::IndirectArgument | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination;
    indirectBufferDesc.defaultState = ResourceState::UnorderedAccess;
    indirectBufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> indirectBuffer;
    REQUIRE_CALL(device->createBuffer(indirectBufferDesc, nullptr, indirectBuffer.writeRef()));

    // Dispatch arguments to write: (x, y, z) thread group counts
    // dispatch0: 1x1x1 = 1 group = 16 threads
    // dispatch1: 2x1x1 = 2 groups = 32 threads
    // dispatch2: 1x2x1 = 2 groups = 32 threads
    // dispatch3: 2x2x1 = 4 groups = 64 threads
    uint32_t threadGroupCounts[dispatchCount][3] = {
        {1, 1, 1},
        {2, 1, 1},
        {1, 2, 1},
        {2, 2, 1},
    };

    // Execute: first write dispatch args on GPU, then run indirect dispatches
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        // First pass: write dispatch arguments using GPU
        {
            auto passEncoder = commandEncoder->beginComputePass();
            for (int i = 0; i < dispatchCount; i++)
            {
                auto rootObject = passEncoder->bindPipeline(writeArgsPipeline);
                ShaderCursor shaderCursor(rootObject);
                shaderCursor["dispatchArgsBuffer"].setBinding(indirectBuffer);
                uint32_t dispatchIndex = i;
                shaderCursor["dispatchIndex"].setData(dispatchIndex);
                uint32_t counts[3] = {threadGroupCounts[i][0], threadGroupCounts[i][1], threadGroupCounts[i][2]};
                shaderCursor["threadGroupCounts"].setData(counts, sizeof(counts));
                passEncoder->dispatchCompute(1, 1, 1);
            }
            passEncoder->end();
        }

        // Second pass: execute indirect dispatches
        {
            auto passEncoder = commandEncoder->beginComputePass();
            for (int i = 0; i < dispatchCount; i++)
            {
                auto rootObject = passEncoder->bindPipeline(pipeline);
                ShaderCursor shaderCursor(rootObject);
                shaderCursor["outputBuffer"].setBinding(outputBuffer);
                uint32_t dispatchIndex = i;
                shaderCursor["dispatchIndex"].setData(dispatchIndex);
                passEncoder->dispatchComputeIndirect({indirectBuffer, i * sizeof(IndirectDispatchArguments)});
            }
            passEncoder->end();
        }

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    // Verify results
    // With numthreads(4, 4, 1) = 16 threads per group:
    // dispatch0: 1x1x1 = 1 group = 16 threads
    // dispatch1: 2x1x1 = 2 groups = 32 threads
    // dispatch2: 1x2x1 = 2 groups = 32 threads
    // dispatch3: 2x2x1 = 4 groups = 64 threads
    compareComputeResult(device, outputBuffer, makeArray<uint32_t>(16, 32, 32, 64));
}

// Test dispatchComputeIndirect with zero dispatch dimensions.
// This verifies the implementation handles edge cases correctly.
GPU_TEST_CASE("compute-indirect-zero", D3D12 | Vulkan | Metal | CUDA)
{
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-compute-indirect", "computeMain", program.writeRef()));

    ComPtr<IShaderProgram> writeArgsProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-indirect", "writeDispatchArgs", writeArgsProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = program.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    ComputePipelineDesc writeArgsPipelineDesc = {};
    writeArgsPipelineDesc.program = writeArgsProgram.get();
    ComPtr<IComputePipeline> writeArgsPipeline;
    REQUIRE_CALL(device->createComputePipeline(writeArgsPipelineDesc, writeArgsPipeline.writeRef()));

    // Create output buffer
    BufferDesc outputBufferDesc = {};
    outputBufferDesc.size = sizeof(uint32_t);
    outputBufferDesc.format = Format::Undefined;
    outputBufferDesc.elementSize = sizeof(uint32_t);
    outputBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                             BufferUsage::CopySource;
    outputBufferDesc.defaultState = ResourceState::UnorderedAccess;
    outputBufferDesc.memoryType = MemoryType::DeviceLocal;

    uint32_t initialData = 0;
    ComPtr<IBuffer> outputBuffer;
    REQUIRE_CALL(device->createBuffer(outputBufferDesc, &initialData, outputBuffer.writeRef()));

    // Create indirect argument buffer (will be written by GPU with zero dispatch)
    BufferDesc indirectBufferDesc = {};
    indirectBufferDesc.size = sizeof(IndirectDispatchArguments);
    indirectBufferDesc.elementSize = sizeof(IndirectDispatchArguments);
    indirectBufferDesc.usage =
        BufferUsage::IndirectArgument | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination;
    indirectBufferDesc.defaultState = ResourceState::UnorderedAccess;
    indirectBufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> indirectBuffer;
    REQUIRE_CALL(device->createBuffer(indirectBufferDesc, nullptr, indirectBuffer.writeRef()));

    // Execute: write zero dispatch args, then run indirect dispatch
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        // First pass: write zero dispatch arguments using GPU
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(writeArgsPipeline);
            ShaderCursor shaderCursor(rootObject);
            shaderCursor["dispatchArgsBuffer"].setBinding(indirectBuffer);
            uint32_t dispatchIndex = 0;
            shaderCursor["dispatchIndex"].setData(dispatchIndex);
            uint32_t counts[3] = {0, 0, 0};
            shaderCursor["threadGroupCounts"].setData(counts, sizeof(counts));
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        // Second pass: execute indirect dispatch with zero dimensions
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(pipeline);
            ShaderCursor shaderCursor(rootObject);
            shaderCursor["outputBuffer"].setBinding(outputBuffer);
            uint32_t dispatchIndex = 0;
            shaderCursor["dispatchIndex"].setData(dispatchIndex);
            passEncoder->dispatchComputeIndirect({indirectBuffer, 0});
            passEncoder->end();
        }

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    // Verify no threads ran (output should still be 0)
    compareComputeResult(device, outputBuffer, makeArray<uint32_t>(0));
}

// Test dispatchComputeIndirect with non-zero buffer offset.
// This verifies the offset parameter is handled correctly.
GPU_TEST_CASE("compute-indirect-offset", D3D12 | Vulkan | Metal | CUDA)
{
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-compute-indirect", "computeMain", program.writeRef()));

    ComPtr<IShaderProgram> writeArgsProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-indirect", "writeDispatchArgs", writeArgsProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = program.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    ComputePipelineDesc writeArgsPipelineDesc = {};
    writeArgsPipelineDesc.program = writeArgsProgram.get();
    ComPtr<IComputePipeline> writeArgsPipeline;
    REQUIRE_CALL(device->createComputePipeline(writeArgsPipelineDesc, writeArgsPipeline.writeRef()));

    // Create output buffer
    BufferDesc outputBufferDesc = {};
    outputBufferDesc.size = sizeof(uint32_t);
    outputBufferDesc.format = Format::Undefined;
    outputBufferDesc.elementSize = sizeof(uint32_t);
    outputBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                             BufferUsage::CopySource;
    outputBufferDesc.defaultState = ResourceState::UnorderedAccess;
    outputBufferDesc.memoryType = MemoryType::DeviceLocal;

    uint32_t initialData = 0;
    ComPtr<IBuffer> outputBuffer;
    REQUIRE_CALL(device->createBuffer(outputBufferDesc, &initialData, outputBuffer.writeRef()));

    // Create indirect argument buffer with space for 2 dispatch args (index 0 unused, index 1 used)
    // This tests that the offset parameter works correctly
    BufferDesc indirectBufferDesc = {};
    indirectBufferDesc.size = 2 * sizeof(IndirectDispatchArguments);
    indirectBufferDesc.elementSize = sizeof(IndirectDispatchArguments);
    indirectBufferDesc.usage =
        BufferUsage::IndirectArgument | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination;
    indirectBufferDesc.defaultState = ResourceState::UnorderedAccess;
    indirectBufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> indirectBuffer;
    REQUIRE_CALL(device->createBuffer(indirectBufferDesc, nullptr, indirectBuffer.writeRef()));

    // Execute: write dispatch args at index 1, then dispatch using offset
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        // First pass: write dispatch arguments at index 1 using GPU
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(writeArgsPipeline);
            ShaderCursor shaderCursor(rootObject);
            shaderCursor["dispatchArgsBuffer"].setBinding(indirectBuffer);
            uint32_t dispatchIndex = 1; // Write to second slot
            shaderCursor["dispatchIndex"].setData(dispatchIndex);
            uint32_t counts[3] = {3, 1, 1}; // 3x1x1 = 3 groups = 48 threads
            shaderCursor["threadGroupCounts"].setData(counts, sizeof(counts));
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        // Second pass: execute indirect dispatch with offset to second slot
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(pipeline);
            ShaderCursor shaderCursor(rootObject);
            shaderCursor["outputBuffer"].setBinding(outputBuffer);
            uint32_t dispatchIndex = 0;
            shaderCursor["dispatchIndex"].setData(dispatchIndex);
            // Use offset to point to second dispatch args slot
            passEncoder->dispatchComputeIndirect({indirectBuffer, sizeof(IndirectDispatchArguments)});
            passEncoder->end();
        }

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    // Verify correct number of threads ran (3 groups * 16 threads = 48)
    compareComputeResult(device, outputBuffer, makeArray<uint32_t>(48));
}

static void runComputeIndirectShaderRead(IDevice* device, QueueType queueType)
{
    ComPtr<IShaderProgram> writeProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-indirect", "writeDispatchRecord", writeProgram.writeRef()));

    ComPtr<IShaderProgram> consumeProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-indirect", "consumeDispatchRecord", consumeProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = writeProgram.get();
    ComPtr<IComputePipeline> writePipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, writePipeline.writeRef()));

    pipelineDesc.program = consumeProgram.get();
    ComPtr<IComputePipeline> consumePipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, consumePipeline.writeRef()));

    BufferDesc recordDesc = {};
    recordDesc.size = 8 * sizeof(uint32_t);
    recordDesc.elementSize = recordDesc.size;
    recordDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::ShaderResource | BufferUsage::IndirectArgument |
                       BufferUsage::CopyDestination;
    recordDesc.defaultState = ResourceState::UnorderedAccess;
    recordDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> recordBuffer;
    REQUIRE_CALL(device->createBuffer(recordDesc, nullptr, recordBuffer.writeRef()));

    BufferDesc outputDesc = {};
    outputDesc.size = sizeof(uint32_t);
    outputDesc.elementSize = sizeof(uint32_t);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource | BufferUsage::CopyDestination;
    outputDesc.defaultState = ResourceState::UnorderedAccess;
    outputDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> outputBuffer;
    REQUIRE_CALL(device->createBuffer(outputDesc, nullptr, outputBuffer.writeRef()));

    ComPtr<IBuffer> outputReadback;
    if (queueType == QueueType::Compute)
    {
        BufferDesc readbackDesc = {};
        readbackDesc.size = sizeof(uint32_t);
        readbackDesc.usage = BufferUsage::CopyDestination;
        readbackDesc.defaultState = ResourceState::CopyDestination;
        readbackDesc.memoryType = MemoryType::ReadBack;
        REQUIRE_CALL(device->createBuffer(readbackDesc, nullptr, outputReadback.writeRef()));
    }

    auto queue = device->getQueue(queueType);
    auto commandEncoder = queue->createCommandEncoder();

    {
        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(writePipeline);
        ShaderCursor cursor(rootObject);
        cursor["dispatchRecords"].setBinding(recordBuffer);
        cursor["itemCount"].setData(uint32_t(41));
        cursor["itemStart"].setData(uint32_t(32));
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();
    }

    {
        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(consumePipeline);
        ShaderCursor cursor(rootObject);
        cursor["dispatchRecords"].setBinding(recordBuffer);
        cursor["outputBuffer"].setBinding(outputBuffer);
        passEncoder->dispatchComputeIndirect({recordBuffer, 0});
        passEncoder->end();
    }

    if (outputReadback)
        commandEncoder->copyBuffer(outputReadback, 0, outputBuffer, 0, sizeof(uint32_t));

    REQUIRE_CALL(queue->submit(commandEncoder->finish()));
    REQUIRE_CALL(queue->waitOnHost());

    if (outputReadback)
    {
        void* mappedData = nullptr;
        REQUIRE_CALL(device->mapBuffer(outputReadback, CpuAccessMode::Read, &mappedData));
        CHECK(*static_cast<uint32_t const*>(mappedData) == 73);
        REQUIRE_CALL(device->unmapBuffer(outputReadback));
    }
    else
    {
        compareComputeResult(device, outputBuffer, makeArray<uint32_t>(73));
    }
}

GPU_TEST_CASE("compute-indirect-shader-read", D3D12 | Vulkan)
{
    runComputeIndirectShaderRead(device, QueueType::Graphics);
}

GPU_TEST_CASE("compute-indirect-shader-read-compute-queue", D3D12 | Vulkan)
{
    runComputeIndirectShaderRead(device, QueueType::Compute);
}
