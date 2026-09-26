#include "testing.h"
#include "rhi-shared.h"

#if SLANG_RHI_ENABLE_VULKAN
#include "vulkan/vk-pipeline.h"
#include "vulkan/vk-shader-table.h"
#endif
#if SLANG_RHI_ENABLE_D3D12
#include "d3d12/d3d12-pipeline.h"
#include "d3d12/d3d12-shader-table.h"
#endif

using namespace rhi;
using namespace rhi::testing;

namespace {

// A virtual pipeline only has a concrete pipeline, which keys the shader table cache, after its first dispatch.
Pipeline* getConcretePipeline(IRayTracingPipeline* pipeline)
{
    auto* rayTracingPipeline = checked_cast<RayTracingPipeline*>(pipeline);
    return rayTracingPipeline->isVirtual() ? rayTracingPipeline->getConcretePipeline() : rayTracingPipeline;
}

// The cache is keyed by the raw pipeline address, so it must own the pipeline: otherwise a new pipeline
// allocated at the same address would be served the stale shader identifiers.
template<typename TableImpl, typename PipelineImpl>
void checkCacheRetainsPipeline(IShaderTable* shaderTable, IRayTracingPipeline* pipeline)
{
    auto* table = checked_cast<TableImpl*>(shaderTable);
    auto* concretePipeline = checked_cast<PipelineImpl*>(getConcretePipeline(pipeline));
    REQUIRE(concretePipeline != nullptr);

    const uint64_t referencesBefore = concretePipeline->getReferenceCount();
    auto* pipelineData = table->getPipelineData(concretePipeline);
    REQUIRE(pipelineData != nullptr);
    CHECK_EQ(concretePipeline->getReferenceCount(), referencesBefore + 1);

    CHECK_EQ(table->getPipelineData(concretePipeline), pipelineData);
    CHECK_EQ(concretePipeline->getReferenceCount(), referencesBefore + 1);
}

void checkCacheRetainsPipeline(IDevice* device, IShaderTable* shaderTable, IRayTracingPipeline* pipeline)
{
    switch (device->getDeviceType())
    {
#if SLANG_RHI_ENABLE_VULKAN
    case DeviceType::Vulkan:
        checkCacheRetainsPipeline<vk::ShaderTableImpl, vk::RayTracingPipelineImpl>(shaderTable, pipeline);
        return;
#endif
#if SLANG_RHI_ENABLE_D3D12
    case DeviceType::D3D12:
        checkCacheRetainsPipeline<d3d12::ShaderTableImpl, d3d12::RayTracingPipelineImpl>(shaderTable, pipeline);
        return;
#endif
    default:
        FAIL("no shader table implementation for this device type");
    }
}

} // namespace

// Test verifies that a shader table works correctly when used with multiple
// pipelines that have the same entry point names but different entry point
// parameters.
GPU_TEST_CASE("ray-tracing-shader-table-multi-pipeline", ALL)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    if (device->getDeviceType() == DeviceType::CUDA)
        SKIP("CUDA/OptiX uses __ldg to load entrypoint parameters which uses non-coherent read-only data cache");

    // Pipeline A: rayGen(output, value) -> output[i] = value + i
    ComPtr<IShaderProgram> programA;
    REQUIRE_CALL(loadProgram(device, "test-ray-tracing-shader-table-multi-pipeline-a", "rayGen", programA.writeRef()));

    ComPtr<IRayTracingPipeline> pipelineA;
    {
        RayTracingPipelineDesc pipelineDesc = {};
        pipelineDesc.program = programA;
        REQUIRE_CALL(device->createRayTracingPipeline(pipelineDesc, pipelineA.writeRef()));
    }

    // Pipeline B: rayGen(output, value, multiplier) -> output[i] = value + i * multiplier
    ComPtr<IShaderProgram> programB;
    REQUIRE_CALL(loadProgram(device, "test-ray-tracing-shader-table-multi-pipeline-b", "rayGen", programB.writeRef()));

    ComPtr<IRayTracingPipeline> pipelineB;
    {
        RayTracingPipelineDesc pipelineDesc = {};
        pipelineDesc.program = programB;
        REQUIRE_CALL(device->createRayTracingPipeline(pipelineDesc, pipelineB.writeRef()));
    }

    // Create a single shader table with the "rayGen" entry point.
    // This shader table will be used with both pipelines.
    ComPtr<IShaderTable> shaderTable;
    {
        ShaderTableDesc shaderTableDesc = {};
        const char* rayGenNames[] = {"rayGen"};
        shaderTableDesc.program = programA;
        shaderTableDesc.rayGenShaderCount = SLANG_COUNT_OF(rayGenNames);
        shaderTableDesc.rayGenShaderEntryPointNames = rayGenNames;
        REQUIRE_CALL(device->createShaderTable(shaderTableDesc, shaderTable.writeRef()));
    }

    const uint32_t width = 2;
    const uint32_t height = 2;

    ComPtr<IBuffer> outputBuffer;
    {
        BufferDesc bufferDesc;
        bufferDesc.size = width * height * sizeof(uint32_t);
        bufferDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
        outputBuffer = device->createBuffer(bufferDesc, nullptr);
        REQUIRE(outputBuffer != nullptr);
    }

    auto queue = device->getQueue(QueueType::Graphics);

    // --- Use pipeline B first ---
    // This populates the shader table's per-pipeline data for pipeline B.
    {
        auto commandEncoder = queue->createCommandEncoder();
        auto passEncoder = commandEncoder->beginRayTracingPass();
        auto rootObject = passEncoder->bindPipeline(pipelineB, shaderTable);
        auto cursor = ShaderCursor(rootObject->getEntryPoint(0));
        cursor["output"].setBinding(outputBuffer);
        cursor["value"].setData<uint32_t>(100);
        cursor["multiplier"].setData<uint32_t>(3);
        passEncoder->dispatchRays(0, width, height, 1);
        passEncoder->end();
        queue->submit(commandEncoder->finish());
    }

    // rayGen B: output[i] = value + i * multiplier = 100 + i * 3
    compareComputeResult(device, outputBuffer, std::array<uint32_t, 4>{100, 103, 106, 109});

    // --- Now use pipeline A ---
    // Before the fix, this would use the strides/offsets computed for pipeline B,
    // leading to incorrect shader record addressing.
    {
        auto commandEncoder = queue->createCommandEncoder();
        auto passEncoder = commandEncoder->beginRayTracingPass();
        auto rootObject = passEncoder->bindPipeline(pipelineA, shaderTable);
        auto cursor = ShaderCursor(rootObject->getEntryPoint(0));
        cursor["output"].setBinding(outputBuffer);
        cursor["value"].setData<uint32_t>(1);
        passEncoder->dispatchRays(0, width, height, 1);
        passEncoder->end();
        queue->submit(commandEncoder->finish());
    }

    // rayGen A: output[i] = value + i = 1 + i
    compareComputeResult(device, outputBuffer, std::array<uint32_t, 4>{1, 2, 3, 4});

    // --- Go back to pipeline B ---
    // Verify pipeline B still works correctly after using pipeline A.
    {
        auto commandEncoder = queue->createCommandEncoder();
        auto passEncoder = commandEncoder->beginRayTracingPass();
        auto rootObject = passEncoder->bindPipeline(pipelineB, shaderTable);
        auto cursor = ShaderCursor(rootObject->getEntryPoint(0));
        cursor["output"].setBinding(outputBuffer);
        cursor["value"].setData<uint32_t>(50);
        cursor["multiplier"].setData<uint32_t>(10);
        passEncoder->dispatchRays(0, width, height, 1);
        passEncoder->end();
        queue->submit(commandEncoder->finish());
    }

    // rayGen B: output[i] = value + i * multiplier = 50 + i * 10
    compareComputeResult(device, outputBuffer, std::array<uint32_t, 4>{50, 60, 70, 80});

    // Recorded commands retain the pipeline themselves, so query a table that never entered a command.
    ComPtr<IShaderTable> unusedShaderTable;
    {
        ShaderTableDesc shaderTableDesc = {};
        const char* rayGenNames[] = {"rayGen"};
        shaderTableDesc.program = programA;
        shaderTableDesc.rayGenShaderCount = SLANG_COUNT_OF(rayGenNames);
        shaderTableDesc.rayGenShaderEntryPointNames = rayGenNames;
        REQUIRE_CALL(device->createShaderTable(shaderTableDesc, unusedShaderTable.writeRef()));
    }
    checkCacheRetainsPipeline(device, unusedShaderTable, pipelineA);
    checkCacheRetainsPipeline(device, unusedShaderTable, pipelineB);
}
