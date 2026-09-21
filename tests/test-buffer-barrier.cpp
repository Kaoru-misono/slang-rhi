#include "testing.h"
#include "../src/state-tracking.h"

using namespace rhi;
using namespace rhi::testing;

class StateTrackingTestBuffer : public Buffer
{
public:
    StateTrackingTestBuffer(ResourceState defaultState)
        : Buffer(nullptr, makeDesc(defaultState))
    {
    }

    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override { return 0; }

private:
    static BufferDesc makeDesc(ResourceState defaultState)
    {
        BufferDesc desc = {};
        desc.size = 32;
        desc.memoryType = MemoryType::DeviceLocal;
        desc.usage = BufferUsage::ShaderResource | BufferUsage::IndirectArgument | BufferUsage::UnorderedAccess |
                     BufferUsage::CopySource;
        desc.defaultState = defaultState;
        return desc;
    }
};

TEST_CASE("buffer-read-state-policy-by-queue")
{
    CHECK(getBufferReadStatePolicy(QueueType::Graphics) == BufferReadStatePolicy::Graphics);
    CHECK(getBufferReadStatePolicy(QueueType::Compute) == BufferReadStatePolicy::Compute);
    CHECK(getBufferReadStatePolicy(QueueType::Transfer) == BufferReadStatePolicy::None);
}

TEST_CASE("buffer-state-tracking-compatible-read-set")
{
    SUBCASE("graphics policy merges every approved read role")
    {
        ResourceState const mergeableStates[] = {
            ResourceState::VertexBuffer,
            ResourceState::IndexBuffer,
            ResourceState::ConstantBuffer,
            ResourceState::ShaderResource,
            ResourceState::IndirectArgument,
        };

        for (ResourceState initialState : mergeableStates)
        {
            for (ResourceState requiredState : mergeableStates)
            {
                StateTrackingTestBuffer buffer(initialState);
                StateTracking tracking;
                tracking.requireBufferState(&buffer, requiredState, BufferReadStatePolicy::Graphics);

                if (initialState == requiredState)
                {
                    CHECK(tracking.getBufferBarriers().empty());
                    continue;
                }

                REQUIRE(tracking.getBufferBarriers().size() == 1);
                BufferBarrier const& barrier = tracking.getBufferBarriers()[0];
                BufferStateSet expectedStates(initialState);
                expectedStates.add(requiredState);
                CHECK(barrier.stateAfter == expectedStates);
            }
        }
    }

    SUBCASE("read requirements merge in either order")
    {
        StateTrackingTestBuffer shaderFirstBuffer(ResourceState::ShaderResource);
        StateTracking shaderFirstTracking;
        shaderFirstTracking.requireBufferState(
            &shaderFirstBuffer,
            ResourceState::IndirectArgument,
            BufferReadStatePolicy::Graphics
        );

        REQUIRE(shaderFirstTracking.getBufferBarriers().size() == 1);
        const auto& shaderFirstBarrier = shaderFirstTracking.getBufferBarriers()[0];
        CHECK(shaderFirstBarrier.stateBefore.isExactly(ResourceState::ShaderResource));
        CHECK(shaderFirstBarrier.stateAfter.contains(ResourceState::ShaderResource));
        CHECK(shaderFirstBarrier.stateAfter.contains(ResourceState::IndirectArgument));

        StateTrackingTestBuffer indirectFirstBuffer(ResourceState::IndirectArgument);
        StateTracking indirectFirstTracking;
        indirectFirstTracking.requireBufferState(
            &indirectFirstBuffer,
            ResourceState::ShaderResource,
            BufferReadStatePolicy::Graphics
        );

        REQUIRE(indirectFirstTracking.getBufferBarriers().size() == 1);
        const auto& indirectFirstBarrier = indirectFirstTracking.getBufferBarriers()[0];
        CHECK(indirectFirstBarrier.stateBefore.isExactly(ResourceState::IndirectArgument));
        CHECK(indirectFirstBarrier.stateAfter == shaderFirstBarrier.stateAfter);
    }

    SUBCASE("contained read requirements do not emit barriers")
    {
        StateTrackingTestBuffer buffer(ResourceState::ShaderResource);
        StateTracking tracking;
        tracking.requireBufferState(&buffer, ResourceState::IndirectArgument, BufferReadStatePolicy::Graphics);
        tracking.clearBarriers();

        tracking.requireBufferState(&buffer, ResourceState::ShaderResource, BufferReadStatePolicy::Graphics);
        tracking.requireBufferState(&buffer, ResourceState::IndirectArgument, BufferReadStatePolicy::Graphics);

        CHECK(tracking.getBufferBarriers().empty());
    }

    SUBCASE("read accumulation follows the backend queue policy")
    {
        StateTrackingTestBuffer computeBuffer(ResourceState::ShaderResource);
        StateTracking computeTracking;
        computeTracking.requireBufferState(
            &computeBuffer,
            ResourceState::IndirectArgument,
            BufferReadStatePolicy::Compute
        );
        REQUIRE(computeTracking.getBufferBarriers().size() == 1);
        BufferStateSet computeExpectedStates(ResourceState::ShaderResource);
        computeExpectedStates.add(ResourceState::IndirectArgument);
        CHECK(computeTracking.getBufferBarriers()[0].stateAfter == computeExpectedStates);

        StateTrackingTestBuffer computeConstantBuffer(ResourceState::ConstantBuffer);
        StateTracking computeConstantTracking;
        computeConstantTracking.requireBufferState(
            &computeConstantBuffer,
            ResourceState::ShaderResource,
            BufferReadStatePolicy::Compute
        );
        REQUIRE(computeConstantTracking.getBufferBarriers().size() == 1);
        BufferStateSet computeConstantExpectedStates(ResourceState::ConstantBuffer);
        computeConstantExpectedStates.add(ResourceState::ShaderResource);
        CHECK(computeConstantTracking.getBufferBarriers()[0].stateAfter == computeConstantExpectedStates);

        StateTrackingTestBuffer incompatibleComputeBuffer(ResourceState::IndexBuffer);
        StateTracking incompatibleComputeTracking;
        incompatibleComputeTracking.requireBufferState(
            &incompatibleComputeBuffer,
            ResourceState::ShaderResource,
            BufferReadStatePolicy::Compute
        );
        REQUIRE(incompatibleComputeTracking.getBufferBarriers().size() == 1);
        CHECK(
            incompatibleComputeTracking.getBufferBarriers()[0].stateAfter.isExactly(ResourceState::ShaderResource)
        );

        StateTrackingTestBuffer noMergeBuffer(ResourceState::ShaderResource);
        StateTracking noMergeTracking;
        noMergeTracking.requireBufferState(
            &noMergeBuffer,
            ResourceState::IndirectArgument,
            BufferReadStatePolicy::None
        );
        REQUIRE(noMergeTracking.getBufferBarriers().size() == 1);
        CHECK(noMergeTracking.getBufferBarriers()[0].stateAfter.isExactly(ResourceState::IndirectArgument));
    }

    SUBCASE("released buffers are not restored to their default state")
    {
        StateTrackingTestBuffer buffer(ResourceState::ShaderResource);
        StateTracking tracking;
        tracking.requireBufferState(&buffer, ResourceState::IndirectArgument, BufferReadStatePolicy::Graphics);
        tracking.clearBarriers();

        tracking.forgetBufferState(&buffer);
        tracking.requireDefaultStates();

        CHECK(tracking.getBufferBarriers().empty());
    }

    SUBCASE("acquired state is adopted without a redundant transition")
    {
        StateTrackingTestBuffer buffer(ResourceState::UnorderedAccess);
        StateTracking tracking;

        tracking.assumeBufferState(&buffer, ResourceState::ShaderResource);
        CHECK(tracking.getBufferBarriers().empty());

        tracking.requireDefaultStates();
        REQUIRE(tracking.getBufferBarriers().size() == 1);
        CHECK(tracking.getBufferBarriers()[0].stateBefore.isExactly(ResourceState::ShaderResource));
        CHECK(tracking.getBufferBarriers()[0].stateAfter.isExactly(ResourceState::UnorderedAccess));
    }

    SUBCASE("explicit and non-mergeable requirements replace the read set")
    {
        StateTrackingTestBuffer buffer(ResourceState::ShaderResource);
        StateTracking tracking;
        tracking.requireBufferState(&buffer, ResourceState::IndirectArgument, BufferReadStatePolicy::Graphics);
        tracking.clearBarriers();

        tracking.setBufferState(&buffer, ResourceState::ShaderResource);
        REQUIRE(tracking.getBufferBarriers().size() == 1);
        CHECK(tracking.getBufferBarriers()[0].stateAfter.isExactly(ResourceState::ShaderResource));

        tracking.clearBarriers();
        tracking.requireBufferState(&buffer, ResourceState::IndirectArgument, BufferReadStatePolicy::Graphics);
        tracking.clearBarriers();
        tracking.requireBufferState(&buffer, ResourceState::CopySource, BufferReadStatePolicy::Graphics);
        REQUIRE(tracking.getBufferBarriers().size() == 1);
        CHECK(tracking.getBufferBarriers()[0].stateAfter.isExactly(ResourceState::CopySource));
    }

    SUBCASE("repeated UAV requirements retain the ordering barrier")
    {
        StateTrackingTestBuffer buffer(ResourceState::UnorderedAccess);
        StateTracking tracking;

        tracking.requireBufferState(&buffer, ResourceState::UnorderedAccess, BufferReadStatePolicy::Graphics);

        REQUIRE(tracking.getBufferBarriers().size() == 1);
        CHECK(tracking.getBufferBarriers()[0].stateBefore.isExactly(ResourceState::UnorderedAccess));
        CHECK(tracking.getBufferBarriers()[0].stateAfter.isExactly(ResourceState::UnorderedAccess));

        tracking.clearBarriers();
        tracking.setBufferState(&buffer, ResourceState::UnorderedAccess);

        REQUIRE(tracking.getBufferBarriers().size() == 1);
        CHECK(tracking.getBufferBarriers()[0].stateBefore.isExactly(ResourceState::UnorderedAccess));
        CHECK(tracking.getBufferBarriers()[0].stateAfter.isExactly(ResourceState::UnorderedAccess));
    }

    SUBCASE("default restoration narrows a compatible read set")
    {
        StateTrackingTestBuffer buffer(ResourceState::ShaderResource);
        StateTracking tracking;
        tracking.requireBufferState(&buffer, ResourceState::IndirectArgument, BufferReadStatePolicy::Graphics);
        tracking.clearBarriers();

        tracking.requireDefaultStates();

        REQUIRE(tracking.getBufferBarriers().size() == 1);
        const auto& barrier = tracking.getBufferBarriers()[0];
        BufferStateSet expectedBefore(ResourceState::ShaderResource);
        expectedBefore.add(ResourceState::IndirectArgument);
        CHECK(barrier.stateBefore == expectedBefore);
        CHECK(barrier.stateAfter.isExactly(ResourceState::ShaderResource));
    }

    SUBCASE("pending barriers retain their original before state and latest after state")
    {
        StateTrackingTestBuffer buffer(ResourceState::ShaderResource);
        StateTracking tracking;
        tracking.requireBufferState(&buffer, ResourceState::IndirectArgument, BufferReadStatePolicy::Graphics);
        tracking.requireBufferState(&buffer, ResourceState::CopySource, BufferReadStatePolicy::Graphics);
        tracking.requireDefaultStates();

        REQUIRE(tracking.getBufferBarriers().size() == 1);
        const auto& barrier = tracking.getBufferBarriers()[0];
        CHECK(barrier.stateBefore.isExactly(ResourceState::ShaderResource));
        CHECK(barrier.stateAfter.isExactly(ResourceState::ShaderResource));
    }
}

struct Shader
{
    ComPtr<IShaderProgram> program;
    slang::ProgramLayout* reflection = nullptr;
    ComputePipelineDesc pipelineDesc = {};
    ComPtr<IComputePipeline> pipeline;
};

ComPtr<IBuffer> createFloatBuffer(
    IDevice* device,
    bool unorderedAccess,
    size_t elementCount,
    float* initialData = nullptr
)
{
    BufferDesc bufferDesc = {};
    bufferDesc.size = elementCount * sizeof(float);
    bufferDesc.format = Format::Undefined;
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.memoryType = MemoryType::DeviceLocal;
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination | BufferUsage::CopySource;
    if (unorderedAccess)
        bufferDesc.usage |= BufferUsage::UnorderedAccess;
    bufferDesc.defaultState = unorderedAccess ? ResourceState::UnorderedAccess : ResourceState::ShaderResource;
    ComPtr<IBuffer> buffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, (void*)initialData, buffer.writeRef()));
    return buffer;
}

GPU_TEST_CASE("buffer-barrier", ALL)
{
    Shader programA;
    Shader programB;
    REQUIRE_CALL(
        loadAndLinkProgram(device, "test-buffer-barrier", "computeA", programA.program.writeRef(), &programA.reflection)
    );
    REQUIRE_CALL(
        loadAndLinkProgram(device, "test-buffer-barrier", "computeB", programB.program.writeRef(), &programB.reflection)
    );
    programA.pipelineDesc.program = programA.program.get();
    programB.pipelineDesc.program = programB.program.get();
    REQUIRE_CALL(device->createComputePipeline(programA.pipelineDesc, programA.pipeline.writeRef()));
    REQUIRE_CALL(device->createComputePipeline(programB.pipelineDesc, programB.pipeline.writeRef()));

    float initialData[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ComPtr<IBuffer> inputBuffer = createFloatBuffer(device, false, 4, initialData);
    ComPtr<IBuffer> intermediateBuffer = createFloatBuffer(device, true, 4);
    ComPtr<IBuffer> outputBuffer = createFloatBuffer(device, true, 4);

    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        // Write inputBuffer to intermediateBuffer
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(programA.pipeline);
            ShaderCursor cursor(rootObject->getEntryPoint(0));
            cursor["inBuffer"].setBinding(inputBuffer);
            cursor["outBuffer"].setBinding(intermediateBuffer);
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        // Resource transition is automatically handled.

        // Write intermediateBuffer to outputBuffer
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(programB.pipeline);
            ShaderCursor cursor(rootObject->getEntryPoint(0));
            cursor["inBuffer"].setBinding(intermediateBuffer);
            cursor["outBuffer"].setBinding(outputBuffer);
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    compareComputeResult(device, outputBuffer, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f));
}

// TODO: Currently disabled because the race condition will not ALWAYS materialize, making the test unreliable.
#if 0
GPU_TEST_CASE("buffer-no-barrier-race-condition", ALL)
{
    Shader programA;
    Shader programB;
    REQUIRE_CALL(loadAndLinkProgram(device, "test-buffer-barrier", "computeA", programA.program, &programA.reflection));
    REQUIRE_CALL(loadAndLinkProgram(device, "test-buffer-barrier", "computeB", programB.program, &programB.reflection));
    programA.pipelineDesc.program = programA.program.get();
    programB.pipelineDesc.program = programB.program.get();
    REQUIRE_CALL(device->createComputePipeline(programA.pipelineDesc, programA.pipeline.writeRef()));
    REQUIRE_CALL(device->createComputePipeline(programB.pipelineDesc, programB.pipeline.writeRef()));

    float initialData[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ComPtr<IBuffer> inputBuffer = createFloatBuffer(device, false, 4, initialData);
    ComPtr<IBuffer> intermediateBuffer = createFloatBuffer(device, true, 4);
    ComPtr<IBuffer> outputBuffer = createFloatBuffer(device, true, 4);

    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        // Write inputBuffer to intermediateBuffer
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(programA.pipeline);
            ShaderCursor cursor(rootObject->getEntryPoint(0));
            cursor["inBuffer"].setBinding(inputBuffer);
            cursor["outBuffer"].setBinding(intermediateBuffer);
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        // Write intermediateBuffer to outputBuffer
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(programB.pipeline);
            ShaderCursor cursor(rootObject->getEntryPoint(0));
            cursor["inBuffer"].setBinding(intermediateBuffer);
            cursor["outBuffer"].setBinding(outputBuffer);
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        // Disable state tracking for the submit
        testing::gDebugDisableStateTracking = true;
        queue->submit(commandEncoder->finish());
        testing::gDebugDisableStateTracking = false;
        queue->waitOnHost();
    }

    // We expect the 2 platforms that do explicit state tracking normally to fail,
    // as we disabled it for the submit.
    bool expectFailure = device->getDeviceType() == DeviceType::D3D12 || device->getDeviceType() == DeviceType::Vulkan;
    compareComputeResult(device, outputBuffer, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f), expectFailure);
}
#endif

GPU_TEST_CASE("buffer-global-barrier", D3D12 | Vulkan)
{
    Shader programA;
    Shader programB;
    REQUIRE_CALL(
        loadAndLinkProgram(device, "test-buffer-barrier", "computeA", programA.program.writeRef(), &programA.reflection)
    );
    REQUIRE_CALL(
        loadAndLinkProgram(device, "test-buffer-barrier", "computeB", programB.program.writeRef(), &programB.reflection)
    );
    programA.pipelineDesc.program = programA.program.get();
    programB.pipelineDesc.program = programB.program.get();
    REQUIRE_CALL(device->createComputePipeline(programA.pipelineDesc, programA.pipeline.writeRef()));
    REQUIRE_CALL(device->createComputePipeline(programB.pipelineDesc, programB.pipeline.writeRef()));

    float initialData[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ComPtr<IBuffer> inputBuffer = createFloatBuffer(device, false, 4, initialData);
    ComPtr<IBuffer> intermediateBuffer = createFloatBuffer(device, true, 4);
    ComPtr<IBuffer> outputBuffer = createFloatBuffer(device, true, 4);

    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
        auto queue = device->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        // Write inputBuffer to intermediateBuffer
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(programA.pipeline);
            ShaderCursor cursor(rootObject->getEntryPoint(0));
            cursor["inBuffer"].setBinding(inputBuffer);
            cursor["outBuffer"].setBinding(intermediateBuffer);
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        // Explicitly add a global barrier to the encoder, ensuring all
        // previous memory operations are visibile before starting the next
        // pass.
        commandEncoder->globalBarrier();

        // Write intermediateBuffer to outputBuffer
        {
            auto passEncoder = commandEncoder->beginComputePass();
            auto rootObject = passEncoder->bindPipeline(programB.pipeline);
            ShaderCursor cursor(rootObject->getEntryPoint(0));
            cursor["inBuffer"].setBinding(intermediateBuffer);
            cursor["outBuffer"].setBinding(outputBuffer);
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
        }

        // Disable state tracking for the submit
        testing::gDebugDisableStateTracking = true;
        queue->submit(commandEncoder->finish());
        testing::gDebugDisableStateTracking = false;
        queue->waitOnHost();
    }

    compareComputeResult(device, outputBuffer, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f));
}
