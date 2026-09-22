#include "testing.h"

#include "../src/device.h"
#include "../src/core/deferred.h"

using namespace rhi;
using namespace rhi::testing;

namespace {
struct DeleteProbe : DeviceChild
{
    int& count;
    DeferredDeleteQueue* queue;
    DeleteProbe(int& count, DeferredDeleteQueue* queue = nullptr)
        : DeviceChild(nullptr), count(count), queue(queue)
    {}
    ~DeleteProbe()
    {
        ++count;
        if (queue)
            queue->add(new DeleteProbe(count), {});
    }
};
}

TEST_CASE("deferred-delete-independent-completions")
{
    DeferredDeleteQueue queue;
    int count = 0;
    queue.add(new DeleteProbe(count), {0, 2, 0});
    queue.add(new DeleteProbe(count, &queue), {1, 0, 0});
    queue.collect({1, 0, 0});
    CHECK(count == 2);
    CHECK(queue.size() == 1);
    queue.collect({1, 2, 0});
    CHECK(count == 3);
    CHECK(queue.size() == 0);
}

GPU_TEST_CASE("deferred-delete-idle-compute", D3D12 | Vulkan)
{
    auto graphics = device->getQueue(QueueType::Graphics);
    auto compute = device->getQueue(QueueType::Compute);
    REQUIRE_CALL(graphics->waitOnHost());
    REQUIRE_CALL(compute->waitOnHost());
    auto* impl = getUnderlyingDevice(device);
    impl->collectGarbage();
    auto baseline = gResourceCount.load();

    auto gate = device->createFence({});
    auto done = device->createFence({});
    REQUIRE(gate);
    REQUIRE(done);
    // Always unblock submitted work, including on a failing assertion.
    SLANG_RHI_DEFERRED({ gate->setCurrentValue(1); compute->waitOnHost(); });
    BufferDesc desc{};
    desc.size = 256;
    desc.usage = BufferUsage::CopySource;
    desc.defaultState = ResourceState::CopySource;
    desc.memoryType = MemoryType::Upload;
    auto source = device->createBuffer(desc);
    desc.usage = BufferUsage::CopyDestination;
    desc.defaultState = ResourceState::CopyDestination;
    desc.memoryType = MemoryType::ReadBack;
    auto destination = device->createBuffer(desc);
    REQUIRE(source);
    REQUIRE(destination);
    auto encoder = compute->createCommandEncoder();
    encoder->copyBuffer(destination, 0, source, 0, 256);
    auto commands = encoder->finish();
    REQUIRE(commands);
    ICommandBuffer* commandList[] = {commands};
    IFence* waits[] = {gate};
    IFence* signals[] = {done};
    uint64_t values[] = {1};
    SubmitDesc submit{};
    submit.commandBuffers = commandList;
    submit.commandBufferCount = 1;
    submit.waitFences = waits;
    submit.waitFenceValues = values;
    submit.waitFenceCount = 1;
    submit.signalFences = signals;
    submit.signalFenceValues = values;
    submit.signalFenceCount = 1;
    REQUIRE_CALL(compute->submit(submit));
    commands.setNull();
    encoder.setNull();
    source.setNull();
    destination.setNull();
    impl->collectGarbage();
    CHECK(gResourceCount.load() >= baseline + 2);
    REQUIRE_CALL(gate->setCurrentValue(1));
    REQUIRE_CALL(device->waitForFences(1, signals, values, true, 5'000'000'000));
    // No compute queue operation occurs between completion and these graphics maintenance points.
    for (int i = 0; i < 8; ++i)
        REQUIRE_CALL(graphics->waitOnHost());
    CHECK(gResourceCount.load() <= baseline);
    CHECK(impl->m_deferredDeletes.size() == 0);
}

GPU_TEST_CASE("deferred-delete-view-and-fence", D3D12 | Vulkan)
{
    auto compute = device->getQueue(QueueType::Compute);
    auto* impl = getUnderlyingDevice(device);
    REQUIRE_CALL(device->getQueue(QueueType::Graphics)->waitOnHost());
    REQUIRE_CALL(compute->waitOnHost());
    auto gate = device->createFence({});
    auto done = device->createFence({});
    auto temporary = device->createFence({});
    REQUIRE(gate);
    REQUIRE(done);
    REQUIRE(temporary);
    SLANG_RHI_DEFERRED({ gate->setCurrentValue(1); compute->waitOnHost(); });

    TextureDesc textureDesc{};
    textureDesc.format = Format::RGBA8Unorm;
    textureDesc.size = {4, 4, 1};
    textureDesc.usage = TextureUsage::ShaderResource;
    auto texture = device->createTexture(textureDesc);
    REQUIRE(texture);
    auto view = texture->createView({});
    REQUIRE(view);
    DescriptorHandle handle{};
    REQUIRE_CALL(view->getDescriptorHandle(DescriptorHandleAccess::Read, &handle));
    IFence* waits[] = {gate};
    IFence* signals[] = {temporary, done};
    uint64_t values[] = {1, 1};
    SubmitDesc submit{};
    submit.waitFences = waits;
    submit.waitFenceValues = values;
    submit.waitFenceCount = 1;
    submit.signalFences = signals;
    submit.signalFenceValues = values;
    submit.signalFenceCount = 2;
    REQUIRE_CALL(compute->submit(submit));
    auto count = gResourceCount.load();
    view.setNull();
    texture.setNull();
    temporary.setNull();
    impl->collectGarbage();
    CHECK(gResourceCount.load() == count);
    CHECK(impl->m_deferredDeletes.size() >= 3);
    REQUIRE_CALL(gate->setCurrentValue(1));
    IFence* completion[] = {done};
    REQUIRE_CALL(device->waitForFences(1, completion, values, true, 5'000'000'000));
    REQUIRE_CALL(device->getQueue(QueueType::Graphics)->waitOnHost());
    CHECK(gResourceCount.load() <= count - 2);
    CHECK(impl->m_deferredDeletes.size() == 0);
}

GPU_TEST_CASE("deferred-delete-view-device-release-orders", (D3D12 | Vulkan) | DontCreateDevice)
{
    for (int order = 0; order < 3; ++order)
    {
        auto baseline = gResourceCount.load();
        DeviceDesc deviceDesc{};
        deviceDesc.deviceType = ctx->deviceType;
        deviceDesc.adapter = getSelectedDeviceAdapter(ctx->deviceType);
        ComPtr<IDevice> local;
        REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
        auto sampler = local->createSampler({});
        REQUIRE(sampler);
        TextureDesc desc{};
        desc.format = Format::RGBA8Unorm;
        desc.usage = TextureUsage::ShaderResource;
        desc.sampler = sampler;
        auto texture = local->createTexture(desc);
        REQUIRE(texture);
        auto defaultView = texture->getDefaultView();
        auto customView = texture->createView({});
        REQUIRE(defaultView);
        REQUIRE(customView);
        DescriptorHandle handle{};
        REQUIRE_CALL(defaultView->getDescriptorHandle(DescriptorHandleAccess::Read, &handle));
        REQUIRE_CALL(customView->getDescriptorHandle(DescriptorHandleAccess::Read, &handle));
        if (local->hasFeature(Feature::CombinedTextureSampler))
            REQUIRE_CALL(customView->getCombinedTextureSamplerDescriptorHandle(&handle));
        sampler.setNull();
        if (order == 0)
            local.setNull();
        if (order != 2)
            texture.setNull();
        customView.setNull();
        defaultView.setNull();
        texture.setNull();
        local.setNull();
        CHECK(gResourceCount.load() == baseline);
    }
}

// This test verifies that the deferred delete mechanism keeps GPU resources alive until the next submit.
GPU_TEST_CASE("deferred-delete", ALL)
{
    // Determine which resource types are deferred delete for the given device type.
    // This is implementation defined, but we need to know for testing purposes.
    bool deferredBuffer = false;
    bool deferredTexture = false;
    bool deferredSampler = false;
    bool deferredAccel = false;

    switch (device->getDeviceType())
    {
    case ::DeviceType::D3D11:
        // D3D11 already handles deferred release internally.
        break;
    case ::DeviceType::D3D12:
        // D3D12 deferred deletes all resources.
        deferredBuffer = true;
        deferredTexture = true;
        deferredSampler = true;
        deferredAccel = true;
        break;
    case ::DeviceType::Vulkan:
        // Vulkan deferred deletes all resources.
        deferredBuffer = true;
        deferredTexture = true;
        deferredSampler = true;
        deferredAccel = true;
        break;
    case ::DeviceType::Metal:
        // Metal deferred deletes all resources.
        deferredBuffer = true;
        deferredTexture = true;
        deferredSampler = true;
        deferredAccel = true;
        break;
    case ::DeviceType::CPU:
        // CPU devices dispatch commands synchronously, so resources can be deleted immediately.
        break;
    case ::DeviceType::CUDA:
        // CUDA deferred deletes all resources except samplers which have no GPU representation.
        deferredBuffer = true;
        deferredTexture = true;
        deferredAccel = true;
        break;
    case ::DeviceType::WGPU:
        // WGPU already handles deferred release internally.
        break;
    default:
        break;
    }

    auto queue = device->getQueue(QueueType::Graphics);
    uint64_t countBegin = gResourceCount;
    uint64_t countBefore = gResourceCount;

    // Create and release a buffer.
    {
        BufferDesc bufferDesc = {};
        bufferDesc.size = 256;
        bufferDesc.memoryType = MemoryType::DeviceLocal;
        bufferDesc.usage = BufferUsage::ShaderResource;

        ComPtr<IBuffer> buffer;
        REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, buffer.writeRef()));
        buffer = nullptr;
    }

    // Check buffer is still alive due to deferred delete.
    if (deferredBuffer)
    {
        CHECK_GT(gResourceCount, countBefore);
    }
    countBefore = gResourceCount;

    // Create and release a texture.
    {
        TextureDesc textureDesc = {};
        textureDesc.type = TextureType::Texture2D;
        textureDesc.format = Format::RGBA8Unorm;
        textureDesc.size = {16, 16, 1};
        textureDesc.memoryType = MemoryType::DeviceLocal;
        textureDesc.usage = TextureUsage::ShaderResource;

        ComPtr<ITexture> texture;
        REQUIRE_CALL(device->createTexture(textureDesc, nullptr, texture.writeRef()));
        texture = nullptr;
    }

    // Check texture is still alive due to deferred delete.
    if (deferredTexture)
    {
        CHECK_GT(gResourceCount, countBefore);
    }
    countBefore = gResourceCount;

    // Create and release a sampler.
    {
        SamplerDesc samplerDesc = {};

        ComPtr<ISampler> sampler;
        REQUIRE_CALL(device->createSampler(samplerDesc, sampler.writeRef()));
        sampler = nullptr;
    }

    // Check sampler is still alive due to deferred delete.
    if (deferredSampler)
    {
        CHECK_GT(gResourceCount, countBefore);
    }
    countBefore = gResourceCount;

    // Create and release an acceleration structure (if supported).
    if (device->hasFeature(Feature::AccelerationStructure))
    {
        AccelerationStructureDesc accelDesc = {};
        accelDesc.size = 1024;

        ComPtr<IAccelerationStructure> accel;
        REQUIRE_CALL(device->createAccelerationStructure(accelDesc, accel.writeRef()));
        accel = nullptr;
    }

    // Check acceleration structure is still alive due to deferred delete (if supported).
    if (deferredAccel && device->hasFeature(Feature::AccelerationStructure))
    {
        CHECK_GT(gResourceCount, countBefore);
    }
    countBefore = gResourceCount;

    // Do a submit - this should trigger executeDeferredDeletes.
    {
        auto encoder = queue->createCommandEncoder();
        queue->submit(encoder->finish());
    }

    // CUDA retires deferred resources lazily; wait to drain them on every backend.
    REQUIRE_CALL(queue->waitOnHost());

    // All deferred resources should now be deleted.
    CHECK_LE(gResourceCount.load(), countBegin);
}


// Dependencies released with a resource must not wait for unrelated later work.
GPU_TEST_CASE("deferred-delete-dependencies", D3D12 | Vulkan | Metal)
{
    auto queue = device->getQueue(QueueType::Graphics);
    REQUIRE_CALL(queue->waitOnHost());
    ComPtr<IFence> gate;
    REQUIRE_CALL(device->createFence({}, gate.writeRef()));
    const uint64_t countBegin = gResourceCount.load();

    SUBCASE("texture-sampler")
    {
        ComPtr<ISampler> sampler;
        REQUIRE_CALL(device->createSampler({}, sampler.writeRef()));
        TextureDesc desc = {};
        desc.format = Format::RGBA8Unorm;
        desc.usage = TextureUsage::ShaderResource;
        desc.sampler = sampler;
        ComPtr<ITexture> texture;
        REQUIRE_CALL(device->createTexture(desc, nullptr, texture.writeRef()));
        // Exercise the cached view without retaining a consumer reference.
        REQUIRE(texture->getDefaultView());
        sampler.setNull();
        texture.setNull();
    }
    SUBCASE("acceleration-structure-buffer")
    {
        if ((device->getDeviceType() != DeviceType::D3D12 && device->getDeviceType() != DeviceType::Vulkan) ||
            !device->hasFeature(Feature::AccelerationStructure))
            return;
        AccelerationStructureDesc desc = {};
        desc.size = 1024;
        ComPtr<IAccelerationStructure> accel;
        REQUIRE_CALL(device->createAccelerationStructure(desc, accel.writeRef()));
        accel.setNull();
    }
    SUBCASE("micromap-buffer")
    {
        if ((device->getDeviceType() != DeviceType::D3D12 && device->getDeviceType() != DeviceType::Vulkan) ||
            !device->hasFeature(Feature::OpacityMicromap))
            return;
        MicromapDesc desc = {};
        desc.size = 1024;
        ComPtr<IMicromap> micromap;
        REQUIRE_CALL(device->createMicromap(desc, micromap.writeRef()));
        micromap.setNull();
    }

    CHECK_GT(gResourceCount.load(), countBegin);

    // Submission retires old resources while this new submission remains blocked.
    // A dependency first released during retirement would acquire the new ID.
    IFence* waitFences[] = {gate};
    uint64_t waitValues[] = {1};
    SubmitDesc submitDesc = {};
    submitDesc.waitFenceCount = 1;
    submitDesc.waitFences = waitFences;
    submitDesc.waitFenceValues = waitValues;
    Result submitResult = queue->submit(submitDesc);
    const uint64_t countAfterSubmit = gResourceCount.load();

    // Unblock the queue before any assertion can abort the test.
    REQUIRE_CALL(gate->setCurrentValue(1));
    REQUIRE_CALL(queue->waitOnHost());
    REQUIRE_CALL(submitResult);
    CHECK_EQ(countAfterSubmit, countBegin);
    CHECK_EQ(gResourceCount.load(), countBegin);
}

// Stress test that verifies deferred delete works correctly with actual GPU work.
// This creates temporary buffers, uses them in compute shaders, and releases them.
// If deferred delete isn't working, the GPU would read from deleted buffers and produce wrong results.
GPU_TEST_CASE("deferred-delete-stress", ALL)
{
    ComPtr<IShaderProgram> writeValueProgram;
    REQUIRE_CALL(loadProgram(device, "test-deferred-delete", "writeValue", writeValueProgram.writeRef()));

    ComputePipelineDesc writeValuePipelineDesc = {};
    writeValuePipelineDesc.program = writeValueProgram.get();
    ComPtr<IComputePipeline> writeValuePipeline;
    REQUIRE_CALL(device->createComputePipeline(writeValuePipelineDesc, writeValuePipeline.writeRef()));

    ComPtr<IShaderProgram> accumulateProgram;
    REQUIRE_CALL(loadProgram(device, "test-deferred-delete", "accumulate", accumulateProgram.writeRef()));

    ComputePipelineDesc accumulatePipelineDesc = {};
    accumulatePipelineDesc.program = accumulateProgram.get();
    ComPtr<IComputePipeline> accumulatePipeline;
    REQUIRE_CALL(device->createComputePipeline(accumulatePipelineDesc, accumulatePipeline.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);

    static const size_t kEntryCount = 1024 * 1024;

    // Create accumulation buffer initialized to 0.
    BufferDesc accumBufferDesc = {};
    accumBufferDesc.size = kEntryCount * sizeof(uint32_t);
    accumBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    accumBufferDesc.memoryType = MemoryType::DeviceLocal;

    std::vector<uint32_t> zeroData(kEntryCount, 0);
    ComPtr<IBuffer> accumBuffer;
    REQUIRE_CALL(device->createBuffer(accumBufferDesc, zeroData.data(), accumBuffer.writeRef()));

    // Buffer desc for temporary buffers.
    BufferDesc tempBufferDesc = {};
    tempBufferDesc.size = kEntryCount * sizeof(uint32_t);
    tempBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess;
    tempBufferDesc.memoryType = MemoryType::DeviceLocal;

    const uint32_t iterations = 100;

    for (uint32_t i = 0; i < iterations; ++i)
    {
        // Create a temporary buffer.
        ComPtr<IBuffer> tempBuffer;
        REQUIRE_CALL(device->createBuffer(tempBufferDesc, nullptr, tempBuffer.writeRef()));

        // Submit 1: Write iteration counter to temp buffer.
        {
            auto encoder = queue->createCommandEncoder();
            auto pass = encoder->beginComputePass();
            auto rootObject = pass->bindPipeline(writeValuePipeline);
            ShaderCursor cursor(rootObject);
            cursor["buffer"].setBinding(tempBuffer);
            cursor["value"].setData(i);
            pass->dispatchCompute(kEntryCount / 256, 1, 1);
            pass->end();
            queue->submit(encoder->finish());
        }

        // Submit 2: Add temp buffer value to accumulation buffer.
        {
            auto encoder = queue->createCommandEncoder();
            auto pass = encoder->beginComputePass();
            auto rootObject = pass->bindPipeline(accumulatePipeline);
            ShaderCursor cursor(rootObject);
            cursor["accumBuffer"].setBinding(accumBuffer);
            cursor["srcBuffer"].setBinding(tempBuffer);
            pass->dispatchCompute(kEntryCount / 256, 1, 1);
            pass->end();
            queue->submit(encoder->finish());
        }

        // Release the temp buffer - it should be deferred since GPU may still be using it.
        tempBuffer = nullptr;
    }

    queue->waitOnHost();

    std::vector<uint32_t> resultData(kEntryCount);
    device->readBuffer(accumBuffer, 0, kEntryCount * sizeof(uint32_t), resultData.data());
    // Expected result: sum of 0 + 1 + 2 + ... + (iterations-1) = iterations * (iterations - 1) / 2
    uint32_t expected = iterations * (iterations - 1) / 2;
    for (size_t i = 0; i < kEntryCount; ++i)
        CHECK_EQ(resultData[i], expected);
}


GPU_TEST_CASE("device-lost-terminal-reclamation", (D3D12 | Vulkan) | DontCreateDevice)
{
    auto baseline = gResourceCount.load();
    DeviceDesc desc{};
    desc.deviceType = ctx->deviceType;
    desc.adapter = getSelectedDeviceAdapter(ctx->deviceType);
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(desc, local.writeRef()));
    auto queue = local->getQueue(QueueType::Graphics);
    auto fence = local->createFence({});
    auto buffer = local->createBuffer({.size = 256, .usage = BufferUsage::CopyDestination});
    REQUIRE(buffer);
    REQUIRE(fence);
    REQUIRE_CALL(queue->waitOnHost());
    auto* impl = getUnderlyingDevice(local);
    impl->collectGarbage();

    // The GPU is idle before simulating terminal loss. Unreachable requirements model
    // work whose completion can no longer be proven.
    int deleted = 0;
    impl->m_deferredDeletes.add(new DeleteProbe(deleted, &impl->m_deferredDeletes), {UINT64_MAX, 0, 0});
    impl->collectGarbage();
    CHECK(deleted == 0);
    impl->m_deviceLost = true;
    buffer.setNull();
    impl->collectGarbage();
    CHECK(deleted == 2);
    CHECK(impl->m_deferredDeletes.size() == 0);
    uint64_t value = 123;
    CHECK(SLANG_FAILED(fence->getCurrentValue(&value)));
    CHECK(value == 0);
    SubmitDesc submit{};
    CHECK(SLANG_FAILED(queue->submit(submit)));
    CHECK(SLANG_FAILED(queue->waitOnHost()));
    CHECK(!queue->createCommandEncoder());
    fence.setNull();
    queue.setNull();
    local.setNull();
    CHECK(gResourceCount.load() == baseline);
}

GPU_TEST_CASE("command-buffer-single-use-and-pool-reuse", (D3D12 | Vulkan) | DontCreateDevice)
{
    DeviceDesc deviceDesc{};
    deviceDesc.deviceType = ctx->deviceType;
    deviceDesc.adapter = getSelectedDeviceAdapter(ctx->deviceType);
    // The duplicate-command-buffer rejection lives in the debug layer, and this device routes its
    // validation errors to the null callback instead of failing the test.
    deviceDesc.enableValidation = true;
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
    auto queue = local->getQueue(QueueType::Compute);
    auto gate = local->createFence({});
    REQUIRE(gate);
    SLANG_RHI_DEFERRED({ gate->setCurrentValue(1); queue->waitOnHost(); });
    auto encoder = queue->createCommandEncoder();
    auto commands = encoder->finish();
    encoder.setNull();
    REQUIRE(commands);
    NativeHandle originalHandle{};
    REQUIRE_CALL(commands->getNativeHandle(&originalHandle));

    ICommandBuffer* buffers[] = {commands, commands};
    SubmitDesc desc{};
    desc.commandBuffers = buffers;
    desc.commandBufferCount = 2;
    CHECK(queue->submit(desc) == SLANG_E_INVALID_ARG);

    desc.commandBufferCount = 1;
    IFence* waits[] = {gate};
    uint64_t values[] = {1};
    desc.waitFences = waits;
    desc.waitFenceValues = values;
    desc.waitFenceCount = 1;
    REQUIRE_CALL(queue->submit(desc));
    CHECK(queue->submit(desc) == SLANG_E_INVALID_ARG);
    REQUIRE_CALL(gate->setCurrentValue(1));
    REQUIRE_CALL(queue->waitOnHost());
    desc.waitFenceCount = 0;
    CHECK(queue->submit(desc) == SLANG_E_INVALID_ARG);

    auto secondEncoder = queue->createCommandEncoder();
    auto secondCommands = secondEncoder->finish();
    secondEncoder.setNull();
    REQUIRE(secondCommands);
    NativeHandle secondHandle{};
    REQUIRE_CALL(secondCommands->getNativeHandle(&secondHandle));
    CHECK(secondHandle.value != originalHandle.value);
    CHECK(queue->submit(desc) == SLANG_E_INVALID_ARG);
    REQUIRE_CALL(queue->submit(secondCommands));
    REQUIRE_CALL(queue->waitOnHost());

    // Reuse is permitted only after the old external handle has been released.
    commands.setNull();
    auto reusedEncoder = queue->createCommandEncoder();
    auto reusedCommands = reusedEncoder->finish();
    reusedEncoder.setNull();
    REQUIRE(reusedCommands);
    NativeHandle reusedHandle{};
    REQUIRE_CALL(reusedCommands->getNativeHandle(&reusedHandle));
    CHECK(reusedHandle.value == originalHandle.value);
    REQUIRE_CALL(queue->submit(reusedCommands));
    REQUIRE_CALL(queue->waitOnHost());
}

GPU_TEST_CASE("submit-consumes-command-buffer-on-device-loss", (D3D12 | Vulkan) | DontCreateDevice)
{
    auto baseline = gResourceCount.load();
    DeviceDesc deviceDesc{};
    deviceDesc.deviceType = ctx->deviceType;
    deviceDesc.adapter = getSelectedDeviceAdapter(ctx->deviceType);
    ComPtr<IDevice> local;
    REQUIRE_CALL(getRHI()->createDevice(deviceDesc, local.writeRef()));
    auto queue = local->getQueue(QueueType::Compute);
    auto* impl = getUnderlyingDevice(local);
    auto encoder = queue->createCommandEncoder();
    auto commands = encoder->finish();
    encoder.setNull();
    REQUIRE(commands);
    ICommandBuffer* buffers[] = {commands};
    SubmitDesc desc{};
    desc.commandBuffers = buffers;
    desc.commandBufferCount = 1;

    impl->m_deviceLost = true;
    CHECK(SLANG_FAILED(queue->submit(desc)));
    // The recording is consumed either way, and a lost device hands out no new ones to replay it.
    CHECK(SLANG_FAILED(queue->submit(desc)));
    CHECK(!queue->createCommandEncoder());

    commands.setNull();
    impl->collectGarbage();
    CHECK(impl->m_deferredDeletes.size() == 0);
    queue.setNull();
    local.setNull();
    CHECK(gResourceCount.load() == baseline);
}

#include <thread>

GPU_TEST_CASE("command-buffer-concurrent-single-use", D3D12 | Vulkan)
{
    auto queue = device->getQueue(QueueType::Compute);
    auto gate = device->createFence({});
    REQUIRE(gate);
    SLANG_RHI_DEFERRED({ gate->setCurrentValue(1); queue->waitOnHost(); });
    auto encoder = queue->createCommandEncoder();
    auto commands = encoder->finish();
    REQUIRE(commands);
    ICommandBuffer* buffers[] = {commands};
    IFence* waits[] = {gate};
    uint64_t values[] = {1};
    SubmitDesc desc{};
    desc.commandBuffers = buffers;
    desc.commandBufferCount = 1;
    desc.waitFences = waits;
    desc.waitFenceValues = values;
    desc.waitFenceCount = 1;
    Result results[] = {SLANG_FAIL, SLANG_FAIL};
    std::atomic<bool> start{false};
    auto submit = [&](size_t index)
    {
        while (!start.load())
            std::this_thread::yield();
        results[index] = queue->submit(desc);
    };
    std::thread first(submit, 0);
    std::thread second(submit, 1);
    start = true;
    first.join();
    second.join();
    CHECK(SLANG_SUCCEEDED(results[0]) != SLANG_SUCCEEDED(results[1]));
    for (size_t i = 0; i < 2; ++i)
    {
        CHECK(results[i] == (SLANG_SUCCEEDED(results[i]) ? SLANG_OK : SLANG_E_INVALID_ARG));
    }
    REQUIRE_CALL(gate->setCurrentValue(1));
    REQUIRE_CALL(queue->waitOnHost());
}
