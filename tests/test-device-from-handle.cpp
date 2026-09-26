#include "test-compute-trivial.h"
#include <slang-rhi/acceleration-structure-utils.h>

using namespace rhi;
using namespace rhi::testing;

static void runDeviceFromHandle(IDevice* device, trivialCompute::BindingModel model)
{
    DeviceNativeHandles handles;
    CHECK_CALL(device->getNativeDeviceHandles(&handles));

    ComPtr<IDevice> newDevice;
    DeviceDesc newDeviceDesc = {};
    newDeviceDesc.deviceType = device->getDeviceType();
    newDeviceDesc.existingDeviceHandles.handles[0] = handles.handles[0];
    if (newDeviceDesc.deviceType == DeviceType::Vulkan)
    {
        newDeviceDesc.existingDeviceHandles.handles[1] = handles.handles[1];
        newDeviceDesc.existingDeviceHandles.handles[2] = handles.handles[2];
    }
    newDeviceDesc.slang.slangGlobalSession = device->getSlangSession()->getGlobalSession();
    auto searchPaths = getSlangSearchPaths();
    newDeviceDesc.slang.searchPaths = searchPaths.data();
    newDeviceDesc.slang.searchPathCount = searchPaths.size();
    CHECK_CALL(getRHI()->createDevice(newDeviceDesc, newDevice.writeRef()));

    auto pipeline = trivialCompute::createPipeline(newDevice, model);

    const int numberCount = 4;
    float initialData[] = {0.0f, 1.0f, 2.0f, 3.0f};
    BufferDesc bufferDesc = {};
    bufferDesc.size = numberCount * sizeof(float);
    bufferDesc.format = Format::Undefined;
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopyDestination |
                       BufferUsage::CopySource;
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    bufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> buffer;
    REQUIRE_CALL(newDevice->createBuffer(bufferDesc, (void*)initialData, buffer.writeRef()));

    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
        auto queue = newDevice->getQueue(QueueType::Graphics);
        auto commandEncoder = queue->createCommandEncoder();

        auto passEncoder = commandEncoder->beginComputePass();
        trivialCompute::bindBuffer(newDevice, passEncoder, pipeline, buffer);
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();

        queue->submit(commandEncoder->finish());
        queue->waitOnHost();
    }

    compareComputeResult(newDevice, buffer, makeArray<float>(1.0f, 2.0f, 3.0f, 4.0f));
}

GPU_TEST_CASE("device-from-handle", D3D12 | Vulkan | CUDA)
{
    runDeviceFromHandle(device, trivialCompute::BindingModel::Legacy);
}

GPU_TEST_CASE("device-from-handle-flat", D3D12 | Vulkan)
{
    runDeviceFromHandle(device, trivialCompute::BindingModel::Flat);
}

GPU_TEST_CASE("device-final-release-flat-dependencies", D3D12 | Vulkan | DontCreateDevice)
{
    auto localDevice = createTestingDevice(ctx, ctx->deviceType, false);
    REQUIRE(localDevice != nullptr);
    {
        auto pipeline = trivialCompute::createPipeline(localDevice, trivialCompute::BindingModel::Flat);
        BufferDesc bufferDesc{};
        bufferDesc.size = 4 * sizeof(float);
        bufferDesc.elementSize = sizeof(float);
        bufferDesc.usage = BufferUsage::UnorderedAccess;
        bufferDesc.defaultState = ResourceState::UnorderedAccess;
        ComPtr<IBuffer> buffer;
        REQUIRE_CALL(localDevice->createBuffer(bufferDesc, nullptr, buffer.writeRef()));

        const BindingSetEntry entry{.slot = 0, .resource = buffer};
        BindingSetDesc setDesc{};
        setDesc.layout = pipeline.setLayout;
        setDesc.entries = &entry;
        setDesc.entryCount = 1;
        ComPtr<IBindingSet> set;
        REQUIRE_CALL(localDevice->createBindingSet(setDesc, set.writeRef()));
        REQUIRE(set != nullptr);
    }
    // No fixture device or cached owner may collect the deferred dependency graph for us.
    // The process's live-device check must succeed after this final release alone.
    localDevice.setNull();
}

GPU_TEST_CASE("device-final-release-tlas-dependencies", D3D12 | Vulkan | DontCreateDevice)
{
    auto localDevice = createTestingDevice(ctx, ctx->deviceType, false);
    REQUIRE(localDevice != nullptr);
    if (!localDevice->hasFeature(Feature::RayTracing))
        SKIP("RayTracing unavailable; TLAS dependency final-release regression not executed");

    {
        // Prebuild queries need valid buffers, but do not read their contents or submit builds.
        BufferDesc vertexDesc{};
        vertexDesc.size = 3 * 3 * sizeof(float);
        vertexDesc.usage = BufferUsage::ShaderResource;
        vertexDesc.defaultState = ResourceState::ShaderResource;
        ComPtr<IBuffer> vertices;
        REQUIRE_CALL(localDevice->createBuffer(vertexDesc, nullptr, vertices.writeRef()));

        AccelerationStructureBuildInput blasInput{};
        blasInput.type = AccelerationStructureBuildInputType::Triangles;
        blasInput.triangles.vertexBuffers[0] = vertices;
        blasInput.triangles.vertexBufferCount = 1;
        blasInput.triangles.vertexFormat = Format::RGB32Float;
        blasInput.triangles.vertexCount = 3;
        blasInput.triangles.vertexStride = 3 * sizeof(float);
        blasInput.triangles.flags = AccelerationStructureGeometryFlags::Opaque;
        AccelerationStructureBuildDesc blasBuild{};
        blasBuild.inputs = &blasInput;
        blasBuild.inputCount = 1;
        AccelerationStructureSizes blasSizes{};
        REQUIRE_CALL(localDevice->getAccelerationStructureSizes(blasBuild, &blasSizes));
        REQUIRE(blasSizes.accelerationStructureSize > 0);
        AccelerationStructureDesc blasDesc{};
        blasDesc.kind = AccelerationStructureKind::BottomLevel;
        blasDesc.size = blasSizes.accelerationStructureSize;
        ComPtr<IAccelerationStructure> blas;
        REQUIRE_CALL(localDevice->createAccelerationStructure(blasDesc, blas.writeRef()));
        REQUIRE(blas != nullptr);

        const auto instanceStride = getAccelerationStructureInstanceDescSize(
            getAccelerationStructureInstanceDescType(localDevice)
        );
        REQUIRE(instanceStride > 0);
        BufferDesc instanceDesc{};
        instanceDesc.size = instanceStride;
        instanceDesc.usage = BufferUsage::ShaderResource;
        instanceDesc.defaultState = ResourceState::ShaderResource;
        ComPtr<IBuffer> instances;
        REQUIRE_CALL(localDevice->createBuffer(instanceDesc, nullptr, instances.writeRef()));
        AccelerationStructureBuildInput tlasInput{};
        tlasInput.type = AccelerationStructureBuildInputType::Instances;
        tlasInput.instances.instanceBuffer = instances;
        tlasInput.instances.instanceStride = uint32_t(instanceStride);
        tlasInput.instances.instanceCount = 1;
        AccelerationStructureBuildDesc tlasBuild{};
        tlasBuild.inputs = &tlasInput;
        tlasBuild.inputCount = 1;
        AccelerationStructureSizes tlasSizes{};
        REQUIRE_CALL(localDevice->getAccelerationStructureSizes(tlasBuild, &tlasSizes));
        REQUIRE(tlasSizes.accelerationStructureSize > 0);
        ComPtr<IAccelerationStructure> tlas;
        {
            IAccelerationStructure* dependencies[] = {blas};
            AccelerationStructureDependenciesDesc dependenciesDesc{};
            dependenciesDesc.structureCount = 1;
            dependenciesDesc.structures = dependencies;
            AccelerationStructureDesc tlasDesc{};
            tlasDesc.kind = AccelerationStructureKind::TopLevel;
            tlasDesc.size = tlasSizes.accelerationStructureSize;
            tlasDesc.next = &dependenciesDesc;
            REQUIRE_CALL(localDevice->createAccelerationStructure(tlasDesc, tlas.writeRef()));
        }
        REQUIRE(tlas != nullptr);
        CHECK(tlas->getDesc().next == nullptr);
        blas.setNull();
        tlas.setNull();
    }
    // Leave all reclamation to final Device destruction, not a wait/collect after releasing the TLAS.
    localDevice.setNull();
}
