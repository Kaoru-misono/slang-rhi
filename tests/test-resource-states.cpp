#include "testing.h"

#include <set>

using namespace rhi;
using namespace rhi::testing;

static void runBufferResourceStates(IDevice* device)
{
    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-dummy", "computeMain", shaderProgram.writeRef()));


    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);

    BufferUsage bufferUsage = BufferUsage::VertexBuffer | BufferUsage::IndexBuffer | BufferUsage::ConstantBuffer |
                              BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                              BufferUsage::IndirectArgument | BufferUsage::CopySource | BufferUsage::CopyDestination |
                              //   BufferUsage::AccelerationStructure | BufferUsage::AccelerationStructureBuildInput |
                              BufferUsage::ShaderTable;

    std::set<ResourceState> allowedStates = {
        ResourceState::VertexBuffer,
        ResourceState::IndexBuffer,
        ResourceState::ConstantBuffer,
        ResourceState::ShaderResource,
        ResourceState::UnorderedAccess,
        ResourceState::IndirectArgument,
        ResourceState::CopySource,
        ResourceState::CopyDestination,
        // ResourceState::AccelerationStructure,
        // ResourceState::AccelerationStructureBuildInput,
    };

    BufferDesc bufferDesc = {};
    bufferDesc.size = 256;
    bufferDesc.format = Format::Undefined;
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.usage = bufferUsage;
    bufferDesc.memoryType = MemoryType::DeviceLocal;
    ComPtr<IBuffer> buffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, nullptr, buffer.writeRef()));

    for (ResourceState state : allowedStates)
    {
        auto commandEncoder = queue->createCommandEncoder();
        commandEncoder->setBufferState(buffer, state);
        auto passEncoder = commandEncoder->beginComputePass();
        REQUIRE(passEncoder->bindPipeline(pipeline) != nullptr);
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();
        REQUIRE_CALL(queue->submit(commandEncoder->finish()));
    }

    REQUIRE_CALL(queue->waitOnHost());
}

GPU_TEST_CASE("buffer-resource-states", D3D12 | Vulkan)
{
    runBufferResourceStates(device);
}

static void runTextureResourceStates(IDevice* device)
{
    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-dummy", "computeMain", shaderProgram.writeRef()));


    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    auto queue = device->getQueue(QueueType::Graphics);

    for (uint32_t i = 1; i < (uint32_t)Format::_Count; ++i)
    {
        auto format = (Format)i;
        FormatSupport formatSupport = FormatSupport::None;
        REQUIRE_CALL(device->getFormatSupport(format, &formatSupport));

        if (!is_set(formatSupport, FormatSupport::Texture))
            continue;

        TextureUsage textureUsage = TextureUsage::CopySource | TextureUsage::CopyDestination |
                                    TextureUsage::ResolveSource | TextureUsage::ResolveDestination;

        std::set<ResourceState> allowedStates = {
            ResourceState::ResolveSource,
            ResourceState::ResolveDestination,
            ResourceState::CopySource,
            ResourceState::CopyDestination
        };

        if (is_set(formatSupport, FormatSupport::RenderTarget))
        {
            textureUsage |= TextureUsage::RenderTarget;
            allowedStates.insert(ResourceState::RenderTarget);
        }
        if (is_set(formatSupport, FormatSupport::DepthStencil))
        {
            textureUsage |= (TextureUsage::DepthStencil);
            allowedStates.insert(ResourceState::DepthRead);
            allowedStates.insert(ResourceState::DepthWrite);
        }
        if (is_set(formatSupport, FormatSupport::ShaderLoad) || is_set(formatSupport, FormatSupport::ShaderSample))
        {
            textureUsage |= TextureUsage::ShaderResource;
            allowedStates.insert(ResourceState::ShaderResource);
        }
        if (is_set(formatSupport, FormatSupport::ShaderUavLoad) || is_set(formatSupport, FormatSupport::ShaderUavStore))
        {
            textureUsage |= TextureUsage::UnorderedAccess;
            allowedStates.insert(ResourceState::UnorderedAccess);
        }

        TextureDesc texDesc = {};
        texDesc.type = TextureType::Texture2D;
        texDesc.format = format;
        texDesc.size = Extent3D{4, 4, 1};
        texDesc.mipCount = 1;
        texDesc.usage = textureUsage;
        texDesc.memoryType = MemoryType::DeviceLocal;
        ComPtr<ITexture> texture;
        REQUIRE_CALL(device->createTexture(texDesc, nullptr, texture.writeRef()));

        for (ResourceState state : allowedStates)
        {
            auto commandEncoder = queue->createCommandEncoder();
            commandEncoder->setTextureState(texture, state);
            auto passEncoder = commandEncoder->beginComputePass();
            REQUIRE(passEncoder->bindPipeline(pipeline) != nullptr);
            passEncoder->dispatchCompute(1, 1, 1);
            passEncoder->end();
            REQUIRE_CALL(queue->submit(commandEncoder->finish()));
        }

        REQUIRE_CALL(queue->waitOnHost());
    }
}

GPU_TEST_CASE("texture-resource-states", D3D12 | Vulkan)
{
    runTextureResourceStates(device);
}
