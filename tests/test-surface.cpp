#if SLANG_RHI_BUILD_TESTS_WITH_GLFW
#include "testing.h"
#include <cstring>

#if SLANG_WINDOWS_FAMILY
#define GLFW_EXPOSE_NATIVE_WIN32
#elif SLANG_LINUX_FAMILY
#define GLFW_EXPOSE_NATIVE_X11
#elif SLANG_APPLE_FAMILY
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <slang-rhi/glfw.h>

// X11 defines None as a macro after slang-rhi.h has already removed it.
#ifdef None
#undef None
#endif

using namespace rhi;
using namespace rhi::testing;

struct Vertex
{
    float position[3];
    float color[3];
};

static const int kVertexCount = 3;
static const Vertex kVertexData[kVertexCount] = {
    // Triangle 1
    {{-0.5, -0.5, 0}, {1, 0, 0}},
    {{+0.5, -0.5, 0}, {0, 1, 0}},
    {{+0.0, +0.5, 0}, {0, 0, 1}},
};

static bool hasMonitor()
{
    int count;
    glfwGetMonitors(&count);
    return count > 0;
}

struct SurfaceTest
{
    ComPtr<IDevice> device;
    ComPtr<ICommandQueue> queue;

    GLFWwindow* window;
    ComPtr<ISurface> surface;
    uint32_t configurationCount = 0;

    virtual Format getSurfaceFormat() { return surface->getInfo().preferredFormat; }
    virtual TextureUsage getSurfaceUsage() { return TextureUsage::None; }
    virtual void initResources() = 0;
    virtual void renderFrame(ITexture* texture, uint32_t width, uint32_t height, uint32_t frameIndex) = 0;

    void init(IDevice* device_)
    {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        this->window = glfwCreateWindow(512, 512, "test-surface", nullptr, nullptr);

        this->device = device_;
        this->queue = device->getQueue(QueueType::Graphics);
        REQUIRE(this->queue);
        this->surface = device->createSurface(getWindowHandleFromGLFW(window));
        REQUIRE(this->surface);
        CHECK(is_set(this->surface->getInfo().supportedUsage, TextureUsage::Present));

        initResources();
    }

    void shutdown()
    {
        this->surface = nullptr;

        glfwDestroyWindow(window);
    }

    void configureSurface(uint32_t width, uint32_t height)
    {
        queue->waitOnHost();

        SurfaceConfig config = {};
        config.format = getSurfaceFormat();
        // Exercise default usage resolution on the first configuration and explicit usage
        // validation on subsequent configurations.
        config.usage = configurationCount++ == 0 ? TextureUsage::None : getSurfaceUsage();
        config.width = width;
        config.height = height;
        config.vsync = false;
        REQUIRE_CALL(surface->configure(config));

        CHECK(surface->getConfig());
        CHECK(surface->getConfig()->width == width);
        CHECK(surface->getConfig()->height == height);
        CHECK(surface->getConfig()->usage != TextureUsage::None);
        CHECK(surface->getConfig()->usage == (surface->getConfig()->usage & surface->getInfo().supportedUsage));
        if (config.usage != TextureUsage::None)
            CHECK(surface->getConfig()->usage == config.usage);
    }

    void run()
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        configureSurface(width, height);

        // Number of frames per test stage.
        const uint32_t kFrameCount = 10;

        // Render initial frames.
        for (uint32_t i = 0; i < kFrameCount; ++i)
        {
            glfwPollEvents();
            ComPtr<ITexture> texture = surface->acquireNextImage();
            CHECK(texture->getDesc().size.width == width);
            CHECK(texture->getDesc().size.height == height);
            CHECK(texture->getDesc().usage == surface->getConfig()->usage);
            renderFrame(texture, width, height, i);
            surface->present();
        }

        // Resize window.
        glfwSetWindowSize(window, 700, 700);
        glfwGetFramebufferSize(window, &width, &height);
        configureSurface(width, height);

        // Render resized frames.
        for (uint32_t i = 0; i < kFrameCount; ++i)
        {
            glfwPollEvents();
            ComPtr<ITexture> texture = surface->acquireNextImage();
            CHECK(texture->getDesc().size.width == width);
            CHECK(texture->getDesc().size.height == height);
            CHECK(texture->getDesc().usage == surface->getConfig()->usage);
            renderFrame(texture, width, height, i);
            surface->present();
        }

        // Minimize window & unconfigure surface.
        glfwIconifyWindow(window);
        queue->waitOnHost();
        REQUIRE_CALL(surface->unconfigure());
        for (uint32_t i = 0; i < kFrameCount; ++i)
        {
            glfwPollEvents();
        }

        // Restore window.
        glfwRestoreWindow(window);
        glfwGetFramebufferSize(window, &width, &height);
        configureSurface(width, height);

        // Render frames after restoring.
        for (uint32_t i = 0; i < kFrameCount; ++i)
        {
            glfwPollEvents();
            ComPtr<ITexture> texture = surface->acquireNextImage();
            CHECK(texture->getDesc().size.width == width);
            CHECK(texture->getDesc().size.height == height);
            CHECK(texture->getDesc().usage == surface->getConfig()->usage);
            renderFrame(texture, width, height, i);
            surface->present();
        }

        queue->waitOnHost();
    }
};

struct RenderSurfaceTest : SurfaceTest
{
    ComPtr<IBuffer> vertexBuffer;
    ComPtr<IRenderPipeline> pipeline;

    TextureUsage getSurfaceUsage() override { return TextureUsage::Present | TextureUsage::RenderTarget; }

    void initResources() override
    {
        VertexStreamDesc vertexStreams[] = {
            {sizeof(Vertex), InputSlotClass::PerVertex, 0},
        };

        InputElementDesc inputElements[] = {
            // Vertex buffer data
            {"POSITION", 0, Format::RGB32Float, offsetof(Vertex, position), 0},
            {"COLOR", 0, Format::RGB32Float, offsetof(Vertex, color), 0},
        };
        InputLayoutDesc inputLayoutDesc = {};
        inputLayoutDesc.inputElementCount = SLANG_COUNT_OF(inputElements);
        inputLayoutDesc.inputElements = inputElements;
        inputLayoutDesc.vertexStreamCount = SLANG_COUNT_OF(vertexStreams);
        inputLayoutDesc.vertexStreams = vertexStreams;
        auto inputLayout = device->createInputLayout(inputLayoutDesc);
        REQUIRE(inputLayout);

        BufferDesc vertexBufferDesc;
        vertexBufferDesc.size = kVertexCount * sizeof(Vertex);
        vertexBufferDesc.usage = BufferUsage::VertexBuffer;
        vertexBufferDesc.defaultState = ResourceState::VertexBuffer;
        vertexBuffer = device->createBuffer(vertexBufferDesc, &kVertexData[0]);
        REQUIRE(vertexBuffer);

        ComPtr<IShaderProgram> shaderProgram;
        REQUIRE_CALL(
            loadProgram(device, "test-surface-render", {"vertexMain", "fragmentMain"}, shaderProgram.writeRef())
        );

        ColorTargetDesc colorTarget = {};
        colorTarget.format = getSurfaceFormat();

        RenderPipelineDesc pipelineDesc = {};
        pipelineDesc.program = shaderProgram.get();
        pipelineDesc.inputLayout = inputLayout;
        pipelineDesc.depthStencil.depthTestEnable = false;
        pipelineDesc.depthStencil.depthWriteEnable = false;
        pipelineDesc.targets = &colorTarget;
        pipelineDesc.targetCount = 1;
        REQUIRE_CALL(device->createRenderPipeline(pipelineDesc, pipeline.writeRef()));
    }

    void renderFrame(ITexture* texture, uint32_t width, uint32_t height, uint32_t frameIndex) override
    {
        ComPtr<ITextureView> textureView = device->createTextureView(texture, {});

        auto commandEncoder = queue->createCommandEncoder();

        RenderPassColorAttachment colorAttachment = {};
        colorAttachment.view = textureView;
        colorAttachment.loadOp = LoadOp::Clear;

        RenderPassDesc renderPass = {};
        renderPass.colorAttachments = &colorAttachment;
        renderPass.colorAttachmentCount = 1;

        auto passEncoder = commandEncoder->beginRenderPass(renderPass);
        passEncoder->bindPipeline(pipeline);

        RenderState renderState = {};
        renderState.viewports[0] = Viewport::fromSize(width, height);
        renderState.viewportCount = 1;
        renderState.scissorRects[0] = ScissorRect::fromSize(width, height);
        renderState.scissorRectCount = 1;
        renderState.vertexBuffers[0] = vertexBuffer;
        renderState.vertexBufferCount = 1;
        passEncoder->setRenderState(renderState);

        DrawArguments drawArgs = {};
        drawArgs.vertexCount = kVertexCount;
        passEncoder->draw(drawArgs);

        passEncoder->end();
        queue->submit(commandEncoder->finish());
    }
};

struct ComputeSurfaceTest : SurfaceTest
{
    ComPtr<ITexture> renderTexture;
    ComPtr<IComputePipeline> pipeline;

    Format getSurfaceFormat() override
    {
        // Choose an non-sRGB format to allow compute shader to write to the surface.
        const SurfaceInfo& info = surface->getInfo();
        for (uint32_t i = 0; i < info.formatCount; ++i)
        {
            switch (info.formats[i])
            {
            case Format::RGBA8Unorm:
            case Format::BGRA8Unorm:
            case Format::BGRX8Unorm:
                return info.formats[i];
            default:
                break;
            }
        }
        return info.preferredFormat;
    }

    TextureUsage getSurfaceUsage() override
    {
        // Exercise explicit storage usage on every backend where both the surface and selected
        // format support it.
        if (is_set(surface->getInfo().supportedUsage, TextureUsage::UnorderedAccess))
        {
            FormatSupport formatSupport = {};
            REQUIRE_CALL(device->getFormatSupport(getSurfaceFormat(), &formatSupport));
            if (is_set(formatSupport, FormatSupport::ShaderUavStore))
                return TextureUsage::Present | TextureUsage::UnorderedAccess;
        }
        return TextureUsage::None;
    }

    void initResources() override
    {
        ComPtr<IShaderProgram> shaderProgram;
        REQUIRE_CALL(loadProgram(device, "test-surface-compute", "computeMain", shaderProgram.writeRef()));

        ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.program = shaderProgram.get();
        REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));
    }

    void renderFrame(ITexture* texture, uint32_t width, uint32_t height, uint32_t frameIndex) override
    {
        bool allowUnorderedAccess = is_set(texture->getDesc().usage, TextureUsage::UnorderedAccess);

        if (!allowUnorderedAccess && (!renderTexture || renderTexture->getDesc().size.width != width ||
                                      renderTexture->getDesc().size.height != height))
        {
            TextureDesc textureDesc = {};
            textureDesc.size.width = width;
            textureDesc.size.height = height;
            textureDesc.format = getSurfaceFormat();
            textureDesc.mipCount = 1;
            textureDesc.usage = TextureUsage::UnorderedAccess | TextureUsage::CopySource;
            renderTexture = device->createTexture(textureDesc);
            REQUIRE(renderTexture);
        }

        auto commandEncoder = queue->createCommandEncoder();
        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        ShaderCursor cursor(rootObject->getEntryPoint(0));
        cursor["texture"].setBinding(allowUnorderedAccess ? texture : renderTexture);
        uint32_t dim[2] = {width, height};
        cursor["dim"].setData(dim, sizeof(dim));
        passEncoder->dispatchCompute(width, height, 1);
        passEncoder->end();
        if (!allowUnorderedAccess)
        {
            Offset3D offset = {0, 0, 0};
            commandEncoder->copyTexture(
                texture,
                {0, 0, 0, 0},
                offset,
                renderTexture,
                {0, 0, 0, 0},
                offset,
                Extent3D::kWholeTexture
            );
        }
        queue->submit(commandEncoder->finish());
    }
};

struct NoRenderSurfaceTest : SurfaceTest
{
    void initResources() override {}
    void renderFrame(ITexture* texture, uint32_t width, uint32_t height, uint32_t frameIndex) override {}
};

template<typename Test>
void testSurface(IDevice* device)
{
    glfwInit();
    if (!hasMonitor())
    {
        SKIP("No monitor attached");
    }

    Test t;
    t.init(device);
    t.run();
    t.shutdown();

    glfwTerminate();
}

GPU_TEST_CASE("surface-render", D3D11 | D3D12 | Vulkan | Metal | WGPU)
{
    CHECK(device->hasFeature(Feature::Surface));
    testSurface<RenderSurfaceTest>(device);
}

// skip WGPU: RWTexture binding fails
GPU_TEST_CASE("surface-compute", D3D11 | D3D12 | Vulkan | Metal | CUDA)
{
    if (device->getDeviceType() == DeviceType::CUDA && SLANG_RHI_ENABLE_VULKAN == 0)
    {
        SKIP("CUDA surface requires Vulkan");
    }
    CHECK(device->hasFeature(Feature::Surface));
    testSurface<ComputeSurfaceTest>(device);
}

GPU_TEST_CASE("surface-no-render", D3D11 | D3D12 | Vulkan | Metal | CUDA)
{
    if (device->getDeviceType() == DeviceType::CUDA && SLANG_RHI_ENABLE_VULKAN == 0)
    {
        SKIP("CUDA surface requires Vulkan");
    }
    CHECK(device->hasFeature(Feature::Surface));
    testSurface<NoRenderSurfaceTest>(device);
}

struct ManualFrameSurfaceTest : NoRenderSurfaceTest
{
    SurfaceConfig config = {};

    TextureUsage getSurfaceUsage() override { return TextureUsage::Present | TextureUsage::RenderTarget; }

    void configureToWindow()
    {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        REQUIRE(width > 0);
        REQUIRE(height > 0);

        config.format = getSurfaceFormat();
        config.usage = getSurfaceUsage();
        config.width = static_cast<uint32_t>(width);
        config.height = static_cast<uint32_t>(height);
        config.vsync = false;
        REQUIRE_CALL(surface->configure(config));
    }

    ComPtr<ITexture> acquire()
    {
        glfwPollEvents();
        ComPtr<ITexture> texture;
        REQUIRE_CALL(surface->acquireNextImage(texture.writeRef()));
        REQUIRE(texture);
        return texture;
    }

    void recordAndDiscard(ITexture* texture)
    {
        ComPtr<ICommandEncoder> encoder;
        REQUIRE_CALL(queue->createCommandEncoder(encoder.writeRef()));
        REQUIRE(encoder);
        encoder->setTextureState(texture, ResourceState::RenderTarget);
        ComPtr<ICommandBuffer> commandBuffer;
        REQUIRE_CALL(encoder->finish(commandBuffer.writeRef()));
        REQUIRE(commandBuffer);
    }

    void submitClear(ITexture* texture, float value)
    {
        ComPtr<ITextureView> view;
        REQUIRE_CALL(device->createTextureView(texture, {}, view.writeRef()));

        RenderPassColorAttachment colorAttachment = {};
        colorAttachment.view = view;
        colorAttachment.loadOp = LoadOp::Clear;
        colorAttachment.clearValue[0] = value;
        colorAttachment.clearValue[3] = 1.f;
        RenderPassDesc renderPass = {};
        renderPass.colorAttachments = &colorAttachment;
        renderPass.colorAttachmentCount = 1;

        ComPtr<ICommandEncoder> encoder;
        REQUIRE_CALL(queue->createCommandEncoder(encoder.writeRef()));
        encoder->beginRenderPass(renderPass)->end();
        ComPtr<ICommandBuffer> commandBuffer;
        REQUIRE_CALL(encoder->finish(commandBuffer.writeRef()));
        REQUIRE_CALL(queue->submit(commandBuffer));
    }

    ComPtr<IBuffer> createBuffer(uint32_t value, BufferUsage usage, ResourceState defaultState)
    {
        BufferDesc desc = {};
        desc.size = sizeof(value);
        desc.usage = usage;
        desc.defaultState = defaultState;
        ComPtr<IBuffer> buffer;
        REQUIRE_CALL(device->createBuffer(desc, &value, buffer.writeRef()));
        REQUIRE(buffer);
        return buffer;
    }

    void submitCopy(IBuffer* dst, IBuffer* src)
    {
        ComPtr<ICommandEncoder> encoder;
        REQUIRE_CALL(queue->createCommandEncoder(encoder.writeRef()));
        encoder->copyBuffer(dst, 0, src, 0, sizeof(uint32_t));
        ComPtr<ICommandBuffer> commandBuffer;
        REQUIRE_CALL(encoder->finish(commandBuffer.writeRef()));
        REQUIRE_CALL(queue->submit(commandBuffer));
    }

    void checkBuffer(IBuffer* buffer, uint32_t expected)
    {
        ComPtr<ISlangBlob> blob;
        REQUIRE_CALL(device->readBuffer(buffer, 0, sizeof(expected), blob.writeRef()));
        REQUIRE(blob);
        REQUIRE(blob->getBufferSize() == sizeof(expected));
        CHECK(std::memcmp(blob->getBufferPointer(), &expected, sizeof(expected)) == 0);
    }

    void verifyPublicQueue(uint32_t pattern)
    {
        ComPtr<IBuffer> src = createBuffer(pattern, BufferUsage::CopySource, ResourceState::CopySource);
        ComPtr<IBuffer> dst =
            createBuffer(0, BufferUsage::CopyDestination | BufferUsage::CopySource, ResourceState::CopyDestination);
        submitCopy(dst, src);
        REQUIRE_CALL(queue->waitOnHost());
        checkBuffer(dst, pattern);
    }
};

struct AbandonAcquiredImageSurfaceTest : ManualFrameSurfaceTest
{
    void run()
    {
        configureToWindow();

        // No unconfigure or queue wait before destroying the acquired surface.
        acquire();
        surface = nullptr;
        verifyPublicQueue(0xA1B2C3D4u);

        REQUIRE_CALL(device->createSurface(getWindowHandleFromGLFW(window), surface.writeRef()));
        REQUIRE(surface);
        REQUIRE_CALL(surface->configure(config));

        recordAndDiscard(acquire());
        REQUIRE_CALL(surface->unconfigure());
        verifyPublicQueue(0xB2C3D4E5u);
        REQUIRE_CALL(surface->configure(config));

        // Reconfiguration itself must cancel the acquired image and discarded recording.
        recordAndDiscard(acquire());
        REQUIRE_CALL(surface->configure(config));
        verifyPublicQueue(0xC3D4E5F6u);

        // Present without any submission, reusing every image after recovery.
        for (uint32_t i = 0; i < 6; ++i)
        {
            ComPtr<ITexture> texture = acquire();
            CHECK(texture->getDesc().size.width == config.width);
            CHECK(texture->getDesc().size.height == config.height);
            REQUIRE_CALL(surface->present());
        }
        REQUIRE_CALL(queue->waitOnHost());
    }
};

GPU_TEST_CASE("surface-abandon-acquired-image-recovery", D3D12 | Vulkan)
{
    CHECK(device->hasFeature(Feature::Surface));
    testSurface<AbandonAcquiredImageSurfaceTest>(device);
}

// More frames than any swapchain has images, so every image is reused.
static const uint32_t kManualFrameCount = 12;

struct LaterSubmitSurfaceTest : ManualFrameSurfaceTest
{
    void run()
    {
        configureToWindow();
        ComPtr<IBuffer> src = createBuffer(0xD4E5F6A7u, BufferUsage::CopySource, ResourceState::CopySource);
        ComPtr<IBuffer> dst =
            createBuffer(0, BufferUsage::CopyDestination | BufferUsage::CopySource, ResourceState::CopyDestination);

        for (uint32_t i = 0; i < kManualFrameCount; ++i)
        {
            ComPtr<ITexture> texture = acquire();
            // The back buffer is written only after an unrelated submission; presentation must
            // still wait for that later write.
            submitCopy(dst, src);
            submitClear(texture, float(i) / kManualFrameCount);
            REQUIRE_CALL(surface->present());
        }
        REQUIRE_CALL(queue->waitOnHost());
        checkBuffer(dst, 0xD4E5F6A7u);
    }
};

GPU_TEST_CASE("surface-present-follows-later-submits", D3D12 | Vulkan)
{
    CHECK(device->hasFeature(Feature::Surface));
    testSurface<LaterSubmitSurfaceTest>(device);
}

struct DiscardedRecordingSurfaceTest : ManualFrameSurfaceTest
{
    void run()
    {
        configureToWindow();
        for (uint32_t i = 0; i < kManualFrameCount; ++i)
        {
            ComPtr<ITexture> texture = acquire();
            recordAndDiscard(texture);
            REQUIRE_CALL(surface->present());
        }
        REQUIRE_CALL(queue->waitOnHost());
    }
};

GPU_TEST_CASE("surface-present-after-discarded-recording", D3D12 | Vulkan)
{
    CHECK(device->hasFeature(Feature::Surface));
    testSurface<DiscardedRecordingSurfaceTest>(device);
}

struct SecondAcquireSurfaceTest : ManualFrameSurfaceTest
{
    void run()
    {
        configureToWindow();
        ComPtr<ITexture> texture = acquire();
        {
            // Debug builds reject this in the debug layer; the backends return SLANG_E_INVALID_ARG.
            ExpectedErrorScope expectedErrors;
            ComPtr<ITexture> second;
            CHECK(SLANG_FAILED(surface->acquireNextImage(second.writeRef())));
            CHECK(!second);
        }
        REQUIRE_CALL(surface->present());

        for (uint32_t i = 0; i < kManualFrameCount; ++i)
        {
            acquire();
            REQUIRE_CALL(surface->present());
        }
        REQUIRE_CALL(queue->waitOnHost());
    }
};

GPU_TEST_CASE("surface-second-acquire-fails", D3D12 | Vulkan)
{
    CHECK(device->hasFeature(Feature::Surface));
    testSurface<SecondAcquireSurfaceTest>(device);
}
#endif // SLANG_RHI_BUILD_TESTS_WITH_GLFW
