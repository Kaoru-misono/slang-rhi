#include "vk-surface.h"
#include "vk-device.h"
#include "vk-command.h"
#include "vk-texture.h"
#include "vk-utils.h"
#include "cocoa-util.h"

#include "core/static_vector.h"
#include "core/reverse-map.h"

#include <algorithm>
#include <exception>
#include <vector>

namespace rhi::vk {

static Format translateVkFormat(VkFormat format)
{
    return reverseMapLookup<Format, VkFormat, Format::Undefined, Format::_Count>(getVkFormat, format);
}

static VkImageUsageFlags getSwapchainImageUsage(TextureUsage usage)
{
    // Present is a surface semantic, not a VkImageUsageFlagBits value. The generic Vulkan
    // texture-usage translation maps it to TRANSFER_SRC, which is not appropriate here unless
    // CopySource was explicitly requested.
    VkImageUsageFlags result = _calcImageUsageFlags(usage & ~TextureUsage::Present);
    if (result == 0 && is_set(usage, TextureUsage::Present))
        result = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    return result;
}

static Result querySwapchainImageUsageSupport(
    DeviceImpl* device,
    Format format,
    VkImageUsageFlags imageUsage,
    bool* outSupported
)
{
    *outSupported = false;
    if (imageUsage == 0)
        return SLANG_OK;

    VkPhysicalDeviceImageFormatInfo2 imageInfo = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    imageInfo.format = getVkFormat(format);
    imageInfo.type = VK_IMAGE_TYPE_2D;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = imageUsage;

    VkImageFormatProperties2 imageProperties = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    VkResult result = device->m_api.vkGetPhysicalDeviceImageFormatProperties2(
        device->m_api.m_physicalDevice,
        &imageInfo,
        &imageProperties
    );
    if (result == VK_ERROR_FORMAT_NOT_SUPPORTED)
        return SLANG_OK;
    SLANG_VK_RETURN_ON_FAIL_REPORT(result, device);

    *outSupported = true;
    return SLANG_OK;
}

SurfaceImpl::~SurfaceImpl()
{
    auto& api = m_device->m_api;

    if (SLANG_FAILED(destroySwapchain()))
    {
        m_device->printError("Vulkan surface destruction could not establish completion.");
        std::terminate();
    }

    if (m_surface)
    {
        api.vkDestroySurfaceKHR(api.m_instance, m_surface, nullptr);
    }

#if SLANG_APPLE_FAMILY
    CocoaUtil::destroyMetalLayer(m_metalLayer);
#endif
}

Result SurfaceImpl::init(DeviceImpl* device, WindowHandle windowHandle)
{
    m_device = device;
    m_windowHandle = windowHandle;

    auto& api = m_device->m_api;

    switch (windowHandle.type)
    {
#if SLANG_WINDOWS_FAMILY
    case WindowHandleType::HWND:
    {
        VkWin32SurfaceCreateInfoKHR surfaceCreateInfo = {};
        surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        surfaceCreateInfo.hinstance = ::GetModuleHandle(nullptr);
        surfaceCreateInfo.hwnd = (HWND)windowHandle.handleValues[0];
        SLANG_VK_RETURN_ON_FAIL_REPORT(
            api.vkCreateWin32SurfaceKHR(api.m_instance, &surfaceCreateInfo, nullptr, &m_surface),
            m_device
        );
        break;
    }
#elif SLANG_APPLE_FAMILY
    case WindowHandleType::NSWindow:
    {
        m_metalLayer = CocoaUtil::createMetalLayer((void*)windowHandle.handleValues[0]);
        VkMetalSurfaceCreateInfoEXT surfaceCreateInfo = {};
        surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        surfaceCreateInfo.pLayer = (CAMetalLayer*)m_metalLayer;
        SLANG_VK_RETURN_ON_FAIL_REPORT(
            api.vkCreateMetalSurfaceEXT(api.m_instance, &surfaceCreateInfo, nullptr, &m_surface),
            m_device
        );
        break;
    }
#elif SLANG_LINUX_FAMILY

#if SLANG_ANDROID
    case WindowHandleType::AndroidWindow:
    {
        VkAndroidSurfaceCreateInfoKHR surfaceCreateInfo = {};
        surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
        surfaceCreateInfo.window = (ANativeWindow*)windowHandle.handleValues[0];
        SLANG_VK_RETURN_ON_FAIL_REPORT(
            api.vkCreateAndroidSurfaceKHR(api.m_instance, &surfaceCreateInfo, nullptr, &m_surface),
            m_device
        );
        break;
    }
#else
    case WindowHandleType::XlibWindow:
    {
        VkXlibSurfaceCreateInfoKHR surfaceCreateInfo = {};
        surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        surfaceCreateInfo.dpy = (Display*)windowHandle.handleValues[0];
        surfaceCreateInfo.window = (Window)windowHandle.handleValues[1];
        SLANG_VK_RETURN_ON_FAIL_REPORT(
            api.vkCreateXlibSurfaceKHR(api.m_instance, &surfaceCreateInfo, nullptr, &m_surface),
            m_device
        );
        break;
    }
#endif

#endif
    default:
        return SLANG_E_INVALID_HANDLE;
    }

    VkBool32 supported = false;
    api.vkGetPhysicalDeviceSurfaceSupportKHR(api.m_physicalDevice, m_device->m_queueFamilyIndex, m_surface, &supported);
    if (!supported)
    {
        return SLANG_FAIL;
    }

    uint32_t formatCount = 0;
    api.vkGetPhysicalDeviceSurfaceFormatsKHR(api.m_physicalDevice, m_surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    api.vkGetPhysicalDeviceSurfaceFormatsKHR(api.m_physicalDevice, m_surface, &formatCount, surfaceFormats.data());

    Format preferredFormat = Format::Undefined;
    for (uint32_t i = 0; i < formatCount; ++i)
    {
        Format format = translateVkFormat(surfaceFormats[i].format);
        if (format != Format::Undefined)
            m_supportedFormats.push_back(format);
        if (preferredFormat == Format::Undefined &&
            (format == Format::RGBA8UnormSrgb || format == Format::BGRA8UnormSrgb || format == Format::BGRX8UnormSrgb))
            preferredFormat = format;
    }
    if (preferredFormat == Format::Undefined && !m_supportedFormats.empty())
    {
        preferredFormat = m_supportedFormats.front();
    }

    m_info.preferredFormat = preferredFormat;
    // Derive the supported usage from the surface capabilities.
    VkSurfaceCapabilitiesKHR surfaceCaps = {};
    SLANG_VK_RETURN_ON_FAIL_REPORT(
        api.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(api.m_physicalDevice, m_surface, &surfaceCaps),
        m_device
    );
    m_info.supportedUsage = TextureUsage::Present;
    if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        m_info.supportedUsage |= TextureUsage::RenderTarget;
    if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT)
        m_info.supportedUsage |= TextureUsage::UnorderedAccess;
    if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT)
        m_info.supportedUsage |= TextureUsage::ShaderResource;
    if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        m_info.supportedUsage |= TextureUsage::CopySource;
    if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        m_info.supportedUsage |= TextureUsage::CopyDestination;
    m_info.formats = m_supportedFormats.data();
    m_info.formatCount = (uint32_t)m_supportedFormats.size();

    return SLANG_OK;
}

Result SurfaceImpl::createSwapchain()
{
    auto& api = m_device->m_api;

    // It is necessary to query the caps -> otherwise the LunarG verification layer will
    // issue an error. The reported supportedUsageFlags are also used below to validate the
    // requested image usage.
    VkSurfaceCapabilitiesKHR surfaceCaps = {};
    SLANG_VK_RETURN_ON_FAIL_REPORT(
        api.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(api.m_physicalDevice, m_surface, &surfaceCaps),
        m_device
    );

    VkExtent2D imageExtent = {};
    if (surfaceCaps.currentExtent.width != UINT32_MAX)
    {
        imageExtent = surfaceCaps.currentExtent;
    }
    else
    {
        imageExtent.width =
            std::clamp(m_config.width, surfaceCaps.minImageExtent.width, surfaceCaps.maxImageExtent.width);
        imageExtent.height =
            std::clamp(m_config.height, surfaceCaps.minImageExtent.height, surfaceCaps.maxImageExtent.height);
    }

    if (imageExtent.width == 0 || imageExtent.height == 0)
    {
        return SLANG_FAIL;
    }

    uint32_t imageCount = std::max(m_config.desiredImageCount, surfaceCaps.minImageCount);
    if (surfaceCaps.maxImageCount > 0)
    {
        imageCount = std::min(imageCount, surfaceCaps.maxImageCount);
    }

    VkSurfaceTransformFlagBitsKHR preTransform =
        (surfaceCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
            ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
            : surfaceCaps.currentTransform;

    VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(surfaceCaps.supportedCompositeAlpha & compositeAlpha))
    {
        static const VkCompositeAlphaFlagBitsKHR kCompositeAlphaModes[] = {
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
        };
        for (VkCompositeAlphaFlagBitsKHR mode : kCompositeAlphaModes)
        {
            if (surfaceCaps.supportedCompositeAlpha & mode)
            {
                compositeAlpha = mode;
                break;
            }
        }
    }

    m_config.width = imageExtent.width;
    m_config.height = imageExtent.height;

    // Query available present modes.
    uint32_t presentModeCount = 0;
    api.vkGetPhysicalDeviceSurfacePresentModesKHR(api.m_physicalDevice, m_surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    api.vkGetPhysicalDeviceSurfacePresentModesKHR(
        api.m_physicalDevice,
        m_surface,
        &presentModeCount,
        presentModes.data()
    );

    // Choose present mode.
    static const VkPresentModeKHR kVsyncOffModes[] = {
        VK_PRESENT_MODE_IMMEDIATE_KHR,
        VK_PRESENT_MODE_MAILBOX_KHR,
        VK_PRESENT_MODE_FIFO_KHR,
        VK_PRESENT_MODE_MAX_ENUM_KHR
    };
    static const VkPresentModeKHR kVsyncOnModes[] = {
        VK_PRESENT_MODE_FIFO_KHR,
        VK_PRESENT_MODE_FIFO_RELAXED_KHR,
        VK_PRESENT_MODE_IMMEDIATE_KHR,
        VK_PRESENT_MODE_MAILBOX_KHR,
        VK_PRESENT_MODE_MAX_ENUM_KHR
    };
    const VkPresentModeKHR* checkPresentModes = m_config.vsync ? kVsyncOnModes : kVsyncOffModes;
    VkPresentModeKHR selectedPresentMode = VK_PRESENT_MODE_MAX_ENUM_KHR;
    for (int i = 0; checkPresentModes[i] != VK_PRESENT_MODE_MAX_ENUM_KHR; ++i)
    {
        if (std::find(presentModes.begin(), presentModes.end(), checkPresentModes[i]) != presentModes.end())
        {
            selectedPresentMode = checkPresentModes[i];
            break;
        }
    }
    if (selectedPresentMode == VK_PRESENT_MODE_MAX_ENUM_KHR)
    {
        return SLANG_FAIL;
    }

    VkFormat format = getVkFormat(m_config.format);
    VkSwapchainKHR oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainCreateInfoKHR swapchainDesc = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swapchainDesc.surface = m_surface;
    swapchainDesc.minImageCount = imageCount;
    swapchainDesc.imageFormat = format;
    swapchainDesc.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchainDesc.imageExtent = imageExtent;
    swapchainDesc.imageArrayLayers = 1;
    swapchainDesc.imageUsage = getSwapchainImageUsage(m_config.usage);
    if (swapchainDesc.imageUsage != (swapchainDesc.imageUsage & surfaceCaps.supportedUsageFlags))
    {
        m_device->printError("Surface does not support the requested usage.");
        return SLANG_E_INVALID_ARG;
    }
    swapchainDesc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainDesc.preTransform = preTransform;
    swapchainDesc.compositeAlpha = compositeAlpha;
    swapchainDesc.presentMode = selectedPresentMode;
    swapchainDesc.clipped = VK_TRUE;
    swapchainDesc.oldSwapchain = oldSwapchain;

    SLANG_VK_RETURN_ON_FAIL_REPORT(
        api.vkCreateSwapchainKHR(api.m_device, &swapchainDesc, nullptr, &m_swapchain),
        m_device
    );

    uint32_t swapchainImageCount = 0;
    api.vkGetSwapchainImagesKHR(api.m_device, m_swapchain, &swapchainImageCount, nullptr);
    std::vector<VkImage> swapchainImages(swapchainImageCount);
    api.vkGetSwapchainImagesKHR(api.m_device, m_swapchain, &swapchainImageCount, swapchainImages.data());

    for (uint32_t i = 0; i < swapchainImageCount; i++)
    {
        TextureDesc textureDesc = {};
        textureDesc.type = TextureType::Texture2D;
        textureDesc.size.width = m_config.width;
        textureDesc.size.height = m_config.height;
        textureDesc.size.depth = 1;
        textureDesc.arrayLength = 1;
        textureDesc.mipCount = 1;
        textureDesc.format = m_config.format;
        textureDesc.usage = m_config.usage;
        textureDesc.defaultState = ResourceState::Present;
        RefPtr<TextureImpl> texture = new TextureImpl(m_device, textureDesc);
        texture->m_image = swapchainImages[i];
        texture->m_imageMemory = 0;
        texture->m_vkformat = format;
        texture->m_shouldDestroyImage = false;
        texture->m_isSwapchainTexture = true;
        m_textures.push_back(texture);
    }

    m_imageLayoutInitialized.resize(swapchainImageCount, false);
    m_presentSemaphores.resize(swapchainImageCount, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo semaphoreCreateInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (VkSemaphore& semaphore : m_presentSemaphores)
    {
        SLANG_VK_RETURN_ON_FAIL_REPORT(
            api.vkCreateSemaphore(api.m_device, &semaphoreCreateInfo, nullptr, &semaphore),
            m_device
        );
    }

    VkFenceCreateInfo fenceCreateInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    SLANG_VK_RETURN_ON_FAIL_REPORT(
        api.vkCreateFence(api.m_device, &fenceCreateInfo, nullptr, &m_acquireFence),
        m_device
    );

    return SLANG_OK;
}

Result SurfaceImpl::destroySwapchain()
{
    m_configured = false;
    // Every native object below is created after the swapchain.
    if (m_swapchain == VK_NULL_HANDLE)
        return SLANG_OK;

    auto& api = m_device->m_api;
    VkResult waitResult;
    {
        std::lock_guard<std::mutex> lock(*m_device->m_queue->m_nativeQueueMutex);
        waitResult = api.vkQueueWaitIdle(m_device->m_queue->m_queue);
    }
    if (waitResult != VK_SUCCESS)
    {
        reportVulkanError(waitResult, "vkQueueWaitIdle", SLANG_RHI_SOURCE_LOCATION(), m_device);
        // Match the existing lost-device idle proof: loss permits destruction, not successful work.
        if (waitResult != VK_ERROR_DEVICE_LOST)
            return SLANG_FAIL;
    }
    if (m_acquireFencePending)
    {
        waitResult = api.vkWaitForFences(api.m_device, 1, &m_acquireFence, VK_TRUE, UINT64_MAX);
        if (waitResult != VK_SUCCESS)
        {
            reportVulkanError(waitResult, "vkWaitForFences", SLANG_RHI_SOURCE_LOCATION(), m_device);
            if (waitResult != VK_ERROR_DEVICE_LOST)
                return SLANG_FAIL;
        }
        m_acquireFencePending = false;
    }
    // For vkQueueWaitIdle, a present counts as complete once the presentation engine holds its
    // wait semaphore payloads, so the idle wait above also retires every present semaphore.
    m_textures.clear();
    for (VkSemaphore semaphore : m_presentSemaphores)
    {
        if (semaphore != VK_NULL_HANDLE)
            api.vkDestroySemaphore(api.m_device, semaphore, nullptr);
    }
    m_presentSemaphores.clear();
    m_imageLayoutInitialized.clear();
    if (m_acquireFence != VK_NULL_HANDLE)
    {
        api.vkDestroyFence(api.m_device, m_acquireFence, nullptr);
        m_acquireFence = VK_NULL_HANDLE;
    }
    m_presentSemaphoreSignaled = false;
    m_currentTextureIndex = kNoImage;
    api.vkDestroySwapchainKHR(api.m_device, m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
    return SLANG_OK;
}

Result SurfaceImpl::initializeImageLayout(TextureImpl* texture)
{
    auto& api = m_device->m_api;
    CommandQueueImpl* queue = m_device->m_queue.get();
    RefPtr<CommandBufferImpl> commandBuffer;
    SLANG_RETURN_ON_FAIL(queue->getOrCreateCommandBuffer(commandBuffer.writeRef()));

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    SLANG_VK_RETURN_ON_FAIL_REPORT(api.vkBeginCommandBuffer(commandBuffer->m_commandBuffer, &beginInfo), m_device);

    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = texture->m_image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // Recordings leave the present state from ALL_COMMANDS, which chains after this transition.
    api.vkCmdPipelineBarrier(
        commandBuffer->m_commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier
    );
    SLANG_VK_RETURN_ON_FAIL_REPORT(api.vkEndCommandBuffer(commandBuffer->m_commandBuffer), m_device);

    ICommandBuffer* commandBuffers[] = {commandBuffer.get()};
    SubmitDesc submitDesc = {};
    submitDesc.commandBuffers = commandBuffers;
    submitDesc.commandBufferCount = 1;
    return queue->submit(submitDesc);
}

Result SurfaceImpl::configure(const SurfaceConfig& config)
{
    SLANG_RETURN_ON_FAIL(validateConfig(config));
    SurfaceConfig resolvedConfig = config;
    if (resolvedConfig.format == Format::Undefined)
    {
        resolvedConfig.format = m_info.preferredFormat;
    }
    if (resolvedConfig.usage == TextureUsage::None)
    {
        // Do not auto-add UnorderedAccess here: a format's optimal-tiling features may report
        // storage support while the swapchain still rejects VK_IMAGE_USAGE_STORAGE_BIT for that
        // format (e.g. *_SRGB), tripping VUID-VkSwapchainCreateInfoKHR-imageFormat-01778. Apps
        // that need storage on the swapchain must request it explicitly; configure() then
        // validates it against the format below.
        resolvedConfig.usage = (TextureUsage::Present | TextureUsage::RenderTarget | TextureUsage::CopyDestination) &
                               m_info.supportedUsage;

        // CopyDestination is useful for the compute fallback, but it is not required for a
        // presentable surface. Drop it when the selected format does not support the complete
        // optimal-tiling usage combination.
        bool usageSupported = false;
        SLANG_RETURN_ON_FAIL(querySwapchainImageUsageSupport(
            m_device,
            resolvedConfig.format,
            getSwapchainImageUsage(resolvedConfig.usage),
            &usageSupported
        ));
        if (!usageSupported)
        {
            resolvedConfig.usage &= ~TextureUsage::CopyDestination;
        }
    }
    else
    {
        if (resolvedConfig.usage != (resolvedConfig.usage & m_info.supportedUsage))
        {
            m_device->printError("Surface does not support the requested usage.");
            return SLANG_E_INVALID_ARG;
        }
    }

    bool usageSupported = false;
    SLANG_RETURN_ON_FAIL(querySwapchainImageUsageSupport(
        m_device,
        resolvedConfig.format,
        getSwapchainImageUsage(resolvedConfig.usage),
        &usageSupported
    ));
    if (!usageSupported)
    {
        m_device->printError("Surface format does not support the requested usage.");
        return SLANG_E_INVALID_ARG;
    }

    SLANG_RETURN_ON_FAIL(destroySwapchain());
    if (m_device->m_deviceLost)
        return SLANG_FAIL;
    setConfig(resolvedConfig);
    SLANG_RETURN_ON_FAIL(createSwapchain());
    m_configured = true;

    return SLANG_OK;
}

Result SurfaceImpl::unconfigure()
{
    return destroySwapchain();
}

Result SurfaceImpl::acquireNextImage(ITexture** outTexture)
{
    *outTexture = nullptr;

    if (!m_configured || m_device->m_deviceLost)
    {
        return SLANG_FAIL;
    }
    if (m_currentTextureIndex != kNoImage)
        return SLANG_E_INVALID_ARG;

    auto& api = m_device->m_api;

    uint32_t imageIndex = 0;
    VkResult result =
        api.vkAcquireNextImageKHR(api.m_device, m_swapchain, UINT64_MAX, VK_NULL_HANDLE, m_acquireFence, &imageIndex);
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        reportVulkanError(result, "vkAcquireNextImageKHR", SLANG_RHI_SOURCE_LOCATION(), m_device);
        return SLANG_FAIL;
    }
    m_currentTextureIndex = imageIndex;
    m_acquireFencePending = true;

    // Waiting on the host until the presentation engine releases the image lets any later submission
    // write it without an acquire semaphore. From here on, a failure leaves the image held and the
    // surface unconfigured; teardown releases it.
    result = api.vkWaitForFences(api.m_device, 1, &m_acquireFence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS)
    {
        reportVulkanError(result, "vkWaitForFences", SLANG_RHI_SOURCE_LOCATION(), m_device);
        m_configured = false;
        return SLANG_FAIL;
    }
    m_acquireFencePending = false;
    result = api.vkResetFences(api.m_device, 1, &m_acquireFence);
    if (result != VK_SUCCESS)
    {
        reportVulkanError(result, "vkResetFences", SLANG_RHI_SOURCE_LOCATION(), m_device);
        m_configured = false;
        return SLANG_FAIL;
    }

    TextureImpl* texture = m_textures[imageIndex];
    if (!m_imageLayoutInitialized[imageIndex])
    {
        if (SLANG_FAILED(initializeImageLayout(texture)))
        {
            m_configured = false;
            return SLANG_FAIL;
        }
        m_imageLayoutInitialized[imageIndex] = true;
    }

    returnComPtr(outTexture, texture);
    return SLANG_OK;
}

Result SurfaceImpl::present()
{
    auto& api = m_device->m_api;

    if (!m_configured || m_device->m_deviceLost)
    {
        return SLANG_FAIL;
    }
    if (m_currentTextureIndex == kNoImage)
    {
        return SLANG_FAIL;
    }

    VkSemaphore presentSemaphore = m_presentSemaphores[m_currentTextureIndex];

    // The surface issues the last submission before presenting. Its signal covers every earlier
    // submission on the queue, however many of them wrote the image. A present that enqueued nothing
    // leaves the semaphore signaled, so a retry must not signal it again.
    if (!m_presentSemaphoreSignaled)
    {
        SLANG_RETURN_ON_FAIL(m_device->m_queue->submitSignalingBinarySemaphore(SubmitDesc{}, presentSemaphore));
        m_presentSemaphoreSignaled = true;
    }

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchain;
    presentInfo.pImageIndices = &m_currentTextureIndex;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &presentSemaphore;
    VkResult result;
    {
        std::lock_guard<std::mutex> lock(*m_device->m_queue->m_nativeQueueMutex);
        result = api.vkQueuePresentKHR(m_device->m_queue->m_queue, &presentInfo);
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        reportVulkanError(result, "vkQueuePresentKHR", SLANG_RHI_SOURCE_LOCATION(), m_device);
        if (result != VK_ERROR_OUT_OF_HOST_MEMORY && result != VK_ERROR_OUT_OF_DEVICE_MEMORY)
            m_configured = false;
        return SLANG_FAIL;
    }
    m_presentSemaphoreSignaled = false;
    m_currentTextureIndex = kNoImage;
    return SLANG_OK;
}

Result DeviceImpl::createSurface(WindowHandle windowHandle, ISurface** outSurface)
{
    RefPtr<SurfaceImpl> surface = new SurfaceImpl();
    SLANG_RETURN_ON_FAIL(surface->init(this, windowHandle));
    returnComPtr(outSurface, surface);
    return SLANG_OK;
}

} // namespace rhi::vk
