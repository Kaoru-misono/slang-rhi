#pragma once

#include "vk-base.h"

#include "core/short_vector.h"

#include <vector>

namespace rhi::vk {

class SurfaceImpl : public Surface
{
public:
    RefPtr<DeviceImpl> m_device;
    WindowHandle m_windowHandle;
    std::vector<Format> m_supportedFormats;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    short_vector<RefPtr<TextureImpl>> m_textures;
    // Indexed by swapchain image: reacquiring an image proves its previous present wait has executed.
    short_vector<VkSemaphore> m_presentSemaphores;
    // Swapchain images start undefined; each moves to the present layout on its first acquisition.
    short_vector<bool> m_imageLayoutInitialized;
    // Signaled once the presentation engine releases the acquired image; acquisition waits for it on the host.
    VkFence m_acquireFence = VK_NULL_HANDLE;
    bool m_acquireFencePending = false;
    bool m_presentSemaphoreSignaled = false;

    static constexpr uint32_t kNoImage = uint32_t(-1);
    uint32_t m_currentTextureIndex = kNoImage;
#if SLANG_APPLE_FAMILY
    void* m_metalLayer;
#endif

public:
    ~SurfaceImpl();

    Result init(DeviceImpl* device, WindowHandle windowHandle);
    Result createSwapchain();
    Result destroySwapchain();
    Result initializeImageLayout(TextureImpl* texture);

    virtual SLANG_NO_THROW Result SLANG_MCALL configure(const SurfaceConfig& config) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL unconfigure() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL acquireNextImage(ITexture** outTexture) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL present() override;
};

} // namespace rhi::vk
