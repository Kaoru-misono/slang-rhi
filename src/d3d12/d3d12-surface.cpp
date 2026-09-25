#include "d3d12-surface.h"
#include "d3d12-device.h"
#include "d3d12-texture.h"

namespace rhi::d3d12 {

Result SurfaceImpl::init(DeviceImpl* device, WindowHandle windowHandle)
{
    m_device = device;
    m_queue = m_device->m_queue->m_d3dQueue;
    m_dxgiFactory = device->m_dxgiFactory;
    SLANG_RETURN_ON_FAIL(D3DSurface::init(windowHandle, DXGI_SWAP_EFFECT_FLIP_DISCARD, false));
    SLANG_D3D_RETURN_ON_FAIL_REPORT(
        m_device->m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_fence.writeRef())),
        m_device
    );
    return SLANG_OK;
}

void SurfaceImpl::createSwapchainTextures(uint32_t count)
{
    while (m_frameEvents.size() < count)
    {
        m_frameEvents.push_back(
            CreateEventEx(nullptr, FALSE, CREATE_EVENT_INITIAL_SET | CREATE_EVENT_MANUAL_RESET, EVENT_ALL_ACCESS)
        );
    }

    for (uint32_t i = 0; i < count; i++)
    {
        ComPtr<ID3D12Resource> d3dResource;
        m_swapChain->GetBuffer(i, IID_PPV_ARGS(d3dResource.writeRef()));
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
        texture->m_resource.setResource(d3dResource.get());
        texture->m_format = getFormatMapping(textureDesc.format).rtvFormat;
        texture->m_isTypeless = false;
        texture->m_defaultState = D3D12_RESOURCE_STATE_PRESENT;
        texture->m_isSwapchainTexture = true;
        m_textures.push_back(texture);

        SetEvent(m_frameEvents[i]);
    }

    m_swapChain->QueryInterface(m_swapChain3.writeRef());
}

SurfaceImpl::~SurfaceImpl()
{
    // A present may still have an event registered for this value; the handle must outlive it.
    if (m_fence && m_fence->GetCompletedValue() < fenceValue)
        m_fence->SetEventOnCompletion(fenceValue, nullptr);
    for (HANDLE event : m_frameEvents)
        CloseHandle(event);
}

Result SurfaceImpl::configure(const SurfaceConfig& config)
{
    m_swapChain3.setNull();
    m_imageAcquired = false;
    for (auto event : m_frameEvents)
        SetEvent(event);
    return D3DSurface::configure(config);
}

Result SurfaceImpl::unconfigure()
{
    m_imageAcquired = false;
    return D3DSurface::unconfigure();
}

Result SurfaceImpl::acquireNextImage(ITexture** outTexture)
{
    *outTexture = nullptr;
    if (!m_configured)
    {
        return SLANG_FAIL;
    }
    // Only present or reconfiguration re-arms the frame event, so a second wait would never return.
    if (m_imageAcquired)
    {
        return SLANG_E_INVALID_ARG;
    }
    auto result = (int)m_swapChain3->GetCurrentBackBufferIndex();
    WaitForSingleObject(m_frameEvents[result], INFINITE);
    ResetEvent(m_frameEvents[result]);
    m_imageAcquired = true;
    returnComPtrCopy(outTexture, m_textures[result]);
    return SLANG_OK;
}

Result SurfaceImpl::present()
{
    if (!m_configured)
    {
        return SLANG_FAIL;
    }
    SLANG_D3D_RETURN_ON_FAIL_REPORT(
        m_fence->SetEventOnCompletion(fenceValue, m_frameEvents[m_swapChain3->GetCurrentBackBufferIndex()]),
        m_device
    );
    SLANG_RETURN_ON_FAIL(D3DSurface::present());
    m_imageAcquired = false;
    fenceValue++;
    m_queue->Signal(m_fence, fenceValue);
    return SLANG_OK;
}

} // namespace rhi::d3d12
