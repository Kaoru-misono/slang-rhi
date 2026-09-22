#include "d3d12-fence.h"
#include "d3d12-device.h"

namespace rhi::d3d12 {

FenceImpl::FenceImpl(Device* device, const FenceDesc& desc)
    : Fence(device, desc)
{
}

void FenceImpl::deleteThis()
{
    getDevice<DeviceImpl>()->deferDelete(this);
}

FenceImpl::~FenceImpl()
{
    if (m_sharedHandle)
    {
#if SLANG_WINDOWS_FAMILY
        ::CloseHandle((HANDLE)m_sharedHandle.value);
#endif
    }
}

Result FenceImpl::init()
{
    DeviceImpl* device = getDevice<DeviceImpl>();

    SLANG_D3D_RETURN_ON_FAIL_REPORT(
        device->m_device->CreateFence(
            m_desc.initialValue,
            m_desc.isShared ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(m_fence.writeRef())
        ),
        device
    );
    if (m_desc.label)
    {
        m_fence->SetName(string::to_wstring(m_desc.label).c_str());
    }
    return SLANG_OK;
}

Result FenceImpl::getCurrentValue(uint64_t* outValue)
{
    uint64_t value = m_fence->GetCompletedValue();
    if (value == UINT64_MAX || getDevice()->m_deviceLost)
    {
        getDevice()->m_deviceLost = true;
        *outValue = 0;
        return SLANG_FAIL;
    }
    *outValue = value;
    return SLANG_OK;
}

Result FenceImpl::setCurrentValue(uint64_t value)
{
    SLANG_D3D_RETURN_ON_FAIL_REPORT(m_fence->Signal(value), m_device);
    return SLANG_OK;
}

Result FenceImpl::getNativeHandle(NativeHandle* outHandle)
{
    outHandle->type = NativeHandleType::D3D12Fence;
    outHandle->value = (uint64_t)m_fence.get();
    return SLANG_OK;
}

Result FenceImpl::getSharedHandle(NativeHandle* outHandle)
{
#if !SLANG_WINDOWS_FAMILY
    return SLANG_E_NOT_AVAILABLE;
#else
    DeviceImpl* device = getDevice<DeviceImpl>();

    if (!m_sharedHandle)
    {
        HANDLE handle = NULL;
        SLANG_D3D_RETURN_ON_FAIL_REPORT(
            device->m_device->CreateSharedHandle(m_fence, NULL, GENERIC_ALL, nullptr, &handle),
            device
        );
        m_sharedHandle = NativeHandle{NativeHandleType::Win32, (uint64_t)handle};
    }

    *outHandle = m_sharedHandle;
    return SLANG_OK;
#endif
}

} // namespace rhi::d3d12
