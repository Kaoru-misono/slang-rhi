#include "d3d12-query.h"
#include "d3d12-device.h"
#include "d3d12-command.h"
#include "d3d12-buffer.h"
#include "d3d12-utils.h"

namespace rhi::d3d12 {

namespace {

CommandQueueImpl* getQueryQueue(DeviceImpl* device, uint32_t queue)
{
    return QueueType(queue) == QueueType::Compute ? device->m_computeQueue.get() : device->m_queue.get();
}

bool areQuerySubmissionsFinished(DeviceImpl* device, const QueryPool::QueryRangeInfo& info)
{
    for (uint32_t queue = 0; queue < QueryPool::kQueueTypeCount; ++queue)
    {
        if (info.submissionIDs[queue] != 0 &&
            getQueryQueue(device, queue)->updateLastFinishedID() < info.submissionIDs[queue])
        {
            return false;
        }
    }
    return true;
}

Result waitForQuerySubmissions(DeviceImpl* device, const QueryPool::QueryRangeInfo& info)
{
    for (uint32_t queue = 0; queue < QueryPool::kQueueTypeCount; ++queue)
    {
        if (info.submissionIDs[queue] != 0)
        {
            SLANG_RETURN_ON_FAIL(
                getQueryQueue(device, queue)->waitForSequence(info.submissionIDs[queue], kTimeoutInfinite)
            );
        }
    }
    return SLANG_OK;
}

} // namespace

QueryPoolImpl::QueryPoolImpl(Device* device, const QueryPoolDesc& desc)
    : QueryPool(device, desc)
{
}

QueryPoolImpl::~QueryPoolImpl()
{
    for (uint32_t queue = 0; queue < kQueueTypeCount; ++queue)
    {
        if (m_mappedReadBackData[queue])
        {
            D3D12_RANGE writtenRange = {0, 0};
            m_readBackBuffers[queue].getResource()->Unmap(0, &writtenRange);
            m_mappedReadBackData[queue] = nullptr;
        }
    }
}

Result QueryPoolImpl::init()
{
    DeviceImpl* device = getDevice<DeviceImpl>();

    // Translate query type.
    D3D12_QUERY_HEAP_DESC heapDesc = {};
    heapDesc.Count = (UINT)m_desc.count;
    heapDesc.NodeMask = 1;
    switch (m_desc.type)
    {
    case QueryType::Timestamp:
        heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        m_queryType = D3D12_QUERY_TYPE_TIMESTAMP;
        break;
    default:
        return SLANG_E_INVALID_ARG;
    }

    // Create query heap.
    auto d3dDevice = device->m_device;
    SLANG_D3D_RETURN_ON_FAIL_REPORT(
        d3dDevice->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(m_queryHeap.writeRef())),
        device
    );

    if (m_desc.label)
    {
        m_queryHeap->SetName(string::to_wstring(m_desc.label).c_str());
    }

    D3D12_HEAP_PROPERTIES heapProps = makeHeapProperties(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC resourceDesc = {};
    initBufferDesc(sizeof(uint64_t) * m_desc.count, resourceDesc);
    D3D12_RANGE readRange = {0, sizeof(uint64_t) * m_desc.count};
    for (uint32_t queue = 0; queue < kQueueTypeCount; ++queue)
    {
        SLANG_RETURN_ON_FAIL(m_readBackBuffers[queue].initCommitted(
            d3dDevice,
            heapProps,
            D3D12_HEAP_FLAG_NONE,
            resourceDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            device->m_allocator
        ));
        SLANG_D3D_RETURN_ON_FAIL_REPORT(
            m_readBackBuffers[queue].getResource()->Map(
                0,
                &readRange,
                reinterpret_cast<void**>(&m_mappedReadBackData[queue])
            ),
            device
        );
    }

    return SLANG_OK;
}

Result QueryPoolImpl::getResultState(uint32_t queryIndex, uint32_t count, QueryResultState* outState)
{
    if (!outState || !isValidQueryRange(queryIndex, count))
    {
        return SLANG_E_INVALID_ARG;
    }

    QueryRangeInfo queryInfo = getQueryRangeInfo(queryIndex, count);
    if (queryInfo.state == QueryResultState::Reset)
    {
        *outState = QueryResultState::Reset;
        return SLANG_OK;
    }
    if (queryInfo.state == QueryResultState::Resolved)
    {
        *outState = QueryResultState::Resolved;
        return SLANG_OK;
    }

    if (!areQuerySubmissionsFinished(getDevice<DeviceImpl>(), queryInfo))
    {
        *outState = QueryResultState::Pending;
        return SLANG_OK;
    }

    markQueryRangeResolved(queryIndex, count, queryInfo);
    *outState = QueryResultState::Resolved;

    return SLANG_OK;
}

Result QueryPoolImpl::getResult(uint32_t queryIndex, uint32_t count, uint64_t* outData)
{
    if (!outData || !isValidQueryRange(queryIndex, count))
    {
        return SLANG_E_INVALID_ARG;
    }

    QueryRangeInfo queryInfo = getQueryRangeInfo(queryIndex, count);
    if (queryInfo.state == QueryResultState::Reset)
    {
        return SLANG_FAIL;
    }
    if (count == 0)
    {
        return SLANG_OK;
    }

    SLANG_RETURN_ON_FAIL(waitForQuerySubmissions(getDevice<DeviceImpl>(), queryInfo));

    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t queue = uint32_t(getQuerySlotQueue(queryIndex + i));
        memcpy(outData + i, m_mappedReadBackData[queue] + sizeof(uint64_t) * (queryIndex + i), sizeof(uint64_t));
    }
    markQueryRangeResolved(queryIndex, count, queryInfo);

    return SLANG_OK;
}

void QueryPoolImpl::writeTimestamp(ID3D12GraphicsCommandList* cmdList, uint32_t index)
{
    cmdList->EndQuery(m_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, index);
}

IQueryPool* PlainBufferProxyQueryPoolImpl::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == IQueryPool::getTypeGuid())
        return static_cast<IQueryPool*>(this);
    return nullptr;
}

PlainBufferProxyQueryPoolImpl::PlainBufferProxyQueryPoolImpl(Device* device, const QueryPoolDesc& desc)
    : QueryPool(device, desc)
{
}

PlainBufferProxyQueryPoolImpl::~PlainBufferProxyQueryPoolImpl()
{
    if (m_mappedReadBackData)
    {
        D3D12_RANGE writtenRange = {0, 0};
        m_readBackBuffer.getResource()->Unmap(0, &writtenRange);
        m_mappedReadBackData = nullptr;
    }
}

Result PlainBufferProxyQueryPoolImpl::init(uint32_t stride)
{
    DeviceImpl* device = getDevice<DeviceImpl>();
    uint64_t size = uint64_t(m_desc.count) * stride;

    ComPtr<IBuffer> buffer;
    BufferDesc bufferDesc = {};
    bufferDesc.defaultState = ResourceState::CopySource;
    bufferDesc.elementSize = 0;
    bufferDesc.size = size;
    bufferDesc.format = Format::Undefined;
    bufferDesc.usage = BufferUsage::UnorderedAccess;
    SLANG_RETURN_ON_FAIL(device->createBuffer(bufferDesc, nullptr, buffer.writeRef()));
    m_buffer = checked_cast<BufferImpl*>(buffer.get());

    D3D12_HEAP_PROPERTIES heapProps = makeHeapProperties(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC resourceDesc = {};
    initBufferDesc(size, resourceDesc);
    SLANG_RETURN_ON_FAIL(m_readBackBuffer.initCommitted(
        device->m_device,
        heapProps,
        D3D12_HEAP_FLAG_NONE,
        resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        device->m_allocator
    ));

    D3D12_RANGE readRange = {0, size};
    SLANG_D3D_RETURN_ON_FAIL_REPORT(
        m_readBackBuffer.getResource()->Map(0, &readRange, reinterpret_cast<void**>(&m_mappedReadBackData)),
        device
    );

    m_queryType = m_desc.type;
    m_device = device;
    m_stride = stride;
    m_count = (uint32_t)m_desc.count;
    return SLANG_OK;
}

Result PlainBufferProxyQueryPoolImpl::getResultState(uint32_t queryIndex, uint32_t count, QueryResultState* outState)
{
    if (!outState || !isValidQueryRange(queryIndex, count))
    {
        return SLANG_E_INVALID_ARG;
    }

    QueryRangeInfo queryInfo = getQueryRangeInfo(queryIndex, count);
    if (queryInfo.state == QueryResultState::Reset)
    {
        *outState = QueryResultState::Reset;
        return SLANG_OK;
    }
    if (queryInfo.state == QueryResultState::Resolved)
    {
        *outState = QueryResultState::Resolved;
        return SLANG_OK;
    }

    if (!areQuerySubmissionsFinished(getDevice<DeviceImpl>(), queryInfo))
    {
        *outState = QueryResultState::Pending;
        return SLANG_OK;
    }

    markQueryRangeResolved(queryIndex, count, queryInfo);
    *outState = QueryResultState::Resolved;

    return SLANG_OK;
}

Result PlainBufferProxyQueryPoolImpl::getResult(uint32_t queryIndex, uint32_t count, uint64_t* outData)
{
    if (!outData || !isValidQueryRange(queryIndex, count))
    {
        return SLANG_E_INVALID_ARG;
    }

    QueryRangeInfo queryInfo = getQueryRangeInfo(queryIndex, count);
    if (queryInfo.state == QueryResultState::Reset)
    {
        return SLANG_FAIL;
    }
    if (count == 0)
    {
        return SLANG_OK;
    }

    SLANG_RETURN_ON_FAIL(waitForQuerySubmissions(getDevice<DeviceImpl>(), queryInfo));

    memcpy(outData, m_mappedReadBackData + uint64_t(m_stride) * queryIndex, uint64_t(m_stride) * count);
    markQueryRangeResolved(queryIndex, count, queryInfo);

    return SLANG_OK;
}

} // namespace rhi::d3d12
