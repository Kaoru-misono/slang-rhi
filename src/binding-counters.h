#pragma once

#if SLANG_RHI_ENABLE_BINDING_COUNTERS
#include <slang-rhi.h>
#include <cstdint>
#include <chrono>

namespace rhi {
struct BindingCounters
{
    uint64_t commandBufferId = 0;
    uint64_t tableHits = 0;
    uint64_t tableMisses = 0;
    uint64_t samplerHits = 0;
    uint64_t samplerMisses = 0;
    uint64_t descriptorsWritten = 0;
    uint64_t cbAllocations = 0;
    uint64_t trackedRefs = 0;
    uint64_t stateEntries = 0;
    uint64_t resolveNs = 0;
    uint64_t recordInclusiveNs = 0;
    uint64_t closeNs = 0;
    uint64_t resolveCalls = 0;
    uint64_t recordCalls = 0;
    uint64_t closeCalls = 0;
};
extern SLANG_RHI_API thread_local BindingCounters* activeBindingCounters;

struct BindingCounterTimer
{
    uint64_t* total;
    std::chrono::steady_clock::time_point start;
    BindingCounterTimer(uint64_t BindingCounters::*duration, uint64_t BindingCounters::*calls)
        : total(activeBindingCounters ? &(activeBindingCounters->*duration) : nullptr),
          start(std::chrono::steady_clock::now())
    {
        if (activeBindingCounters) ++(activeBindingCounters->*calls);
    }
    ~BindingCounterTimer()
    {
        if (total) *total += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    }
};

struct BindingCounterScope
{
    SLANG_RHI_API BindingCounterScope(BindingCounters& counters);
    SLANG_RHI_API ~BindingCounterScope();
    BindingCounters* previous;
};

struct BindingCounterFinish : BindingCounterScope
{
    BindingCounterFinish(BindingCounters& counters, const char* backend, uint32_t queue);
    ~BindingCounterFinish();
    BindingCounters& counters;
    const char* backend;
    uint32_t queue;
    bool success = false;
};
}
#define SLANG_RHI_BINDING_COUNT(field, amount) \
    do { if (::rhi::activeBindingCounters) ::rhi::activeBindingCounters->field += (amount); } while (0)
#define SLANG_RHI_BINDING_TIMER(duration, calls) \
    ::rhi::BindingCounterTimer bindingCounterTimer(&::rhi::BindingCounters::duration, &::rhi::BindingCounters::calls)
#define SLANG_RHI_BINDING_SCOPE(buffer) ::rhi::BindingCounterScope bindingCounterScope((buffer)->m_bindingCounters)
#define SLANG_RHI_BINDING_FINISH(buffer, backend, queue) \
    ::rhi::BindingCounterFinish bindingCounterFinish((buffer)->m_bindingCounters, backend, uint32_t(queue))
#define SLANG_RHI_BINDING_FINISH_SUCCESS() bindingCounterFinish.success = true
#define SLANG_RHI_BINDING_RESET() m_bindingCounters = {}
#else
#define SLANG_RHI_BINDING_TIMER(duration, calls) do {} while (0)
#define SLANG_RHI_BINDING_COUNT(field, amount) do {} while (0)
#define SLANG_RHI_BINDING_SCOPE(buffer) do {} while (0)
#define SLANG_RHI_BINDING_FINISH(buffer, backend, queue) do {} while (0)
#define SLANG_RHI_BINDING_FINISH_SUCCESS() do {} while (0)
#define SLANG_RHI_BINDING_RESET() do {} while (0)
#endif
