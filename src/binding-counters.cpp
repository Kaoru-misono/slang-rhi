#include "binding-counters.h"

#if SLANG_RHI_ENABLE_BINDING_COUNTERS
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace rhi {
thread_local BindingCounters* activeBindingCounters = nullptr;
static std::atomic<uint64_t> nextCommandBufferId{1};

BindingCounterScope::BindingCounterScope(BindingCounters& counters)
    : previous(activeBindingCounters)
{
    if (!counters.commandBufferId)
        counters.commandBufferId = nextCommandBufferId.fetch_add(1, std::memory_order_relaxed);
    activeBindingCounters = &counters;
}
BindingCounterScope::~BindingCounterScope() { activeBindingCounters = previous; }

BindingCounterFinish::BindingCounterFinish(BindingCounters& counters, const char* backend, uint32_t queue)
    : BindingCounterScope(counters), counters(counters), backend(backend), queue(queue)
{}

BindingCounterFinish::~BindingCounterFinish()
{
    // Output is opt-in and never overwrites an earlier measurement.
    struct Sink
    {
        std::mutex mutex;
        FILE* file = nullptr;
        uint64_t sequence = 0;
        Sink()
        {
            const char* path = std::getenv("SLANG_RHI_BINDING_COUNTERS_FILE");
            if (!path || !*path)
                path = SLANG_RHI_BINDING_COUNTERS_PATH;
            if (path && *path)
            {
                file = std::fopen(path, "wx");
                if (!file)
                    std::fprintf(stderr, "slang-rhi: cannot exclusively create binding counters file: %s\n", path);
            }
        }
        ~Sink() { if (file) std::fclose(file); }
    };
    static Sink sink;
    std::lock_guard<std::mutex> lock(sink.mutex);
    if (!sink.file)
        return;
    const auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    auto u = [](uint64_t value) { return static_cast<unsigned long long>(value); };
    std::fprintf(sink.file,
        "{\"schema\":1,\"sequence\":%llu,\"unix_us\":%lld,\"backend\":\"%s\","
        "\"command_buffer_id\":%llu,\"queue\":%u,\"finish_success\":%s,\"counts\":{"
        "\"table_hits\":%llu,\"table_misses\":%llu,\"sampler_hits\":%llu,\"sampler_misses\":%llu,"
        "\"descriptors_written\":%llu,\"cb_allocations\":%llu,\"tracked_refs\":%llu,\"state_entries\":%llu},\"timing_ns\":{\"resolve\":%llu,\"record_inclusive\":%llu,\"close\":%llu},"
        "\"phase_calls\":{\"resolve\":%llu,\"record\":%llu,\"close\":%llu}}\n",
        u(++sink.sequence), static_cast<long long>(timestamp), backend, u(counters.commandBufferId), queue,
        success ? "true" : "false", u(counters.tableHits), u(counters.tableMisses),
        u(counters.samplerHits), u(counters.samplerMisses), u(counters.descriptorsWritten),
        u(counters.cbAllocations), u(counters.trackedRefs), u(counters.stateEntries),
        u(counters.resolveNs), u(counters.recordInclusiveNs), u(counters.closeNs),
        u(counters.resolveCalls), u(counters.recordCalls), u(counters.closeCalls));
    std::fflush(sink.file);
}
}
#endif
