#include "testing.h"
#include "../src/binding-counters.h"

using namespace rhi;
using namespace rhi::testing;

TEST_CASE("binding-counters-recording-isolation")
{
#if SLANG_RHI_ENABLE_BINDING_COUNTERS
    BindingCounters first;
    BindingCounters second;
    auto* previous = activeBindingCounters;
    {
        BindingCounterScope a(first);
        SLANG_RHI_BINDING_COUNT(descriptorsWritten, 3);
        {
            BindingCounterScope b(second);
            SLANG_RHI_BINDING_COUNT(descriptorsWritten, 7);
            SLANG_RHI_BINDING_COUNT(tableMisses, 1);
        }
        SLANG_RHI_BINDING_COUNT(descriptorsWritten, 2);
    }
    CHECK(activeBindingCounters == previous);
    CHECK(first.descriptorsWritten == 5);
    CHECK(second.descriptorsWritten == 7);
    CHECK(first.tableMisses == 0);
    CHECK(second.tableMisses == 1);
    CHECK(first.commandBufferId != second.commandBufferId);
    const uint64_t oldId = first.commandBufferId;
    first = {};
    {
        BindingCounterScope a(first);
        CHECK(first.commandBufferId != oldId);
        CHECK(first.descriptorsWritten == 0);
    }
#else
    int evaluations = 0;
    SLANG_RHI_BINDING_COUNT(descriptorsWritten, ++evaluations);
    CHECK(evaluations == 0);
#endif
}
