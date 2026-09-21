// The shader increments a counter in the output buffer for each thread that runs.
// By using different indirect dispatch arguments, we can verify the correct number of threads are dispatched.

#include "../include/slang-rhi-device.h"

[shader("compute")]
[numthreads(4, 4, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID, RWByteAddressBuffer outputBuffer, uniform uint dispatchIndex)
{
    // Each thread increments the counter at its dispatch index
    // This allows us to verify that the correct number of threads ran for each dispatch
    outputBuffer.InterlockedAdd(dispatchIndex * sizeof(uint), 1);
}

// Entry point to write dispatch arguments from the GPU
[shader("compute")]
[numthreads(1, 1, 1)]
void writeDispatchArgs(
    RWStructuredBuffer<rhi::IndirectDispatchArguments> dispatchArgsBuffer,
    uniform uint dispatchIndex,
    uniform uint3 threadGroupCounts
)
{
    dispatchArgsBuffer[dispatchIndex].threadGroupCountX = threadGroupCounts.x;
    dispatchArgsBuffer[dispatchIndex].threadGroupCountY = threadGroupCounts.y;
    dispatchArgsBuffer[dispatchIndex].threadGroupCountZ = threadGroupCounts.z;
}

struct IndirectDispatchRecord
{
    uint threadGroupCountX;
    uint threadGroupCountY;
    uint threadGroupCountZ;
    uint itemCount;
    uint itemStart;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void writeDispatchRecord(
    RWStructuredBuffer<IndirectDispatchRecord> dispatchRecords,
    uniform uint itemCount,
    uniform uint itemStart
)
{
    IndirectDispatchRecord record = {};
    record.threadGroupCountX = 1;
    record.threadGroupCountY = 1;
    record.threadGroupCountZ = 1;
    record.itemCount = itemCount;
    record.itemStart = itemStart;
    dispatchRecords[0] = record;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void consumeDispatchRecord(
    StructuredBuffer<IndirectDispatchRecord> dispatchRecords,
    RWStructuredBuffer<uint> outputBuffer
)
{
    IndirectDispatchRecord record = dispatchRecords[0];
    outputBuffer[0] = record.itemCount + record.itemStart;
}
