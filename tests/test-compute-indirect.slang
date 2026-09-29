#if defined(INDIRECT_ENTRY_computeMain) || defined(INDIRECT_ENTRY_writeDispatchArgs) || defined(INDIRECT_ENTRY_writeDispatchRecord) || defined(INDIRECT_ENTRY_consumeDispatchRecord)
// Flat binding layout; each entry point is compiled alone through its INDIRECT_ENTRY_ macro.
// The shader increments a counter in the output buffer for each thread that runs.
// By using different indirect dispatch arguments, we can verify the correct number of threads are dispatched.

#include "../include/slang-rhi-device.h"

struct ComputeResources
{
    RWByteAddressBuffer outputBuffer;
};

struct WriteArgsResources
{
    RWStructuredBuffer<rhi::IndirectDispatchArguments> dispatchArgsBuffer;
};

struct DispatchConstants
{
    uint dispatchIndex;
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

#if defined(INDIRECT_ENTRY_computeMain)
struct ComputeConstants
{
    uint dispatchIndex;
    uint padding0;
    uint padding1;
    uint padding2;
};
ConstantBuffer<ComputeConstants> constants;

[shader("compute")]
[numthreads(4, 4, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID, ParameterBlock<ComputeResources> resources)
{
    // Each thread increments the counter at its dispatch index
    // This allows us to verify that the correct number of threads ran for each dispatch
    resources.outputBuffer.InterlockedAdd(constants.dispatchIndex * sizeof(uint), 1);
}
#endif

#if defined(INDIRECT_ENTRY_writeDispatchArgs)
ConstantBuffer<DispatchConstants> constants;

// Entry point to write dispatch arguments from the GPU
[shader("compute")]
[numthreads(1, 1, 1)]
void writeDispatchArgs(
    ParameterBlock<WriteArgsResources> resources
)
{
    resources.dispatchArgsBuffer[constants.dispatchIndex].threadGroupCountX = constants.groupCountX;
    resources.dispatchArgsBuffer[constants.dispatchIndex].threadGroupCountY = constants.groupCountY;
    resources.dispatchArgsBuffer[constants.dispatchIndex].threadGroupCountZ = constants.groupCountZ;
}
#endif

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

struct WriteRecordResources
{
    RWStructuredBuffer<IndirectDispatchRecord> dispatchRecords;
};

struct ConsumeRecordResources
{
    StructuredBuffer<IndirectDispatchRecord> dispatchRecords;
    RWStructuredBuffer<uint> outputBuffer;
};

#if defined(INDIRECT_ENTRY_writeDispatchRecord)
struct RecordConstants
{
    uint itemCount;
    uint itemStart;
    uint padding0;
    uint padding1;
};
ConstantBuffer<RecordConstants> constants;

[shader("compute")]
[numthreads(1, 1, 1)]
void writeDispatchRecord(
    ParameterBlock<WriteRecordResources> resources
)
{
    IndirectDispatchRecord record = {};
    record.threadGroupCountX = 1;
    record.threadGroupCountY = 1;
    record.threadGroupCountZ = 1;
    record.itemCount = constants.itemCount;
    record.itemStart = constants.itemStart;
    resources.dispatchRecords[0] = record;
}
#endif

#if defined(INDIRECT_ENTRY_consumeDispatchRecord)
[shader("compute")]
[numthreads(1, 1, 1)]
void consumeDispatchRecord(
    ParameterBlock<ConsumeRecordResources> resources
)
{
    IndirectDispatchRecord record = resources.dispatchRecords[0];
    resources.outputBuffer[0] = record.itemCount + record.itemStart;
}
#endif
#else
// Legacy shader-object layout: loose entry-point parameters.
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
#endif
