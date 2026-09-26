// uint16-buffer.slang - Simple shader that takes a buffer of uint16 type and increments all elements by 1.

// This is to verify that the slang-rhi can set correct buffer strides for structured buffer bindings.

struct Resources
{
    RWStructuredBuffer<uint16_t> buffer;
};

ParameterBlock<Resources> resources;

[shader("compute")]
[numthreads(4,1,1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID)
{
    var input = resources.buffer[sv_dispatchThreadID.x];
    resources.buffer[sv_dispatchThreadID.x] = uint16_t(input + 1);
}
