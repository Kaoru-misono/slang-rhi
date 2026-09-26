// compute-trivial.slang - Simple shader that takes a buffer and increments all elements by 1.

#if defined(TRIVIAL_ENTRY_computeMain)
struct Resources
{
    RWStructuredBuffer<float> buffer;
};
struct Constants
{
    float value;
    float padding0;
    float padding1;
    float padding2;
};
ParameterBlock<Resources> resources;
ConstantBuffer<Constants> constants;

[shader("compute")]
[numthreads(4,1,1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID)
{
    var input = resources.buffer[sv_dispatchThreadID.x];
    resources.buffer[sv_dispatchThreadID.x] = input + 1.0f + constants.value;
}
#endif

#if defined(TRIVIAL_ENTRY_readTexture)
struct Resources
{
    Texture2D<float> inputTexture;
    RWStructuredBuffer<float> outputBuffer;
};
ParameterBlock<Resources> resources;
[shader("compute")]
[numthreads(1, 1, 1)]
void readTexture()
{
    resources.outputBuffer[0] = resources.inputTexture.Load(int3(0, 0, 0));
}
#endif

#if defined(TRIVIAL_ENTRY_writeTexture)
struct Resources
{
    RWTexture2D<float> outputTexture;
};
ParameterBlock<Resources> resources;
[shader("compute")]
[numthreads(1, 1, 1)]
void writeTexture()
{
    resources.outputTexture[int2(0, 0)] = 42.0f;
}
#endif

#if !defined(TRIVIAL_ENTRY_computeMain) && !defined(TRIVIAL_ENTRY_readTexture) && !defined(TRIVIAL_ENTRY_writeTexture)
// Legacy shader-object layout: loose globals and entry-point resources.
uniform RWStructuredBuffer<float> buffer;
uniform float value;

[shader("compute")]
[numthreads(4,1,1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID)
{
    var input = buffer[sv_dispatchThreadID.x];
    buffer[sv_dispatchThreadID.x] = input + 1.0f + value;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void readTexture(Texture2D<float> inputTexture, RWStructuredBuffer<float> outputBuffer)
{
    outputBuffer[0] = inputTexture.Load(int3(0, 0, 0));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void writeTexture(RWTexture2D<float> outputTexture)
{
    outputTexture[int2(0, 0)] = 42.0f;
}
#endif
