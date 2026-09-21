// compute-trivial.slang - Simple shader that takes a buffer and increments all elements by 1.

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
