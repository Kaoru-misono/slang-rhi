#if defined(RESOURCE_TRACKING_FLAT)
struct GlobalResources
{
    StructuredBuffer<float> buffer;
    Texture2D<float> texture;
    SamplerState sampler;
};

struct EntryResources
{
    StructuredBuffer<float> buffer;
    Texture2D<float> texture;
    SamplerState sampler;
    RWStructuredBuffer<float> resultBuffer;
};

ParameterBlock<GlobalResources> globals;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(ParameterBlock<EntryResources> entry)
{
    entry.resultBuffer[0] = globals.buffer[0];
    entry.resultBuffer[1] = globals.texture.SampleLevel(globals.sampler, float2(0.25), 0.0);
    entry.resultBuffer[2] = entry.buffer[0];
    entry.resultBuffer[3] = entry.texture.SampleLevel(entry.sampler, float2(0.5), 0.0);
}
#else
// Legacy shader-object layout: loose global and entry-point resources.
StructuredBuffer<float> globalBuffer;
Texture2D<float> globalTexture;
SamplerState globalSampler;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain( StructuredBuffer<float> buffer, Texture2D<float> texture, SamplerState sampler, RWStructuredBuffer<float> resultBuffer)
{
    resultBuffer[0] = globalBuffer[0];
    resultBuffer[1] = globalTexture.SampleLevel(globalSampler, float2(0.25), 0.0);
    resultBuffer[2] = buffer[0];
    resultBuffer[3] = texture.SampleLevel(sampler, float2(0.5), 0.0);
}
#endif
