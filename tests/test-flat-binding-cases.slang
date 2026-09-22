// One parameter shape in several variants, selected by preprocessor macros, so every positive
// flat-binding case compiles from this module:
//   INLINE_CONSTANTS  the constants reach the shader as push constants instead of a constant buffer
//   WIDE_CONSTANTS    the constant block gains a uint4, so only the constants layout differs
//   BINDLESS_HANDLE   the constant block carries a descriptor handle to a buffer outside any set,
//                     and the set/handle transition entry points are compiled
//   SAMPLER_COUNT     the parameter block holds an array of samplers instead of the default scene

#ifndef INLINE_CONSTANTS
#define INLINE_CONSTANTS 0
#endif
#ifndef WIDE_CONSTANTS
#define WIDE_CONSTANTS 0
#endif
#ifndef BINDLESS_HANDLE
#define BINDLESS_HANDLE 0
#endif
#ifndef SAMPLER_COUNT
#define SAMPLER_COUNT 0
#endif

struct Constants
{
    uint offset;
    uint multiplier;
    uint bias;
    uint tag;
#if WIDE_CONSTANTS
    uint4 extra;
#endif
#if BINDLESS_HANDLE
    // Only .xy is the handle; the block stays a multiple of 16 bytes for the portable subset.
    uint4 handle;
#endif
};

#if INLINE_CONSTANTS
[[vk::push_constant]] ConstantBuffer<Constants> constants;
#else
ConstantBuffer<Constants> constants;
#endif

#if SAMPLER_COUNT
struct Scene
{
    Texture2D<float4> texture;
    SamplerState samplers[SAMPLER_COUNT];
    RWStructuredBuffer<uint> output;
};
#else
struct Scene
{
    StructuredBuffer<uint> input;
    RWStructuredBuffer<uint> output;
    Texture2D<float4> texture;
    SamplerState sampler;
};
#endif

ParameterBlock<Scene> scene;

#if SAMPLER_COUNT

[shader("compute")]
[numthreads(1, 1, 1)]
void csSamplers(uint3 id : SV_DispatchThreadID)
{
    uint sum = 0;
    for (uint i = 0; i < SAMPLER_COUNT; ++i)
        sum += uint(scene.texture.SampleLevel(scene.samplers[i], float2(0.5, 0.5), 0).x);
    scene.output[constants.offset + id.x] = sum * constants.multiplier + constants.bias;
}

#else

uint sampled()
{
    return uint(scene.texture.SampleLevel(scene.sampler, float2(0.5, 0.5), 0).x);
}

uint contribution(uint index)
{
    uint value = scene.input[index] * constants.multiplier + constants.bias + sampled();
#if WIDE_CONSTANTS
    value += constants.extra.x + constants.extra.w;
#endif
#if BINDLESS_HANDLE
    ByteAddressBuffer.Handle extra = ByteAddressBuffer.Handle(constants.handle.xy);
    value += extra.Load<uint>(index * 4);
#endif
    return value;
}

// `tag` scales whatever is already in the output slot, so consecutive dispatches observe each
// other's writes while a single dispatch with tag == 0 is a plain write.
[shader("compute")]
[numthreads(1, 1, 1)]
void csMain(uint3 id : SV_DispatchThreadID)
{
    uint slot = constants.offset + id.x;
    scene.output[slot] = scene.output[slot] * constants.tag + contribution(id.x);
}

struct Varying
{
    float4 position : SV_Position;
    nointerpolation float4 value : COLOR0;
};

[shader("vertex")]
Varying vsMain(uint id : SV_VertexID)
{
    Varying result;
    result.position = float4(id == 2 ? 3 : -1, id == 1 ? 3 : -1, 0, 1);
    result.value = float4(float(scene.input[0] * constants.multiplier), 0, 0, 0);
    return result;
}

[shader("fragment")]
float4 psMain(Varying input) : SV_Target
{
    return input.value + float4(float(constants.bias + constants.tag + sampled()), 0, 0, 1);
}

#if BINDLESS_HANDLE

// Producers and consumers for the state-transition case. Each pair hands one buffer from a
// dispatch that writes it to a dispatch that reads it, once through a parameter block entry and
// once through a descriptor handle, so a missing transition shows up in the readback.

[shader("compute")]
[numthreads(1, 1, 1)]
void csSetWrite(uint3 id : SV_DispatchThreadID)
{
    scene.output[id.x] = scene.input[id.x] * constants.multiplier + constants.bias;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csHandleRead(uint3 id : SV_DispatchThreadID)
{
    ByteAddressBuffer.Handle source = ByteAddressBuffer.Handle(constants.handle.xy);
    scene.output[id.x] = source.Load<uint>(id.x * 4) * constants.tag;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csHandleWrite(uint3 id : SV_DispatchThreadID)
{
    RWByteAddressBuffer.Handle target = RWByteAddressBuffer.Handle(constants.handle.xy);
    target.Store<uint>(id.x * 4, scene.input[id.x] * constants.multiplier + constants.bias);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csSetRead(uint3 id : SV_DispatchThreadID)
{
    scene.output[id.x] = scene.input[id.x] * constants.tag;
}

#endif

#endif
