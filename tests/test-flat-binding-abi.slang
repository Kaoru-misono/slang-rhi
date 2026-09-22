#ifndef NATIVE_LAYOUT
#define NATIVE_LAYOUT 0
#endif

struct Nested
{
    float3 weights;
    uint tag;
};

struct Params
{
    float3 direction;
    float scale;
    float4 samples[2];
    float4x4 matrix;
    Nested nested;
};

#if INLINE_CONSTANTS
[[vk::push_constant]] ConstantBuffer<Params> parameters;
#elif NATIVE_LAYOUT
[[vk::binding(0, 1)]] ConstantBuffer<Params> parameters : register(b0, space1);
#else
// Legacy ShaderObject supports one own Vulkan set and drops this CBV's explicit D3D12 space.
// The flat-binding implementation must separately validate constants in set/space 1.
[[vk::binding(1, 0)]] ConstantBuffer<Params> parameters : register(b0, space0);
#endif

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> output : register(u0, space0);

float4 evaluate(uint index)
{
    return mul(parameters.matrix, parameters.samples[index])
        + float4(parameters.direction * parameters.scale, float(parameters.nested.tag))
        + float4(parameters.nested.weights, 1.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csMain(uint3 id : SV_DispatchThreadID)
{
    output[id.x] = evaluate(id.x);
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
    result.value = evaluate(0);
    return result;
}

[shader("fragment")]
float4 psMain(Varying input) : SV_Target
{
    return input.value + evaluate(1);
}
