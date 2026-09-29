// Shared ray-tracing oracle for ShaderObject snapshots and shader-table retention.

[raypayload]
struct RayPayload
{
    float4 color : read(caller) : write(caller, closesthit, miss);
};

struct Scene
{
    RaytracingAccelerationStructure tlas;
    RWTexture2D<float4> output;
};
#if defined(SCENE_SET)
[[vk::binding(0, SCENE_SET)]]
#endif
ParameterBlock<Scene> scene;

struct Params
{
    uint2 dims;
    float hitValue;
    float missValue;
};
#if !defined(INLINE_CONSTANTS) || INLINE_CONSTANTS
[[vk::push_constant]]
#endif
ConstantBuffer<Params> params;

[shader("raygeneration")]
void rayGenMain()
{
    uint2 pixel = DispatchRaysIndex().xy;
    if (any(pixel >= params.dims))
        return;

    // Ray origins stay clear of the triangle edges so the hit pattern is exact on both backends.
    RayDesc ray;
    ray.Origin = float3(0.05 + 0.25 * pixel.x, 0.05 + 0.25 * pixel.y, 0.0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayPayload payload = {float4(0, 0, 0, 0)};
    TraceRay(scene.tlas, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);
    scene.output[pixel] = payload.color;
}

[shader("raygeneration")]
void rayGenAlternate()
{
    uint2 pixel = DispatchRaysIndex().xy;
    if (any(pixel >= params.dims))
        return;

    RayDesc ray;
    ray.Origin = float3(0.05 + 0.25 * pixel.x, 0.05 + 0.25 * pixel.y, 0.0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayPayload payload = {float4(0, 0, 0, 0)};
    TraceRay(scene.tlas, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);
    scene.output[pixel] = payload.color + float4(0.125, 0.125, 0.125, 0);
}

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.color = float4(params.missValue, params.missValue, params.missValue, 1);
}

[shader("closesthit")]
void closestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.color = float4(params.hitValue, params.hitValue, params.hitValue, 1);
}
