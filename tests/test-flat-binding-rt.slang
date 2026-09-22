// Ray tracing in the flat binding model: every resource lives in one ParameterBlock and the only
// ordinary data is the push constant block, so the pipeline needs no shader object.

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
ParameterBlock<Scene> scene;

struct Params
{
    uint2 dims;
    float hitValue;
    float missValue;
};
[[vk::push_constant]] ConstantBuffer<Params> params;

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
