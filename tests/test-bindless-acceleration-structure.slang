[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uniform RaytracingAccelerationStructure.Handle scene, uniform RWStructuredBuffer<uint> result)
{
    RayDesc ray;
    ray.Origin = float3(0.25, 0.25, 0.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 100.0;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed())
    {
    }
    result[0] = query.CommittedStatus();
}
