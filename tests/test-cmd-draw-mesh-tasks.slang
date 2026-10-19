struct MeshVertex
{
    float4 position : SV_Position;
};

// Group x covers column x of four columns spanning the target.
[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void meshMain(uint3 groupId: SV_GroupID, OutputVertices<MeshVertex, 4> vertices, OutputIndices<uint3, 2> triangles)
{
    SetMeshOutputCounts(4, 2);
    float left = -1.0 + 0.5 * groupId.x;
    float right = left + 0.5;
    vertices[0].position = float4(left, -1.0, 0.5, 1.0);
    vertices[1].position = float4(right, -1.0, 0.5, 1.0);
    vertices[2].position = float4(right, 1.0, 0.5, 1.0);
    vertices[3].position = float4(left, 1.0, 0.5, 1.0);
    triangles[0] = uint3(0, 1, 2);
    triangles[1] = uint3(0, 2, 3);
}

[shader("fragment")]
float4 fragmentMain() : SV_Target
{
    return float4(1.0, 0.0, 0.0, 1.0);
}
