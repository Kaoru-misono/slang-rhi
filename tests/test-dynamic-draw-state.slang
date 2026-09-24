[shader("vertex")]
float4 vsMain(uint id : SV_VertexID) : SV_Position
{
    float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(p[id], 0.5, 1);
}
[shader("fragment")]
float4 psWhite() : SV_Target0 { return float4(1, 1, 1, 1); }
[shader("fragment")]
float4 psRed() : SV_Target0 { return float4(1, 0, 0, 1); }
