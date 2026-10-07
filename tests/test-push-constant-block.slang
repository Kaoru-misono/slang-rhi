import push_constants;

struct Block
{
    uint4 values;
    RWByteAddressBuffer output;
};
struct Root
{
    uint4 dispatch;
};

ParameterBlock<Block> params;
[PushConstants] [[vk::push_constant]] ConstantBuffer<Root> root;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    let p = params;
    p.output.Store<uint>(root.dispatch.x * 4, p.values.x + root.dispatch.y);
}
struct Vertex
{
    float4 position : SV_Position;
    nointerpolation uint value : VALUE;
};
[shader("vertex")]
Vertex vertexMain(uint id : SV_VertexID)
{
    let p = params;
    Vertex v;
    v.position = float4(float2((id << 1) & 2, id & 2) * 2 - 1, 0, 1);
    v.value = p.values.x + root.dispatch.y;
    return v;
}
struct Extra { uint bias; };
[shader("fragment")]
float4 fragmentMain(Vertex v, uniform ConstantBuffer<Extra> extra) : SV_Target
{
    let p = params;
    return float4(v.value, p.values.y + root.dispatch.z + extra.bias, 0, 1);
}
