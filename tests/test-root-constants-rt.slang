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
[[vk::push_constant]] ConstantBuffer<Root> root;
[shader("raygeneration")]
void rayMain()
{
    let p = params;
    p.output.Store<uint>(root.dispatch.x * 4, p.values.x + root.dispatch.y);
}
