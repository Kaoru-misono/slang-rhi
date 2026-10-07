import push_constants;

struct GlobalValues
{
    uint4 values;
};
struct Step
{
    uint slot;
    uint value;
    uint unused;
};
struct Outputs
{
    RWByteAddressBuffer output;
};

// The global constant buffer takes b0, so the push-constant block is reflected at b1.
ConstantBuffer<GlobalValues> globals;
[PushConstants] [[vk::push_constant]] ConstantBuffer<Step> step;
RWByteAddressBuffer output;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    output.Store<uint>(step.slot * 4, globals.values.x + step.value);
}

// The entry point's parameter block takes space1, while the push-constant block stays at b1 in space0.
[shader("compute")]
[numthreads(1, 1, 1)]
void computeWithBlock(uniform ParameterBlock<Outputs> outputs)
{
    outputs.output.Store<uint>(step.slot * 4, globals.values.x + step.value);
}
