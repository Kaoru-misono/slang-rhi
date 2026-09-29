struct Params { uint index; uint value; uint2 padding; };
struct Resources { RWStructuredBuffer<uint> output; };
ParameterBlock<Resources> resources;
[shader("raygeneration")]
void raygenMain(uniform Params params)
{
    resources.output[params.index] += params.value;
}
