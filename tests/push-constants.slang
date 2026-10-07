// HLSL reflection keeps no trace of `[[vk::push_constant]]`, so D3D12 finds a program's push-constant block by
// this attribute. Slang only reflects a user attribute whose struct comes from a module.
[__AttributeUsage(_AttributeTargets.Var)]
public struct PushConstantsAttribute {};
