#pragma once

#include "testing.h"
#include <initializer_list>
#include <string>

namespace rhi::testing::trivialCompute {

struct Pipeline
{
    ComPtr<IBindingSetLayout> setLayout;
    ComPtr<IPipelineLayout> layout;
    ComPtr<IComputePipeline> pipeline;
};

inline Pipeline createPipeline(IDevice* device, const char* entryPoint = "computeMain")
{
    REQUIRE(entryPoint != nullptr);
    const bool compute = std::string(entryPoint) == "computeMain";
    const bool read = std::string(entryPoint) == "readTexture";
    REQUIRE((compute || read || std::string(entryPoint) == "writeTexture"));
    const std::string source = std::string("#define TRIVIAL_ENTRY_") + entryPoint +
        " 1\n#include \"test-compute-trivial.slang\"\n";
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadComputeProgramFromSource(device, source, program.writeRef()));

    const BindingSetLayoutEntry computeEntries[] = {{BindingKind::RWStructuredBuffer, 1, "buffer"}};
    const BindingSetLayoutEntry readEntries[] = {
        {BindingKind::Texture2D, 1, "inputTexture"},
        {BindingKind::RWStructuredBuffer, 1, "outputBuffer"},
    };
    const BindingSetLayoutEntry writeEntries[] = {{BindingKind::RWTexture2D, 1, "outputTexture"}};
    BindingSetLayoutDesc setDesc{};
    setDesc.entries = compute ? computeEntries : read ? readEntries : writeEntries;
    setDesc.entryCount = read ? 2 : 1;
    Pipeline result;
    REQUIRE_CALL(device->createBindingSetLayout(setDesc, result.setLayout.writeRef()));
    PipelineLayoutSetDesc set{"resources", result.setLayout};
    PipelineLayoutDesc layoutDesc{};
    layoutDesc.program = program;
    layoutDesc.sets = &set;
    layoutDesc.setCount = 1;
    layoutDesc.constantsSize = compute ? 4 * sizeof(float) : 0;
    REQUIRE_CALL(device->createPipelineLayout(layoutDesc, result.layout.writeRef()));
    REQUIRE(result.layout->getConstantsSize() == layoutDesc.constantsSize);
    ComputePipelineDesc pipelineDesc{};
    pipelineDesc.program = program;
    pipelineDesc.layout = result.layout;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, result.pipeline.writeRef()));
    return result;
}

inline void bind(
    IDevice* device,
    IComputePassEncoder* encoder,
    const Pipeline& pipeline,
    std::initializer_list<BindingSetEntry> entries,
    float value = 0.f
)
{
    BindingSetDesc setDesc{};
    setDesc.layout = pipeline.setLayout;
    setDesc.entries = entries.begin();
    setDesc.entryCount = uint32_t(entries.size());
    ComPtr<IBindingSet> set;
    REQUIRE_CALL(device->createBindingSet(setDesc, set.writeRef()));
    const float constants[4] = {value, 0.f, 0.f, 0.f};
    IBindingSet* sets[] = {set};
    FlatBindingDesc binding{};
    binding.constantsSize = pipeline.layout->getConstantsSize();
    REQUIRE((binding.constantsSize == 0 || binding.constantsSize == sizeof(constants)));
    binding.constants = binding.constantsSize ? constants : nullptr;
    binding.sets = sets;
    binding.setCount = 1;
    REQUIRE_CALL(encoder->bindPipeline(pipeline.pipeline, binding));
}

inline void bindBuffer(IDevice* device, IComputePassEncoder* encoder, const Pipeline& pipeline, IBuffer* buffer, float value = 0.f)
{
    bind(device, encoder, pipeline, {{.slot = 0, .resource = buffer}}, value);
}

} // namespace rhi::testing::trivialCompute
