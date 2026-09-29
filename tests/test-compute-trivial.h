#pragma once

#include "testing.h"
#include <initializer_list>
#include <string>

namespace rhi::testing::trivialCompute {

struct ResourceBinding
{
    uint32_t slot;
    ISlangUnknown* resource;
};

struct Pipeline
{
    bool compute = false;
    /// Parameter names by binding slot, shared by both binding models.
    const char* const* names = nullptr;
    ComPtr<IComputePipeline> pipeline;
};

inline Pipeline createPipeline(IDevice* device, const char* entryPoint = "computeMain")
{
    REQUIRE(entryPoint != nullptr);
    const bool compute = std::string(entryPoint) == "computeMain";
    const bool read = std::string(entryPoint) == "readTexture";
    REQUIRE((compute || read || std::string(entryPoint) == "writeTexture"));
    static const char* const computeNames[] = {"buffer"};
    static const char* const readNames[] = {"inputTexture", "outputBuffer"};
    static const char* const writeNames[] = {"outputTexture"};
    Pipeline result;
    result.compute = compute;
    result.names = compute ? computeNames : read ? readNames : writeNames;

    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", entryPoint, program.writeRef()));
    REQUIRE_CALL(device->createComputePipeline({.program = program}, result.pipeline.writeRef()));
    return result;
}

inline rhi::Binding toLegacyBinding(ISlangUnknown* resource)
{
    ComPtr<IBuffer> buffer;
    if (SLANG_SUCCEEDED(resource->queryInterface(IBuffer::getTypeGuid(), (void**)buffer.writeRef())))
        return rhi::Binding(buffer.get());
    ComPtr<ITextureView> view;
    REQUIRE_CALL(resource->queryInterface(ITextureView::getTypeGuid(), (void**)view.writeRef()));
    return rhi::Binding(view.get());
}

inline void bind(
    IDevice* device,
    IComputePassEncoder* encoder,
    const Pipeline& pipeline,
    std::initializer_list<ResourceBinding> entries,
    float value = 0.f
)
{
    auto rootObject = encoder->bindPipeline(pipeline.pipeline);
    REQUIRE(rootObject != nullptr);
    ShaderCursor cursor(rootObject);
    for (const ResourceBinding& entry : entries)
        REQUIRE_CALL(cursor[pipeline.names[entry.slot]].setBinding(toLegacyBinding(entry.resource)));
    if (pipeline.compute)
        REQUIRE_CALL(cursor["value"].setData(value));
}

inline void bindBuffer(IDevice* device, IComputePassEncoder* encoder, const Pipeline& pipeline, IBuffer* buffer, float value = 0.f)
{
    bind(device, encoder, pipeline, {{.slot = 0, .resource = buffer}}, value);
}

} // namespace rhi::testing::trivialCompute
