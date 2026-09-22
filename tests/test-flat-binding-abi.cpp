#include "testing.h"
#include "flat-binding-test-data.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

using namespace rhi;
using namespace rhi::testing;

namespace {
using flat::Nested;
using flat::Params;

auto createSession(IDevice* device, bool inlineConstants) -> ComPtr<slang::ISession>
{
    auto deviceSession = device->getSlangSession();
    auto* globalSession = deviceSession->getGlobalSession();
    REQUIRE(std::string_view(globalSession->getBuildTagString()).find("2026.17.1") != std::string_view::npos);

    slang::TargetDesc target{};
    target.format = device->getDeviceType() == DeviceType::Vulkan ? SLANG_SPIRV : SLANG_DXIL;
    if (target.format == SLANG_DXIL)
    {
        target.profile = globalSession->findProfile("sm_6_0");
    }
    target.forceGLSLScalarBufferLayout = true;
    slang::CompilerOptionEntry options[2]{};
    options[0].name = slang::CompilerOptionName::EmitSpirvDirectly;
    options[0].value.kind = slang::CompilerOptionValueKind::Int;
    options[0].value.intValue0 = 1;
    options[1].name = slang::CompilerOptionName::MatrixLayoutRow;
    options[1].value.kind = slang::CompilerOptionValueKind::Int;
    options[1].value.intValue0 = 1;
    slang::PreprocessorMacroDesc macro{"INLINE_CONSTANTS", inlineConstants ? "1" : "0"};
    auto searchPaths = getSlangSearchPaths();
    slang::SessionDesc desc{};
    desc.targets = &target;
    desc.targetCount = 1;
    desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    desc.searchPaths = searchPaths.data();
    desc.searchPathCount = searchPaths.size();
    desc.preprocessorMacros = &macro;
    desc.preprocessorMacroCount = 1;
    desc.compilerOptionEntries = options;
    desc.compilerOptionEntryCount = std::size(options);
    ComPtr<slang::ISession> session;
    REQUIRE_CALL(globalSession->createSession(desc, session.writeRef()));
    return session;
}

auto field(slang::TypeLayoutReflection* type, const char* name) -> slang::VariableLayoutReflection*
{
    const auto index = type->findFieldIndexByName(name);
    REQUIRE(index >= 0);
    return type->getFieldByIndex(static_cast<unsigned int>(index));
}

auto checkConstantSchema(slang::TypeLayoutReflection* reflected) -> void
{
    std::string diagnostic;
    REQUIRE_CALL(validateConstantLayout(reflected, flat::paramsSchema(), diagnostic));
    CHECK(diagnostic.empty());

    auto expectRejected = [&](const ConstantTypeDesc& schema, const char* path, const char* reason)
    {
        CHECK(SLANG_FAILED(validateConstantLayout(reflected, schema, diagnostic)));
        CHECK(diagnostic.find(path) != std::string::npos);
        CHECK(diagnostic.find(reason) != std::string::npos);
    };
    auto schema = flat::paramsSchema();
    schema.size += 16;
    expectRejected(schema, "constants", "size");
    schema = flat::paramsSchema();
    schema.alignment = 4;
    expectRejected(schema, "constants", "alignment");
    schema.alignment = 256;
    expectRejected(schema, "constants", "alignment");

    // Every mismatch keeps the total size unchanged: sizeof alone cannot validate ABI.
    std::vector<ConstantFieldDesc> fields(schema.fields, schema.fields + schema.fieldCount);
    schema = flat::paramsSchema();
    schema.fields = fields.data();
    schema.fieldCount = uint32_t(fields.size());
    auto original = fields;
    fields[0].offset = 4;
    expectRejected(schema, "constants.direction", "offset");
    fields = original;
    auto changed = *fields[1].type;
    changed.scalar = slang::TypeReflection::ScalarType::UInt32;
    fields[1].type = &changed;
    expectRejected(schema, "constants.scale", "32-bit");
    fields = original;
    changed = *fields[0].type;
    changed.alignment = 16;
    fields[0].type = &changed;
    expectRejected(schema, "constants.direction", "alignment");
    fields = original;
    changed = *fields[2].type;
    changed.elementStride = 32;
    fields[2].type = &changed;
    expectRejected(schema, "constants.samples", "stride");
    changed = *original[2].type;
    changed.elementCount = 1;
    expectRejected(schema, "constants.samples", "count");
    fields = original;
    changed = *fields[3].type;
    changed.matrixLayout = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
    fields[3].type = &changed;
    expectRejected(schema, "constants.matrix", "row/column");
    fields = original;
    auto nested = *fields[4].type;
    std::vector<ConstantFieldDesc> nestedFields(nested.fields, nested.fields + nested.fieldCount);
    nestedFields[0].name = "missingTag";
    nested.fields = nestedFields.data();
    fields[4].type = &nested;
    expectRejected(schema, "constants.nested.missingTag", "not found");
    fields = original;
    fields[1].name = "direction";
    expectRejected(schema, "constants.direction", "duplicate");
    fields = original;
    fields[1].offset = 8;
    expectRejected(schema, "constants.scale", "overlapping");
    fields = original;
    fields[4].offset = SIZE_MAX;
    expectRejected(schema, "constants.nested", "beyond");
    fields = original;
    schema.fieldCount = uint32_t(fields.size() - 1);
    expectRejected(schema, "constants", "field count");
    REQUIRE_CALL(validateConstantLayout(reflected, flat::paramsSchema(), diagnostic));
    CHECK(diagnostic.empty());
}

auto checkReflection(slang::ProgramLayout* program, DeviceType target, bool inlineConstants, bool graphics) -> void
{
    REQUIRE(program != nullptr);
    CHECK(program->getEntryPointCount() == (graphics ? 2u : 1u));
    auto* globals = program->getGlobalParamsVarLayout();
    REQUIRE(globals != nullptr);
    auto* scope = globals->getTypeLayout();
    REQUIRE(scope != nullptr);
    // No loose ordinary globals: the only constant storage is the explicit parameters block.
    REQUIRE(scope->getKind() == slang::TypeReflection::Kind::Struct);
    CHECK(scope->getSize() == 0);
    auto* parameters = field(scope, "parameters");
    const auto category = inlineConstants                ? slang::ParameterCategory::PushConstantBuffer
                          : target == DeviceType::Vulkan ? slang::ParameterCategory::DescriptorTableSlot
                                                         : slang::ParameterCategory::ConstantBuffer;
    CHECK(parameters->getCategory() == category);
    CHECK(
        globals->getOffset(category) + parameters->getOffset(category) ==
        (!inlineConstants && target == DeviceType::Vulkan ? 1u : 0u)
    );
    CHECK(globals->getBindingSpace(category) + parameters->getBindingSpace(category) == 0);
    if (inlineConstants)
    {
        CHECK(parameters->getTypeLayout()->getSize(slang::ParameterCategory::DescriptorTableSlot) == 0);
    }

    auto* block = parameters->getTypeLayout();
    REQUIRE(block->getKind() == slang::TypeReflection::Kind::ConstantBuffer);
    REQUIRE(block->getContainerVarLayout() != nullptr);
    REQUIRE(block->getElementVarLayout() != nullptr);
    CHECK(block->getContainerVarLayout()->getOffset(category) == 0);
    CHECK(block->getElementVarLayout()->getOffset() == 0);
    auto* params = block->getElementVarLayout()->getTypeLayout();
    REQUIRE(params != nullptr);
    checkConstantSchema(params);
    CHECK(params->getSize() == sizeof(Params));
    CHECK(params->getAlignment(SLANG_PARAMETER_CATEGORY_UNIFORM) <= alignof(Params));
    CHECK(field(params, "direction")->getOffset() == offsetof(Params, direction));
    CHECK(field(params, "scale")->getOffset() == offsetof(Params, scale));
    CHECK(field(params, "samples")->getOffset() == offsetof(Params, samples));
    CHECK(field(params, "matrix")->getOffset() == offsetof(Params, matrix));
    CHECK(field(params, "nested")->getOffset() == offsetof(Params, nested));

    auto* samples = field(params, "samples")->getTypeLayout();
    CHECK(samples->getElementCount() == 2);
    CHECK(samples->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM) == sizeof(Params::samples[0]));
    CHECK(samples->getSize() == sizeof(Params::samples));
    auto* matrix = field(params, "matrix")->getTypeLayout();
    CHECK(matrix->getSize() == sizeof(Params::matrix));
    CHECK(matrix->getRowCount() == 4);
    CHECK(matrix->getColumnCount() == 4);
    CHECK(matrix->getMatrixLayoutMode() == SLANG_MATRIX_LAYOUT_ROW_MAJOR);
    auto* nested = field(params, "nested")->getTypeLayout();
    CHECK(nested->getSize() == sizeof(Nested));
    CHECK(field(nested, "tag")->getOffset() == offsetof(Nested, tag));
    CHECK(field(nested, "weights")->getOffset() == offsetof(Nested, weights));

    auto* output = field(scope, "output");
    const auto outputCategory = target == DeviceType::Vulkan ? slang::ParameterCategory::DescriptorTableSlot
                                                             : slang::ParameterCategory::UnorderedAccess;
    CHECK(globals->getOffset(outputCategory) + output->getOffset(outputCategory) == 0);
    CHECK(globals->getBindingSpace(outputCategory) + output->getBindingSpace(outputCategory) == 0);
}

auto bindParameters(IShaderObject* root, const Params& params, IBuffer* output) -> void
{
    REQUIRE(root != nullptr);
    ShaderCursor cursor(root);
    ShaderCursor data;
    REQUIRE_CALL(cursor["parameters"].getDereferenced(data));
    REQUIRE_CALL(data.setData(params));
    REQUIRE_CALL(cursor["output"].setBinding(output));
}

// This is an ABI/GPU baseline through ShaderObject, not a root-CBV or dynamic-UBO implementation test.
auto runBaseline(IDevice* device, bool inlineConstants) -> void
{
    auto session = createSession(device, inlineConstants);
    const Params params{
        {1.f, 2.f, 3.f},
        2.f,
        {{1.f, 2.f, 3.f, 4.f}, {-2.f, 1.f, 0.f, 3.f}},
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}, {13.f, 14.f, 15.f, 16.f}},
        {17u, {5.f, 7.f, 11.f}}
    };
    const auto expectedCompute = makeArray<float>(37.f, 81.f, 127.f, 168.f, 19.f, 31.f, 45.f, 54.f);
    const auto expectedPixel = makeArray<float>(56.f, 112.f, 172.f, 222.f);

    BufferDesc outputDesc{};
    outputDesc.size = sizeof(float) * expectedCompute.size();
    outputDesc.elementSize = sizeof(float) * 4;
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    outputDesc.defaultState = ResourceState::UnorderedAccess;
    auto output = device->createBuffer(outputDesc);
    REQUIRE(output != nullptr);
    auto queue = device->getQueue(QueueType::Graphics);

    ComPtr<IShaderProgram> computeProgram;
    slang::ProgramLayout* computeReflection = nullptr;
    REQUIRE_CALL(loadAndLinkProgram(
        device,
        session,
        "test-flat-binding-abi",
        "csMain",
        computeProgram.writeRef(),
        &computeReflection
    ));
    checkReflection(computeReflection, device->getDeviceType(), inlineConstants, false);
    ComputePipelineDesc computeDesc{};
    computeDesc.program = computeProgram;
    auto computePipeline = device->createComputePipeline(computeDesc);
    REQUIRE(computePipeline != nullptr);
    {
        auto encoder = queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        bindParameters(pass->bindPipeline(computePipeline), params, output);
        pass->dispatchCompute(2, 1, 1);
        pass->end();
        auto commands = encoder->finish();
        REQUIRE(commands != nullptr);
        REQUIRE_CALL(queue->submit(commands));
        REQUIRE_CALL(queue->waitOnHost());
    }
    compareComputeResult(device, output, expectedCompute);

    ComPtr<IShaderProgram> renderProgram;
    slang::ProgramLayout* renderReflection = nullptr;
    REQUIRE_CALL(loadAndLinkProgram(
        device,
        session,
        "test-flat-binding-abi",
        {"vsMain", "psMain"},
        renderProgram.writeRef(),
        &renderReflection
    ));
    checkReflection(renderReflection, device->getDeviceType(), inlineConstants, true);
    ColorTargetDesc target{};
    target.format = Format::RGBA32Float;
    RenderPipelineDesc renderDesc{};
    renderDesc.program = renderProgram;
    renderDesc.targets = &target;
    renderDesc.targetCount = 1;
    renderDesc.depthStencil.depthTestEnable = false;
    renderDesc.depthStencil.depthWriteEnable = false;
    renderDesc.rasterizer.cullMode = CullMode::None;
    auto renderPipeline = device->createRenderPipeline(renderDesc);
    REQUIRE(renderPipeline != nullptr);
    TextureDesc textureDesc{};
    textureDesc.type = TextureType::Texture2D;
    textureDesc.size = {2, 2, 1};
    textureDesc.format = Format::RGBA32Float;
    textureDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    textureDesc.defaultState = ResourceState::RenderTarget;
    auto texture = device->createTexture(textureDesc);
    REQUIRE(texture != nullptr);
    auto view = device->createTextureView(texture, {});
    REQUIRE(view != nullptr);
    {
        auto encoder = queue->createCommandEncoder();
        RenderPassColorAttachment attachment{};
        attachment.view = view;
        attachment.loadOp = LoadOp::Clear;
        attachment.storeOp = StoreOp::Store;
        RenderPassDesc passDesc{};
        passDesc.colorAttachments = &attachment;
        passDesc.colorAttachmentCount = 1;
        auto pass = encoder->beginRenderPass(passDesc);
        bindParameters(pass->bindPipeline(renderPipeline), params, output);
        RenderState state{};
        state.viewports[0] = Viewport::fromSize(2, 2);
        state.viewportCount = 1;
        state.scissorRects[0] = ScissorRect::fromSize(2, 2);
        state.scissorRectCount = 1;
        pass->setRenderState(state);
        DrawArguments draw{};
        draw.vertexCount = 3;
        pass->draw(draw);
        pass->end();
        auto commands = encoder->finish();
        REQUIRE(commands != nullptr);
        REQUIRE_CALL(queue->submit(commands));
        REQUIRE_CALL(queue->waitOnHost());
    }
    ComPtr<ISlangBlob> pixels;
    SubresourceLayout layout{};
    REQUIRE_CALL(device->readTexture(texture, 0, 0, pixels.writeRef(), &layout));
    REQUIRE(layout.rowPitch >= sizeof(float) * 8);
    REQUIRE(pixels->getBufferSize() >= layout.rowPitch + sizeof(float) * 8);
    for (size_t y = 0; y < 2; ++y)
    {
        for (size_t x = 0; x < 2; ++x)
        {
            std::array<float, 4> pixel{};
            const auto* source = static_cast<const std::byte*>(pixels->getBufferPointer());
            std::memcpy(pixel.data(), source + y * layout.rowPitch + x * sizeof(pixel), sizeof(pixel));
            compareResultFuzzy(pixel.data(), expectedPixel.data(), pixel.size());
        }
    }
}
} // namespace

GPU_TEST_CASE("flat-binding-abi-global-buffered", D3D12 | Vulkan)
{
    runBaseline(device, false);
}

GPU_TEST_CASE("flat-binding-abi-global-inline", Vulkan)
{
    runBaseline(device, true);
}
