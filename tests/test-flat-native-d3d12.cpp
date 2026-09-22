#include "testing.h"
#include "flat-binding-test-data.h"
#include "constant-layout.h"

#if SLANG_RHI_ENABLE_D3D12
#include <d3d12.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

using namespace rhi;
using namespace rhi::testing;

namespace {
using flat::Nested;
using flat::Params;

auto createSession(IDevice* device) -> ComPtr<slang::ISession>
{
    auto deviceSession = device->getSlangSession();
    auto* global = deviceSession->getGlobalSession();
    REQUIRE(std::string_view(global->getBuildTagString()).find("2026.17.1") != std::string_view::npos);
    slang::TargetDesc target{};
    target.format = SLANG_DXIL;
    target.profile = global->findProfile("sm_6_0");
    target.forceGLSLScalarBufferLayout = true;
    slang::CompilerOptionEntry columnMajor{};
    columnMajor.name = slang::CompilerOptionName::MatrixLayoutColumn;
    columnMajor.value.intValue0 = 1;
    slang::PreprocessorMacroDesc macros[]{{"NATIVE_LAYOUT", "1"}, {"INLINE_CONSTANTS", "0"}};
    auto paths = getSlangSearchPaths();
    slang::SessionDesc desc{};
    desc.targets = &target;
    desc.targetCount = 1;
    desc.searchPaths = paths.data();
    desc.searchPathCount = paths.size();
    desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
    desc.preprocessorMacros = macros;
    desc.preprocessorMacroCount = std::size(macros);
    desc.compilerOptionEntries = &columnMajor;
    desc.compilerOptionEntryCount = 1;
    ComPtr<slang::ISession> session;
    REQUIRE_CALL(global->createSession(desc, session.writeRef()));
    return session;
}

auto linkProgram(slang::ISession* session, const std::vector<const char*>& names) -> ComPtr<slang::IComponentType>
{
    ComPtr<ISlangBlob> diagnostics;
    auto* module = session->loadModule("test-flat-binding-abi", diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    REQUIRE(module != nullptr);
    std::vector<ComPtr<slang::IEntryPoint>> entries;
    std::vector<slang::IComponentType*> components{module};
    for (const auto* name : names)
    {
        ComPtr<slang::IEntryPoint> entry;
        REQUIRE_CALL(module->findEntryPointByName(name, entry.writeRef()));
        components.emplace_back(entry.get());
        entries.emplace_back(std::move(entry));
    }
    ComPtr<slang::IComponentType> composite;
    auto result = session->createCompositeComponentType(
        components.data(),
        components.size(),
        composite.writeRef(),
        diagnostics.writeRef()
    );
    diagnoseIfNeeded(diagnostics);
    REQUIRE_CALL(result);
    ComPtr<slang::IComponentType> linked;
    result = composite->link(linked.writeRef(), diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    REQUIRE_CALL(result);

    auto* globals = linked->getLayout()->getGlobalParamsVarLayout();
    auto* type = globals->getTypeLayout();
    const auto index = type->findFieldIndexByName("parameters");
    REQUIRE(index >= 0);
    auto* parameters = type->getFieldByIndex(static_cast<unsigned int>(index));
    CHECK(parameters->getCategory() == slang::ParameterCategory::ConstantBuffer);
    CHECK(parameters->getOffset(slang::ParameterCategory::ConstantBuffer) == 0);
    CHECK(parameters->getBindingSpace(slang::ParameterCategory::ConstantBuffer) == 1);
    CHECK(parameters->getTypeLayout()->getElementTypeLayout()->getSize() == sizeof(Params));
    std::string diagnostic;
    REQUIRE_CALL(validateConstantLayout(parameters->getTypeLayout()->getElementTypeLayout(), diagnostic));
    return linked;
}

auto entryCode(slang::IComponentType* program, SlangInt index) -> ComPtr<ISlangBlob>
{
    ComPtr<ISlangBlob> code;
    ComPtr<ISlangBlob> diagnostics;
    const auto result = program->getEntryPointCode(index, 0, code.writeRef(), diagnostics.writeRef());
    diagnoseIfNeeded(diagnostics);
    REQUIRE_CALL(result);
    REQUIRE(code != nullptr);
    return code;
}

auto createRootSignature(ID3D12Device* device) -> ComPtr<ID3D12RootSignature>
{
    auto module = GetModuleHandleW(L"d3d12.dll");
    REQUIRE(module != nullptr);
    const auto address = GetProcAddress(module, "D3D12SerializeRootSignature");
    REQUIRE(address != nullptr);
    auto serialize = std::bit_cast<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(address);
    D3D12_ROOT_PARAMETER parameters[2]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameters[0].Descriptor = {0, 1};
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters[1].Descriptor = {0, 0};
    parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = static_cast<UINT>(std::size(parameters));
    desc.pParameters = parameters;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> errors;
    const auto result = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.writeRef(), errors.writeRef());
    if (errors)
    {
        INFO(static_cast<const char*>(errors->GetBufferPointer()));
    }
    REQUIRE_CALL(result);
    ComPtr<ID3D12RootSignature> root;
    REQUIRE_CALL(
        device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(root.writeRef()))
    );
    return root;
}

struct Commands
{
    ID3D12RootSignature* root;
    ID3D12PipelineState* pipeline;
    D3D12_GPU_VIRTUAL_ADDRESS constants;
    D3D12_GPU_VIRTUAL_ADDRESS output;
    D3D12_CPU_DESCRIPTOR_HANDLE target;
    bool graphics;
};

auto SLANG_MCALL recordNative(const ExecuteCallbackContext* context, void*, const void* userData, Size size) -> void
{
    REQUIRE(context->nativeHandle.type == NativeHandleType::D3D12GraphicsCommandList);
    REQUIRE(size == sizeof(Commands));
    Commands commands{};
    std::memcpy(&commands, userData, sizeof(commands));
    auto* list = reinterpret_cast<ID3D12GraphicsCommandList*>(context->nativeHandle.value);
    list->SetPipelineState(commands.pipeline);
    if (!commands.graphics)
    {
        list->SetComputeRootSignature(commands.root);
        list->SetComputeRootConstantBufferView(0, commands.constants);
        list->SetComputeRootUnorderedAccessView(1, commands.output);
        list->Dispatch(2, 1, 1);
        return;
    }
    list->SetGraphicsRootSignature(commands.root);
    list->SetGraphicsRootConstantBufferView(0, commands.constants);
    list->SetGraphicsRootUnorderedAccessView(1, commands.output);
    D3D12_VIEWPORT viewport{0.f, 0.f, 2.f, 2.f, 0.f, 1.f};
    D3D12_RECT scissor{0, 0, 2, 2};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &commands.target, FALSE, nullptr);
    const float clear[4]{};
    list->ClearRenderTargetView(commands.target, clear, 0, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
}
} // namespace

GPU_TEST_CASE("flat-native-root-cbv-snapshots", D3D12)
{
    DeviceNativeHandles handles{};
    REQUIRE_CALL(device->getNativeDeviceHandles(&handles));
    REQUIRE(handles.handles[0].type == NativeHandleType::D3D12Device);
    auto* native = reinterpret_cast<ID3D12Device*>(handles.handles[0].value);
    auto root = createRootSignature(native);
    auto session = createSession(device);
    auto computeProgram = linkProgram(session, {"csMain"});
    auto computeCode = entryCode(computeProgram, 0);
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc{};
    computeDesc.pRootSignature = root;
    computeDesc.CS = {computeCode->getBufferPointer(), computeCode->getBufferSize()};
    ComPtr<ID3D12PipelineState> computePipeline;
    REQUIRE_CALL(native->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(computePipeline.writeRef())));

    auto graphicsProgram = linkProgram(session, {"vsMain", "psMain"});
    auto vertexCode = entryCode(graphicsProgram, 0);
    auto fragmentCode = entryCode(graphicsProgram, 1);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC graphicsDesc{};
    graphicsDesc.pRootSignature = root;
    graphicsDesc.VS = {vertexCode->getBufferPointer(), vertexCode->getBufferSize()};
    graphicsDesc.PS = {fragmentCode->getBufferPointer(), fragmentCode->getBufferSize()};
    graphicsDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    graphicsDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    graphicsDesc.RasterizerState.DepthClipEnable = TRUE;
    graphicsDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    graphicsDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    graphicsDesc.DepthStencilState.FrontFace =
        {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};
    graphicsDesc.DepthStencilState.BackFace = graphicsDesc.DepthStencilState.FrontFace;
    auto& blend = graphicsDesc.BlendState.RenderTarget[0];
    blend.SrcBlend = D3D12_BLEND_ONE;
    blend.DestBlend = D3D12_BLEND_ZERO;
    blend.BlendOp = D3D12_BLEND_OP_ADD;
    blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlendAlpha = D3D12_BLEND_ZERO;
    blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    graphicsDesc.SampleMask = std::numeric_limits<UINT>::max();
    graphicsDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    graphicsDesc.NumRenderTargets = 1;
    graphicsDesc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    graphicsDesc.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> graphicsPipeline;
    REQUIRE_CALL(native->CreateGraphicsPipelineState(&graphicsDesc, IID_PPV_ARGS(graphicsPipeline.writeRef())));

    Params params{
        {1.f, 2.f, 3.f},
        2.f,
        {{1.f, 2.f, 3.f, 4.f}, {-2.f, 1.f, 0.f, 3.f}},
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}, {13.f, 14.f, 15.f, 16.f}},
        {17u, {5.f, 7.f, 11.f}}
    };
    constexpr size_t stride = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    std::array<std::byte, 2 * stride> snapshots{};
    std::memcpy(snapshots.data(), &params, sizeof(params));
    params.scale = 3.f;
    params.nested.tag = 19;
    std::memcpy(snapshots.data() + stride, &params, sizeof(params));
    BufferDesc constantDesc{};
    constantDesc.size = snapshots.size();
    constantDesc.memoryType = MemoryType::Upload;
    constantDesc.usage = BufferUsage::ConstantBuffer;
    constantDesc.defaultState = ResourceState::ConstantBuffer;
    auto constants = device->createBuffer(constantDesc, snapshots.data());
    REQUIRE(constants != nullptr);
    snapshots.fill(std::byte{0});
    params = {};
    REQUIRE(constants->getDeviceAddress() % stride == 0);
    BufferDesc outputDesc{};
    outputDesc.size = 16 * sizeof(float);
    outputDesc.elementSize = 4 * sizeof(float);
    outputDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    outputDesc.defaultState = ResourceState::UnorderedAccess;
    auto output = device->createBuffer(outputDesc);
    REQUIRE(output != nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heapDesc.NumDescriptors = 2;
    ComPtr<ID3D12DescriptorHeap> heap;
    REQUIRE_CALL(native->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(heap.writeRef())));
    const auto descriptorStride = native->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    std::array<ComPtr<ITexture>, 2> targets;
    for (size_t i = 0; i < targets.size(); ++i)
    {
        TextureDesc desc{};
        desc.type = TextureType::Texture2D;
        desc.size = {2, 2, 1};
        desc.format = Format::RGBA32Float;
        desc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
        desc.defaultState = ResourceState::RenderTarget;
        targets[i] = device->createTexture(desc);
        REQUIRE(targets[i] != nullptr);
        NativeHandle targetHandle{};
        REQUIRE_CALL(targets[i]->getNativeHandle(&targetHandle));
        REQUIRE(targetHandle.type == NativeHandleType::D3D12Resource);
        auto handle = heap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += i * descriptorStride;
        native->CreateRenderTargetView(reinterpret_cast<ID3D12Resource*>(targetHandle.value), nullptr, handle);
    }

    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();
    // The test owns native PSOs, descriptors and resources until the single submission completes.
    for (size_t i = 0; i < targets.size(); ++i)
    {
        auto target = heap->GetCPUDescriptorHandleForHeapStart();
        target.ptr += i * descriptorStride;
        Commands commands{
            root.get(),
            computePipeline.get(),
            constants->getDeviceAddress() + i * stride,
            output->getDeviceAddress() + i * 8 * sizeof(float),
            target,
            false
        };
        ExecuteCallbackDesc callback{};
        callback.callback = recordNative;
        callback.userData = &commands;
        callback.userDataSize = sizeof(commands);
        encoder->setBufferState(output, ResourceState::UnorderedAccess);
        encoder->executeCallback(callback);
        commands.pipeline = graphicsPipeline;
        commands.graphics = true;
        encoder->setTextureState(targets[i], ResourceState::RenderTarget);
        encoder->executeCallback(callback);
    }
    auto commands = encoder->finish();
    REQUIRE(commands != nullptr);
    REQUIRE_CALL(queue->submit(commands));
    REQUIRE_CALL(queue->waitOnHost());
    compareComputeResult(
        device,
        output,
        makeArray<
            float>(97.f, 111.f, 127.f, 138.f, 49.f, 55.f, 63.f, 66.f, 98.f, 113.f, 130.f, 140.f, 50.f, 57.f, 66.f, 68.f)
    );
    const std::array<std::array<float, 4>, 2> expected{{{146.f, 166.f, 190.f, 204.f}, {148.f, 170.f, 196.f, 208.f}}};
    for (size_t i = 0; i < targets.size(); ++i)
    {
        ComPtr<ISlangBlob> pixels;
        SubresourceLayout layout{};
        REQUIRE_CALL(device->readTexture(targets[i], 0, 0, pixels.writeRef(), &layout));
        REQUIRE(layout.rowPitch >= 8 * sizeof(float));
        REQUIRE(pixels->getBufferSize() >= layout.rowPitch + 8 * sizeof(float));
        for (size_t y = 0; y < 2; ++y)
        {
            for (size_t x = 0; x < 2; ++x)
            {
                std::array<float, 4> pixel{};
                const auto* source = static_cast<const std::byte*>(pixels->getBufferPointer());
                std::memcpy(pixel.data(), source + y * layout.rowPitch + x * sizeof(pixel), sizeof(pixel));
                compareResultFuzzy(pixel.data(), expected[i].data(), pixel.size());
            }
        }
    }
}
#endif
