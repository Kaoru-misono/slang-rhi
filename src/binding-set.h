#pragma once

#include <slang-rhi.h>

#include "core/common.h"
#include "core/struct-holder.h"

#include "reference.h"
#include "device-child.h"
#include "rhi-shared-fwd.h"

#include <vector>

namespace rhi {

/// Reflected detail recorded on a layout entry when a pipeline layout validates this layout against
/// a program. A set layout can be created before any pipeline layout refers to it, so both members
/// keep their unknown value until then.
struct BindingSetLayoutEntryReflection
{
    /// Element stride of a StructuredBuffer/RWStructuredBuffer entry, 0 while unknown.
    uint32_t structuredBufferStride = 0;
    /// Element format class of a Buffer/RWBuffer entry, meaningful only when `formatKindKnown`.
    FormatKind formatKind = FormatKind::Integer;
    bool formatKindKnown = false;
};

class BindingSetLayout : public IBindingSetLayout, public DeviceChild
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    IBindingSetLayout* getInterface(const Guid& guid);

public:
    BindingSetLayout(Device* device, const BindingSetLayoutDesc& desc);
    virtual ~BindingSetLayout() override;

    /// Builds the native layout from `m_desc`.
    virtual Result initNative() = 0;

    /// Number of individual bindings, counting every element of an array entry.
    uint32_t getBindingCount() const { return m_bindingCount; }
    /// Index of the first binding of `entryIndex` within the flattened binding space.
    uint32_t getBindingOffset(uint32_t entryIndex) const { return m_bindingOffsets[entryIndex]; }

    // IBindingSetLayout implementation
    virtual SLANG_NO_THROW const BindingSetLayoutDesc& SLANG_MCALL getDesc() override { return m_desc; }

public:
    BindingSetLayoutDesc m_desc;
    StructHolder m_descHolder;
    std::vector<uint32_t> m_bindingOffsets;
    std::vector<BindingSetLayoutEntryReflection> m_reflection;
    uint32_t m_bindingCount = 0;
};

/// One resolved binding of a set. Exactly one of the resource references is set, matching the kind
/// of the layout entry it belongs to.
struct BindingSetBinding
{
    RefPtr<Sampler> sampler;
    RefPtr<TextureView> textureView;
    RefPtr<Buffer> buffer;
    BufferRange bufferRange = kEntireBuffer;
    RefPtr<AccelerationStructure> accelerationStructure;
};

class BindingSet : public IBindingSet, public DeviceChild
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    IBindingSet* getInterface(const Guid& guid);

public:
    BindingSet(Device* device, BindingSetLayout* layout);
    virtual ~BindingSet() override;

    /// Resolves `desc` against the layout and precomputes the access list. Reports the first
    /// problem through the device callback.
    Result init(const BindingSetDesc& desc);

    /// Writes the native descriptors for the bindings `init` resolved.
    virtual Result initNative() = 0;

    // IBindingSet implementation
    virtual SLANG_NO_THROW IBindingSetLayout* SLANG_MCALL getLayout() override { return m_layout; }

public:
    RefPtr<BindingSetLayout> m_layout;
    /// One entry per binding of the layout, in flattened binding order.
    std::vector<BindingSetBinding> m_bindings;
    /// The states the bound resources must be in, derived from the layout kinds. The resources are
    /// kept alive by m_bindings, so the raw pointers stay valid for the lifetime of the set.
    std::vector<ResourceAccess> m_accesses;
    StructHolder m_descHolder;
    const char* m_label = nullptr;
};

/// State a binding of `kind` requires. Samplers and acceleration structures need no transition and
/// map to ResourceState::Undefined.
ResourceState getBindingKindResourceState(BindingKind kind);

/// True when `kind` binds a sampler, a texture view, a buffer or an acceleration structure.
bool isSamplerBindingKind(BindingKind kind);
bool isTextureBindingKind(BindingKind kind);
bool isBufferBindingKind(BindingKind kind);

} // namespace rhi
