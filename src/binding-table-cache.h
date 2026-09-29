#pragma once

#include "shader-object.h"
#include "binding-counters.h"
#include <memory>
#include <unordered_map>

namespace rhi {

// A closure stops at PB edges. Entry points share the root table.
template<typename Visitor>
bool visitBindingClosure(ShaderObject* object, bool root, Visitor& visitor)
{
    if (!visitor(object))
        return false;
    if (!object)
        return true;
    for (uint32_t i = 0; i < object->m_layout->getBindingRangeCount(); ++i)
    {
        const auto& range = object->m_layout->getBindingRange(i);
        switch (range.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ExistentialValue:
        case slang::BindingType::PushConstant:
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
            if (range.subObjectIndex == uint32_t(-1))
                break;
            for (uint32_t j = 0; j < range.count; ++j)
                if (range.subObjectIndex + j < object->m_objects.size() &&
                    !visitBindingClosure(object->m_objects[range.subObjectIndex + j], false, visitor))
                    return false;
            break;
        default:
            break;
        }
    }
    if (root)
        for (auto& entry : static_cast<RootShaderObject*>(object)->m_entryPoints)
            if (!visitBindingClosure(entry, false, visitor))
                return false;
    return true;
}

struct BindingClosureKey
{
    RefPtr<ShaderObjectLayout> layout;
    std::vector<ShaderObjectID> objects;

    bool matches(ShaderObject* object, bool root) const
    {
        size_t index = 0;
        auto compare = [&](ShaderObject* node)
        {
            ShaderObjectID id = node ? ShaderObjectID{node->m_uid, node->m_version} : ShaderObjectID{0, 0};
            return index < objects.size() && objects[index++] == id;
        };
        return visitBindingClosure(object, root, compare) && index == objects.size();
    }

    void capture(ShaderObject* object, bool root, ShaderObjectLayout* specializedLayout,
                 std::vector<InternalRefPtr<RefObject>>& retained)
    {
        layout = specializedLayout;
        auto append = [&](ShaderObject* node)
        {
            objects.push_back(node ? ShaderObjectID{node->m_uid, node->m_version} : ShaderObjectID{0, 0});
            if (node)
            {
                for (const auto& owner : node->m_descriptorOwners)
                {
                    retained.emplace_back(owner.resource);
                    SLANG_RHI_BINDING_COUNT(trackedRefs, 1);
                }
                for (const auto& slot : node->m_slots)
                {
                    if (slot.resource)
                    {
                        retained.emplace_back(slot.resource);
                        SLANG_RHI_BINDING_COUNT(trackedRefs, 1);
                    }
                    if (slot.resource2)
                    {
                        retained.emplace_back(slot.resource2);
                        SLANG_RHI_BINDING_COUNT(trackedRefs, 1);
                    }
                }
            }
            return true;
        };
        visitBindingClosure(object, root, append);
    }
};

template<typename Entry>
struct BindingTableCache
{
    // Hash is only an accelerator; full closure equality remains authoritative.
    std::unordered_multimap<size_t, std::unique_ptr<Entry>> entries;

    static void hashObject(size_t& hash, ShaderObjectID id)
    {
        hash ^= size_t(id.uid) + size_t(0x9e3779b9u) + (hash << 6) + (hash >> 2);
        hash ^= size_t(id.version) + size_t(0x9e3779b9u) + (hash << 6) + (hash >> 2);
    }

    Entry* lookup(ShaderObject* object, bool root, ShaderObjectLayout* layout)
    {
        size_t hash = reinterpret_cast<size_t>(layout);
        auto append = [&](ShaderObject* node)
        {
            hashObject(hash, node ? ShaderObjectID{node->m_uid, node->m_version} : ShaderObjectID{0, 0});
            return true;
        };
        visitBindingClosure(object, root, append);
        auto range = entries.equal_range(hash);
        for (auto it = range.first; it != range.second; ++it)
            if (it->second->key.layout.get() == layout && it->second->key.matches(object, root))
                return it->second.get();
        return nullptr;
    }
    void publish(std::unique_ptr<Entry> entry)
    {
        size_t hash = reinterpret_cast<size_t>(entry->key.layout.get());
        for (auto id : entry->key.objects)
            hashObject(hash, id);
        entries.emplace(hash, std::move(entry));
    }
    void reset() { entries.clear(); }
};

} // namespace rhi
