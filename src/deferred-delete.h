#pragma once

#include "device-child.h"

#include <array>
#include <mutex>
#include <vector>

namespace rhi {

class DeferredDeleteQueue
{
public:
    /// One completion value per queue that can retain work: graphics, compute, transfer.
    static constexpr size_t kQueueCount = 3;
    using Completion = std::array<uint64_t, kQueueCount>;

private:
    struct Entry
    {
        DeviceChild* object;
        Completion required;
    };

    mutable std::mutex m_mutex;
    std::vector<Entry> m_entries;

public:
    void add(DeviceChild* object, const Completion& required)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries.emplace_back(Entry{object, required});
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_entries.size();
    }

    void discardAfterDeviceLoss()
    {
        for (;;)
        {
            std::vector<Entry> abandoned;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                abandoned.swap(m_entries);
            }
            if (abandoned.empty())
                return;
            for (const auto& entry : abandoned)
                delete entry.object;
        }
    }

    void collect(const Completion& completed)
    {
        for (;;)
        {
            std::vector<DeviceChild*> ready;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                size_t retained = 0;
                for (const auto& entry : m_entries)
                {
                    bool finished = true;
                    for (size_t i = 0; i < completed.size(); ++i)
                    {
                        finished &= entry.required[i] <= completed[i];
                    }
                    if (finished)
                    {
                        ready.emplace_back(entry.object);
                    }
                    else
                    {
                        m_entries[retained++] = entry;
                    }
                }
                m_entries.resize(retained);
            }
            if (ready.empty())
                return;
            // Destructors may enqueue dependencies; never run them under the list lock.
            for (auto* object : ready)
            {
                delete object;
            }
        }
    }
};

} // namespace rhi
