#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Non-owning, owner-scoped identity. Reused slots never revive an old binding.
struct ContentHandle {
    std::size_t slot=0;
    std::uint64_t generation=0;
    bool operator==(ContentHandle const& other) const {
        return slot==other.slot && generation==other.generation;
    }
};

// Metadata for the existing content owner, not another cache or buffer owner.
// The caller registers only stable entries and releases a handle before erasing
// its entry. No handle lookup allocates or extends the content lifetime.
template<class Content> class ResidentContent {
    struct Slot {Content* value=nullptr;std::uint64_t generation=0;std::size_t next=0;};
    std::vector<Slot> slots;
    std::size_t free=0,limit;
    std::uint64_t serial=0;
public:
    explicit ResidentContent(std::size_t capacity):limit(capacity){}
    ContentHandle bind(Content& value) {
        if(serial==~std::uint64_t(0))return {};
        std::size_t index=free;
        if(index==slots.size()){
            if(index==limit)return {};
            slots.push_back({});free=slots.size();
        }else free=slots[index].next;
        slots[index]={&value,++serial,slots.size()};
        return {index,serial};
    }
    Content* resolve(ContentHandle handle) const {
        if(!handle.generation || handle.slot>=slots.size())return nullptr;
        auto const& slot=slots[handle.slot];
        return slot.generation==handle.generation?slot.value:nullptr;
    }
    void release(ContentHandle handle) {
        if(!resolve(handle))return;
        auto& slot=slots[handle.slot];slot.value=nullptr;slot.generation=0;
        slot.next=free;free=handle.slot;
    }
    void clear() {
        std::vector<Slot>().swap(slots);free=0;
        // Preserve serial across device/content resets in this owner.
    }
    std::size_t bytes() const {return slots.capacity()*sizeof(Slot);}
};
} }
