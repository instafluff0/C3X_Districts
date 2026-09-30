#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <atomic>
#include <unordered_map>
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

// Retired generations remain charged until the last selected view releases
// them. The ledger can outlive a device/cache reset; it owns no content itself.
struct ResidentRetirement {
    std::atomic<std::size_t> bytes{0},peak{0};
};
struct ResidentRetirementToken {
    std::shared_ptr<ResidentRetirement> ledger;
    std::size_t bytes=0;
    ResidentRetirementToken()=default;
    ResidentRetirementToken(ResidentRetirementToken const&)=delete;
    ResidentRetirementToken& operator=(ResidentRetirementToken const&)=delete;
    void retire(std::shared_ptr<ResidentRetirement> const& owner,std::size_t charge){
        if(ledger)return;
        ledger=owner;resize(charge);
    }
    void resize(std::size_t charge){
        if(!ledger)return;
        auto current=ledger->bytes.fetch_add(charge-bytes)+charge-bytes;bytes=charge;
        auto prior=ledger->peak.load();while(current>prior && !ledger->peak.compare_exchange_weak(prior,current)){}
    }
    ~ResidentRetirementToken(){if(ledger)ledger->bytes-=bytes;}
};

// One resource lease per immutable generation, never per draw or occurrence.
// Copies share the bounded selection, including FRESH's retained draw records.
class ResidentSelection {
    struct Owners:std::unordered_map<std::uint64_t,std::shared_ptr<void>>{
        ResidentRetirementToken charge;
        std::size_t bytes()const{return sizeof(Owners)+bucket_count()*sizeof(void*)+
            size()*(sizeof(value_type)+32);}
    };
    std::shared_ptr<Owners> owners;
    std::shared_ptr<ResidentRetirement> ledger;
public:
    static constexpr std::size_t generation_limit=16384; // two owners per captured occurrence
    explicit ResidentSelection(std::shared_ptr<ResidentRetirement> budget={}):ledger(std::move(budget)){}
    bool retain(ContentHandle handle,std::shared_ptr<void> owner){
        if(!owner || !handle.generation)return false;
        if(owners && owners->count(handle.generation))return true;
        if(owners && owners->size()==generation_limit)return false;
        if(!owners || owners.use_count()>1){
            auto next=std::make_shared<Owners>();if(owners)next->insert(owners->begin(),owners->end());
            if(ledger)next->charge.retire(ledger,next->bytes());owners=std::move(next);
        }
        owners->try_emplace(handle.generation,std::move(owner));owners->charge.resize(owners->bytes());return true;
    }
    void clear(){owners.reset();}
    std::size_t size()const{return owners?owners->size():0;}
    std::size_t bytes()const{return owners?owners->bytes():0;}
};

// Metadata for the existing cache, with weak identity for immutable resource
// generations. resolve() borrows cache metadata; lease() explicitly retains
// resources only. The caller releases each binding before erasing its metadata.
template<class Content> class ResidentContent {
    struct Slot {Content* value=nullptr;std::uint64_t generation=0;std::size_t next=0;std::weak_ptr<void> owner;};
    std::vector<Slot> slots;
    std::size_t free=0,limit;
    std::uint64_t serial=0;
public:
    explicit ResidentContent(std::size_t capacity):limit(capacity){}
    ContentHandle bind(Content& value,std::shared_ptr<void> const& owner={}) {
        if(serial==~std::uint64_t(0))return {};
        std::size_t index=free;
        if(index==slots.size()){
            if(index==limit)return {};
            slots.push_back({});free=slots.size();
        }else free=slots[index].next;
        slots[index]={&value,++serial,slots.size(),owner};
        return {index,serial};
    }
    std::shared_ptr<void> lease(ContentHandle handle)const{
        return resolve(handle)?slots[handle.slot].owner.lock():std::shared_ptr<void>{};
    }
    Content* resolve(ContentHandle handle) const {
        if(!handle.generation || handle.slot>=slots.size())return nullptr;
        auto const& slot=slots[handle.slot];
        return slot.generation==handle.generation?slot.value:nullptr;
    }
    void release(ContentHandle handle) {
        if(!resolve(handle))return;
        auto& slot=slots[handle.slot];slot.value=nullptr;slot.generation=0;slot.owner.reset();
        slot.next=free;free=handle.slot;
    }
    void clear() {
        std::vector<Slot>().swap(slots);free=0;
        // Preserve serial across device/content resets in this owner.
    }
    std::size_t bytes() const {return slots.capacity()*sizeof(Slot);}
};
} }
