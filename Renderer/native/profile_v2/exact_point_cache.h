#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace c3x_renderer { namespace profile_v2 {
// Exact float coordinates, with no rounding/interpolation. Compilation-local
// scratch is recycled between tiles; clearing also prevents dependency reuse
// across owners. Capacity is capped at 16,384 slots per table (half occupied).
// At the capacity limit a miss simply evaluates without caching.
template<class Value> class ExactPointCache {
    struct Entry {std::uint64_t key=0;unsigned generation=0;Value value{};};
    std::vector<Entry> entries;
    unsigned generation=1;
    std::size_t count=0;
    std::size_t bucket(std::uint64_t key) const {
        key^=key>>33;key*=0xff51afd7ed558ccdull;key^=key>>33;
        return std::size_t(key)&(entries.size()-1);
    }
    void grow() {
        auto previous=std::move(entries);
        entries.resize(previous.empty()?256:previous.size()*2);
        for(auto const& entry:previous) if(entry.generation==generation) {
            auto i=bucket(entry.key);
            while(entries[i].generation==generation)i=(i+1)&(entries.size()-1);
            entries[i]=entry;
        }
    }
public:
    std::size_t hits=0,misses=0;
    void clear() {
        count=0;
        if(++generation==0) {for(auto& entry:entries)entry.generation=0;generation=1;}
    }
    std::size_t bytes() const {return entries.capacity()*sizeof(Entry);}
    template<class Compute> Value get(float x,float y,Compute compute) {
        std::uint32_t a=0,b=0;
        if(x!=0)std::memcpy(&a,&x,sizeof(a));
        if(y!=0)std::memcpy(&b,&y,sizeof(b));
        std::uint64_t key=(std::uint64_t(a)<<32)|b;
        if(entries.empty() || (count*2>=entries.size() && entries.size()<16384))grow();
        auto i=bucket(key);
        while(entries[i].generation==generation) {
            if(entries[i].key==key) {++hits;return entries[i].value;}
            i=(i+1)&(entries.size()-1);
        }
        ++misses;Value value=compute();
        if(count*2<entries.size()) {entries[i]={key,generation,value};++count;}
        return value;
    }
};
} }
