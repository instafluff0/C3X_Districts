#pragma once
#include <vector>
#include <cstdint>
#include <limits>

namespace c3x_renderer { namespace render_core {
// Exact sample keys. A frame pins the union of its consumers before either
// pass draws, so another sample cannot overwrite a borrowed result. Overflow
// uses the caller's scratch path; it never evicts a current-frame borrower.
// Word-vector keys carry a hash so a frame of several hundred part lookups
// compares one integer per entry instead of whole key vectors.
inline std::uint64_t frame_sample_hash(std::vector<std::uint64_t> const& key){
    std::uint64_t h=0x9e3779b97f4a7c15ull^key.size();
    for(auto word:key){h^=word+0x9e3779b97f4a7c15ull+(h<<6)+(h>>2);}
    return h;
}
template<class Key>inline std::uint64_t frame_sample_hash(Key const&){return 0;}
template<class Key,class Value,unsigned Capacity>class FrameSampleCache {
public:
    struct Entry {Key key;Value value;std::uint64_t used=0;bool prepared=false,valid=false;std::uint64_t hash=0;};
    static constexpr unsigned unavailable=std::numeric_limits<unsigned>::max();
private:
    std::vector<Entry> entries;
    std::uint64_t frame=0;
public:
    void begin(){if(++frame==0){entries.clear();frame=1;}}
    unsigned select(Key const& key){
        auto hash=frame_sample_hash(key);
        for(unsigned i=0;i<entries.size();++i)if(entries[i].hash==hash&&entries[i].key==key){entries[i].used=frame;return i;}
        unsigned index=unavailable;
        if(entries.size()<Capacity){index=unsigned(entries.size());entries.emplace_back();}
        else for(unsigned i=0;i<entries.size();++i)
            if(entries[i].used!=frame && (index==unavailable || entries[i].used<entries[index].used))index=i;
        if(index==unavailable)return index;
        auto& entry=entries[index];entry.key=key;entry.hash=hash;entry.used=frame;entry.prepared=entry.valid=false;
        return index;
    }
    Entry& operator[](unsigned index){return entries[index];}
    Entry const& operator[](unsigned index)const{return entries[index];}
    unsigned size()const{return unsigned(entries.size());}
};
} }
