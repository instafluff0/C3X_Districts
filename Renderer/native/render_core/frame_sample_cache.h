#pragma once
#include <vector>
#include <cstdint>
#include <limits>

namespace c3x_renderer { namespace render_core {
// Exact sample keys. A frame pins the union of its consumers before either
// pass draws, so another sample cannot overwrite a borrowed result. Overflow
// uses the caller's scratch path; it never evicts a current-frame borrower.
template<class Key,class Value,unsigned Capacity>class FrameSampleCache {
public:
    struct Entry {Key key;Value value;std::uint64_t used=0;bool prepared=false,valid=false;};
    static constexpr unsigned unavailable=std::numeric_limits<unsigned>::max();
private:
    std::vector<Entry> entries;
    std::uint64_t frame=0;
public:
    void begin(){if(++frame==0){entries.clear();frame=1;}}
    unsigned select(Key const& key){
        for(unsigned i=0;i<entries.size();++i)if(entries[i].key==key){entries[i].used=frame;return i;}
        unsigned index=unavailable;
        if(entries.size()<Capacity){index=unsigned(entries.size());entries.emplace_back();}
        else for(unsigned i=0;i<entries.size();++i)
            if(entries[i].used!=frame && (index==unavailable || entries[i].used<entries[index].used))index=i;
        if(index==unavailable)return index;
        auto& entry=entries[index];entry.key=key;entry.used=frame;entry.prepared=entry.valid=false;
        return index;
    }
    Entry& operator[](unsigned index){return entries[index];}
    Entry const& operator[](unsigned index)const{return entries[index];}
    unsigned size()const{return unsigned(entries.size());}
};
} }
