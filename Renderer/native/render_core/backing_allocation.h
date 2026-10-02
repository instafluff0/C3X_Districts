#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Physical ranges in the bounded session file. Invalidated generations return
// their space immediately; neighboring holes coalesce and a free tail shrinks.
class BackingAllocation {
    std::map<std::uint64_t,std::uint64_t> holes;
    std::uint64_t extent=0,live=0,limit;
public:
    explicit BackingAllocation(std::uint64_t capacity):limit(capacity){}
    std::uint64_t high_water()const{return extent;}
    std::uint64_t live_bytes()const{return live;}
    std::uint64_t capacity()const{return limit;}
    void clear(){holes.clear();extent=live=0;}
    bool allocate(std::uint64_t bytes,std::uint64_t& offset){
        if(!bytes || bytes>limit-live)return false;
        auto best=holes.end();
        for(auto it=holes.begin();it!=holes.end();++it)
            if(it->second>=bytes && (best==holes.end() || it->second<best->second))best=it;
        if(best!=holes.end()){
            offset=best->first;auto spare=best->second-bytes;holes.erase(best);
            if(spare)holes.emplace(offset+bytes,spare);
        }else{
            if(bytes>limit-extent)return false;
            offset=extent;extent+=bytes;
        }
        live+=bytes;return true;
    }
    void release(std::uint64_t offset,std::uint64_t bytes){
        if(!bytes)return;
        live-=bytes;auto next=holes.lower_bound(offset);
        if(next!=holes.begin()){
            auto previous=std::prev(next);
            if(previous->first+previous->second==offset){offset=previous->first;bytes+=previous->second;holes.erase(previous);}
        }
        if(next!=holes.end() && offset+bytes==next->first){bytes+=next->second;holes.erase(next);}
        if(offset+bytes==extent)extent=offset;
        else holes.emplace(offset,bytes);
    }
    // Rebuild after an in-place compaction, including a partial I/O failure.
    // Live entries are the sole authority; orphaned ranges become reusable.
    bool reset_layout(std::vector<std::pair<std::uint64_t,std::uint64_t>> ranges){
        std::sort(ranges.begin(),ranges.end());
        std::map<std::uint64_t,std::uint64_t> next;std::uint64_t end=0,total=0;
        for(auto const& range:ranges){
            if(!range.second || range.first<end || range.first>limit || range.second>limit-range.first)return false;
            if(range.first>end)next.emplace(end,range.first-end);
            end=range.first+range.second;total+=range.second;
        }
        holes=std::move(next);extent=end;live=total;return true;
    }
};
}}
