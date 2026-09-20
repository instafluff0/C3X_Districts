#pragma once
#include "resident_content.h"
#include <algorithm>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Non-owning eviction order for the existing GPU cache. One sorted pass replaces
// a full cache scan for every allocation reclaimed during destination streaming.
// Generations and current-frame pins are checked again before every eviction.
class ResidencyCandidates {
    struct Candidate {ContentHandle handle;std::uint64_t used;bool animated;};
    std::vector<Candidate> candidates;
    std::size_t cursor=0;
    std::uint64_t epoch=0;
public:
    std::uint64_t rebuilds=0,examined=0;
    void clear(){std::vector<Candidate>().swap(candidates);cursor=0;epoch=0;}
    std::size_t bytes()const{return candidates.capacity()*sizeof(Candidate);}
    template<class Cache,class Owner,class Priority>
    ContentHandle next(Cache const& cache,Owner const& owner,std::uint64_t current,Priority priority){
        if(epoch!=current){candidates.clear();cursor=0;epoch=current;}
        for(unsigned pass=0;pass<2;++pass){
            while(cursor<candidates.size()){
                auto const candidate=candidates[cursor++];++examined;
                auto value=owner.resolve(candidate.handle);
                if(value && value->last_used!=current && value->last_used==candidate.used &&
                   bool(priority(*value))==candidate.animated)return candidate.handle;
            }
            if(pass)return {};
            candidates.clear();cursor=0;candidates.reserve(cache.size());
            for(auto const& entry:cache){auto const& value=entry.second;
                if(value.binding.generation && value.last_used!=current)
                    candidates.push_back({value.binding,value.last_used,bool(priority(value))});}
            std::stable_sort(candidates.begin(),candidates.end(),[](auto const& a,auto const& b){
                return a.animated!=b.animated?a.animated<b.animated:a.used<b.used;});
            ++rebuilds;
        }
        return {};
    }
};
}}
