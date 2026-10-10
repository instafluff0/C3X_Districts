#pragma once
#include "resident_content.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Non-owning eviction order for the existing GPU cache. One sorted pass replaces
// a full cache scan for every allocation reclaimed during destination streaming.
// Generations and current-frame pins are checked again before every eviction.
// A shared component (one region's natural ground) ranks as recently used as
// its newest resident dependent and, on a tie, after it: evicting it first
// invalidated every dependent tile still resident, which then had to be
// rebuilt (performance review, section 55).
class ResidencyCandidates {
    struct Candidate {ContentHandle handle;std::uint64_t used,order;bool animated,component;};
    std::vector<Candidate> candidates;
    std::size_t cursor=0;
    std::uint64_t epoch=0;
public:
    std::uint64_t rebuilds=0,examined=0;
    void clear(){std::vector<Candidate>().swap(candidates);cursor=0;epoch=0;}
    std::size_t bytes()const{return candidates.capacity()*sizeof(Candidate);}
    template<class Cache,class Owner,class Priority>
    ContentHandle next(Cache const& cache,Owner const& owner,std::uint64_t current,Priority priority){
        return next(cache,owner,current,priority,[](auto const&){return ContentHandle{};});
    }
    // dependency(value): the shared component this value draws with, if any.
    template<class Cache,class Owner,class Priority,class Dependency>
    ContentHandle next(Cache const& cache,Owner const& owner,std::uint64_t current,Priority priority,Dependency dependency){
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
            std::unordered_map<std::size_t,std::pair<std::uint64_t,std::uint64_t>> held; // slot -> generation, newest dependent use
            for(auto const& entry:cache){auto const& value=entry.second;
                auto shared=dependency(value);if(!shared.generation)continue;
                auto& newest=held[shared.slot];
                if(newest.first!=shared.generation)newest={shared.generation,0};
                newest.second=std::max(newest.second,value.last_used);}
            for(auto const& entry:cache){auto const& value=entry.second;
                if(!value.binding.generation || value.last_used==current)continue;
                auto found=held.find(value.binding.slot);
                bool component=found!=held.end() && found->second.first==value.binding.generation;
                if(component && found->second.second==current)continue; // a dependent is in this frame
                candidates.push_back({value.binding,value.last_used,
                    component?std::max(value.last_used,found->second.second):value.last_used,bool(priority(value)),component});}
            std::stable_sort(candidates.begin(),candidates.end(),[](auto const& a,auto const& b){
                return a.animated!=b.animated?a.animated<b.animated:a.order!=b.order?a.order<b.order:a.component<b.component;});
            ++rebuilds;
        }
        return {};
    }
};
}}
