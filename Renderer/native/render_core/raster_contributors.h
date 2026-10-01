#pragma once
#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// Retained pixels depend on contributing content, not the camera's entire
// selection. Proofs contain no GPU resources. Admission is bounded; rejection
// makes the caller draw again rather than silently omit a dependency.
template<class Proof,std::size_t KeyWords=14> struct RasterContributors {
    using Key=std::array<std::uint64_t,KeyWords>;
    struct Hash {std::size_t operator()(Key const& key)const{
        std::uint64_t h=14695981039346656037ull;
        for(auto value:key){h^=value;h*=1099511628211ull;}return std::size_t(h);
    }};
    std::unordered_set<Key,Hash> draws;
    std::unordered_map<std::uint64_t,std::shared_ptr<Proof const>> proofs;
    std::unordered_map<std::uint64_t,std::uint64_t> visibility;
    static constexpr std::size_t limit=16u*1024u*1024u;
    bool complete=true;
    std::size_t bytes()const{return draws.size()*(sizeof(Key)+48)+proofs.size()*64+visibility.size()*64+
        (draws.bucket_count()+proofs.bucket_count()+visibility.bucket_count())*sizeof(void*);}
    void clear(){draws.clear();proofs.clear();visibility.clear();complete=true;}
    bool contains(Key const& key)const{return draws.count(key)!=0;}
    bool add(Key const& key,std::shared_ptr<Proof const> proof,std::uint64_t tile,std::uint64_t revision){
        if(!complete || !proof || bytes()>limit-1024){complete=false;return false;}
        draws.insert(key);proofs.try_emplace(key[0],std::move(proof));visibility[tile]=revision;
        complete=bytes()<=limit;return complete;
    }
    template<class ContentValid,class VisibilityRevision>bool valid(ContentValid content,VisibilityRevision current)const{
        if(!complete)return false;
        for(auto const& item:proofs)if(!content(*item.second))return false;
        for(auto const& item:visibility)if(current(item.first)!=item.second)return false;
        return true;
    }
};
} }
