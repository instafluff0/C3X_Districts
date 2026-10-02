#pragma once
#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include "raster_dependency_revisions.h"

namespace c3x_renderer { namespace render_core {
// Retained pixels depend on contributing content, not the camera's entire
// selection. Proofs contain no GPU resources. Admission is bounded; rejection
// makes the caller draw again rather than silently omit a dependency.
template<class Proof,std::size_t KeyWords=14> struct RasterContributors {
    using Key=std::array<std::uint64_t,KeyWords>;
    using Revisions=RasterDependencyRevisions;
    using ValidationKey=std::array<std::uint64_t,40>;
    struct ValidationCounts {std::uint64_t full=0,content=0,visibility=0,membership=0,regions=0,reused=0,changes=0;};
    struct Hash {std::size_t operator()(Key const& key)const{
        std::uint64_t h=14695981039346656037ull;
        for(auto value:key){h^=value;h*=1099511628211ull;}return std::size_t(h);
    }};
    std::unordered_set<Key,Hash> draws;
    std::unordered_map<std::uint64_t,std::shared_ptr<Proof const>> proofs;
    std::unordered_map<std::uint64_t,std::uint64_t> visibility;
    std::unordered_set<Revisions::Key,Revisions::Hash> dependencies;
    mutable ValidationCounts validation_counts;
    ValidationKey validated_key{};
    Revisions::Checkpoint validated_revision{};
    bool dependencies_complete=false,validated=false;
    static constexpr std::size_t limit=16u*1024u*1024u;
    bool complete=true;
    std::size_t bytes()const{return draws.size()*(sizeof(Key)+48)+proofs.size()*64+visibility.size()*64+
        dependencies.size()*(sizeof(Revisions::Key)+48)+
        (draws.bucket_count()+proofs.bucket_count()+visibility.bucket_count()+dependencies.bucket_count())*sizeof(void*);}
    void clear(){draws.clear();proofs.clear();visibility.clear();dependencies.clear();complete=true;dependencies_complete=validated=false;}
    bool contains(Key const& key)const{return draws.count(key)!=0;}
    bool add(Key const& key,std::shared_ptr<Proof const> proof,std::uint64_t tile,std::uint64_t revision){
        if(!complete || !proof || bytes()>limit-1024){complete=false;return false;}
        validated=false;dependencies_complete=false;
        draws.insert(key);proofs.try_emplace(key[0],std::move(proof));visibility[tile]=revision;
        complete=bytes()<=limit;return complete;
    }
    bool watch(Revisions::Domain domain,std::uint64_t id){
        if(!complete || bytes()>limit-1024){complete=false;return false;}
        if(dependencies.insert({domain,id}).second)validated=false;
        complete=bytes()<=limit;return complete;
    }
    void finish_dependencies(){dependencies_complete=complete;}
    template<class Validate>bool validate(Revisions const& revisions,ValidationKey const& key,Validate exact){
        ++validation_counts.regions;
        if(complete&&dependencies_complete&&validated&&validated_key==key&&
           revisions.unchanged(validated_revision,dependencies,validation_counts.changes)){
            validated_revision=revisions.checkpoint();++validation_counts.reused;return true;
        }
        validated=false;++validation_counts.full;
        auto before=revisions.checkpoint();bool result=complete&&exact();auto after=revisions.checkpoint();
        if(result&&dependencies_complete&&before.owner==after.owner&&before.sequence==after.sequence){
            validated_key=key;validated_revision=after;validated=true;
        }return result;
    }
    template<class ContentValid,class VisibilityRevision>bool valid(ContentValid content,VisibilityRevision current)const{
        if(!complete)return false;
        for(auto const& item:proofs){++validation_counts.content;if(!content(*item.second))return false;}
        for(auto const& item:visibility){++validation_counts.visibility;if(current(item.first)!=item.second)return false;}
        return true;
    }
};
} }
