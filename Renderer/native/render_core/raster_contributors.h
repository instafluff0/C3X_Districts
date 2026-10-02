#pragma once
#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <new>
#include <chrono>
#include "raster_dependency_revisions.h"

namespace c3x_renderer { namespace render_core {
// Retained pixels depend on contributing content, not the camera's entire
// selection. Proofs contain no GPU resources. Admission is bounded; rejection
// makes the caller draw again rather than silently omit a dependency.
template<class Proof,std::size_t KeyWords=14> struct RasterContributors {
    using Key=std::array<std::uint64_t,KeyWords>;
    using Revisions=RasterDependencyRevisions;
    using ValidationKey=std::array<std::uint64_t,40>;
    struct ValidationCounts {std::uint64_t full=0,content=0,visibility=0,membership=0,regions=0,reused=0,changes=0;
        std::uint64_t proof_registrations=0,dependency_watch_calls=0,source_expansions=0,source_reuses=0;
        double append_ms=0;};
    struct Hash {std::size_t operator()(Key const& key)const{
        std::uint64_t h=14695981039346656037ull;
        for(auto value:key){h^=value;h*=1099511628211ull;}return std::size_t(h);
    }};
    // Slow complete validation marks the existing bounded entries. Counting
    // distinct visits rejects removed contributors without an idle-frame set.
    std::unordered_map<Key,std::uint64_t,Hash> draws;
    std::uint64_t membership_epoch=0;
    std::size_t membership_seen=0;
    std::unordered_map<std::uint64_t,std::shared_ptr<Proof const>> proofs;
    std::unordered_map<std::uint64_t,std::uint64_t> visibility;
    std::unordered_set<Revisions::Key,Revisions::Hash> dependencies;
    // Page input owners are immutable after publication. Retaining their owners
    // makes identity reuse safe until this consumer is cleared, including ABA.
    std::unordered_map<void const*,std::shared_ptr<void const>> dependency_sources;
    mutable ValidationCounts validation_counts;
    ValidationKey validated_key{};
    Revisions::Checkpoint validated_revision{};
    bool dependencies_complete=false,validated=false;
    static constexpr std::size_t limit=16u*1024u*1024u;
    bool complete=true;
    std::size_t bytes()const{return draws.size()*(sizeof(Key)+sizeof(std::uint64_t)+48)+proofs.size()*64+visibility.size()*64+
        dependencies.size()*(sizeof(Revisions::Key)+48)+dependency_sources.size()*80+
        (draws.bucket_count()+proofs.bucket_count()+visibility.bucket_count()+dependencies.bucket_count()+dependency_sources.bucket_count())*sizeof(void*);}
    void clear(){draws.clear();proofs.clear();visibility.clear();dependencies.clear();dependency_sources.clear();complete=true;dependencies_complete=validated=false;membership_epoch=membership_seen=0;}
    bool contains(Key const& key)const{return draws.count(key)!=0;}
    void begin_membership(){
        membership_seen=0;
        if(++membership_epoch==0){for(auto& draw:draws)draw.second=0;membership_epoch=1;}
    }
    bool visit_membership(Key const& key){
        if(!complete || !membership_epoch)return false;
        auto found=draws.find(key);if(found==draws.end())return false;
        if(found->second!=membership_epoch){found->second=membership_epoch;++membership_seen;}
        return true;
    }
    bool exact_membership()const{return complete && membership_epoch && membership_seen==draws.size();}
    bool add(Key const& key,std::shared_ptr<Proof const> proof,std::uint64_t tile,std::uint64_t revision,bool* new_proof=nullptr){
        if(new_proof)*new_proof=false;
        if(!complete || !proof || bytes()>limit-1024){complete=false;return false;}
        validated=false;dependencies_complete=false;
        draws.try_emplace(key,0);bool inserted=proofs.try_emplace(key[0],std::move(proof)).second;visibility[tile]=revision;
        complete=bytes()<=limit;if(inserted)++validation_counts.proof_registrations;
        if(new_proof)*new_proof=complete&&inserted;return complete;
    }
    bool watch(Revisions::Domain domain,std::uint64_t id){
        ++validation_counts.dependency_watch_calls;
        if(!complete || bytes()>limit-1024){complete=false;return false;}
        if(dependencies.insert({domain,id}).second)validated=false;
        complete=bytes()<=limit;return complete;
    }
    template<class Source,class Register>bool watch_source(std::shared_ptr<Source> const& source,Register register_inputs){
        if(!complete || !source){complete=false;return false;}
        if(dependency_sources.count(source.get())){++validation_counts.source_reuses;return true;}
        if(bytes()>limit-1024){complete=false;return false;}
        try {
            dependency_sources.emplace(source.get(),std::shared_ptr<void const>(source));
            if(bytes()>limit){complete=false;return false;}
            ++validation_counts.source_expansions;
            if(!register_inputs(*source)){complete=false;return false;}
        }catch(std::bad_alloc const&){complete=false;return false;}
        return true;
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
