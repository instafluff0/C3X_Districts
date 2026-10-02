#pragma once
#include "geometry_draws.h"
#include "shared_instance_submission.h"
#include <set>

namespace c3x_renderer { namespace render_core {
// One bounded submission list for the current fresh consumers. Exact keys
// deduplicate placement requirements, without changing any pass's draw order.
// The caller's current membership lease owns the borrowed immutable sources.
template<class Chunk> class BodyPlacementRequirements {
public:
    using Owner=SharedInstanceSubmission;
    using Draw=GeometryDrawRecord<Chunk>;
    struct Entry {Owner::Key key{};Draw draw;unsigned layer=0,count=0;};
    std::vector<Entry> entries;
    unsigned visits=0,duplicates=0;
    mutable unsigned coverage_probes=0;
    mutable unsigned coverage_reuses=0;
    mutable std::uint64_t coverage_total_probes=0,coverage_total_reuses=0;
private:
    // Sorted indices compare the single stored exact key. Do not duplicate
    // 24-word keys in a second node allocation under the joint allowance.
    struct Order {
        using is_transparent=void;
        std::vector<Entry> const* entries;
        bool operator()(unsigned a,unsigned b)const{return (*entries)[a].key<(*entries)[b].key;}
        bool operator()(unsigned a,Owner::Key const& b)const{return (*entries)[a].key<b;}
        bool operator()(Owner::Key const& a,unsigned b)const{return a<(*entries)[b].key;}
    };
    std::set<unsigned,Order> lookup{Order{&entries}};
    Owner::CpuLease charge;
    mutable std::weak_ptr<Owner::Generation const> covered_generation;
    mutable bool covered_result=false;
public:
    BodyPlacementRequirements()=default;
    BodyPlacementRequirements(BodyPlacementRequirements const&)=delete;
    BodyPlacementRequirements& operator=(BodyPlacementRequirements const&)=delete;
    void clear(){entries.clear();lookup.clear();charge.reset();covered_generation.reset();
        std::vector<Entry>().swap(entries);visits=duplicates=coverage_probes=coverage_reuses=0;}
    std::size_t bytes()const{return charge?charge->bytes():0;}
    template<class Ref>bool add(Owner& owner,unsigned layer,Ref const& draw,Owner::Key const& key){
        ++visits;auto const& mesh=draw.content();
        if(!mesh.instances || mesh.instances->empty())return true;
        auto found=lookup.find(key);
        if(found!=lookup.end()){
            ++duplicates;return entries[*found].count==mesh.instances->size();
        }
        if(entries.size()==Owner::entry_limit || mesh.instances->size()>Owner::record_limit)return false;
        auto capacity=entries.capacity();
        if(entries.size()==capacity)capacity=std::min<std::size_t>(Owner::entry_limit,std::max<std::size_t>(64,capacity*2));
        auto bytes=sizeof(*this)+capacity*sizeof(Entry)+(lookup.size()+1)*(sizeof(unsigned)+64u);
        if(!charge)charge=owner.retain_metadata(bytes);
        else if(!owner.resize_metadata(charge,bytes))return false;
        if(!charge)return false;
        try{
            entries.reserve(capacity);Entry entry;entry.key=key;entry.layer=layer;entry.count=unsigned(mesh.instances->size());
            entry.draw=Draw(mesh);if(draw.occurrence)entry.draw=*draw.occurrence;
            entries.push_back(std::move(entry));lookup.emplace(unsigned(entries.size()-1));covered_generation.reset();return true;
        }catch(...){clear();return false;}
    }
    bool covers(Owner::Generation const& candidate)const{
        for(auto const& entry:entries){++coverage_probes;++coverage_total_probes;
            if(candidate.find(entry.key).count!=entry.count)return false;
        }return true;
    }
    bool covers(Owner::Lease const& candidate)const{
        if(!candidate)return false;
        if(!candidate->complete)return covers(*candidate);
        if(covered_generation.lock()==candidate){++coverage_reuses;++coverage_total_reuses;return covered_result;}
        covered_result=covers(*candidate);covered_generation=candidate;return covered_result;
    }
    template<class Visit>void visit(Visit const& callback)const{
        for(auto const& entry:entries)callback(entry.layer,entry.key,entry.draw,entry.count);
    }
};
} }
