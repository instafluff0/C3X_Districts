#pragma once
#include "shadow_sampling_grid.h"
#include <vector>
#include <array>
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <functional>
#include <iterator>
#include <cstring>
#include "raster_contributors.h"

namespace c3x_renderer { namespace render_core {
// One registration per active immutable producer. Camera membership and light
// projection do not change its dependencies. Distinct immutable river input
// owners retain their own lifetimes but share exactly equal normalized watches.
template<class Proof> struct ShadowCasterProofs {
    ShadowCasterProofs()=default;
    ShadowCasterProofs(ShadowCasterProofs const&)=delete;
    ShadowCasterProofs& operator=(ShadowCasterProofs const&)=delete;
    ShadowCasterProofs(ShadowCasterProofs&&)=delete;
    ShadowCasterProofs& operator=(ShadowCasterProofs&&)=delete;
    using Raster=RasterContributors<Proof,20>;
    using Key=typename Raster::Key;
    using Hash=typename Raster::Hash;
    using ValidationCounts=typename Raster::ValidationCounts;
    using Revisions=RasterDependencyRevisions;
    static constexpr std::size_t limit=Raster::limit;
    struct Less {bool operator()(Revisions::Key const& a,Revisions::Key const& b)const{
        return a.domain==b.domain?a.id<b.id:unsigned(a.domain)<unsigned(b.domain);}};
    struct Producer {
        std::shared_ptr<Proof const> proof;
        std::vector<Revisions::Key> dependencies;
        std::vector<void const*> sources;
        std::uint64_t tile=0,visibility=0,seen=0;
        bool valid=false;
    };
    struct DependencyList {std::vector<Revisions::Key> keys;std::size_t refs=0;std::uint64_t hash=0;};
    struct Source {std::shared_ptr<void const> owner;DependencyList* dependencies=nullptr;std::size_t refs=0;};
    std::map<std::uint64_t,Producer> producers;
    std::map<void const*,Source> sources;
    std::multimap<std::uint64_t,DependencyList> dependency_lists;
    std::vector<Revisions::Key> source_scratch;
    ValidationCounts validation_counts;
    Revisions::Checkpoint checkpoint{};
    std::array<std::uint64_t,4> context{};
    std::uint64_t epoch=0;
    bool complete=true;
    bool valid_all=false;
    bool registration_open=false;
    std::size_t allocated=sizeof(*this);
    Producer* registering=nullptr;
    Source* expanding=nullptr;
    std::function<bool(std::size_t)> admit;
    std::size_t bytes()const{return allocated;}
    void clear(){producers.clear();sources.clear();dependency_lists.clear();std::vector<Revisions::Key>().swap(source_scratch);allocated=sizeof(*this);checkpoint={};epoch=0;complete=true;valid_all=false;registration_open=false;registering=nullptr;expanding=nullptr;}
    bool reserve_bytes(std::size_t extra){
        if(!complete || extra>limit || allocated>limit-extra || !admit || !admit(allocated+extra)){complete=false;return false;}
        allocated+=extra;return true;
    }
    template<class T>bool append(std::vector<T>& values,T value){
        if(values.size()==values.capacity()){
            auto capacity=std::max(std::size_t(4),values.capacity()*2);
            auto extra=(capacity-values.capacity())*sizeof(T);
            if(!reserve_bytes(extra))return false;
            try{values.reserve(capacity);}catch(std::bad_alloc const&){allocated-=extra;return complete=false;}
        }
        values.push_back(value);return true;
    }
    bool watch(Revisions::Domain domain,std::uint64_t id){
        ++validation_counts.dependency_watch_calls;
        if(!registering)return complete=false;
        return append(expanding?source_scratch:registering->dependencies,Revisions::Key{domain,id});
    }
    static void normalize(std::vector<Revisions::Key>& keys){
        std::sort(keys.begin(),keys.end(),Less{});keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
    }
    static std::uint64_t dependency_hash(std::vector<Revisions::Key> const& keys){
        std::uint64_t hash=14695981039346656037ull;
        for(auto const& key:keys){hash^=std::uint64_t(key.domain);hash*=1099511628211ull;hash^=key.id;hash*=1099511628211ull;}
        return hash;
    }
    void release_scratch(){allocated-=source_scratch.capacity()*sizeof(Revisions::Key);std::vector<Revisions::Key>().swap(source_scratch);}
    bool intern_source(Source& source){
        normalize(source_scratch);auto hash=dependency_hash(source_scratch);
        auto matches=dependency_lists.equal_range(hash);
        for(auto i=matches.first;i!=matches.second;++i)if(i->second.keys==source_scratch){
            source.dependencies=&i->second;++i->second.refs;release_scratch();return true;
        }
        auto extra=sizeof(DependencyList)+sizeof(std::uint64_t)+64;
        if(!reserve_bytes(extra))return false;
        try{
            auto found=dependency_lists.emplace(hash,DependencyList{});
            found->second.keys.swap(source_scratch);found->second.refs=1;found->second.hash=hash;
            source.dependencies=&found->second;return true;
        }catch(std::bad_alloc const&){allocated-=extra;return complete=false;}
    }
    void erase_source(typename std::map<void const*,Source>::iterator found){
        auto* list=found->second.dependencies;
        if(list && !--list->refs){
            auto matches=dependency_lists.equal_range(list->hash);
            for(auto i=matches.first;i!=matches.second;++i)if(&i->second==list){
                allocated-=sizeof(DependencyList)+sizeof(std::uint64_t)+64+list->keys.capacity()*sizeof(Revisions::Key);
                dependency_lists.erase(i);break;
            }
        }
        allocated-=sizeof(Source)+sizeof(void const*)+64;sources.erase(found);
    }
    template<class T,class Register>bool watch_source(std::shared_ptr<T> const& owner,Register register_inputs){
        if(!complete || !owner || !registering || expanding)return complete=false;
        if(std::find(registering->sources.begin(),registering->sources.end(),owner.get())!=registering->sources.end())return true;
        auto found=sources.find(owner.get());
        if(found==sources.end()){
            if(!reserve_bytes(sizeof(Source)+sizeof(void const*)+64))return false;
            try{found=sources.emplace(owner.get(),Source{}).first;}
            catch(std::bad_alloc const&){allocated-=sizeof(Source)+sizeof(void const*)+64;return complete=false;}
            found->second.owner=owner;
            expanding=&found->second;++validation_counts.source_expansions;
            bool result=false;
            try{result=register_inputs(*owner);}catch(std::bad_alloc const&){complete=false;}
            expanding=nullptr;
            if(!result || !complete || !intern_source(found->second)){
                erase_source(found);release_scratch();return complete=false;
            }
        }else ++validation_counts.source_reuses;
        if(!append(registering->sources,static_cast<void const*>(owner.get()))){if(!found->second.refs)erase_source(found);return false;}
        ++found->second.refs;return true;
    }
    void erase(typename std::map<std::uint64_t,Producer>::iterator found,bool rollback=false){
        for(auto source:found->second.sources){auto i=sources.find(source);if(i!=sources.end()&&!--i->second.refs&&(!registration_open||rollback))erase_source(i);}
        allocated-=sizeof(Producer)+sizeof(std::uint64_t)+64+found->second.dependencies.capacity()*sizeof(Revisions::Key)+found->second.sources.capacity()*sizeof(void const*);
        producers.erase(found);
    }
    template<class Admit>void begin(std::array<std::uint64_t,4> const& next,Revisions const& revisions,Admit reserve){
        admit=reserve;
        if(!complete || context!=next){clear();context=next;}
        if(producers.empty())checkpoint=revisions.checkpoint();
        registration_open=true;
        if(++epoch==0){for(auto& p:producers)p.second.seen=0;epoch=1;}
    }
    void mark(std::uint64_t generation){auto found=producers.find(generation);if(found!=producers.end())found->second.seen=epoch;}
    template<class Register,class Valid>bool add(std::uint64_t generation,std::shared_ptr<Proof const> proof,
            std::uint64_t tile,std::uint64_t visibility,Register register_inputs,Valid valid){
        ++validation_counts.membership;
        if(!complete || !proof)return complete=false;
        auto found=producers.find(generation);
        // The immutable proof owner's tile cannot move. A different supplied
        // tile under that same owner cannot safely renew its old watch keys.
        if(found!=producers.end()&&found->second.proof.get()==proof.get()&&found->second.tile!=tile){valid_all=false;return complete=false;}
        if(found!=producers.end()&&found->second.proof.get()!=proof.get()){erase(found);found=producers.end();}
        if(found==producers.end()){
            if(!reserve_bytes(sizeof(Producer)+sizeof(std::uint64_t)+64))return false;
            found=producers.emplace(generation,Producer{}).first;
            auto& producer=found->second;producer.proof=std::move(proof);producer.tile=tile;producer.visibility=visibility;
            registering=&producer;++validation_counts.proof_registrations;
            bool result=register_inputs(*producer.proof,*this);registering=nullptr;
            if(!result || !complete){erase(found,true);return complete=false;}
            normalize(producer.dependencies);
            ++validation_counts.content;++validation_counts.visibility;
            producer.valid=valid(*producer.proof); // one exact check, never append then recheck all producers
        }else if(found->second.visibility!=visibility){
            // Current selected casters may have a newer visibility snapshot
            // while their immutable content/dependency owners remain exact.
            // Refresh the snapshot only with a mandatory subsequent full check.
            found->second.visibility=visibility;found->second.valid=false;valid_all=false;
        }
        found->second.seen=epoch;return complete;
    }
    void finish(bool final=true){
        valid_all=true;for(auto i=producers.begin();i!=producers.end();){auto next=std::next(i);if(i->second.seen!=epoch)erase(i);else valid_all=i->second.valid&&valid_all;i=next;}
        if(final){registration_open=false;for(auto i=sources.begin();i!=sources.end();){if(i->second.refs){++i;continue;}
            auto next=std::next(i);erase_source(i);i=next;}}
    }
    template<class Valid,class Visibility>bool validate(Revisions const& revisions,Valid valid,Visibility visibility){
        ++validation_counts.regions;if(!complete)return false;
        std::array<Revisions::Key,Revisions::capacity> changed{};std::size_t count=0;
        struct Visits {decltype(changed)& keys;std::size_t& size;std::size_t count(Revisions::Key const& key)const{keys[size++]=key;return 0;}} visits{changed,count};
        auto before=revisions.checkpoint();
        bool incremental=revisions.unchanged(checkpoint,visits,validation_counts.changes);
        if(incremental && !count && valid_all){checkpoint=before;++validation_counts.reused;return true;}
        bool result=true,checked=false;
        // Broad journals make per-producer/source selection more work than an
        // exact full check. Keep sparse edits selective and no-change reuse.
        for(auto& item:producers){auto& producer=item.second;bool touched=!incremental || count>16 || !producer.valid;
            auto includes=[&](auto const& keys,Revisions::Key const& key){return std::binary_search(keys.begin(),keys.end(),key,Less{});};
            for(std::size_t i=0;!touched&&i<count;++i){touched=includes(producer.dependencies,changed[i]);
                for(auto source:producer.sources){if(touched)break;touched=includes(sources.at(source).dependencies->keys,changed[i]);}}
            if(touched){checked=true;++validation_counts.content;++validation_counts.visibility;
                producer.valid=valid(*producer.proof)&&visibility(producer.tile)==producer.visibility;}
            result=producer.valid&&result;
        }
        auto after=revisions.checkpoint();
        if(before.owner!=after.owner || before.sequence!=after.sequence)return false;
        checkpoint=after;valid_all=result;if(checked)++validation_counts.full;else ++validation_counts.reused;return result;
    }
};
// Exact current caster proofs own no geometry or former camera generations.
// The fixed atlas slices survive logical-grid movement; only completed pages
// can be selected again. Sampling density/light/wrap/scope are explicit facts.
template<class Key> struct ShadowPageContents {
    using Grid=ShadowSamplingGrid;
    using Context=std::array<std::uint64_t,32>;
    using Inputs=std::array<std::vector<Key>,Grid::max_pages>;
    using OccurrenceId=std::uint32_t;
    struct Page {std::array<int,2> coordinate{};Context context{};std::vector<OccurrenceId> contributors;bool valid=false,sorted=true;};
    std::array<Page,Grid::max_pages> pages{};
    std::array<unsigned,Grid::max_pages> slots{};
    std::array<bool,Grid::max_pages> reused{};
    std::uint64_t hits=0,rebuilt=0,refused=0;
    struct Occurrence {std::array<float,4> bounds{};std::uint64_t seen=0;std::uint32_t pages=0;OccurrenceId id=0;};
    std::map<Key,Occurrence> occurrences;
    OccurrenceId last_id=0;
    std::array<float,12> projection_light{};
    std::uint64_t epoch=0,projections=0,projection_reuses=0,page_tests=0,contributor_edits=0,page_sorts=0;
    std::uint32_t preserved=0;
    bool projection_valid=false;
    Grid membership_grid;
    bool same_page_grid=false;
    bool membership_complete=false;
    std::size_t bytes()const{std::size_t result=sizeof(*this)+occurrences.size()*(sizeof(Key)+sizeof(Occurrence)+64);
        for(auto const& page:pages)result+=page.contributors.capacity()*sizeof(OccurrenceId);return result;}
    void clear(){for(auto& page:pages){page.valid=false;std::vector<OccurrenceId>().swap(page.contributors);}occurrences.clear();last_id=0;slots={};reused={};projection_valid=false;membership_complete=false;epoch=0;}
    static bool intersects(std::array<float,4> const& bounds,std::array<float,4> const& page){
        return !(bounds[2]<page[0]||bounds[0]>page[0]+page[2]||bounds[3]<page[1]||bounds[1]>page[1]+page[3]);
    }
    // Bind logical pages to existing coordinates before editing contributors.
    // A failed draw keeps its exact input vector but never a completed proof.
    void begin_incremental(Grid const& grid,Context const& context,std::array<float,12> const& light){
        reused={};preserved=0;std::array<bool,Grid::max_pages> used{},matched{};
        for(unsigned logical=0;logical<grid.pages();++logical)for(unsigned physical=0;physical<Grid::max_pages;++physical){
            auto const& page=pages[physical];if(!used[physical]&&page.coordinate==grid.page(logical)&&page.context==context){
                slots[logical]=physical;used[physical]=matched[logical]=true;preserved|=1u<<physical;break;}}
        for(unsigned logical=0;logical<grid.pages();++logical)if(!matched[logical]){
            unsigned physical=0;while(used[physical])++physical;used[physical]=true;slots[logical]=physical;
            auto& page=pages[physical];page.valid=false;page.coordinate=grid.page(logical);page.context=context;page.contributors.clear();page.sorted=false;
        }
        // Unrequested slices are not maintained by the current membership
        // edits. They must not re-enter later with an obsolete completed proof.
        for(unsigned physical=0;physical<Grid::max_pages;++physical)if(!used[physical]){pages[physical].valid=false;pages[physical].contributors.clear();}
        same_page_grid=membership_complete&&membership_grid.valid&&membership_grid.low==grid.low&&membership_grid.count==grid.count&&
            membership_grid.quality_span==grid.quality_span&&std::all_of(matched.begin(),matched.begin()+grid.pages(),[](bool value){return value;});
        if(!projection_valid || std::memcmp(projection_light.data(),light.data(),sizeof(light))){for(auto& page:pages){page.valid=false;page.contributors.clear();page.sorted=false;}occurrences.clear();last_id=0;preserved=0;projection_light=light;projection_valid=true;}
        if(!preserved)same_page_grid=false;
        membership_grid=grid;
        membership_complete=false;
        if(++epoch==0){for(auto& item:occurrences)item.second.seen=0;epoch=1;}
    }
    void mark(Key const& key){auto found=occurrences.find(key);if(found!=occurrences.end()&&found->second.seen!=epoch){
        found->second.pages&=preserved;found->second.seen=epoch;}}
    template<class Admit>bool edit(unsigned physical,OccurrenceId id,bool insert,Admit admit){
        auto& page=pages[physical];
        if(insert){
            // The occurrence mask records each successful append immediately.
            // A retry skips that bit, so cold insertion needs no page scan.
            if(page.contributors.size()==page.contributors.capacity()){
                auto capacity=std::max(std::size_t(4),page.contributors.capacity()*2);
                // reserve allocates the new vector before releasing the old
                // capacity. Both remain charged during that replacement.
                if(!admit(bytes()+capacity*sizeof(OccurrenceId)))return false;
                page.contributors.reserve(capacity);}
            page.contributors.push_back(id);
        }else {auto position=page.sorted?std::lower_bound(page.contributors.begin(),page.contributors.end(),id):std::find(page.contributors.begin(),page.contributors.end(),id);
            if(position==page.contributors.end()||*position!=id)return true;
            page.contributors.erase(std::remove(page.contributors.begin(),page.contributors.end(),id),page.contributors.end());}
        page.valid=false;page.sorted=false;++contributor_edits;return true;
    }
    template<class Project,class Admit>bool update(Key const& key,Grid const& grid,Project project,Admit admit){
        auto found=occurrences.find(key);bool existing=found!=occurrences.end();
        if(found==occurrences.end()){
            // No ID can name a different exact key while any page proof lives.
            // Exhaustion invalidates all pages before IDs can restart.
            if(last_id==UINT32_MAX){clear();return false;}
            if(!admit(bytes()+sizeof(Key)+sizeof(Occurrence)+64))return false;
            found=occurrences.emplace(key,Occurrence{project(),0,0,last_id+1}).first;last_id=found->second.id;++projections;
        }else ++projection_reuses;
        auto& occurrence=found->second;
        if(occurrence.seen!=epoch){occurrence.pages&=preserved;occurrence.seen=epoch;}
        std::uint32_t next=existing&&same_page_grid?occurrence.pages:0;
        if(!existing||!same_page_grid)for(unsigned logical=0;logical<grid.pages();++logical){++page_tests;
            if(intersects(occurrence.bounds,grid.page_box(logical)))next|=1u<<slots[logical];}
        auto old=occurrence.pages;
        for(unsigned physical=0;physical<Grid::max_pages;++physical)if((old^next)&(1u<<physical)){
            if(!edit(physical,occurrence.id,bool(next&(1u<<physical)),admit)){same_page_grid=false;return false;}
            // Keep partial successes visible to retry and retirement even when
            // a later page cannot be admitted and no completed proof exists.
            occurrence.pages^=1u<<physical;
        }
        occurrence.pages=next;return true;
    }
    template<class Admit>bool retire_missing(Admit admit){
        for(auto i=occurrences.begin();i!=occurrences.end();){if(i->second.seen==epoch){++i;continue;}
            auto old=i->second.pages&preserved;for(unsigned physical=0;physical<Grid::max_pages;++physical)
                if((old&(1u<<physical))&&!edit(physical,i->second.id,false,admit))return false;
            i=occurrences.erase(i);}
        return true;
    }
    template<class Admit>bool finish_incremental(Grid const& grid,bool proved,Admit admit){
        if(!retire_missing(admit))return false;
        for(unsigned logical=0;logical<grid.pages();++logical){auto& page=pages[slots[logical]];
            if(!page.sorted){std::sort(page.contributors.begin(),page.contributors.end());page.contributors.erase(std::unique(page.contributors.begin(),page.contributors.end()),page.contributors.end());page.sorted=true;++page_sorts;}
            reused[logical]=proved&&page.valid;if(reused[logical])++hits;}
        membership_complete=true;
        return true;
    }
    bool complete_incremental(unsigned logical,bool proved=true){pages[slots[logical]].valid=proved;++rebuilt;return true;}
    std::array<float,4> const* projected(Key const& key)const{auto found=occurrences.find(key);return found==occurrences.end()?nullptr:&found->second.bounds;}
    // Resource-free forced-oracle callers supply ordered full exact keys. The
    // production incremental path compares collision-free IDs directly.
    bool exact_inputs(unsigned physical,std::vector<Key> const& inputs)const{
        auto const& ids=pages[physical].contributors;
        if(!pages[physical].sorted || ids.size()!=inputs.size())return false;
        std::size_t next=0;
        for(auto const& entry:occurrences){auto range=std::equal_range(ids.begin(),ids.end(),entry.second.id);
            for(auto i=range.first;i!=range.second;++i){if(next==inputs.size() || inputs[next]!=entry.first)return false;++next;}}
        return next==inputs.size();
    }
    void prune_oracle_keys(){
        for(auto i=occurrences.begin();i!=occurrences.end();){bool used=false;
            for(auto const& page:pages)if(page.sorted?std::binary_search(page.contributors.begin(),page.contributors.end(),i->second.id):
                    std::find(page.contributors.begin(),page.contributors.end(),i->second.id)!=page.contributors.end()){used=true;break;}
            if(used)++i;else i=occurrences.erase(i);}
    }
    void select(Grid const& grid,Context const& context,Inputs const& inputs,bool proved){
        projection_valid=false;reused={};std::array<bool,Grid::max_pages> used{};
        // Claim every exact old page first, so recycling a leaving slice cannot
        // overwrite a later requested page during a shift in either axis.
        for(unsigned logical=0;logical<grid.pages();++logical){
            if(!proved)continue;
            for(unsigned physical=0;physical<Grid::max_pages;++physical){auto const& page=pages[physical];
                if(!used[physical]&&page.valid&&page.coordinate==grid.page(logical)&&page.context==context&&exact_inputs(physical,inputs[logical])){
                    slots[logical]=physical;used[physical]=reused[logical]=true;++hits;break;
                }
            }
        }
        for(unsigned logical=0;logical<grid.pages();++logical)if(!reused[logical]){
            unsigned physical=0;while(used[physical])++physical;
            slots[logical]=physical;used[physical]=true;pages[physical].valid=false;
        }
        for(unsigned physical=0;physical<Grid::max_pages;++physical)if(!used[physical]){pages[physical].valid=false;pages[physical].contributors.clear();}
        prune_oracle_keys();membership_complete=false;
    }
    bool complete(unsigned logical,Grid const& grid,Context const& context,std::vector<Key> const& inputs,bool proved=true){
        projection_valid=false;auto& page=pages[slots[logical]];page.valid=false;
        try{
            // Legacy oracle completion has no borrowed key pointers and keeps
            // only keys still referenced by the fixed physical page slots.
            std::vector<OccurrenceId> ids;ids.reserve(inputs.size());
            for(auto const& key:inputs){auto found=occurrences.find(key);
                if(found==occurrences.end()){
                    if(last_id==UINT32_MAX){clear();return false;}
                    found=occurrences.emplace(key,Occurrence{{},0,0,last_id+1}).first;last_id=found->second.id;
                }ids.push_back(found->second.id);
            }
            std::sort(ids.begin(),ids.end());page.contributors.swap(ids);page.coordinate=grid.page(logical);page.context=context;page.valid=proved;page.sorted=true;
            prune_oracle_keys();++rebuilt;return true;
        }
        catch(...){prune_oracle_keys();++refused;return false;}
    }
};
} }
