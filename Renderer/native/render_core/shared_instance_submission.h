#pragma once
#include "../../lab/shared/natural/instance.h"
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Immutable selected placements, shared by main/reflection/shadow consumers.
// Source mesh owners retain their existing independent 32 MiB limits; this
// owner adds a 32 MiB joint CPU staging/metadata/GPU limit, including old pinned
// generations. Thus rigid + natural sources + selected instances are bounded
// by 96 MiB of logical source GPU buffers plus instance CPU/GPU storage, before
// separately accounted source CPU packs, prepared terrain and scene targets.
class SharedInstanceSubmission {
public:
    using Instance=fidelity::MeshInstance;
    // Full body/caster facts fit directly; no hash equality can admit a range.
    using Key=std::array<std::uint64_t,24>;
    static constexpr std::size_t budget=32u*1024u*1024u;
    static constexpr unsigned record_limit=65536,entry_limit=16384;
    struct Range {
        unsigned first=0,count=0;
        explicit operator bool()const{return count!=0;}
        bool contiguous(Range const& next)const{return first+count==next.first;}
    };
    struct Ledger {
        std::atomic<std::size_t> bytes{0},cpu{0},gpu{0},peak{0};
    };
private:
    struct Charge {
        std::shared_ptr<Ledger> ledger;
        std::size_t cpu=0,gpu=0;
        explicit Charge(std::shared_ptr<Ledger> owner):ledger(std::move(owner)){}
        bool resize(std::size_t next_cpu,std::size_t next_gpu){
            auto previous=cpu+gpu,next=next_cpu+next_gpu;
            if(next>previous){
                auto extra=next-previous,retained_count=ledger->bytes.load();
                do{if(extra>budget-retained_count)return false;}
                while(!ledger->bytes.compare_exchange_weak(retained_count,retained_count+extra));
                auto high=ledger->peak.load();
                while(high<retained_count+extra && !ledger->peak.compare_exchange_weak(high,retained_count+extra)){}
            }else ledger->bytes.fetch_sub(previous-next);
            if(next_cpu>cpu)ledger->cpu.fetch_add(next_cpu-cpu);else ledger->cpu.fetch_sub(cpu-next_cpu);
            if(next_gpu>gpu)ledger->gpu.fetch_add(next_gpu-gpu);else ledger->gpu.fetch_sub(gpu-next_gpu);
            cpu=next_cpu;gpu=next_gpu;return true;
        }
        ~Charge(){ledger->bytes.fetch_sub(cpu+gpu);ledger->cpu.fetch_sub(cpu);ledger->gpu.fetch_sub(gpu);}
    };
public:
    // These references prove residency without pinning an old view or mesh.
    // The active required content lease still owns the sources used by draws.
    struct RetainedSource {
        std::weak_ptr<void const> source;
        std::array<std::uint64_t,2> owner{}; // resident slot and immutable generation
        std::array<std::uint64_t,2> canonical_source{};
        bool canonical=false;
    };
    struct Generation {
        Key identity{};
        ID3D11Buffer* buffer=nullptr;
        ID3D11ShaderResourceView* view=nullptr;
        std::map<Key,Range> ranges;
        // A source placement may have several native wrapped occurrences.
        // Shadow shaders need its canonical world placement only, so the first
        // occurrence range is sufficient and carries the same exact material.
        std::map<std::array<std::uint64_t,2>,Range> sources;
        std::vector<Instance> staging;
        std::vector<Instance> placements;
        struct Transfer {unsigned first=0,source=0,count=0;};
        // Replacement staging contains changed values only. An old immutable
        // generation stays pinned until its validated ranges are copied.
        std::vector<Transfer> copies,updates;
        std::shared_ptr<Generation const> copy_source;
        bool delta=false;
        // A range already owns the exact 24-word key. Its unique starting
        // record indexes the weak source proof without storing that key twice.
        std::map<unsigned,RetainedSource> retained_sources;
        bool retain_placements=false;
        std::shared_ptr<void const> content;
        unsigned records=0;
        std::uint64_t owner_epoch=0;
        ID3D11Device* device_cookie=nullptr; // borrowed identity; resources own the device lifetime
        bool complete=false;
    private:
        friend class SharedInstanceSubmission;
        Charge charge;
        explicit Generation(std::shared_ptr<Ledger> ledger):charge(std::move(ledger)){}
        std::size_t metadata(std::size_t entries,std::size_t source_count,
                std::size_t retained_count=0)const{
            // Includes conservative node/allocator overhead, not just payloads.
            return sizeof(Generation)+entries*(sizeof(Key)+sizeof(Range)+64u)+
                source_count*(sizeof(std::array<std::uint64_t,2>)+sizeof(Range)+64u)+
                retained_count*(sizeof(unsigned)+sizeof(RetainedSource)+64u);
        }
        std::size_t transfer_bytes()const{return (copies.capacity()+updates.capacity())*sizeof(Transfer);}
    public:
        ~Generation(){if(view)view->Release();if(buffer)buffer->Release();}
        Range find(Key const& key)const{auto found=ranges.find(key);return found==ranges.end()?Range{}:found->second;}
        static std::array<std::uint64_t,2> source_key(void const* key,float material){
            std::uint32_t bits=0;std::memcpy(&bits,&material,sizeof(bits));
            return {std::uint64_t(reinterpret_cast<std::uintptr_t>(key)),bits};
        }
        Range source(void const* key,float material)const{
            auto found=sources.find(source_key(key,material));
            return found==sources.end()?Range{}:found->second;
        }
        Range source(void const* key,float material,unsigned expected)const{
            auto range=source(key,material);
            return range.count==expected && range.first<=records && range.count<=records-range.first?range:Range{};
        }
        std::size_t bytes()const{return charge.cpu+charge.gpu;}
        std::size_t gpu_bytes()const{return charge.gpu;}
    };
    using Lease=std::shared_ptr<Generation const>;
    using Builder=std::shared_ptr<Generation>;
    struct IndexedSelection {
        Lease content;
        ID3D11Buffer* buffer=nullptr;
        unsigned count=0;
    private:
        friend class SharedInstanceSubmission;
        Charge charge;
        explicit IndexedSelection(std::shared_ptr<Ledger> ledger):charge(std::move(ledger)){}
    public:
        ~IndexedSelection(){if(buffer)buffer->Release();}
        std::size_t bytes()const{return charge.cpu+charge.gpu;}
    };
    using SelectionLease=std::shared_ptr<IndexedSelection const>;
    struct CpuAllocation {
    private:
        friend class SharedInstanceSubmission;
        Charge charge;
        explicit CpuAllocation(std::shared_ptr<Ledger> ledger):charge(std::move(ledger)){}
    public:
        std::size_t bytes()const{return charge.cpu+charge.gpu;}
    };
    using CpuLease=std::shared_ptr<CpuAllocation const>;
private:
    std::shared_ptr<Ledger> ledger=std::make_shared<Ledger>();
    Lease current;
    Charge selection_charge{ledger};
    std::uint64_t owner_epoch=1;
    bool reserve(Charge& charge,std::size_t cpu,std::size_t gpu){
        return charge.resize(cpu,gpu);
    }
public:
    ID3D11Buffer* selection_buffer=nullptr;
    unsigned selection_cursor=record_limit,selection_offset=0,selection_uploads=0,selection_discards=0;
    std::size_t selection_uploaded_bytes=0;
    unsigned uploads=0,reuses=0,rejected=0;
    std::size_t uploaded_bytes=0;
    unsigned range_reuses=0,carry_visits=0,carried_ranges=0,gpu_copies=0;
    std::size_t packed_records=0,copied_bytes=0,allocated_bytes=0;
    unsigned plan_uploads=0;
    std::size_t plan_uploaded_bytes=0;
    SharedInstanceSubmission()=default;
    SharedInstanceSubmission(SharedInstanceSubmission const&)=delete;
    ~SharedInstanceSubmission(){clear();}
    Lease select(Key const& identity){
        if(current && current->identity==identity){++reuses;return current;}
        return {};
    }
    template<class Required>Lease find_covering(Required required){
        if(valid(current) && required(*current)){++reuses;return current;}
        return {};
    }
    Builder begin(Key const& identity,std::shared_ptr<void const> content={}){
        try{
            Builder next(new Generation(ledger));next->identity=identity;next->content=std::move(content);next->owner_epoch=owner_epoch;
            if(!reserve(next->charge,sizeof(Generation),0)){++rejected;return {};}
            return next;
        }catch(...){++rejected;return {};}
    }
    Builder begin_retained(Key const& identity,std::shared_ptr<void const> content={},bool delta=false,Lease base={}){
        if(base && (!delta || !valid(base) || !base->retain_placements))return {};
        auto next=begin(identity,std::move(content));
        if(next){next->retain_placements=true;next->delta=delta;next->copy_source=std::move(base);}
        return next;
    }
    bool append(Builder const& next,Key const& key,void const* source,
            Instance const* instances,unsigned count,float const* projection,
            float x,float y,float depth,float material,Range& range){
        if(!next || next->charge.ledger!=ledger || next->owner_epoch!=owner_epoch || next->complete || !instances || !count || !source || !projection)return false;
        auto found=next->ranges.find(key);
        if(found!=next->ranges.end()){range=found->second;return range.count==count;}
        if(next->ranges.size()==entry_limit || count>record_limit-next->records){++rejected;return false;}
        auto source_key=Generation::source_key(source,material);
        bool new_source=next->sources.find(source_key)==next->sources.end();
        auto records=next->records+count;
        auto capacity=next->staging.capacity();
        auto staged=next->staging.size()+count;
        if(staged>capacity)capacity=std::min<std::size_t>(record_limit,std::max<std::size_t>(staged,std::max<std::size_t>(64,capacity*2)));
        auto update_capacity=next->updates.capacity();
        bool merge_update=next->delta && !next->updates.empty() &&
            next->updates.back().first+next->updates.back().count==next->records;
        if(next->delta && !merge_update && next->updates.size()==update_capacity)
            update_capacity=std::min<std::size_t>(entry_limit,std::max<std::size_t>(64,update_capacity*2));
        auto old_cpu=next->charge.cpu;
        auto metadata=next->metadata(next->ranges.size()+1,next->sources.size()+unsigned(new_source),next->retained_sources.size());
        // Legacy retained unions keep CPU placements; delta unions stage only
        // changed records. Charge either replacement's future GPU buffer now.
        if(!reserve(next->charge,metadata+capacity*sizeof(Instance)+
                (next->copies.capacity()+update_capacity)*sizeof(Generation::Transfer),
                next->retain_placements?records*sizeof(Instance):0)){++rejected;return false;}
        try{
            next->staging.reserve(capacity);
            if(next->delta){next->updates.reserve(update_capacity);
                if(merge_update)next->updates.back().count+=count;
                else next->updates.push_back({next->records,unsigned(next->staging.size()),count});}
            range={next->records,count};
            next->ranges.emplace(key,range);
            if(new_source)next->sources.emplace(source_key,range);
            for(unsigned i=0;i<count;++i){auto value=instances[i];
                std::memcpy(value.projection,projection,sizeof(value.projection));
                value.view[0]=x;value.view[1]=y;value.view[2]=depth;value.view[3]=material;
                next->staging.push_back(value);
            }
            next->records=records;packed_records+=count;return true;
        }catch(...){
            // A partially built generation cannot be published or reused.
            next->complete=true;next->charge.resize(std::max(old_cpu,next->metadata(next->ranges.size(),next->sources.size(),next->retained_sources.size())+
                next->staging.capacity()*sizeof(Instance)+next->transfer_bytes()),next->charge.gpu);++rejected;return false;
        }
    }
    // The caller supplies its current resident/dependency proof. Equal source
    // pointers alone never authorize reuse after eviction or slot recycling.
    bool reuse_range(Builder const& next,Key const& key,unsigned count,RetainedSource const& proof,Range& range){
        auto base=next&&next->copy_source?next->copy_source:current;
        if(!next || next->charge.ledger!=ledger || next->owner_epoch!=owner_epoch || next->complete || !next->delta ||
                !valid(base) || !base->retain_placements || !count)return false;
        auto source=proof.source.lock();if(!source || !proof.owner[1])return false;
        auto old=base->find(key);if(old.count!=count || old.first>base->records || count>base->records-old.first)return false;
        auto retained=base->retained_sources.find(old.first);
        if(retained==base->retained_sources.end() || retained->second.owner!=proof.owner ||
                retained->second.canonical_source!=proof.canonical_source || retained->second.source.lock()!=source)return false;
        if(auto present=next->find(key)){range=present;return present.count==count;}
        if(!copy_range(next,key,old,proof,false,range))return false;
        ++range_reuses;return true;
    }
private:
    bool copy_range(Builder const& next,Key const& key,Range old,RetainedSource const& proof,bool optional,Range& range){
        auto base=next->copy_source?next->copy_source:current;
        if(!valid(base))return false;
        if(next->ranges.size()==entry_limit || old.count>record_limit-next->records)return false;
        bool new_source=!next->sources.count(proof.canonical_source);
        auto copy_capacity=next->copies.capacity();
        bool merge_copy=!next->copies.empty() && next->copies.back().first+next->copies.back().count==next->records &&
            next->copies.back().source+next->copies.back().count==old.first;
        if(!merge_copy && next->copies.size()==copy_capacity)
            copy_capacity=std::min<std::size_t>(entry_limit,std::max<std::size_t>(64,copy_capacity*2));
        auto records=next->records+old.count;
        auto metadata=next->metadata(next->ranges.size()+1,next->sources.size()+unsigned(new_source),next->retained_sources.size()+1);
        auto cpu=metadata+next->staging.capacity()*sizeof(Instance)+
            (copy_capacity+next->updates.capacity())*sizeof(Generation::Transfer);
        if(optional && cpu+records*sizeof(Instance)>budget/4)return false;
        if(!reserve(next->charge,cpu,records*sizeof(Instance)))return false;
        try{
            next->copies.reserve(copy_capacity);range={next->records,old.count};
            next->ranges.emplace(key,range);next->retained_sources.emplace(range.first,proof);
            if(new_source)next->sources.emplace(proof.canonical_source,range);
            if(merge_copy)next->copies.back().count+=old.count;
            else next->copies.push_back({range.first,old.first,old.count});next->copy_source=base;
            next->records=records;return true;
        }catch(...){next->complete=true;++rejected;return false;}
    }
public:
    bool retain_source(Builder const& next,Key const& key,RetainedSource proof){
        if(!next || next->charge.ledger!=ledger || next->owner_epoch!=owner_epoch || next->complete ||
            !next->retain_placements || !proof.owner[1] || proof.source.expired())return false;
        auto range=next->find(key);if(!range)return false;
        if(next->retained_sources.count(range.first))return true;
        auto metadata=next->metadata(next->ranges.size(),next->sources.size(),next->retained_sources.size()+1);
        if(!reserve(next->charge,metadata+next->staging.capacity()*sizeof(Instance)+next->transfer_bytes(),next->charge.gpu))return false;
        try{next->retained_sources.emplace(range.first,std::move(proof));return true;}
        catch(...){next->complete=true;++rejected;return false;}
    }
    // Required ranges must already be present. Useful resident world ranges
    // are optional and never displace the current request or defeat admission.
    // Callers may retain separate active pass unions under this same ledger;
    // the explicit base is released after upload and never forms a history.
    template<class Valid>bool carry_forward(Builder const& next,Valid valid_source){
        if(!next || next->charge.ledger!=ledger || next->owner_epoch!=owner_epoch || next->complete || !next->retain_placements)return false;
        auto base=next->copy_source?next->copy_source:current;
        if(!valid(base) || !base->retain_placements || (!next->delta && base->placements.size()!=base->records))return true;
        if(base->retained_sources.empty())return true;
        // Keep exact-key order: optional admission and canonical fallback must
        // select the same first eligible range as the previous representation.
        for(auto const& entry:base->ranges){
            ++carry_visits;
            auto retained=base->retained_sources.find(entry.second.first);
            if(retained==base->retained_sources.end())continue;
            auto const& proof=retained->second;
            if(next->ranges.count(entry.first) || proof.source.expired() || !valid_source(proof))continue;
            auto range=entry.second;
            if(!range || range.first>base->records || range.count>base->records-range.first)return false;
            if(next->ranges.size()==entry_limit || range.count>record_limit-next->records)continue;
            if(next->delta){Range copied;
                if(copy_range(next,entry.first,range,proof,true,copied))++carried_ranges;
                else if(next->complete)return false;
                continue;
            }
            // Any validated carried representative supplies the canonical
            // caster values; caster shaders replace occurrence view.xyz.
            // Required source entries were appended first and always win.
            bool new_source=!next->sources.count(proof.canonical_source);
            auto records=next->records+range.count;
            auto capacity=next->staging.capacity();
            if(records>capacity)capacity=std::min<std::size_t>(record_limit,std::max<std::size_t>(records,std::max<std::size_t>(64,capacity*2)));
            auto metadata=next->metadata(next->ranges.size()+1,next->sources.size()+unsigned(new_source),next->retained_sources.size()+1);
            // Optional world ranges leave room for the next required union,
            // its old front and charged selection plans under the joint cap.
            if(metadata+capacity*sizeof(Instance)+records*sizeof(Instance)>budget/4)continue;
            if(!reserve(next->charge,metadata+capacity*sizeof(Instance),records*sizeof(Instance)))continue;
            try{
                next->staging.reserve(capacity);
                Range copied{next->records,range.count};
                next->ranges.emplace(entry.first,copied);next->retained_sources.emplace(copied.first,proof);
                if(new_source)next->sources.emplace(proof.canonical_source,copied);
                next->staging.insert(next->staging.end(),base->placements.begin()+range.first,
                    base->placements.begin()+range.first+range.count);
                next->records=records;++carried_ranges;packed_records+=range.count;
            }catch(...){next->complete=true;++rejected;return false;}
        }
        return true;
    }
    Lease upload(Builder const& next,ID3D11Device* device){
        if(!next || next->charge.ledger!=ledger || next->owner_epoch!=owner_epoch || next->complete || next->delta || !device ||
            (current && current->device_cookie!=device) || next->staging.size()!=next->records)return {};
        auto gpu=std::size_t(next->records)*sizeof(Instance);
        if(!reserve(next->charge,next->charge.cpu,gpu)){++rejected;return {};}
        if(gpu){
            D3D11_BUFFER_DESC desc{};desc.ByteWidth=unsigned(gpu);desc.Usage=D3D11_USAGE_IMMUTABLE;
            desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            desc.StructureByteStride=sizeof(Instance);
            D3D11_SUBRESOURCE_DATA initial{};initial.pSysMem=next->staging.data();
            ID3D11Buffer* buffer=nullptr;
            if(FAILED(device->CreateBuffer(&desc,&initial,&buffer))){next->charge.resize(next->charge.cpu,0);++rejected;return {};}
            ID3D11ShaderResourceView* view=nullptr;
            if(FAILED(device->CreateShaderResourceView(buffer,nullptr,&view))){buffer->Release();next->charge.resize(next->charge.cpu,0);++rejected;return {};}
            next->buffer=buffer;next->view=view;++uploads;uploaded_bytes+=gpu;allocated_bytes+=gpu;
        }
        if(next->retain_placements)next->placements.swap(next->staging);
        else std::vector<Instance>().swap(next->staging);
        next->charge.resize(next->metadata(next->ranges.size(),next->sources.size(),next->retained_sources.size())+
            next->placements.capacity()*sizeof(Instance),gpu);
        next->complete=true;next->device_cookie=device;current=next;return current;
    }
    Lease upload(Builder const& next,ID3D11Device* device,ID3D11DeviceContext* context){
        if(next && !next->delta)return upload(next,device);
        if(!next || next->charge.ledger!=ledger || next->owner_epoch!=owner_epoch || next->complete || !device || !context ||
                (current && current->device_cookie!=device))return {};
        ID3D11Device* context_device=nullptr;context->GetDevice(&context_device);
        bool matches=context_device==device;if(context_device)context_device->Release();if(!matches)return {};
        if(!next->copies.empty() && (!valid(next->copy_source) || next->copy_source->device_cookie!=device))return {};
        std::size_t transferred=0;
        for(auto const& part:next->copies){
            if(!part.count || part.first>next->records || part.count>next->records-part.first ||
                    part.source>next->copy_source->records || part.count>next->copy_source->records-part.source)return {};
            transferred+=part.count;
        }
        for(auto const& part:next->updates){
            if(!part.count || part.first>next->records || part.count>next->records-part.first ||
                    part.source>next->staging.size() || part.count>next->staging.size()-part.source)return {};
            transferred+=part.count;
        }
        if(transferred!=next->records)return {};
        auto gpu=std::size_t(next->records)*sizeof(Instance);
        if(!reserve(next->charge,next->charge.cpu,gpu)){++rejected;return {};}
        if(gpu){
            D3D11_BUFFER_DESC desc{};desc.ByteWidth=unsigned(gpu);desc.Usage=D3D11_USAGE_DEFAULT;
            desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            desc.StructureByteStride=sizeof(Instance);
            ID3D11Buffer* buffer=nullptr;
            if(FAILED(device->CreateBuffer(&desc,nullptr,&buffer))){next->charge.resize(next->charge.cpu,0);++rejected;return {};}
            ID3D11ShaderResourceView* view=nullptr;
            if(FAILED(device->CreateShaderResourceView(buffer,nullptr,&view))){buffer->Release();next->charge.resize(next->charge.cpu,0);++rejected;return {};}
            // Queue copies before publishing. D3D retains resources for queued
            // commands; no wait or mutation of an old consumer's buffer occurs.
            for(auto const& part:next->copies){
                D3D11_BOX box{part.source*unsigned(sizeof(Instance)),0,0,(part.source+part.count)*unsigned(sizeof(Instance)),1,1};
                context->CopySubresourceRegion(buffer,0,part.first*unsigned(sizeof(Instance)),0,0,next->copy_source->buffer,0,&box);
                ++gpu_copies;copied_bytes+=std::size_t(part.count)*sizeof(Instance);
            }
            for(auto const& part:next->updates){
                D3D11_BOX box{part.first*unsigned(sizeof(Instance)),0,0,(part.first+part.count)*unsigned(sizeof(Instance)),1,1};
                context->UpdateSubresource(buffer,0,&box,next->staging.data()+part.source,0,0);
                uploaded_bytes+=std::size_t(part.count)*sizeof(Instance);
            }
            next->buffer=buffer;next->view=view;++uploads;allocated_bytes+=gpu;
        }
        std::vector<Instance>().swap(next->staging);
        std::vector<Generation::Transfer>().swap(next->copies);std::vector<Generation::Transfer>().swap(next->updates);
        next->copy_source.reset();
        next->charge.resize(next->metadata(next->ranges.size(),next->sources.size(),next->retained_sources.size()),gpu);
        next->complete=true;next->device_cookie=device;current=next;return current;
    }
    bool valid(Lease const& content)const{
        return content && content->charge.ledger==ledger && content->owner_epoch==owner_epoch && content->complete &&
            (!content->records || content->view);
    }
    bool valid(SelectionLease const& selection)const{
        return selection && selection->buffer && valid(selection->content);
    }
    bool reserve_index_scratch(){return reserve(selection_charge,record_limit*sizeof(unsigned)*2,selection_charge.gpu);}
    CpuLease retain_metadata(std::size_t bytes,std::size_t gpu_bytes=0){
        if(gpu_bytes>budget-sizeof(CpuAllocation) || bytes>budget-sizeof(CpuAllocation)-gpu_bytes)return {};
        try{std::shared_ptr<CpuAllocation> allocation(new CpuAllocation(ledger));
            if(!reserve(allocation->charge,bytes+sizeof(CpuAllocation),gpu_bytes)){++rejected;return {};}
            return allocation;
        }catch(...){++rejected;return {};}
    }
    bool resize_metadata(CpuLease const& allocation,std::size_t bytes,std::size_t gpu_bytes=0){
        if(!allocation || allocation->charge.ledger!=ledger || gpu_bytes>budget-sizeof(CpuAllocation) || bytes>budget-sizeof(CpuAllocation)-gpu_bytes)return false;
        return const_cast<CpuAllocation*>(allocation.get())->charge.resize(bytes+sizeof(CpuAllocation),gpu_bytes);
    }
    // Warm selection plans keep their four-byte indices. Their GPU storage,
    // caller cache-key metadata and temporary input scratch share the same
    // hard allowance with all pinned placement generations.
    bool can_prepare_selection(unsigned count,std::size_t key_metadata_bytes=0)const{
        if(!count || count>record_limit || key_metadata_bytes>budget-sizeof(IndexedSelection))return false;
        auto needed=sizeof(IndexedSelection)+key_metadata_bytes+std::size_t(count)*sizeof(unsigned)*2;
        return needed<=budget && ledger->bytes.load()<=budget-needed;
    }
    SelectionLease prepare_selection(ID3D11Device* device,Lease const& content,
            unsigned const* values,unsigned count,std::size_t key_metadata_bytes=0){
        if(!device || !valid(content) || device!=content->device_cookie || !values || !count || count>record_limit)return {};
        if(!can_prepare_selection(count,key_metadata_bytes)){++rejected;return {};}
        for(unsigned i=0;i<count;++i)if(values[i]>=content->records)return {};
        auto size=std::size_t(count)*sizeof(unsigned);
        if(key_metadata_bytes>budget-sizeof(IndexedSelection))return {};
        auto metadata=sizeof(IndexedSelection)+key_metadata_bytes;
        try{
            std::shared_ptr<IndexedSelection> selection(new IndexedSelection(ledger));
            if(!reserve(selection->charge,metadata+size,size)){++rejected;return {};}
            D3D11_BUFFER_DESC desc{};desc.ByteWidth=unsigned(size);desc.Usage=D3D11_USAGE_IMMUTABLE;
            desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA initial{};initial.pSysMem=values;
            if(FAILED(device->CreateBuffer(&desc,&initial,&selection->buffer))){++rejected;return {};}
            selection->content=content;selection->count=count;selection->charge.resize(metadata,size);
            ++plan_uploads;plan_uploaded_bytes+=size;return selection;
        }catch(...){++rejected;return {};}
    }
    // Four-byte references are the only per-pass instance upload. Main,
    // reflection and shadow never rebuild or map the 64-byte placement owner.
    bool select_indices(ID3D11Device* device,ID3D11DeviceContext* context,
            Lease const& content,unsigned const* values,unsigned count){
        if(!context || !valid(content) || (device && device!=content->device_cookie) || !content->view || !values || !count || count>record_limit)return false;
        for(unsigned i=0;i<count;++i)if(values[i]>=content->records)return false;
        if(!device){context->GetDevice(&device);auto matches=device==content->device_cookie;device->Release();if(!matches)return false;}
        if(!selection_buffer){
            auto size=record_limit*sizeof(unsigned);
            // Serial pass scratch includes the selected-index vector and
            // bounded contributor/cache-key scratch as well as its GPU stream.
            if(!reserve(selection_charge,size*2,size)){++rejected;return false;}
            D3D11_BUFFER_DESC desc{};desc.ByteWidth=unsigned(size);desc.Usage=D3D11_USAGE_DYNAMIC;
            desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;desc.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            auto result=device->CreateBuffer(&desc,nullptr,&selection_buffer);
            if(FAILED(result)){selection_charge.resize(0,0);++rejected;return false;}
        }
        bool discard=count>record_limit-selection_cursor;
        unsigned start=discard?0:selection_cursor;
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if(FAILED(context->Map(selection_buffer,0,discard?D3D11_MAP_WRITE_DISCARD:D3D11_MAP_WRITE_NO_OVERWRITE,0,&mapped)))return false;
        std::memcpy(static_cast<char*>(mapped.pData)+start*sizeof(unsigned),values,count*sizeof(unsigned));context->Unmap(selection_buffer,0);
        selection_cursor=start+count;selection_offset=start*sizeof(unsigned);++selection_uploads;
        selection_discards+=unsigned(discard);selection_uploaded_bytes+=count*sizeof(unsigned);
        return true;
    }
    void clear(){current.reset();++owner_epoch;if(selection_buffer)selection_buffer->Release();selection_buffer=nullptr;selection_charge.resize(0,0);
        selection_cursor=record_limit;selection_offset=selection_uploads=selection_discards=0;selection_uploaded_bytes=0;
        uploads=reuses=rejected=plan_uploads=0;uploaded_bytes=plan_uploaded_bytes=0;
        range_reuses=carry_visits=carried_ranges=gpu_copies=0;packed_records=copied_bytes=allocated_bytes=0;}
    std::size_t bytes()const{return ledger->bytes.load();}
    std::size_t cpu_bytes()const{return ledger->cpu.load();}
    std::size_t gpu_bytes()const{return ledger->gpu.load();}
    std::size_t peak_bytes()const{return ledger->peak.load();}
};
// Source owners lend this accounting lease alongside their existing COM
// references. Pack/device replacement drops the current lease; selected and
// inflight content keeps the retired allocation charged until its last use.
class SharedSourceResidency {
public:
    static constexpr std::size_t gpu_budget=32u*1024u*1024u,budget=64u*1024u*1024u;
private:
    struct Ledger {std::atomic<std::size_t> bytes{0},gpu{0},peak{0};};
    struct Charge {
        std::shared_ptr<Ledger> ledger;
        std::size_t cpu=0,gpu=0;
        explicit Charge(std::shared_ptr<Ledger> value):ledger(std::move(value)){}
        bool resize(std::size_t next_cpu,std::size_t next_gpu){
            auto extra_gpu=next_gpu>gpu?next_gpu-gpu:0;
            if(extra_gpu){auto total=ledger->gpu.load();
                do{if(extra_gpu>gpu_budget-total)return false;}
                while(!ledger->gpu.compare_exchange_weak(total,total+extra_gpu));}
            auto before=cpu+gpu,after=next_cpu+next_gpu;
            if(after>before){auto extra=after-before,total=ledger->bytes.load();
                do{if(extra>budget-total){ledger->gpu.fetch_sub(extra_gpu);return false;}}
                while(!ledger->bytes.compare_exchange_weak(total,total+extra));
                auto high=ledger->peak.load();
                while(high<total+extra && !ledger->peak.compare_exchange_weak(high,total+extra)){}
            }else ledger->bytes.fetch_sub(before-after);
            if(next_gpu<gpu)ledger->gpu.fetch_sub(gpu-next_gpu);
            cpu=next_cpu;gpu=next_gpu;return true;
        }
        ~Charge(){ledger->bytes.fetch_sub(cpu+gpu);ledger->gpu.fetch_sub(gpu);}
    };
    std::shared_ptr<Ledger> ledger=std::make_shared<Ledger>();
    std::shared_ptr<Charge> current;
public:
    SharedSourceResidency()=default;
    SharedSourceResidency(SharedSourceResidency const&)=delete;
    bool reserve(std::size_t staging_bytes,std::size_t gpu_bytes){
        if(staging_bytes>budget-sizeof(Charge))return false;
        try{if(!current)current=std::make_shared<Charge>(ledger);
            return current->resize(staging_bytes+sizeof(Charge),gpu_bytes);
        }catch(...){return false;}
    }
    std::shared_ptr<void const> pin()const{return current;}
    void clear(){current.reset();}
    std::size_t bytes()const{return ledger->bytes.load();}
    std::size_t gpu_bytes()const{return ledger->gpu.load();}
    std::size_t peak_bytes()const{return ledger->peak.load();}
};
inline HRESULT create_resident_instance_layout(ID3D11Device* device,ID3DBlob* code,ID3D11InputLayout** layout){
    D3D11_INPUT_ELEMENT_DESC elements[]={
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",1,DXGI_FORMAT_R32_UINT,1,0,D3D11_INPUT_PER_INSTANCE_DATA,1}};
    return device->CreateInputLayout(elements,4,code->GetBufferPointer(),code->GetBufferSize(),layout);
}
// These generic rigid material branches return alpha one after authored
// cutout tests. The fractional ground-state family retains feathered alpha
// and remains a barrier, along with flat surfaces and native city materials.
inline bool opaque_rigid_material(float material){
    if(!(material>=7.5f && material<59.5f))return false;
    auto fraction=material-float(int(material));
    return !(material>=20.5f && material<28.5f && fraction>=.295f && fraction<.320f);
}
// Group opaque compatible source draws only through proven independent
// intervening work. Conflict must include conservative main AND reflected
// coverage; alpha, decals, native ordering and equal-depth overlap are barriers.
// The comparison bound limits selection CPU work; exhaustion preserves all
// unperformed commutations instead of weakening their ordering proof.
template<class Item,class Compatible,class Opaque,class Conflict>
unsigned group_independent_opaque(std::vector<Item>& items,Compatible compatible,
        Opaque opaque,Conflict conflict,unsigned comparison_limit=4096){
    unsigned comparisons=0,moved=0;
    for(std::size_t first=0;first<items.size();++first){
        if(!opaque(items[first]))continue;
        auto end=first+1;
        for(auto candidate=end;candidate<items.size();++candidate){
            if(++comparisons>comparison_limit)return moved;
            if(!opaque(items[candidate]) || !compatible(items[first],items[candidate]))continue;
            bool independent=true;
            for(auto between=end;between<candidate;++between){
                if(++comparisons>comparison_limit)return moved;
                if(!opaque(items[between]) || conflict(items[candidate],items[between])){independent=false;break;}
            }
            if(!independent)continue;
            if(candidate!=end){std::rotate(items.begin()+end,items.begin()+candidate,items.begin()+candidate+1);++moved;}
            ++end;
        }
        first=end-1;
    }
    return moved;
}
}}
