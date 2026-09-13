#pragma once

// Optional diagnostics on the D3D owner thread. Queries are retired on later
// submissions with DONOTFLUSH; no query adds a wait to a frame or reset.
#include <array>
#include <cstdint>
#include <d3d11.h>

namespace c3x_renderer { namespace render_core {
struct GpuFrameTelemetry {
    struct Slot {
        ID3D11Query* disjoint=nullptr;
        std::array<ID3D11Query*,3> stamp{};
        std::uint64_t sequence=0;
        bool pending=false, valid=false;
    };
    std::array<Slot,8> slots{};
    Slot* active=nullptr;
    bool failed=false;
    unsigned skipped=0;
    UINT64 last_frequency=0,last_draw_ticks=0,last_copy_ticks=0;
    ~GpuFrameTelemetry(){reset();}
    void reset() {
        active=nullptr;
        for(auto& slot:slots){
            if(slot.disjoint)slot.disjoint->Release();
            for(auto* query:slot.stamp)if(query)query->Release();
            slot={};
        }
        failed=false;skipped=0;
    }
    template<class Report> void poll(ID3D11DeviceContext* context,Report report) {
        for(auto& slot:slots)if(slot.pending){
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint={};
            UINT64 stamp[3]={};
            HRESULT hr=context->GetData(slot.disjoint,&disjoint,sizeof(disjoint),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            if(hr==S_FALSE)continue;
            bool ready=hr==S_OK;
            for(unsigned i=0;i<3 && ready;++i){
                hr=context->GetData(slot.stamp[i],&stamp[i],sizeof(stamp[i]),D3D11_ASYNC_GETDATA_DONOTFLUSH);
                ready=hr==S_OK;
            }
            if(hr==S_FALSE)continue;
            // A completed timestamp must be immutable. Some virtual adapters
            // return the retrieval clock instead; monotonicity alone misses it.
            UINT64 repeated=0;
            if(ready){hr=context->GetData(slot.stamp[0],&repeated,sizeof(repeated),D3D11_ASYNC_GETDATA_DONOTFLUSH);ready=hr==S_OK;}
            if(hr==S_FALSE)continue;
            bool valid=ready && repeated==stamp[0] && slot.valid && !disjoint.Disjoint && disjoint.Frequency &&
                stamp[0]<=stamp[1] && stamp[1]<=stamp[2];
            last_frequency=disjoint.Frequency;
            last_draw_ticks=valid?stamp[1]-stamp[0]:0;last_copy_ticks=valid?stamp[2]-stamp[1]:0;
            report(slot.sequence,valid,valid?double(stamp[1]-stamp[0])*1000/disjoint.Frequency:0,
                   valid?double(stamp[2]-stamp[1])*1000/disjoint.Frequency:0,skipped);
            slot.pending=false;
        }
    }
    bool begin(ID3D11Device* device,ID3D11DeviceContext* context,std::uint64_t sequence) {
        if(failed || active)return false;
        for(auto& slot:slots)if(!slot.pending){
            if(!slot.disjoint){
                D3D11_QUERY_DESC desc={D3D11_QUERY_TIMESTAMP_DISJOINT,0};
                if(FAILED(device->CreateQuery(&desc,&slot.disjoint))){failed=true;return false;}
                desc.Query=D3D11_QUERY_TIMESTAMP;
                for(auto& query:slot.stamp)if(FAILED(device->CreateQuery(&desc,&query))){failed=true;return false;}
            }
            slot.sequence=sequence;slot.valid=false;
            active=&slot;context->Begin(slot.disjoint);context->End(slot.stamp[0]);return true;
        }
        ++skipped;return false;
    }
    void draw_end(ID3D11DeviceContext* context){if(active){context->End(active->stamp[1]);active->valid=true;}}
    void end(ID3D11DeviceContext* context,bool complete=true) {
        if(!active)return;
        if(!active->valid)context->End(active->stamp[1]);
        active->valid=active->valid && complete;
        context->End(active->stamp[2]);context->End(active->disjoint);
        active->pending=true;active=nullptr;
    }
    struct Scope {
        GpuFrameTelemetry& owner;ID3D11DeviceContext* context;
        ~Scope(){owner.end(context,false);}
    };
};

// Bounded optional animation diagnostics. Timestamp intervals measure command
// spans, including possible GPU starvation; they are not hardware busy counters.
// CPU Map wait overlaps this timeline and must never be added to these values.
struct GpuAnimationTelemetry {
    enum Phase : unsigned { background, import, receivers, shadow, body, finish, transfer, phase_count };
    struct Pair { ID3D11Query *begin=nullptr,*end=nullptr;Phase phase=background; };
    struct Slot {
        ID3D11Query *disjoint=nullptr,*begin=nullptr,*end=nullptr;
        std::array<Pair,1024> pairs{};
        unsigned allocated=0,count=0;std::uint64_t sequence=0;
        bool pending=false,open=false,valid=true;
    };
    struct Sample {
        std::uint64_t sequence=0;UINT64 frequency=0,total=0;
        std::array<UINT64,phase_count> ticks{};
        std::array<unsigned,phase_count> counts{};
        bool valid=false;unsigned skipped=0;
    };
    std::array<Slot,2> slots{};Slot* active=nullptr;
    bool failed=false;unsigned skipped=0;
    template<class T>void release(T*& p){if(p){p->Release();p=nullptr;}}
    void reset(){
        active=nullptr;
        for(auto& slot:slots){
            release(slot.disjoint);release(slot.begin);release(slot.end);
            for(auto& pair:slot.pairs){release(pair.begin);release(pair.end);}
            slot={};
        }
        failed=false;skipped=0;
    }
    ~GpuAnimationTelemetry(){reset();}
    bool begin(ID3D11Device* device,ID3D11DeviceContext* context,std::uint64_t sequence,unsigned capacity){
        if(failed || active)return false;
        if(capacity>1024){++skipped;return false;}
        for(auto& slot:slots)if(!slot.pending){
            D3D11_QUERY_DESC desc={D3D11_QUERY_TIMESTAMP_DISJOINT,0};
            auto create=[&](ID3D11Query** query){return *query || !FAILED(device->CreateQuery(&desc,query));};
            bool ok=create(&slot.disjoint);desc.Query=D3D11_QUERY_TIMESTAMP;
            ok=ok && create(&slot.begin) && create(&slot.end);
            while(ok && slot.allocated<capacity){
                auto& pair=slot.pairs[slot.allocated];ok=create(&pair.begin)&&create(&pair.end);
                if(ok)++slot.allocated;
            }
            if(!ok){failed=true;return false;}
            slot.count=0;slot.open=false;slot.valid=true;slot.sequence=sequence;active=&slot;
            context->Begin(slot.disjoint);context->End(slot.begin);return true;
        }
        ++skipped;return false;
    }
    bool pass_begin(ID3D11DeviceContext* context,Phase phase){
        if(!active || active->open)return false; // Outer background owns recursive static work.
        if(active->count>=active->allocated){active->valid=false;return false;}
        auto& pair=active->pairs[active->count];pair.phase=phase;active->open=true;
        context->End(pair.begin);return true;
    }
    void pass_end(ID3D11DeviceContext* context){
        if(active && active->open){context->End(active->pairs[active->count].end);++active->count;active->open=false;}
    }
    void end(ID3D11DeviceContext* context,bool complete=true){
        if(!active)return;
        active->valid=active->valid && complete && !active->open;
        if(active->open)pass_end(context);
        context->End(active->end);context->End(active->disjoint);active->pending=true;active=nullptr;
    }
    template<class Report>void poll(ID3D11DeviceContext* context,Report report){
        for(auto& slot:slots)if(slot.pending){
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint={};
            HRESULT hr=context->GetData(slot.disjoint,&disjoint,sizeof(disjoint),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            if(hr==S_FALSE)continue;
            bool ready=hr==S_OK;UINT64 first=0,last=0;
            auto get=[&](ID3D11Query* query,UINT64& value){
                if(ready){hr=context->GetData(query,&value,sizeof(value),D3D11_ASYNC_GETDATA_DONOTFLUSH);ready=hr==S_OK;}
            };
            get(slot.begin,first);get(slot.end,last);
            Sample sample;sample.sequence=slot.sequence;sample.frequency=disjoint.Frequency;sample.skipped=skipped;
            bool ordered=first<=last;UINT64 preceding=first;
            for(unsigned i=0;i<slot.count && ready;++i){
                UINT64 begin=0,end=0;get(slot.pairs[i].begin,begin);get(slot.pairs[i].end,end);
                ordered=ordered && preceding<=begin && begin<=end && end<=last;preceding=end;
                sample.ticks[slot.pairs[i].phase]+=end-begin;++sample.counts[slot.pairs[i].phase];
            }
            if(hr==S_FALSE)continue;
            UINT64 repeated=0;get(slot.begin,repeated);
            if(hr==S_FALSE)continue;
            sample.valid=ready && first==repeated && slot.valid && ordered && !disjoint.Disjoint && disjoint.Frequency;
            if(sample.valid)sample.total=last-first;else {sample.ticks={};sample.total=0;}
            report(sample);slot.pending=false;
        }
    }
    struct Pass {
        GpuAnimationTelemetry& owner;ID3D11DeviceContext* context;bool started;
        Pass(GpuAnimationTelemetry& o,ID3D11DeviceContext* c,Phase phase,bool enabled=true):owner(o),context(c),started(enabled && o.pass_begin(c,phase)){}
        ~Pass(){if(started)owner.pass_end(context);}
    };
    struct Scope {GpuAnimationTelemetry& owner;ID3D11DeviceContext* context;~Scope(){owner.end(context,false);}};
};

struct AddressSpaceSample {
    std::uint64_t available=0,largest=0,committed=0,reserved=0;
    static AddressSpaceSample capture() {
        AddressSpaceSample result;
        MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);
        if(GlobalMemoryStatusEx(&memory))result.available=memory.ullAvailVirtual;
        std::uintptr_t address=0;MEMORY_BASIC_INFORMATION region={};
        while(VirtualQuery(reinterpret_cast<void const*>(address),&region,sizeof(region))){
            if(region.State==MEM_FREE && region.RegionSize>result.largest)result.largest=region.RegionSize;
            if(region.State==MEM_COMMIT)result.committed+=region.RegionSize;
            if(region.State==MEM_RESERVE)result.reserved+=region.RegionSize;
            auto next=reinterpret_cast<std::uintptr_t>(region.BaseAddress)+region.RegionSize;
            if(next<=address)break;
            address=next;
        }
        return result;
    }
};
} }
