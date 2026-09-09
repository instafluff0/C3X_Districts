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
            bool valid=ready && slot.valid && !disjoint.Disjoint && disjoint.Frequency &&
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
