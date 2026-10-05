#pragma once
#include <d3d11.h>
#include <wrl/client.h>
#include <cstdio>
#include <string>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Profiling-only GPU phase timeline for translation layers whose timestamp
// and event queries cannot be trusted (under Parallels D3D11 every disjoint
// interval is invalid and event queries report completion at submission).
// A readback cannot be faked: each mark copies a 1x1 texture into its own
// staging texture, and collect() maps them in order. The GPU executes one
// queue in order, so a map returns only after every command issued before
// its mark has finished; consecutive return times bound each phase's GPU
// time. Collection serializes CPU and GPU and is enabled only by
// C3X_RENDERER_PROFILE=1; production frames issue no copies.
class GpuEventTimeline {
    using Texture=Microsoft::WRL::ComPtr<ID3D11Texture2D>;
    struct Mark {char const* name;Texture staging;};
    std::vector<Mark> marks;
    std::vector<Texture> pool;
    Texture probe;
    struct Total {char const* name;double ms=0;unsigned count=0;};
    std::vector<Total> totals;
    LARGE_INTEGER frequency{},reported{};
    unsigned frames=0;double frame_ms=0,wait_ms=0;
    bool configured=false;
    static bool texture(ID3D11Device* device,D3D11_USAGE usage,Texture& out){
        D3D11_TEXTURE2D_DESC desc{};desc.Width=desc.Height=desc.MipLevels=desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.SampleDesc.Count=1;desc.Usage=usage;
        desc.CPUAccessFlags=usage==D3D11_USAGE_STAGING?D3D11_CPU_ACCESS_READ:0;
        return SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&out));
    }
public:
    bool enabled=false;
    void configure(){
        if(configured)return;configured=true;
        char value[8]={};
        enabled=GetEnvironmentVariableA("C3X_RENDERER_PROFILE",value,sizeof(value))&&value[0]=='1';
        QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&reported);
    }
    void mark(ID3D11DeviceContext* context,char const* name){
        if(!enabled||!context)return;
        Microsoft::WRL::ComPtr<ID3D11Device> device;context->GetDevice(&device);if(!device)return;
        if(!probe && !texture(device.Get(),D3D11_USAGE_DEFAULT,probe))return;
        Texture staging;
        if(!pool.empty()){staging=pool.back();pool.pop_back();}
        else if(!texture(device.Get(),D3D11_USAGE_STAGING,staging))return;
        context->CopyResource(staging.Get(),probe.Get());marks.push_back({name,staging});
    }
    void begin(ID3D11DeviceContext* context){configure();if(!enabled)return;marks.clear();mark(context,"begin");}
    // Returns a summary line every two seconds, otherwise an empty string.
    // gpu_frame_ms spans the first to the last mark; wait_ms is how long the
    // CPU waited at collection for the frame's GPU work to drain.
    std::string collect(ID3D11DeviceContext* context){
        if(!enabled||!context||marks.empty())return {};
        context->Flush();
        LARGE_INTEGER previous{},first{},start{};QueryPerformanceCounter(&start);
        for(std::size_t i=0;i<marks.size();++i){
            D3D11_MAPPED_SUBRESOURCE mapped{};
            if(SUCCEEDED(context->Map(marks[i].staging.Get(),0,D3D11_MAP_READ,0,&mapped)))context->Unmap(marks[i].staging.Get(),0);
            LARGE_INTEGER now{};QueryPerformanceCounter(&now);
            if(i){double ms=1000.0*double(now.QuadPart-previous.QuadPart)/double(frequency.QuadPart);
                Total* total=nullptr;for(auto& t:totals)if(t.name==marks[i].name){total=&t;break;}
                if(!total){totals.push_back({marks[i].name});total=&totals.back();}
                total->ms+=ms;++total->count;}
            else first=now;
            previous=now;pool.push_back(marks[i].staging);
        }
        frame_ms+=1000.0*double(previous.QuadPart-first.QuadPart)/double(frequency.QuadPart);
        wait_ms+=1000.0*double(previous.QuadPart-start.QuadPart)/double(frequency.QuadPart);++frames;
        marks.clear();
        LARGE_INTEGER now{};QueryPerformanceCounter(&now);
        if(now.QuadPart-reported.QuadPart<frequency.QuadPart*2)return {};
        reported=now;std::string line;char item[96];
        std::snprintf(item,sizeof(item),"frames=%u gpu_frame_ms=%.2f wait_ms=%.2f",frames,frames?frame_ms/frames:0.0,frames?wait_ms/frames:0.0);line+=item;
        for(auto& t:totals){std::snprintf(item,sizeof(item)," %s_ms=%.2f",t.name,t.count?t.ms/t.count:0.0);line+=item;t.ms=0;t.count=0;}
        frames=0;frame_ms=0;wait_ms=0;return line;
    }
};
inline GpuEventTimeline& gpu_timeline(){static GpuEventTimeline timeline;return timeline;}
}}
