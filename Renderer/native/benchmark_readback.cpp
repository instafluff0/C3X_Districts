// Standalone transfer/synchronization floor. No window, game loop or presenter.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

struct Samples {
    std::vector<double> values;
    void report(char const* phase,int width,int height,int copied_height) {
        std::sort(values.begin(),values.end());
        std::printf("TRANSFER phase=%s viewport=%dx%d copied_height=%d samples=%zu median_ms=%.3f p95_ms=%.3f p99_ms=%.3f max_ms=%.3f\n",
            phase,width,height,copied_height,values.size(),(values[49]+values[50])*.5,values[94],values[98],values[99]);
    }
};
int main() {
    ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;D3D_FEATURE_LEVEL level;
    if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr,0,D3D11_SDK_VERSION,&device,&level,&context)))return 1;
    LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
    bool ok=true;
    for(auto size:{std::pair<int,int>{640,480},{2240,1192}}) {
        int width=size.first,height=size.second;
        // Match production's power-of-two targets, but copy only covered pixels.
        unsigned target_width=1,target_height=1;
        while(target_width<unsigned(width))target_width*=2;
        while(target_height<unsigned(height))target_height*=2;
        D3D11_TEXTURE2D_DESC desc={};desc.Width=target_width;desc.Height=target_height;desc.MipLevels=desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.SampleDesc.Count=1;
        desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_RENDER_TARGET;
        ID3D11Texture2D* source=nullptr;ID3D11Texture2D* staging=nullptr;ID3D11RenderTargetView* target=nullptr;
        HRESULT hr=device->CreateTexture2D(&desc,nullptr,&source);
        if(SUCCEEDED(hr))hr=device->CreateRenderTargetView(source,nullptr,&target);
        desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        if(SUCCEEDED(hr))hr=device->CreateTexture2D(&desc,nullptr,&staging);
        if(FAILED(hr)){ok=false;}else{
            float color[]={.25f,.5f,.75f,1};context->ClearRenderTargetView(target,color);
            std::vector<std::uint32_t> pixels(std::size_t(width)*height);
            for(int copied_height:{height,64}) {
                Samples enqueue,map,copy,total;
                for(unsigned step=0;step<101;++step){
                    LARGE_INTEGER start={},submitted={},mapped_at={},end={};QueryPerformanceCounter(&start);
                    D3D11_BOX box={0,0,0,unsigned(width),unsigned(copied_height),1};
                    context->CopySubresourceRegion(staging,0,0,0,0,source,0,&box);
                    QueryPerformanceCounter(&submitted);
                    D3D11_MAPPED_SUBRESOURCE mapped={};hr=context->Map(staging,0,D3D11_MAP_READ,0,&mapped);
                    QueryPerformanceCounter(&mapped_at);
                    if(FAILED(hr)){ok=false;break;}
                    for(int y=0;y<copied_height;++y)std::memcpy(pixels.data()+std::size_t(y)*width,
                        static_cast<char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,std::size_t(width)*4);
                    context->Unmap(staging,0);QueryPerformanceCounter(&end);
                    if(step){
                        double scale=1000.0/frequency.QuadPart;
                        enqueue.values.push_back((submitted.QuadPart-start.QuadPart)*scale);
                        map.values.push_back((mapped_at.QuadPart-submitted.QuadPart)*scale);
                        copy.values.push_back((end.QuadPart-mapped_at.QuadPart)*scale);
                        total.values.push_back((end.QuadPart-start.QuadPart)*scale);
                    }
                }
                if(ok){enqueue.report("enqueue",width,height,copied_height);map.report("map_wait",width,height,copied_height);
                    copy.report("cpu_copy",width,height,copied_height);total.report("total",width,height,copied_height);}
            }
        }
        if(target)target->Release();if(staging)staging->Release();if(source)source->Release();
    }
    context->Release();device->Release();
    std::puts(ok?"PASS standalone transfer floor; excludes rendering and native composition":"FAIL transfer floor");
    return ok?0:1;
}
