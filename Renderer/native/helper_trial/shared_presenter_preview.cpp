#define NOMINMAX
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cstdio>
#include <stdexcept>
#include "../gpu_native_presenter.h"

using Microsoft::WRL::ComPtr;
void verify(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
int main(){
    HWND window=nullptr;
    try{
        window=CreateWindowExW(0,L"STATIC",L"C3X shared presenter preview",WS_POPUP,
                               0,0,64,64,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
        verify(window!=nullptr,"preview window");
        ComPtr<ID3D11Device> producer,consumer;
        ComPtr<ID3D11DeviceContext> producer_context,consumer_context;
        D3D_FEATURE_LEVEL feature={};
        verify(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&producer,&feature,&producer_context)),
            "producer device");
        verify(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&consumer,&feature,&consumer_context)),
            "consumer device");
        ComPtr<ID3D11Device1> consumer1;verify(SUCCEEDED(consumer.As(&consumer1)),"consumer device1");
        D3D11_TEXTURE2D_DESC desc={};desc.Width=desc.Height=64;
        desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags=D3D11_RESOURCE_MISC_SHARED_NTHANDLE|D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
        ComPtr<ID3D11Texture2D> shared;ComPtr<ID3D11RenderTargetView> view;
        verify(SUCCEEDED(producer->CreateTexture2D(&desc,nullptr,&shared)),"shared texture");
        verify(SUCCEEDED(producer->CreateRenderTargetView(shared.Get(),nullptr,&view)),"producer target");
        ComPtr<IDXGIKeyedMutex> mutex;ComPtr<IDXGIResource1> resource;
        verify(SUCCEEDED(shared.As(&mutex))&&SUCCEEDED(shared.As(&resource)),"shared interfaces");
        verify(SUCCEEDED(mutex->AcquireSync(0,1000)),"producer acquire");
        float color[4]={.25f,.5f,.75f,1.f};producer_context->ClearRenderTargetView(view.Get(),color);
        verify(SUCCEEDED(mutex->ReleaseSync(1)),"producer release");producer_context->Flush();
        HANDLE handle=nullptr;
        verify(SUCCEEDED(resource->CreateSharedHandle(nullptr,
            DXGI_SHARED_RESOURCE_READ|DXGI_SHARED_RESOURCE_WRITE,nullptr,&handle))&&handle,"shared handle");
        c3x_gpu_images::NativePresenter presenter;
        verify(presenter.prepare(window,consumer.Get(),64,64,true),"native presenter preparation");
        // adopt_shared consumes the handle, including on failure.
        verify(presenter.adopt_shared(consumer1.Get(),consumer_context.Get(),
            std::uint64_t(std::uintptr_t(handle)),64,64)==C3X_RENDERER_RESULT_OK,"shared frame adoption");
        desc.MiscFlags=desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        verify(SUCCEEDED(consumer->CreateTexture2D(&desc,nullptr,&staging)),"verification staging");
        consumer_context->CopyResource(staging.Get(),presenter.retained());
        D3D11_MAPPED_SUBRESOURCE mapped={};
        verify(SUCCEEDED(consumer_context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped)),"verification readback");
        auto pixel=*static_cast<unsigned const*>(mapped.pData);consumer_context->Unmap(staging.Get(),0);
        auto channel_close=[](unsigned value,unsigned expected){return value+1>=expected&&value<=expected+1;};
        verify(channel_close(pixel&255,191)&&channel_close((pixel>>8)&255,128)&&channel_close((pixel>>16)&255,64)&&
               ((pixel>>24)&255)==255,"imported BGRA pixel");
        presenter.reset();DestroyWindow(window);
        std::puts("PASS x86 presenter imported and displayed a shared BGRA frame");return 0;
    }catch(std::exception const& error){if(window)DestroyWindow(window);
        std::fprintf(stderr,"FAIL shared presenter: %s\n",error.what());return 1;}
}
