#pragma once
#include <windows.h>
#include <d3d11.h>
#include <dcomp.h>
#include <wrl/client.h>

namespace c3x_remote_scene {
// Trial only: x86 owns the HWND and its composition target. The duplicated
// surface handle lets x64 own the swap chain and advance pixels independently.
class DirectSurface {
    Microsoft::WRL::ComPtr<IDCompositionDevice> composition;
    Microsoft::WRL::ComPtr<IDCompositionTarget> target;
    Microsoft::WRL::ComPtr<IDCompositionVisual> visual;
    Microsoft::WRL::ComPtr<IUnknown> wrapper;
    HANDLE surface=nullptr;
    HWND window=nullptr;
    DWORD owner=0;
    unsigned width=0,height=0;
    bool active=false;
public:
    ~DirectSurface(){reset();}
    HANDLE handle()const{return surface;}
    bool caller_thread()const{return !owner||owner==GetCurrentThreadId();}
    bool matches(HWND value,unsigned w,unsigned h)const{
        return surface&&window==value&&width==w&&height==h;
    }
    void reset(){
        if(active&&target){target->SetRoot(nullptr);composition->Commit();composition->WaitForCommitCompletion();}
        visual.Reset();target.Reset();wrapper.Reset();composition.Reset();
        if(surface){CloseHandle(surface);surface=nullptr;}
        window=nullptr;owner=0;width=height=0;active=false;
    }
    bool activate(){
        if(active)return true;
        if(!target||!visual||!composition)return false;
        HRESULT hr=target->SetRoot(visual.Get());
        if(SUCCEEDED(hr))hr=composition->Commit();
        active=SUCCEEDED(hr);return active;
    }
    bool prepare(HWND value,ID3D11Device* device,unsigned w,unsigned h){
        if(matches(value,w,h))return true;
        reset();
        if(!value||!device||!w||!h)return false;
        DWORD process=0;
        if(GetWindowThreadProcessId(value,&process)!=GetCurrentThreadId()||process!=GetCurrentProcessId())
            return false;
        Microsoft::WRL::ComPtr<IDXGIDevice> dxgi;
        HRESULT hr=device->QueryInterface(IID_PPV_ARGS(&dxgi));
        if(SUCCEEDED(hr))hr=DCompositionCreateDevice(dxgi.Get(),IID_PPV_ARGS(&composition));
        if(SUCCEEDED(hr))hr=DCompositionCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS,nullptr,&surface);
        if(SUCCEEDED(hr))hr=composition->CreateSurfaceFromHandle(surface,&wrapper);
        if(SUCCEEDED(hr))hr=composition->CreateTargetForHwnd(value,FALSE,&target);
        if(SUCCEEDED(hr))hr=composition->CreateVisual(&visual);
        if(SUCCEEDED(hr))hr=visual->SetContent(wrapper.Get());
        if(FAILED(hr)){reset();return false;}
        window=value;owner=GetCurrentThreadId();width=w;height=h;return true;
    }
};
}
