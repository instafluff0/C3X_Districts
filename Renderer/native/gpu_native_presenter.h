#pragma once
#include "gpu_frame_api.h"
#include "gpu_image_compositor.h"
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <dcomp.h>
#pragma comment(lib,"dcomp.lib")
namespace c3x_gpu_images {
// One native-owned HWND; no window creation, message loop or redraw callback.
// Native lifecycle changes stay on the caller. The composition swap chain has
// no HWND/message-pump dependency during Present; an independently scheduled
// visual frame can present under the same serialized ownership gate.
class NativePresenter {
    ComPtr<IDXGISwapChain1> swap;
    ComPtr<IDCompositionDevice> composition;
    ComPtr<IDCompositionTarget> composition_target;
    ComPtr<IDCompositionVisual> visual;
    ComPtr<ID3D11Texture2D> back,display;
    ComPtr<ID3D11RenderTargetView> target;
    std::vector<unsigned short> native_pixels;
    std::vector<unsigned> fallback_pixels;
    unsigned native_format=1;
    ComPtr<ID3D11Texture2D> native_upload;ComPtr<ID3D11ShaderResourceView> native_view;
    ImageDisplay native_program;
    HWND window=nullptr;DWORD owner=0;unsigned width=0,height=0;
public:
    bool initialized=false;
    bool caller_thread()const{return !owner||owner==GetCurrentThreadId();}
    void reset(){
        if(composition_target){composition_target->SetRoot(nullptr);if(composition){composition->Commit();composition->WaitForCommitCompletion();}}
        visual.Reset();composition_target.Reset();composition.Reset();
        fallback_pixels.clear();native_pixels.clear();native_view.Reset();native_upload.Reset();target.Reset();display.Reset();back.Reset();swap.Reset();window=nullptr;owner=0;width=height=0;initialized=false;
    }
    // Switching a partial transfer back to GDI must preserve the last displayed
    // pixels outside its rectangle. The CPU compatibility route already owns
    // these bytes; no GPU readback or repaint request is needed.
    void release_native(){
        auto hwnd=window;auto w=width,h=height,format=native_format;auto pixels=std::move(native_pixels);auto rgb=std::move(fallback_pixels);
        reset();
        if(hwnd&&IsWindow(hwnd)&&rgb.size()==std::size_t(w)*h){
            auto dc=GetDC(hwnd);if(dc){BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
                info.bmiHeader.biWidth=LONG(w);info.bmiHeader.biHeight=-LONG(h);info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
                SetDIBitsToDevice(dc,0,0,w,h,0,0,0,h,rgb.data(),&info,DIB_RGB_COLORS);GdiFlush();ReleaseDC(hwnd,dc);}
            return;
        }
        if(hwnd&&IsWindow(hwnd)&&pixels.size()==std::size_t((w+1)&~1u)*h){
            auto dc=GetDC(hwnd);if(dc){struct Info {BITMAPINFOHEADER header;DWORD masks[3];} info={};info.header.biSize=sizeof(BITMAPINFOHEADER);
                info.header.biWidth=LONG(w);info.header.biHeight=-LONG(h);info.header.biPlanes=1;info.header.biBitCount=16;info.header.biCompression=BI_BITFIELDS;
                info.masks[0]=format==2?0xf800:0x7c00;info.masks[1]=format==2?0x7e0:0x3e0;info.masks[2]=0x1f;
                SetDIBitsToDevice(dc,0,0,w,h,0,0,0,h,pixels.data(),reinterpret_cast<BITMAPINFO*>(&info),DIB_RGB_COLORS);GdiFlush();ReleaseDC(hwnd,dc);}
        }
    }
    // Worker-only, and only on an explicit return to native drawing. The live
    // GPU path has no CPU display shadow; transfer exactly the last displayed
    // surface, rather than the possibly newer native working canvas.
    bool preserve_display(ID3D11DeviceContext* context){
        if(!initialized||!display||!native_pixels.empty())return true;
        if(!context)return false;
        ComPtr<ID3D11Device> device;display->GetDevice(&device);
        D3D11_TEXTURE2D_DESC desc={};display->GetDesc(&desc);
        desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;desc.MiscFlags=0;
        ComPtr<ID3D11Texture2D> stage;checked(device->CreateTexture2D(&desc,nullptr,&stage));
        std::vector<unsigned> pixels(std::size_t(width)*height);
        context->CopyResource(stage.Get(),display.Get());D3D11_MAPPED_SUBRESOURCE mapped={};
        checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&mapped));
        for(unsigned y=0;y<height;++y)std::memcpy(pixels.data()+std::size_t(y)*width,static_cast<char*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,width*4);
        context->Unmap(stage.Get(),0);fallback_pixels=std::move(pixels);return true;
    }
    void gpu_written(){native_pixels.clear();fallback_pixels.clear();}
    bool matches(HWND hwnd,ID3D11Device* device,unsigned w,unsigned h){
        if(!swap||window!=hwnd||owner!=GetCurrentThreadId()||width!=w||height!=h)return false;
        ComPtr<ID3D11Device> existing;back->GetDevice(&existing);return device==existing.Get();
    }
    bool prepare(HWND hwnd,ID3D11Device* device,unsigned w,unsigned h,bool full){
        DWORD process=0;
        if(!hwnd||!device||!w||!h||w>2240||h>1260||GetWindowThreadProcessId(hwnd,&process)!=GetCurrentThreadId()||process!=GetCurrentProcessId())return false;
        RECT client={};if(!GetClientRect(hwnd,&client)||client.right!=int(w)||client.bottom!=int(h))return false;
        if(matches(hwnd,device,w,h))return initialized||full;
        if(!full)return false;
        release_native();
        ComPtr<IDXGIDevice> dxgi;checked(device->QueryInterface(IID_PPV_ARGS(&dxgi)));
        ComPtr<IDXGIAdapter> adapter;checked(dxgi->GetAdapter(&adapter));ComPtr<IDXGIFactory2> factory;checked(adapter->GetParent(IID_PPV_ARGS(&factory)));
        // This flip chain targets a composition visual, not Civ III's HWND.
        // Detaching the visual restores ordinary GDI on that same window.
        // Never use CreateSwapChainForHwnd/SetFullscreenState here.
        DXGI_SWAP_CHAIN_DESC1 desc={};desc.Width=w;desc.Height=h;desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count=1;desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.BufferCount=2;
        desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;desc.Scaling=DXGI_SCALING_STRETCH;desc.AlphaMode=DXGI_ALPHA_MODE_IGNORE;
        checked(factory->CreateSwapChainForComposition(device,&desc,nullptr,&swap));checked(swap->GetBuffer(0,IID_PPV_ARGS(&back)));
        checked(DCompositionCreateDevice(dxgi.Get(),IID_PPV_ARGS(&composition)));
        checked(composition->CreateTargetForHwnd(hwnd,FALSE,&composition_target));
        checked(composition->CreateVisual(&visual));checked(visual->SetContent(swap.Get()));
        D3D11_TEXTURE2D_DESC texture={};back->GetDesc(&texture);texture.BindFlags=D3D11_BIND_RENDER_TARGET;texture.MiscFlags=0;
        checked(device->CreateTexture2D(&texture,nullptr,&display));checked(device->CreateRenderTargetView(display.Get(),nullptr,&target));
        window=hwnd;owner=GetCurrentThreadId();width=w;height=h;return true;
    }
    ID3D11RenderTargetView* view()const{return target.Get();}
    ID3D11Texture2D* retained()const{return display.Get();}
    ID3D11Texture2D* buffer()const{return back.Get();}
    // A rare direct-surface -> native partial-transfer handoff seeds the
    // x86 presenter with the exact previous x64 frame before applying 16-bit
    // Civ III pixels. Ordinary direct frames never use this CPU path.
    bool seed_bgra(ID3D11DeviceContext* context,unsigned const* pixels,unsigned w,unsigned h){
        if(!context||!pixels||!display||!back||w!=width||h!=height)return false;
        context->UpdateSubresource(display.Get(),0,nullptr,pixels,w*4,0);
        context->CopyResource(back.Get(),display.Get());context->Flush();gpu_written();
        return present()==C3X_RENDERER_RESULT_OK;
    }
    // The helper owns scene composition; this process owns the Civ III window.
    // The duplicated handle is consumed exactly once. No CPU readback or HWND
    // crosses the process boundary, and no frame is adopted after a mismatch.
    int adopt_shared(ID3D11Device1* device,ID3D11DeviceContext* context,
                     std::uint64_t raw_handle,unsigned w,unsigned h,bool independent=false){
        HANDLE handle=reinterpret_cast<HANDLE>(std::uintptr_t(raw_handle));
        if(!handle)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        struct Close {HANDLE value;~Close(){CloseHandle(value);}} close{handle};
        if(!device||!context||!display||!back||w!=width||h!=height||
           (!independent&&owner!=GetCurrentThreadId())||(independent&&!initialized)||!swap)
            return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        ComPtr<ID3D11Texture2D> source;
        if(FAILED(device->OpenSharedResource1(handle,IID_PPV_ARGS(&source))))return C3X_RENDERER_RESULT_DEVICE_ERROR;
        D3D11_TEXTURE2D_DESC desc={};source->GetDesc(&desc);
        if(desc.Width!=w||desc.Height!=h||desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM)
            return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        ComPtr<IDXGIKeyedMutex> mutex;
        if(FAILED(source.As(&mutex))||mutex->AcquireSync(1,1000)!=S_OK)
            return C3X_RENDERER_RESULT_DEVICE_ERROR;
        context->CopyResource(display.Get(),source.Get());
        context->CopyResource(back.Get(),display.Get());
        auto released=mutex->ReleaseSync(0);context->Flush();
        if(FAILED(released)||FAILED(device->GetDeviceRemovedReason()))return C3X_RENDERER_RESULT_DEVICE_ERROR;
        gpu_written();return present(independent);
    }
    // Worker-only upload of a completed native CPU surface. Keep pixels outside
    // the native transfer rectangle from the previous displayed frame.
    bool upload_screen(ID3D11DeviceContext* context,unsigned short const* pixels,unsigned w,unsigned h,RECT area,unsigned format){
        if(!pixels||!display||!back||w!=width||h!=height||format<1||format>2)return false;
        bool full=area.left==0&&area.top==0&&area.right==int(w)&&area.bottom==int(h);
        if(!full && !initialized)return false;
        // A partial CPU source may overlay a full-color GPU display. Its CPU
        // bytes are a complete fallback only after a full transfer (or another
        // same-format CPU transfer); never fill the untouched region with zeros.
        bool complete_cpu=full||(!native_pixels.empty()&&native_format==format);
        ComPtr<ID3D11Device> device;display->GetDevice(&device);
        if(!native_upload){D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
            d.Format=DXGI_FORMAT_R16_UINT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            checked(device->CreateTexture2D(&d,nullptr,&native_upload));checked(device->CreateShaderResourceView(native_upload.Get(),nullptr,&native_view));}
        unsigned pitch=(w+1)&~1u;fallback_pixels.clear();
        if(complete_cpu){
            native_pixels.resize(std::size_t(pitch)*h);native_format=format;
            for(int y=area.top;y<area.bottom;++y)std::copy(pixels+std::size_t(y)*pitch+area.left,pixels+std::size_t(y)*pitch+area.right,
                native_pixels.data()+std::size_t(y)*pitch+area.left);
        }else native_pixels.clear();
        D3D11_BOX box={unsigned(area.left),unsigned(area.top),0,unsigned(area.right),unsigned(area.bottom),1};
        context->UpdateSubresource(native_upload.Get(),0,&box,pixels+std::size_t(area.top)*pitch+area.left,pitch*2,0);
        if(!native_program.draw(device.Get(),context,native_view.Get(),target.Get(),w,h,area,format))return false;
        context->CopyResource(back.Get(),display.Get());context->Flush();return true;
    }

    int present(bool independent=false){
        if(!swap||(!independent&&owner!=GetCurrentThreadId())||(independent&&!initialized))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        HRESULT hr=swap->Present(0,independent?DXGI_PRESENT_DO_NOT_WAIT:0);
        if(hr==DXGI_ERROR_WAS_STILL_DRAWING)return C3X_RENDERER_RESULT_PENDING;
        if(FAILED(hr))return C3X_RENDERER_RESULT_ERROR;
        if(!initialized){checked(composition_target->SetRoot(visual.Get()));checked(composition->Commit());}
        initialized=true;return C3X_RENDERER_RESULT_OK;
    }
};
}
