#pragma once
#include "gpu_frame_api.h"
#include "gpu_image_compositor.h"
#include <dxgi.h>
namespace c3x_gpu_images {
// One native-owned HWND; no window creation, message loop or redraw callback.
// Create/resize/Present/release run on its caller thread while RendererWorker is
// parked. Only the worker writes the retained display and swap-chain back buffer.
class NativePresenter {
    ComPtr<IDXGISwapChain> swap;
    ComPtr<ID3D11Texture2D> back,display;
    ComPtr<ID3D11RenderTargetView> target;
    std::vector<unsigned short> native_pixels;
    unsigned native_format=1;
    ComPtr<ID3D11Texture2D> native_upload;ComPtr<ID3D11ShaderResourceView> native_view;
    ImageDisplay native_program;
    HWND window=nullptr;DWORD owner=0;unsigned width=0,height=0;
public:
    bool initialized=false;
    bool caller_thread()const{return !owner||owner==GetCurrentThreadId();}
    void reset(){native_pixels.clear();native_view.Reset();native_upload.Reset();target.Reset();display.Reset();back.Reset();swap.Reset();window=nullptr;owner=0;width=height=0;initialized=false;}
    // Switching a partial transfer back to GDI must preserve the last displayed
    // pixels outside its rectangle. The CPU compatibility route already owns
    // these bytes; no GPU readback or repaint request is needed.
    void release_native(){
        auto hwnd=window;auto w=width,h=height,format=native_format;auto pixels=std::move(native_pixels);
        reset();
        if(hwnd&&IsWindow(hwnd)&&pixels.size()==std::size_t((w+1)&~1u)*h){
            auto dc=GetDC(hwnd);if(dc){struct Info {BITMAPINFOHEADER header;DWORD masks[3];} info={};info.header.biSize=sizeof(BITMAPINFOHEADER);
                info.header.biWidth=LONG(w);info.header.biHeight=-LONG(h);info.header.biPlanes=1;info.header.biBitCount=16;info.header.biCompression=BI_BITFIELDS;
                info.masks[0]=format==2?0xf800:0x7c00;info.masks[1]=format==2?0x7e0:0x3e0;info.masks[2]=0x1f;
                SetDIBitsToDevice(dc,0,0,w,h,0,0,0,h,pixels.data(),reinterpret_cast<BITMAPINFO*>(&info),DIB_RGB_COLORS);GdiFlush();ReleaseDC(hwnd,dc);}
        }
    }
    bool matches(HWND hwnd,ID3D11Device* device,unsigned w,unsigned h){
        if(!swap||window!=hwnd||owner!=GetCurrentThreadId()||width!=w||height!=h)return false;
        ComPtr<ID3D11Device> existing;back->GetDevice(&existing);return device==existing.Get();
    }
    bool prepare(HWND hwnd,ID3D11Device* device,unsigned w,unsigned h,bool full){
        DWORD process=0;
        if(!hwnd||!device||!w||!h||w>2240||h>1192||GetWindowThreadProcessId(hwnd,&process)!=GetCurrentThreadId()||process!=GetCurrentProcessId())return false;
        RECT client={};if(!GetClientRect(hwnd,&client)||client.right!=int(w)||client.bottom!=int(h))return false;
        if(matches(hwnd,device,w,h))return initialized||full;
        if(!full)return false;
        release_native();
        ComPtr<IDXGIDevice> dxgi;checked(device->QueryInterface(IID_PPV_ARGS(&dxgi)));
        ComPtr<IDXGIAdapter> adapter;checked(dxgi->GetAdapter(&adapter));ComPtr<IDXGIFactory> factory;checked(adapter->GetParent(IID_PPV_ARGS(&factory)));
        checked(factory->MakeWindowAssociation(hwnd,DXGI_MWA_NO_WINDOW_CHANGES|DXGI_MWA_NO_ALT_ENTER));
        // Blt-model windowed presentation permits clean return to native GDI.
        // Partial native transfers update 'display'; discard buffers never supply
        // preserved pixels. Do not change Civ III's display mode or window size.
        DXGI_SWAP_CHAIN_DESC desc={};desc.BufferDesc.Width=w;desc.BufferDesc.Height=h;desc.BufferDesc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count=1;desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.BufferCount=1;desc.OutputWindow=hwnd;desc.Windowed=TRUE;desc.SwapEffect=DXGI_SWAP_EFFECT_DISCARD;
        checked(factory->CreateSwapChain(device,&desc,&swap));checked(swap->GetBuffer(0,IID_PPV_ARGS(&back)));
        D3D11_TEXTURE2D_DESC texture={};back->GetDesc(&texture);texture.BindFlags=D3D11_BIND_RENDER_TARGET;texture.MiscFlags=0;
        checked(device->CreateTexture2D(&texture,nullptr,&display));checked(device->CreateRenderTargetView(display.Get(),nullptr,&target));
        window=hwnd;owner=GetCurrentThreadId();width=w;height=h;return true;
    }
    ID3D11RenderTargetView* view()const{return target.Get();}
    ID3D11Texture2D* retained()const{return display.Get();}
    ID3D11Texture2D* buffer()const{return back.Get();}
    // Worker-only upload of a completed native CPU surface. Keep pixels outside
    // the native transfer rectangle from the previous displayed frame.
    bool upload_screen(ID3D11DeviceContext* context,unsigned short const* pixels,unsigned w,unsigned h,RECT area,unsigned format){
        if(!pixels||!display||!back||w!=width||h!=height||format<1||format>2)return false;
        bool full=area.left==0&&area.top==0&&area.right==int(w)&&area.bottom==int(h);
        if(!full && (native_pixels.empty() || native_format!=format))return false;
        ComPtr<ID3D11Device> device;display->GetDevice(&device);
        if(!native_upload){D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
            d.Format=DXGI_FORMAT_R16_UINT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            checked(device->CreateTexture2D(&d,nullptr,&native_upload));checked(device->CreateShaderResourceView(native_upload.Get(),nullptr,&native_view));}
        unsigned pitch=(w+1)&~1u;native_pixels.resize(std::size_t(pitch)*h);native_format=format;
        for(int y=area.top;y<area.bottom;++y)std::copy(pixels+std::size_t(y)*pitch+area.left,pixels+std::size_t(y)*pitch+area.right,
            native_pixels.data()+std::size_t(y)*pitch+area.left);
        D3D11_BOX box={unsigned(area.left),unsigned(area.top),0,unsigned(area.right),unsigned(area.bottom),1};
        context->UpdateSubresource(native_upload.Get(),0,&box,pixels+std::size_t(area.top)*pitch+area.left,pitch*2,0);
        if(!native_program.draw(device.Get(),context,native_view.Get(),target.Get(),w,h,area,format))return false;
        context->CopyResource(back.Get(),display.Get());context->Flush();return true;
    }

    int present(){
        if(!swap||owner!=GetCurrentThreadId())return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        HRESULT hr=swap->Present(0,0);
        if(FAILED(hr))return C3X_RENDERER_RESULT_ERROR;
        initialized=true;return C3X_RENDERER_RESULT_OK;
    }
};
}
