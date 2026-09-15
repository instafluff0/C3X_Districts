// Isolated downstream feasibility probe. Never attaches to Civ III or loads its DLLs.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include "gpu_image_compositor.h"

using Microsoft::WRL::ComPtr;
using Pixel = std::uint32_t;
void check(HRESULT result, char const* operation) {
    if (FAILED(result)) { char error[160]; std::snprintf(error,sizeof(error),"%s HRESULT=%08lx",operation,result); throw std::runtime_error(error); }
}
void require(bool result, char const* operation) { if(!result) throw std::runtime_error(operation); }
double now() { LARGE_INTEGER t,f; QueryPerformanceCounter(&t); QueryPerformanceFrequency(&f); return double(t.QuadPart)*1000.0/double(f.QuadPart); }
std::size_t largest_free_region() {
    std::uintptr_t address=0;std::size_t largest=0;MEMORY_BASIC_INFORMATION info={};
    while(VirtualQuery(reinterpret_cast<void*>(address),&info,sizeof(info))==sizeof(info)){
        if(info.State==MEM_FREE)largest=std::max(largest,std::size_t(info.RegionSize));
        auto next=reinterpret_cast<std::uintptr_t>(info.BaseAddress)+info.RegionSize;if(next<=address)break;address=next;
    }return largest;
}
struct Dib {
    int width,height; HDC dc=nullptr; HBITMAP bitmap=nullptr; HGDIOBJ previous=nullptr; Pixel* pixels=nullptr;
    Dib(int w,int h):width(w),height(h) {
        BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=w;
        info.bmiHeader.biHeight=-h;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;info.bmiHeader.biCompression=BI_RGB;
        dc=CreateCompatibleDC(nullptr);require(dc!=nullptr,"CreateCompatibleDC");
        bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,reinterpret_cast<void**>(&pixels),nullptr,0);
        require(bitmap!=nullptr,"CreateDIBSection");previous=SelectObject(dc,bitmap);
    }
    ~Dib(){if(previous)SelectObject(dc,previous);if(bitmap)DeleteObject(bitmap);if(dc)DeleteDC(dc);}
    Dib(Dib const&)=delete;Dib& operator=(Dib const&)=delete;
};
struct Texture {
    ComPtr<ID3D11Texture2D> image; ComPtr<ID3D11RenderTargetView> target; ComPtr<ID3D11ShaderResourceView> view;
};
struct Window {
    HWND handle=nullptr;
    Window(int width,int height) {
        WNDCLASSA cls={};cls.lpfnWndProc=DefWindowProcA;cls.hInstance=GetModuleHandleA(nullptr);cls.lpszClassName="C3XCompositionProbe";
        require(RegisterClassA(&cls)!=0,"RegisterClass"); RECT rect={0,0,width,height};AdjustWindowRect(&rect,WS_OVERLAPPEDWINDOW,FALSE);
        handle=CreateWindowA(cls.lpszClassName,"C3X isolated composition probe",WS_OVERLAPPEDWINDOW,
            0,0,rect.right-rect.left,rect.bottom-rect.top,nullptr,nullptr,cls.hInstance,nullptr);
        require(handle!=nullptr,"CreateWindow");ShowWindow(handle,SW_SHOWNOACTIVATE);
    }
    ~Window(){if(handle)DestroyWindow(handle);UnregisterClassA("C3XCompositionProbe",GetModuleHandleA(nullptr));}
    void pump(){MSG msg;while(PeekMessage(&msg,nullptr,0,0,PM_REMOVE)){TranslateMessage(&msg);DispatchMessage(&msg);}}
};
char const* shader=R"(
Texture2D<uint> image : register(t0);
float4 vs(uint id:SV_VertexID):SV_Position {
    return float4((id==1||id==3)?1:-1,id>=2?-1:1,0,1);
}
float4 ps(float4 p:SV_Position):SV_Target {
    uint v=image.Load(int3(int2(p.xy),0));
    return float4(float3((v>>16)&255,(v>>8)&255,v&255)/255,1);
})";
struct Probe {
    int width,height;Window window;ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;ComPtr<IDXGISwapChain> swap;
    ComPtr<ID3D11Texture2D> back,staging,marker;ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11RasterizerState> raster;
    std::unique_ptr<c3x_gpu_images::Compositor> gpu;
    c3x_gpu_images::Id gpu_map=0,gpu_frame=0,gpu_save=0,gpu_art=0;
    Texture map,frame;ComPtr<IDXGISurface1> surface;
    Dib transfer,canvas,save,art;std::vector<Pixel> publication;std::size_t payload_bytes;
    std::string adapter;
    Probe(int w,int h,char const* map_path):width(w),height(h),window(w,h),transfer(w,h),canvas(w,h),save(w,h),art(256,96),publication(std::size_t(w)*h),payload_bytes(publication.size()*4) {
        check(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context),"CreateDevice");
        ComPtr<IDXGIDevice> dxgi;check(device.As(&dxgi),"DXGIDevice");ComPtr<IDXGIAdapter> a;check(dxgi->GetAdapter(&a),"GetAdapter");DXGI_ADAPTER_DESC ad={};check(a->GetDesc(&ad),"GetDesc");
        char text[256];WideCharToMultiByte(CP_UTF8,0,ad.Description,-1,text,sizeof(text),nullptr,nullptr);adapter=text;
        ComPtr<IDXGIFactory> factory;check(a->GetParent(IID_PPV_ARGS(&factory)),"GetParent");check(factory->MakeWindowAssociation(window.handle,DXGI_MWA_NO_ALT_ENTER),"MakeWindowAssociation");
        map=texture(w,h);frame=texture(w,h,true);check(frame.image.As(&surface),"IDXGISurface1");
        D3D11_TEXTURE2D_DESC sd={};frame.image->GetDesc(&sd);sd.BindFlags=sd.MiscFlags=0;sd.Usage=D3D11_USAGE_STAGING;sd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        check(device->CreateTexture2D(&sd,nullptr,&staging),"Create staging");sd.Width=sd.Height=1;check(device->CreateTexture2D(&sd,nullptr,&marker),"Create completion marker");
        ComPtr<ID3DBlob> vb,pb,error;
        check(D3DCompile(shader,std::strlen(shader),"composition probe",nullptr,nullptr,"vs","vs_4_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&vb,&error),"Compile VS");
        check(D3DCompile(shader,std::strlen(shader),"composition probe",nullptr,nullptr,"ps","ps_4_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&pb,&error),"Compile PS");
        check(device->CreateVertexShader(vb->GetBufferPointer(),vb->GetBufferSize(),nullptr,&vs),"Create VS");
        check(device->CreatePixelShader(pb->GetBufferPointer(),pb->GetBufferSize(),nullptr,&ps),"Create PS");
        D3D11_RASTERIZER_DESC rd={};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;rd.ScissorEnable=TRUE;rd.DepthClipEnable=TRUE;
        check(device->CreateRasterizerState(&rd,&raster),"Create rasterizer");
        HBITMAP source_bitmap=static_cast<HBITMAP>(LoadImageA(nullptr,map_path,IMAGE_BITMAP,0,0,LR_LOADFROMFILE|LR_CREATEDIBSECTION));
        require(source_bitmap!=nullptr,"Load preserved renderer map BMP");HDC source_dc=CreateCompatibleDC(nullptr);require(source_dc!=nullptr,"Create map DC");
        HGDIOBJ old=SelectObject(source_dc,source_bitmap);BITMAP bm={};GetObject(source_bitmap,sizeof(bm),&bm);
        BOOL loaded=StretchBlt(transfer.dc,0,0,w,h,source_dc,0,0,bm.bmWidth,bm.bmHeight,SRCCOPY);
        GdiFlush();SelectObject(source_dc,old);DeleteDC(source_dc);DeleteObject(source_bitmap);require(loaded!=FALSE,"Load map pixels");
        for(std::size_t i=0;i<publication.size();++i)transfer.pixels[i]|=0xff000000;
        context->UpdateSubresource(map.image.Get(),0,nullptr,transfer.pixels,w*4,0);
        std::fill(art.pixels,art.pixels+256*96,0xffff00ff);
        for(int y=0;y<32;++y)for(int x=0;x<32;++x)if((x-16)*(x-16)+(y-16)*(y-16)<200)art.pixels[y*256+x]=0xff80d020;
        SetBkMode(art.dc,TRANSPARENT);SetTextColor(art.dc,RGB(245,245,230));
        LOGFONTA lf={};lf.lfHeight=-18;lf.lfQuality=NONANTIALIASED_QUALITY;strcpy_s(lf.lfFaceName,"Arial");HFONT font=CreateFontIndirectA(&lf);require(font!=nullptr,"Create font");
        HGDIOBJ oldfont=SelectObject(art.dc,font);require(TextOutA(art.dc,0,40,"Complete frame: city / UI",25)!=FALSE,"TextOut");GdiFlush();SelectObject(art.dc,oldfont);DeleteObject(font);
        gpu=std::make_unique<c3x_gpu_images::Compositor>(device.Get(),context.Get());
        using c3x_gpu_images::Format;
        gpu_map=gpu->create(w,h,Format::bgra32);gpu_frame=gpu->create(w,h,Format::bgra32);
        gpu_save=gpu->create(w,h,Format::bgra32);gpu_art=gpu->create(256,96,Format::bgra32);
        require(gpu_map&&gpu_frame&&gpu_save&&gpu_art,"GPU image budget");
        require(gpu->upload(gpu_map,1,transfer.pixels,publication.size()),"Resident map seed");
        require(gpu->upload(gpu_art,1,art.pixels,256*96),"Resident CPU UI source");
        complete();
    }
    ~Probe(){if(context){context->ClearState();context->Flush();}}
    void ensure_presenter() {
        if(swap)return;
        ComPtr<IDXGIDevice> dxgi;check(device.As(&dxgi),"Presenter device");ComPtr<IDXGIAdapter> a;check(dxgi->GetAdapter(&a),"Presenter adapter");
        ComPtr<IDXGIFactory> factory;check(a->GetParent(IID_PPV_ARGS(&factory)),"Presenter factory");
        DXGI_SWAP_CHAIN_DESC desc={};desc.BufferDesc.Width=width;desc.BufferDesc.Height=height;desc.BufferDesc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count=1;desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.BufferCount=1;desc.OutputWindow=window.handle;
        desc.Windowed=TRUE;desc.SwapEffect=DXGI_SWAP_EFFECT_DISCARD;
        check(factory->CreateSwapChain(device.Get(),&desc,&swap),"CreateSwapChain");check(swap->GetBuffer(0,IID_PPV_ARGS(&back)),"GetBuffer");
    }
    Texture texture(int w,int h,bool gdi=false) {
        Texture t;D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
        d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;d.MiscFlags=gdi?D3D11_RESOURCE_MISC_GDI_COMPATIBLE:0;
        check(device->CreateTexture2D(&d,nullptr,&t.image),"Create texture");check(device->CreateRenderTargetView(t.image.Get(),nullptr,&t.target),"Create RTV");
        check(device->CreateShaderResourceView(t.image.Get(),nullptr,&t.view),"Create SRV");return t;
    }
    void unbind(){ID3D11ShaderResourceView* empty=nullptr;context->PSSetShaderResources(0,1,&empty);context->OMSetRenderTargets(0,nullptr,nullptr);}
    void finish_gpu() {
        ID3D11RenderTargetView* target=frame.target.Get();context->OMSetRenderTargets(1,&target,nullptr);context->OMSetBlendState(nullptr,nullptr,~0u);context->OMSetDepthStencilState(nullptr,0);
        D3D11_VIEWPORT viewport={0,0,float(width),float(height),0,1};RECT clip={0,0,width,height};context->RSSetViewports(1,&viewport);context->RSSetState(raster.Get());context->RSSetScissorRects(1,&clip);
        context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);context->VSSetShader(vs.Get(),nullptr,0);context->PSSetShader(ps.Get(),nullptr,0);
        auto read=gpu->view(gpu_frame);context->PSSetShaderResources(0,1,&read);context->Draw(4,0);unbind();
    }
    void cpu_operations(HDC dc,int phase) {
        int saved=SaveDC(dc);require(saved!=0,"SaveDC");RECT clip={13,17,width-11,height-9};IntersectClipRect(dc,clip.left,clip.top,clip.right,clip.bottom);
        RECT panel={24,24,324,108};HBRUSH brush=CreateSolidBrush(RGB(30,42,58));require(brush!=nullptr,"CreateSolidBrush");require(FillRect(dc,&panel,brush)!=0,"FillRect");DeleteObject(brush);
        require(TransparentBlt(dc,32,40,256,40,art.dc,0,36,256,40,RGB(255,0,255))!=FALSE,"Text image blit");
        for(int i=0;i<64;++i){int x=(i*97+phase*7)%(width+24)-12,y=(i*53)%(height+24)-12;
            require(TransparentBlt(dc,x,y,32,32,art.dc,0,0,32,32,RGB(255,0,255))!=FALSE,"Sprite blit");}
        RECT popup={width/3,height/3,width/3+260,height/3+100};
        require(BitBlt(save.dc,popup.left,popup.top,260,100,dc,popup.left,popup.top,SRCCOPY)!=FALSE,"Save background");
        require(PatBlt(dc,popup.left,popup.top,260,100,WHITENESS)!=FALSE,"Popup fill");
        require(TransparentBlt(dc,popup.left+4,popup.top+4,256,40,art.dc,0,36,256,40,RGB(255,0,255))!=FALSE,"Popup text");
        if((phase&1)==0)require(BitBlt(dc,popup.left,popup.top,260,100,save.dc,popup.left,popup.top,SRCCOPY)!=FALSE,"Restore background");
        require(PatBlt(dc,60,height-70,120,32,DSTINVERT)!=FALSE,"Destination raster operation");
        require(RestoreDC(dc,saved)!=FALSE,"RestoreDC");
    }
    void gpu_operations(int phase) {
        using namespace c3x_gpu_images;
        Rect all={0,0,width,height},clip={13,17,width-11,height-9};
        std::vector<Command> commands;commands.reserve(72);
        commands.push_back({Kind::copy,gpu_frame,gpu_map,all,all});
        commands.push_back({Kind::fill,gpu_frame,0,{24,24,324,108},clip,0,0,0xff1e2a3a});
        commands.push_back({Kind::color_key,gpu_frame,gpu_art,{32,40,288,80},clip,0,36,0xffff00ff});
        for(int i=0;i<64;++i){int x=(i*97+phase*7)%(width+24)-12,y=(i*53)%(height+24)-12;
            commands.push_back({Kind::color_key,gpu_frame,gpu_art,{x,y,x+32,y+32},clip,0,0,0xffff00ff});}
        Rect popup={width/3,height/3,width/3+260,height/3+100};
        commands.push_back({Kind::copy,gpu_save,gpu_frame,popup,all,popup.left,popup.top});
        commands.push_back({Kind::fill,gpu_frame,0,popup,clip,0,0,0xffffffff});
        commands.push_back({Kind::color_key,gpu_frame,gpu_art,{popup.left+4,popup.top+4,popup.left+260,popup.top+44},clip,0,36,0xffff00ff});
        if((phase&1)==0)commands.push_back({Kind::copy,gpu_frame,gpu_save,popup,clip,popup.left,popup.top});
        commands.push_back({Kind::invert,gpu_frame,0,{60,height-70,180,height-38},clip,0,0,0x00ffffff});
        require(gpu->submit(commands.data(),commands.size()),"Admitted GPU image transaction");finish_gpu();
    }
    void read(ID3D11Texture2D* source,Pixel* output) {
        unbind();context->CopyResource(staging.Get(),source);D3D11_MAPPED_SUBRESOURCE mapped={};check(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),"Map readback");
        for(int y=0;y<height;++y)std::memcpy(output+std::size_t(y)*width,static_cast<char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,std::size_t(width)*4);
        context->Unmap(staging.Get(),0);
    }
    // Identical diagnostic barrier for all arms; deliberately reported separately.
    // This is not a production readback and does not measure physical scanout.
    double complete() {
        double start=now();unbind();D3D11_BOX box={0,0,0,1,1,1};context->CopySubresourceRegion(marker.Get(),0,0,0,0,frame.image.Get(),0,&box);
        D3D11_MAPPED_SUBRESOURCE m={};check(context->Map(marker.Get(),0,D3D11_MAP_READ,0,&m),"Completion marker");volatile Pixel p=*static_cast<Pixel*>(m.pData);(void)p;context->Unmap(marker.Get(),0);return now()-start;
    }
    std::array<double,6> run(int arm,int phase,bool present) {
        window.pump();if(present)ensure_presenter();double begin=now(),acquire=0,draw=0,upload=0;unbind();if(arm!=2)context->CopyResource(frame.image.Get(),map.image.Get());
        if(arm==0){read(frame.image.Get(),publication.data());std::memcpy(transfer.pixels,publication.data(),payload_bytes);
            require(BitBlt(canvas.dc,0,0,width,height,transfer.dc,0,0,SRCCOPY)!=FALSE,"Native canvas blit");acquire=now()-begin;
            double t=now();cpu_operations(canvas.dc,phase);GdiFlush();draw=now()-t;t=now();
            context->UpdateSubresource(frame.image.Get(),0,nullptr,canvas.pixels,width*4,0);upload=now()-t;
        }else if(arm==1){HDC dc=nullptr;check(surface->GetDC(FALSE,&dc),"GetDC(FALSE)");acquire=now()-begin;
            double t=now();cpu_operations(dc,phase);check(surface->ReleaseDC(nullptr),"ReleaseDC");draw=now()-t;
        }else{acquire=now()-begin;double t=now();gpu_operations(phase);draw=now()-t;}
        double t=now();if(present){context->CopyResource(back.Get(),frame.image.Get());HRESULT hr=swap->Present(0,0);check(hr,"Present");require(hr==S_OK,"Presentation occluded; reject timing");}double presented=now();
        return {presented-begin,acquire,draw,upload,presented-t,complete()};
    }
    void save_bmp(char const* path,std::vector<Pixel> const& pixels){
        BITMAPFILEHEADER f={};BITMAPINFOHEADER h={};f.bfType=0x4d42;f.bfOffBits=sizeof(f)+sizeof(h);f.bfSize=f.bfOffBits+DWORD(payload_bytes);
        h.biSize=sizeof(h);h.biWidth=width;h.biHeight=-height;h.biPlanes=1;h.biBitCount=32;h.biCompression=BI_RGB;
        FILE* file=nullptr;require(fopen_s(&file,path,"wb")==0,"Open BMP");
        bool ok=fwrite(&f,sizeof(f),1,file)==1&&fwrite(&h,sizeof(h),1,file)==1&&fwrite(pixels.data(),payload_bytes,1,file)==1;
        fclose(file);require(ok,"Write BMP");
    }
};
void samples(FILE* f,char const* name,std::vector<double> values){
    std::sort(values.begin(),values.end());double sum=0;for(double v:values)sum+=v;
    std::fprintf(f,"\"%s\":{\"mean\":%.6f,\"median\":%.6f,\"p95\":%.6f,\"max\":%.6f}",name,sum/values.size(),values[values.size()/2],values[(values.size()*95-1)/100],values.back());
}
int main(int argc,char** argv){
    if(argc!=6){std::fprintf(stderr,"usage: benchmark_gpu_composition MAP.bmp RESULTS.json WIDTH HEIGHT PRESENT\n");return 2;}
    try{
        int width=std::stoi(argv[3]),height=std::stoi(argv[4]);require(width>=640&&height>=480&&width<=2240&&height<=1192,"Viewport budget");
        bool present=std::stoi(argv[5])!=0;
        double start=now();Probe p(width,height,argv[1]);double setup=now()-start;
        std::printf("SETUP width=%d height=%d ms=%.3f adapter=%s\n",width,height,setup,p.adapter.c_str());std::fflush(stdout);
        std::vector<Pixel> expected(std::size_t(width)*height),actual(expected.size());
        for(int phase=0;phase<6;++phase){
            p.run(0,phase,false);p.read(p.frame.image.Get(),expected.data());
            for(int arm=1;arm<3;++arm){p.run(arm,phase,false);p.read(p.frame.image.Get(),actual.data());std::size_t differences=0;
                for(std::size_t i=0;i<actual.size();++i)if((actual[i]&0xffffff)!=(expected[i]&0xffffff))++differences;
                std::printf("PARITY phase=%d arm=%d rgb_differences=%zu\n",phase,arm,differences);
                if(differences){p.save_bmp((std::string(argv[2])+".expected.bmp").c_str(),expected);p.save_bmp((std::string(argv[2])+".actual.bmp").c_str(),actual);throw std::runtime_error("RGB parity failure; alpha is not a native display channel");}
            }
        }
        p.save_bmp((std::string(argv[2])+".bmp").c_str(),actual);
        std::array<std::array<std::vector<double>,7>,3> timings;
        double warm=now();for(int arm=0;arm<3;++arm)for(int i=0;i<8;++i)p.run(arm,i,present);warm=now()-warm;
        std::size_t minimum_free=largest_free_region();
        // Three rotations avoid consistently favoring the last arm; all cases serial.
        for(int block=0;block<3;++block)for(int order=0;order<3;++order){int arm=(order+block)%3;
            for(int i=0;i<24;++i){auto t=p.run(arm,i,present);for(unsigned j=0;j<6;++j)timings[arm][j].push_back(t[j]);timings[arm][6].push_back(t[0]+t[5]);}
            minimum_free=std::min(minimum_free,largest_free_region());
            std::printf("BLOCK rotation=%d arm=%d complete\n",block,arm);std::fflush(stdout);}
        require(minimum_free>=512u*1024u*1024u,"32-bit address-space headroom");
        // Off-screen timing still executes one complete presentation as an untimed witness.
        if(!present)p.run(2,5,true);
        FILE* file=nullptr;require(fopen_s(&file,argv[2],"wb")==0,"Open results");
        std::fprintf(file,"{\"schema\":2,\"width\":%d,\"height\":%d,\"present_in_timing\":%s,\"minimum_largest_free_bytes\":%zu,\"setup_ms\":%.3f,\"warmup_ms\":%.3f,\"rgb_parity\":\"exact_6_phases\",\"alpha\":\"ignored_native_display_channel\",\"samples_per_arm\":72,\"arms\":[",width,height,present?"true":"false",minimum_free,setup,warm);
        char const* names[]={"cpu_readback_gdi_upload","dxgi_gdi","gpu_primitives"};char const* phases[]={"submit_ms","acquire_ms","compose_ms","cpu_presenter_upload_ms","present_ms","completion_probe_ms","synchronized_total_ms"};
        for(int arm=0;arm<3;++arm){std::fprintf(file,"%s{\"name\":\"%s\",\"explicit_map_readback_bytes_per_frame\":%zu,\"diagnostic_readback_bytes_per_frame\":4,",arm?",":"",names[arm],arm==0?p.payload_bytes:0);
            for(int i=0;i<7;++i){if(i)std::fputc(',',file);samples(file,phases[i],timings[arm][i]);}
            std::fprintf(file,",\"synchronized_samples_ms\":[");for(std::size_t i=0;i<timings[arm][6].size();++i)std::fprintf(file,"%s%.6f",i?",":"",timings[arm][6][i]);
            std::fprintf(file,"]}");}
        std::fprintf(file,"]}\n");fclose(file);std::puts("PASS complete composition probe; synthetic downstream workload, not gameplay");return 0;
    }catch(std::exception const& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
