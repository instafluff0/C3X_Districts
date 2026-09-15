// Bounded GDI text precision diagnostic. Native font quality is never replaced.
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <cstring>
#include "native_text_raster.h"
#include "gpu_image_compositor.h"
struct Dib {
    HDC dc=CreateCompatibleDC(nullptr);HBITMAP bitmap=nullptr;HGDIOBJ old=nullptr;void* pixels=nullptr;
    Dib(int depth,bool green6){struct Info {BITMAPINFOHEADER h;DWORD masks[3];} info={};
        info.h.biSize=sizeof info.h;info.h.biWidth=320;info.h.biHeight=-48;info.h.biPlanes=1;info.h.biBitCount=WORD(depth);
        if(depth==16){info.h.biCompression=BI_BITFIELDS;info.masks[0]=green6?0xf800:0x7c00;info.masks[1]=green6?0x7e0:0x3e0;info.masks[2]=31;}
        bitmap=CreateDIBSection(dc,reinterpret_cast<BITMAPINFO*>(&info),DIB_RGB_COLORS,&pixels,nullptr,0);
        if(!bitmap||!pixels)throw std::runtime_error("DIB allocation");old=SelectObject(dc,bitmap);
    }
    ~Dib(){SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);}
};
unsigned expand(unsigned c,bool g6){unsigned b=c&31,g=(c>>5)&(g6?63:31),r=(c>>(g6?11:10))&31;
    return ((b<<3)|(b>>2))|((g6?((g<<2)|(g>>4)):((g<<3)|(g>>2)))<<8)|(((r<<3)|(r>>2))<<16);}
unsigned pack(unsigned c,bool g6){return (c>>3&31)|((c>>(g6?10:11)&(g6?63:31))<<5)|((c>>19&31)<<(g6?11:10));}
std::vector<unsigned> read_gpu(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* texture){
    D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> stage;c3x_gpu_images::checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
    D3D11_MAPPED_SUBRESOURCE m={};c3x_gpu_images::checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&m));std::vector<unsigned> pixels(d.Width*d.Height);
    for(unsigned y=0;y<d.Height;++y)std::memcpy(pixels.data()+y*d.Width,static_cast<char*>(m.pData)+y*m.RowPitch,d.Width*4);
    context->Unmap(stage.Get(),0);return pixels;
}
void save_comparison(unsigned short const* native,unsigned const* wide,std::vector<unsigned> const& gpu){
    std::vector<unsigned> pixels(320*48*3);
    for(unsigned n=0;n<320*48;++n){pixels[n]=expand(native[n],false);pixels[320*48+n]=wide[n];pixels[2*320*48+n]=gpu[n];}
    BITMAPFILEHEADER file={};file.bfType=0x4d42;file.bfOffBits=sizeof(file)+sizeof(BITMAPINFOHEADER);file.bfSize=file.bfOffBits+DWORD(pixels.size()*4);
    BITMAPINFOHEADER info={};info.biSize=sizeof(info);info.biWidth=320;info.biHeight=-144;info.biPlanes=1;info.biBitCount=32;
    FILE* output=nullptr;if(fopen_s(&output,"build/gpu-composition/native-text-comparison.bmp","wb")||!output)throw std::runtime_error("comparison output");
    std::fwrite(&file,sizeof(file),1,output);std::fwrite(&info,sizeof(info),1,output);std::fwrite(pixels.data(),4,pixels.size(),output);std::fclose(output);
}
int main(){try {
    using namespace c3x_gpu_images;Microsoft::WRL::ComPtr<ID3D11Device> device;Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL feature;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context));
    Compositor gpu(device.Get(),context.Get());
    for(bool g6:{false,true})for(int quality:{DEFAULT_QUALITY,ANTIALIASED_QUALITY,CLEARTYPE_QUALITY})for(int bk:{TRANSPARENT,OPAQUE}){
        Dib native(16,g6),wide(32,g6);LOGFONTA lf={};lf.lfHeight=-17;lf.lfWeight=700;lf.lfQuality=BYTE(quality);lf.lfOutPrecision=7;
        strcpy_s(lf.lfFaceName,"Arial");auto font=CreateFontIndirectA(&lf);
        auto a=SelectObject(native.dc,font),b=SelectObject(wide.dc,font);
        c3x_native_text::State state;SetBkMode(native.dc,bk);SetTextColor(native.dc,RGB(237,171,55));SetBkColor(native.dc,RGB(47,68,211));
        c3x_native_text::Raster raster;LARGE_INTEGER started,ended,frequency;QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&started);
        bool compiled=c3x_native_text::capture(native.dc,state)&&c3x_native_text::compile(native.dc,state,"Berlin: 6",9,raster);
        QueryPerformanceCounter(&ended);std::printf("compiled=%d curves=%zu area=%ux%u ms=%.3f\n",int(compiled),raster.curves.size()/17,raster.width,raster.height,1000.*(ended.QuadPart-started.QuadPart)/frequency.QuadPart);
        if(!compiled)throw std::runtime_error("native glyph compilation rejected");
        auto glyph=gpu.create(raster.width,raster.height,Format::bgra32),curves=gpu.create(17,unsigned(raster.curves.size()/17),Format::bgra32);
        auto destination=gpu.create(320,48,g6?Format::rgb565:Format::rgb555),detail=gpu.create(320,48,Format::bgra32);
        if(!glyph||!curves||!destination||!detail||!gpu.upload(glyph,1,raster.pixels.data(),raster.pixels.size())||!gpu.upload(curves,1,raster.curves.data(),raster.curves.size()))throw std::runtime_error("glyph GPU upload");
        unsigned mismatch=0,partial=0,changed_high=0,response_error=0,response_pixels=0;
        for(int background=0;background<4;++background){
            std::vector<unsigned short> before(320*48);
            for(unsigned n=0;n<before.size();++n){unsigned c=background==0?0:background==1?0x4210:background==2?0xc210:(n*3137)&65535;
                before[n]=static_cast<unsigned short>(c);static_cast<unsigned short*>(native.pixels)[n]=before[n];static_cast<unsigned*>(wide.pixels)[n]=expand(c,g6);}
            std::vector<unsigned> native_before(before.begin(),before.end()),wide_before(320*48);
            for(unsigned n=0;n<wide_before.size();++n)wide_before[n]=expand(before[n],g6)|0xff000000u;
            gpu.upload(destination,unsigned(background+1),native_before.data(),native_before.size());gpu.upload(detail,unsigned(background+1),wide_before.data(),wide_before.size());
            Rect area={7+raster.left,9+raster.top,7+raster.left+int(raster.width),9+raster.top+int(raster.height)};
            Command commands[2]={{Kind::native_text,destination,glyph,area,{0,0,320,48},0,0,0,curves},{Kind::native_text,detail,glyph,area,{0,0,320,48},0,0,0,curves}};
            if(!gpu.submit(commands,2))throw std::runtime_error("GPU glyph submission");
            auto native_gpu=read_gpu(device.Get(),context.Get(),gpu.texture(destination)),detail_gpu=read_gpu(device.Get(),context.Get(),gpu.texture(detail));
            for(auto dc:{native.dc,wide.dc}){SetBkMode(dc,bk);SetTextColor(dc,RGB(237,171,55));SetBkColor(dc,RGB(47,68,211));TextOutA(dc,7,9,"Berlin: 6",9);}GdiFlush();
            if(!g6&&quality==DEFAULT_QUALITY&&bk==TRANSPARENT&&background==1)save_comparison(static_cast<unsigned short*>(native.pixels),static_cast<unsigned*>(wide.pixels),detail_gpu);
            unsigned fg=pack(0xedab37,g6),bg=pack(0x2f44d3,g6);
            for(unsigned n=0;n<before.size();++n){unsigned got=static_cast<unsigned short*>(native.pixels)[n],rgb=static_cast<unsigned*>(wide.pixels)[n];
                if((got&(g6?65535:32767))!=pack(rgb,g6))++mismatch;
                if(compiled){int x=int(n%320)-7-raster.left,y=int(n/320)-9-raster.top;
                    unsigned response=expand(before[n],g6)|0xff000000u;
                    unsigned packed_response=before[n];
                    if(x>=0&&y>=0&&x<int(raster.width)&&y<int(raster.height)){
                        response=c3x_native_text::apply(raster,unsigned(y)*raster.width+unsigned(x),response,g6,true);
                        packed_response=c3x_native_text::apply(raster,unsigned(y)*raster.width+unsigned(x),before[n],g6,false);
                    }
                    if(response!=detail_gpu[n]||packed_response!=native_gpu[n]){std::printf("pixel=%u expected_native=%04x actual_native=%04x expected_full=%08x actual_full=%08x before=%04x\n",n,packed_response,native_gpu[n],response,detail_gpu[n],unsigned(before[n]));throw std::runtime_error("GPU text differs from compiled response");}
                    unsigned error=0;for(unsigned c=0;c<3;++c)error=std::max(error,unsigned(std::abs(int(response>>(8*c)&255)-int(rgb>>(8*c)&255))));
                    response_error=std::max(response_error,error);response_pixels+=error!=0;
                }
                if(got!=before[n]&&got!=fg&&got!=bg)++partial;
                if(!g6&&(got&32767)==(unsigned(before[n])&32767u)&&got!=before[n])++changed_high;
            }
        }
        std::printf("text format=%s quality=%d background=%d native_vs_bgra=%u partial=%u high_only=%u response_max=%u response_pixels=%u\n",g6?"565":"555",quality,bk,mismatch,partial,changed_high,response_error,response_pixels);
        if(response_error>3)throw std::runtime_error("text exceeds three-level interpolation tolerance");
        gpu.destroy(glyph);gpu.destroy(curves);gpu.destroy(destination);gpu.destroy(detail);
        SelectObject(native.dc,a);SelectObject(wide.dc,b);DeleteObject(font);
    }std::puts("PASS native GPU text: 555/565 and full color, actual GDI smoothing, transparent/opaque, bounded edge error, exact GPU response");return 0;
}catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}}
