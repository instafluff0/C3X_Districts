#define NOMINMAX
#include <windows.h>
#include "gpu_tactical_overlay.h"
#include <cassert>
#include <cstdio>
#include <vector>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"user32.lib")
using Microsoft::WRL::ComPtr;
std::vector<unsigned> tactical_pixels(ID3D11Device* d,ID3D11DeviceContext* c,ID3D11Texture2D* t){
    D3D11_TEXTURE2D_DESC desc={};t->GetDesc(&desc);unsigned w=desc.Width,h=desc.Height;
    desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> read;assert(SUCCEEDED(d->CreateTexture2D(&desc,nullptr,&read)));c->CopyResource(read.Get(),t);
    D3D11_MAPPED_SUBRESOURCE m={};assert(SUCCEEDED(c->Map(read.Get(),0,D3D11_MAP_READ,0,&m)));
    std::vector<unsigned> out(w*h);for(unsigned y=0;y<h;++y)std::memcpy(out.data()+y*w,static_cast<char*>(m.pData)+y*m.RowPitch,w*4);c->Unmap(read.Get(),0);return out;
}
void tactical_bmp(char const* name,std::vector<unsigned> const& pixels,int w,int h){
    BITMAPFILEHEADER f={};f.bfType=0x4d42;f.bfOffBits=sizeof(f)+sizeof(BITMAPINFOHEADER);f.bfSize=f.bfOffBits+w*h*4;
    BITMAPINFOHEADER b={};b.biSize=sizeof(b);b.biWidth=w;b.biHeight=-h;b.biPlanes=1;b.biBitCount=32;
    FILE* out=nullptr;assert(!fopen_s(&out,name,"wb"));fwrite(&f,sizeof(f),1,out);fwrite(&b,sizeof(b),1,out);fwrite(pixels.data(),4,pixels.size(),out);fclose(out);
}
int test_tactical_overlay(){
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL feature;
    assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context)));
    c3x_renderer::tactical::Gpu gpu;unsigned checks=0;
    for(int zoom:{64,128,160,192}){
        c3x_renderer::tactical::Input input;input.ring(320,180,float(zoom),true);input.line(320,180,230,240);input.line(230,240,100,240);input.ring(100,240,float(zoom),false);input.label(100,240,"2",std::max(18.f,float(zoom)*.26f));
        auto copied=input;input.primitives.clear();
        auto frame=[&](double seconds){return tactical_pixels(device.Get(),context.Get(),gpu.draw(device.Get(),context.Get(),copied,{0,0,480,320},seconds));};
        auto a=frame(0),b=frame(1),repeat=frame(0);assert(a==repeat&&a!=b);
        auto packed=tactical_pixels(device.Get(),context.Get(),gpu.packed(device.Get(),context.Get(),copied,{0,0,480,320},0));assert(a==packed);
        unsigned partial=0,opaque=0;for(auto pixel:a){auto alpha=pixel>>24;partial+=alpha>0&&alpha<250;opaque+=alpha>=250;}
        assert(partial>300&&opaque>5);assert(a[0]==0&&a[479]==0);++checks;
        char path[192];sprintf_s(path,"../lab/out/tactical-overlays/current-z%d.bmp",zoom);tactical_bmp(path,a,480,320);
        if(zoom==128)for(int n=0;n<12;++n){auto pixels=frame(double(n)*.2);sprintf_s(path,"../lab/out/tactical-overlays/motion-%02d.bmp",n);tactical_bmp(path,pixels,480,320);}
    }
    std::printf("PASS tactical GPU: %u zooms, immutable input, moving marker, exact time return, antialias coverage and clear exterior\n",checks);return 0;
}

#ifdef C3X_TACTICAL_STANDALONE
int main(){try{return test_tactical_overlay();}catch(std::exception const& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
#endif
