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
        c3x_renderer::tactical::Input input;input.ring(320,180,float(zoom),true);input.line(320,180,230,240);input.line(230,240,100,240);input.ring(100,240,float(zoom),false);input.label(100,240,"2",20.f);
        auto copied=input;input.primitives.clear();
        auto frame=[&](double seconds){return tactical_pixels(device.Get(),context.Get(),gpu.draw(device.Get(),context.Get(),copied,{0,0,480,320},seconds));};
        auto a=frame(0),b=frame(1),endFrame=frame(2.5),repeat=frame(5);assert(a==repeat&&a!=b);
        if(zoom==128){unsigned marker=178*480+357;assert((endFrame[marker]&255)>(a[marker]&255)+150);}
        auto packed=tactical_pixels(device.Get(),context.Get(),gpu.packed(device.Get(),context.Get(),copied,{0,0,480,320},0));assert(a==packed);
        unsigned partial=0,opaque=0;for(auto pixel:a){auto alpha=pixel>>24;partial+=alpha>0&&alpha<250;opaque+=alpha>=250;}
        assert(partial>300&&opaque>5);assert(a[0]==0&&a[479]==0);++checks;
        char path[192];sprintf_s(path,"../lab/out/tactical-overlays/current-z%d.bmp",zoom);tactical_bmp(path,a,480,320);
        if(zoom==128)for(int n=0;n<25;++n){auto pixels=frame(double(n)*.2);sprintf_s(path,"../lab/out/tactical-overlays/motion-%02d.bmp",n);tactical_bmp(path,pixels,480,320);}
    }
    c3x_renderer::tactical::Input route;route.line(20,40,200,40);
    auto flat=tactical_pixels(device.Get(),context.Get(),gpu.draw(device.Get(),context.Get(),route,{0,0,240,80},0));
    assert((flat[40*240+100]>>24)>0 && (flat[45*240+100]>>24)==0);
    c3x_renderer::tactical::Input cursor;cursor.ring(100,40,128,false);
    auto curled=tactical_pixels(device.Get(),context.Get(),gpu.draw(device.Get(),context.Get(),cursor,{0,0,200,80},0));
    auto alpha=[&](int x,int y){return curled[y*200+x]>>24;};
    assert(alpha(59,38)>80&&alpha(59,42)>80&&alpha(56,40)<60);
    assert(alpha(98,18)>80&&alpha(102,18)>80&&alpha(100,18)<60);
    tactical_bmp("../lab/out/tactical-overlays/cursor-only.bmp",curled,200,80);
    std::printf("PASS tactical GPU: %u zooms, curled cursor, eased quarter-turn and dark-white fade, shadowless route, antialias coverage and clear exterior\n",checks);return 0;
}

#ifdef C3X_TACTICAL_STANDALONE
int main(){try{return test_tactical_overlay();}catch(std::exception const& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
#endif
