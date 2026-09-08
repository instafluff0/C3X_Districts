#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "linear_target.h"
#include "test_source_shadow.h"

bool compile(char const* entry,char const* target) {
    ID3DBlob *code=nullptr,*errors=nullptr;
    HRESULT hr=c3x_renderer::render_core::compile_cached(L"terrain_scene.hlsl",
        entry,target,&code,&errors);
    if(errors) { std::fprintf(stderr,"%s",static_cast<char*>(errors->GetBufferPointer())); errors->Release(); }
    if(code) code->Release();
    std::printf("render-core shader %s: %s\n",entry,SUCCEEDED(hr)?"pass":"FAIL");
    return SUCCEEDED(hr);
}
int main() {
    if(!compile("VSIntegrated","vs_5_0") || !compile("PSIntegrated","ps_5_0") ||
        !compile("VSIntegratedFeature","vs_5_0") || !compile("PSIntegratedFeature","ps_5_0")) return 1;
    ID3D11Device* device=nullptr; ID3D11DeviceContext* context=nullptr;
    D3D_FEATURE_LEVEL level;
    HRESULT hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,
        D3D11_SDK_VERSION,&device,&level,&context);
    if(FAILED(hr)) return 2;
    bool passed=test_source_shadow(device,context);
    {
        c3x_renderer::render_core::LinearTarget linear;
        c3x_renderer::render_core::LinearOutput output;
        if(!linear.ensure(device,128,128) || !output.ensure(device)) return 3;
        ID3D11Texture2D *target=nullptr,*readback=nullptr;
        ID3D11RenderTargetView* view=nullptr;
        D3D11_TEXTURE2D_DESC d={}; d.Width=d.Height=128; d.MipLevels=d.ArraySize=1;
        d.SampleDesc.Count=1; d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
        d.BindFlags=D3D11_BIND_RENDER_TARGET;
        hr=device->CreateTexture2D(&d,nullptr,&target);
        if(SUCCEEDED(hr)) hr=device->CreateRenderTargetView(target,nullptr,&view);
        d.Usage=D3D11_USAGE_STAGING; d.BindFlags=0; d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        if(SUCCEEDED(hr)) hr=device->CreateTexture2D(&d,nullptr,&readback);
        if(FAILED(hr)) return 4;
        for(float alpha:{0.f,.25f,1.f}) {
            float clear[]={2.f*alpha,.5f*alpha,.125f*alpha,alpha};
            context->ClearRenderTargetView(linear.target,clear);
            output.draw(context,linear,view,1.25f);
            context->OMSetRenderTargets(0,nullptr,nullptr);
            context->CopyResource(readback,target);
            D3D11_MAPPED_SUBRESOURCE map={};
            if(FAILED(context->Map(readback,0,D3D11_MAP_READ,0,&map))) return 5;
            int expected[4]={};
            for(int c=0;c<3;c++) {
                double value=alpha>0 ? clear[c]/alpha*1.25/3.5 : 0;
                value=value<=.0031308 ? 12.92*value : 1.055*std::pow(value,1/2.4)-.055;
                expected[2-c]=int(std::lround(value*255));
            }
            expected[3]=int(std::lround(alpha*255));
            for(unsigned y=0;y<128;y++) for(unsigned x=0;x<128;x++) for(int c=0;c<4;c++) {
                auto value=static_cast<std::uint8_t*>(map.pData)[y*map.RowPitch+x*4+c];
                if(std::abs(int(value)-expected[c])>1) passed=false;
            }
            context->Unmap(readback,0);
        }
        readback->Release(); view->Release(); target->Release();
    }
    context->ClearState(); context->Release(); device->Release();
    std::printf("render-core MSAA4 linear/premultiplied/transfer: %s\n",passed?"pass":"FAIL");
    return passed?0:6;
}
