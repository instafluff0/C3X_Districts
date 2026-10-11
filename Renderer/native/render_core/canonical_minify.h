#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace c3x_renderer { namespace render_core {
// A settled zoomed step is drawn once, at its presented zoom; the hidden
// canonical (1x) image is filled from that draw instead of a second full
// scene render (review 69). The zoomed draw is the canonical image magnified
// about the frame centre (SceneProjection: display = c + (p - c) * zoom), so a
// canonical pixel is the box average of the zoomed pixels it covers: exact for
// whole zooms (2x2 at 2x, 3x3 at 3x), a ceil(zoom)-tap bilinear box otherwise.
// Only the canonical pixels whose footprint lies inside the zoomed image are
// written; everything that reads the canonical image while zoomed samples
// that centred rectangle (the held and unprojected display paths).
struct CanonicalMinify {
    struct Region {int left=0,top=0,right=0,bottom=0;};
    // The canonical pixels covered by a zoomed image of the same size.
    static Region region(unsigned width,unsigned height,float zoom){
        float cx=float(width/2),cy=float(height/2);
        auto clamp=[](float v,unsigned limit){return int(std::clamp(v,0.f,float(limit)));};
        return {clamp(std::floor(cx-cx/zoom),width),clamp(std::floor(cy-cy/zoom),height),
                clamp(std::ceil(cx+(float(width)-cx)/zoom),width),clamp(std::ceil(cy+(float(height)-cy)/zoom),height)};
    }
    ID3D11VertexShader* vertex=nullptr;ID3D11PixelShader* pixel=nullptr;
    ID3D11Buffer* settings=nullptr;ID3D11SamplerState* sampler=nullptr;ID3D11RasterizerState* rasterizer=nullptr;
    ID3D11Device* owner=nullptr;
    template<class T>void release(T*& p){if(p)p->Release();p=nullptr;}
    void reset(){release(vertex);release(pixel);release(settings);release(sampler);release(rasterizer);owner=nullptr;}
    ~CanonicalMinify(){reset();}
    bool ensure(ID3D11Device* device){
        if(owner!=device)reset();
        if(pixel)return true;
        static char const source[]=R"(
Texture2D<float4> zoomed:register(t0);
SamplerState filtered:register(s0);
cbuffer Minify:register(b0){float2 center;float2 extent;float zoom;float taps;float2 unused;};
float4 VS(uint id:SV_VertexID):SV_Position{float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
float4 PS(float4 position:SV_Position):SV_Target{
 float2 at=center+(position.xy-center)*zoom; // this pixel's centre in the zoomed image
 float4 sum=0;
 for(float y=0;y<taps;++y)for(float x=0;x<taps;++x)
  sum+=zoomed.SampleLevel(filtered,(at+float2(x,y)-(taps-1)*.5)/extent,0);
 return sum/(taps*taps);
})";
        auto compile=[&](char const* entry,char const* target,ID3DBlob** blob){
            ID3DBlob* errors=nullptr;HRESULT hr=D3DCompile(source,sizeof(source)-1,"canonical_minify",nullptr,nullptr,
                entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,blob,&errors);
            if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));errors->Release();}return hr;
        };
        ID3DBlob* blob=nullptr;HRESULT hr=compile("VS","vs_5_0",&blob);
        if(SUCCEEDED(hr))hr=device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vertex);release(blob);
        if(SUCCEEDED(hr))hr=compile("PS","ps_5_0",&blob);
        if(SUCCEEDED(hr))hr=device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&pixel);release(blob);
        D3D11_BUFFER_DESC b={};b.ByteWidth=32;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&settings);
        D3D11_SAMPLER_DESC s={};s.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        s.AddressU=s.AddressV=s.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;s.MaxLOD=D3D11_FLOAT32_MAX;
        if(SUCCEEDED(hr))hr=device->CreateSamplerState(&s,&sampler);
        D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=TRUE;
        if(SUCCEEDED(hr))hr=device->CreateRasterizerState(&r,&rasterizer);
        if(FAILED(hr)){reset();return false;}
        owner=device;return true;
    }
    // `zoomed` and `canonical` are the same size; the context's state is cleared.
    bool draw(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11ShaderResourceView* zoomed,
            ID3D11RenderTargetView* canonical,unsigned width,unsigned height,float zoom){
        if(!zoomed || !canonical || !width || !height || !(zoom>1.f) || !ensure(device))return false;
        auto r=region(width,height,zoom);
        if(r.right<=r.left || r.bottom<=r.top)return false;
        float values[8]={float(width/2),float(height/2),float(width),float(height),zoom,std::ceil(zoom-1e-4f),0,0};
        context->ClearState();
        context->UpdateSubresource(settings,0,nullptr,values,0,0);
        D3D11_VIEWPORT viewport={float(r.left),float(r.top),float(r.right-r.left),float(r.bottom-r.top),0,1};
        context->OMSetRenderTargets(1,&canonical,nullptr);context->RSSetViewports(1,&viewport);context->RSSetState(rasterizer);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);
        context->PSSetConstantBuffers(0,1,&settings);context->PSSetShaderResources(0,1,&zoomed);context->PSSetSamplers(0,1,&sampler);
        context->Draw(3,0);
        context->ClearState();
        return true;
    }
};
}}
