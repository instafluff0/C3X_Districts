#pragma once
#include "render_core/city_site_overlay.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>

namespace c3x_renderer {
// Display-space color pass after tone mapping, before map fog. The actor
// stencil keeps the tile wash below unit bodies, including cutout edges.
class GpuCitySiteOverlay {
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    render_core::CitySiteOverlay input;
    Ptr<ID3D11VertexShader> vertex;
    Ptr<ID3D11PixelShader> pixel;
    Ptr<ID3D11Buffer> settings,buffer;
    Ptr<ID3D11ShaderResourceView> records,actor_stencil;
    Ptr<ID3D11BlendState> blend;
    Ptr<ID3D11RasterizerState> raster;
    Ptr<ID3D11DepthStencilState> depth;
    Ptr<ID3D11RenderTargetView> view;
    ID3D11Texture2D* target=nullptr;
    ID3D11Texture2D* actors=nullptr;
    unsigned capacity=0;
public:
    void reset(){vertex.Reset();pixel.Reset();settings.Reset();buffer.Reset();records.Reset();
        actor_stencil.Reset();blend.Reset();raster.Reset();depth.Reset();view.Reset();
        target=actors=nullptr;capacity=0;input.tiles.clear();}
    bool draw(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* output,
              c3x_renderer_frame_v1 const& frame,ID3D11Texture2D* actor_depth,
              unsigned actor_guard,float zoom){
        if(!input.capture(frame))return false;
        if(input.tiles.empty())return true;
        if(!device||!context||!output||!actor_depth||zoom<1.f||zoom>3.f)return false;
        D3D11_TEXTURE2D_DESC out_desc={},actor_desc={};output->GetDesc(&out_desc);actor_depth->GetDesc(&actor_desc);
        if(out_desc.Width!=unsigned(frame.target_width)||out_desc.Height!=unsigned(frame.target_height)||
           out_desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM||out_desc.SampleDesc.Count!=1||
           actor_desc.Width!=out_desc.Width+2*actor_guard||actor_desc.Height!=out_desc.Height+2*actor_guard||
           actor_desc.Format!=DXGI_FORMAT_R24G8_TYPELESS||actor_desc.SampleDesc.Count<1||
           actor_desc.SampleDesc.Count>8)return false;
        if(!vertex){
            char const* shader=R"(
struct Record{float2 anchor;uint grade,pad;};StructuredBuffer<Record> records:register(t0);
Texture2DMS<uint2> actor_ms:register(t1);Texture2D<uint2> actor_single:register(t2);
cbuffer Settings:register(b0){float4 size;float4 style;float4 pale;float4 deep;float4 actor;};
struct V{float4 position:SV_Position;float2 uv:TEXCOORD0;nointerpolation uint grade:TEXCOORD1;};
V vs(uint id:SV_VertexID,uint instance:SV_InstanceID){
 float2 uv[6]={float2(0,0),float2(1,0),float2(1,1),float2(0,0),float2(1,1),float2(0,1)};
 Record r=records[instance];V v;v.uv=uv[id];v.grade=r.grade;
 float2 p=r.anchor+uv[id]*size.zw;
 p=(p-floor(size.xy*.5))*style.x+floor(size.xy*.5);
 v.position=float4(p.x*2/size.x-1,1-p.y*2/size.y,0,1);return v;
}
float4 ps(V v):SV_Target{
 float2 q=2*v.uv-1;
 float distance=1-abs(q.x)-abs(q.y);
 float antialias=max(fwidth(distance),1.e-5);
 float inside=saturate(distance/antialias+.5-style.z);
 float alpha=inside*style.y;
 int2 at=int2(v.position.xy)+int(style.w);
 if(actor.x==1)alpha*=1-(actor_single.Load(int3(at,0)).y!=0);
 else if(actor.x>1){float covered=0;
  for(uint sample=0;sample<(uint)actor.x;++sample)
   covered+=(actor_ms.Load(at,sample).y!=0)/actor.x;
  alpha*=1-covered;
 }
 float weight=v.grade/10.f;weight=weight*weight*weight;
 float3 color=lerp(pale.rgb,deep.rgb,weight);
 return float4(color*alpha,alpha);
}
)";
            Ptr<ID3DBlob> vs,ps,error;
            if(FAILED(D3DCompile(shader,std::strlen(shader),"city-site overlay",nullptr,nullptr,"vs","vs_5_0",
                                  D3DCOMPILE_ENABLE_STRICTNESS,0,&vs,&error)) ||
               FAILED(D3DCompile(shader,std::strlen(shader),"city-site overlay",nullptr,nullptr,"ps","ps_5_0",
                                  D3DCOMPILE_ENABLE_STRICTNESS,0,&ps,&error)) ||
               FAILED(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex)) ||
               FAILED(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel)))return false;
            D3D11_BUFFER_DESC cb={};cb.ByteWidth=80;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(device->CreateBuffer(&cb,nullptr,&settings)))return false;
            D3D11_BLEND_DESC b={};auto& rt=b.RenderTarget[0];rt.BlendEnable=TRUE;
            rt.SrcBlend=D3D11_BLEND_ONE;rt.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOp=D3D11_BLEND_OP_ADD;
            rt.SrcBlendAlpha=D3D11_BLEND_ZERO;rt.DestBlendAlpha=D3D11_BLEND_ONE;rt.BlendOpAlpha=D3D11_BLEND_OP_ADD;
            rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_RED|D3D11_COLOR_WRITE_ENABLE_GREEN|D3D11_COLOR_WRITE_ENABLE_BLUE;
            D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=TRUE;
            D3D11_DEPTH_STENCIL_DESC d={};d.DepthEnable=FALSE;
            if(FAILED(device->CreateBlendState(&b,&blend))||FAILED(device->CreateRasterizerState(&r,&raster))||
               FAILED(device->CreateDepthStencilState(&d,&depth)))return false;
        }
        if(target!=output){view.Reset();target=nullptr;
            if(FAILED(device->CreateRenderTargetView(output,nullptr,&view)))return false;target=output;}
        if(actors!=actor_depth){actor_stencil.Reset();actors=nullptr;
            D3D11_SHADER_RESOURCE_VIEW_DESC s={};s.Format=DXGI_FORMAT_X24_TYPELESS_G8_UINT;
            s.ViewDimension=actor_desc.SampleDesc.Count==1?D3D11_SRV_DIMENSION_TEXTURE2D:D3D11_SRV_DIMENSION_TEXTURE2DMS;
            if(actor_desc.SampleDesc.Count==1)s.Texture2D.MipLevels=1;
            if(FAILED(device->CreateShaderResourceView(actor_depth,&s,&actor_stencil)))return false;actors=actor_depth;}
        unsigned bytes=unsigned(input.tiles.size()*sizeof(render_core::CitySiteOverlay::Tile));
        if(bytes>capacity){records.Reset();buffer.Reset();capacity=0;
            D3D11_BUFFER_DESC b={};b.ByteWidth=bytes;b.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            b.Usage=D3D11_USAGE_DEFAULT;b.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            b.StructureByteStride=sizeof(render_core::CitySiteOverlay::Tile);
            if(FAILED(device->CreateBuffer(&b,nullptr,&buffer))||
               FAILED(device->CreateShaderResourceView(buffer.Get(),nullptr,&records)))return false;
            capacity=bytes;
        }
        D3D11_BOX range={0,0,0,bytes,1,1};context->UpdateSubresource(buffer.Get(),0,&range,input.tiles.data(),0,0);
        auto white=render_core::CitySiteOverlay::color(0),green=render_core::CitySiteOverlay::color(10);
        float constants[]={float(out_desc.Width),float(out_desc.Height),float(frame.tile_width),float(frame.tile_height),
            zoom,render_core::CitySiteOverlay::fill_alpha,render_core::CitySiteOverlay::inset_pixels,float(actor_guard),
            white[0],white[1],white[2],0,green[0],green[1],green[2],0,float(actor_desc.SampleDesc.Count),0,0,0};
        context->UpdateSubresource(settings.Get(),0,nullptr,constants,0,0);
        context->OMSetRenderTargets(0,nullptr,nullptr);auto output_view=view.Get();
        context->OMSetRenderTargets(1,&output_view,nullptr);context->OMSetBlendState(blend.Get(),nullptr,~0u);
        context->OMSetDepthStencilState(depth.Get(),0);context->RSSetState(raster.Get());
        D3D11_VIEWPORT viewport={0,0,float(out_desc.Width),float(out_desc.Height),0,1};context->RSSetViewports(1,&viewport);
        auto cb=settings.Get();context->VSSetConstantBuffers(0,1,&cb);context->PSSetConstantBuffers(0,1,&cb);
        auto tile_records=records.Get();context->VSSetShaderResources(0,1,&tile_records);
        ID3D11ShaderResourceView* actor_views[]={actor_desc.SampleDesc.Count>1?actor_stencil.Get():nullptr,
            actor_desc.SampleDesc.Count==1?actor_stencil.Get():nullptr};
        context->PSSetShaderResources(1,2,actor_views);
        context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
        context->DrawInstanced(6,unsigned(input.tiles.size()),0,0);context->OMSetRenderTargets(0,nullptr,nullptr);
        tile_records=nullptr;context->VSSetShaderResources(0,1,&tile_records);
        actor_views[0]=actor_views[1]=nullptr;context->PSSetShaderResources(1,2,actor_views);
        return true;
    }
};
}
