#pragma once
#include "render_core/visibility_coverage.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>

namespace c3x_renderer {
// Final map coverage blends directly into the completed output texture.
// Reusable scene color/depth stays untouched; GPU output requires no readback.
class GpuVisibility {
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    Ptr<ID3D11VertexShader> vertex;
    Ptr<ID3D11PixelShader> pixel;
    Ptr<ID3D11Buffer> settings;
    Ptr<ID3D11ShaderResourceView> records;
    Ptr<ID3D11BlendState> blend;
    Ptr<ID3D11RasterizerState> raster;
    Ptr<ID3D11DepthStencilState> depth;
    Ptr<ID3D11Buffer> buffer;
    ID3D11Texture2D* target=nullptr; // view owns the lifetime
    Ptr<ID3D11RenderTargetView> view;
    unsigned width=0,height=0,capacity=0;
public:
    void reset(){vertex.Reset();pixel.Reset();settings.Reset();records.Reset();blend.Reset();raster.Reset();depth.Reset();buffer.Reset();view.Reset();target=nullptr;width=height=capacity=0;}
    bool apply(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* source,
               render_core::VisibilityCoverage const& coverage){
        if(coverage.tiles.empty())return true;
        if(!source)return false;
        if(!vertex){
            char const* shader=R"(
struct Record{float2 anchor;uint cells,pad;};StructuredBuffer<Record> records:register(t0);
cbuffer Settings:register(b0){float4 size;float4 response;};
struct V{float4 position:SV_Position;float2 uv:TEXCOORD0;nointerpolation uint cells:TEXCOORD1;};
V vs(uint vertex:SV_VertexID,uint instance:SV_InstanceID){
 float2 uv[6]={float2(0,0),float2(1,0),float2(1,1),float2(0,0),float2(1,1),float2(0,1)};
 V v;v.uv=uv[vertex];Record r=records[instance];v.cells=r.cells;
 float2 p=r.anchor+float2(1+v.uv.x-v.uv.y,v.uv.x+v.uv.y)*size.zw*.5f;
 v.position=float4(p.x*2/size.x-1,1-p.y*2/size.y,0,1);return v;
}
float value(uint cells,int x,int y,uint threshold){return ((cells>>(2*((y+1)*3+x+1)))&3)>=threshold?1.f:0.f;}
float coverage(uint cells,float2 uv,uint threshold){
 int2 direction=int2(uv.x<.5f?-1:1,uv.y<.5f?-1:1);
 float2 t=saturate(min(uv,1-uv)/response.x),w=.5f*(1-t*t*(3-2*t));
 return lerp(lerp(value(cells,0,0,threshold),value(cells,direction.x,0,threshold),w.x),
             lerp(value(cells,0,direction.y,threshold),value(cells,direction.x,direction.y,threshold),w.x),w.y);
}
float4 ps(V v):SV_Target{
 float explored=coverage(v.cells,v.uv,1),visible=coverage(v.cells,v.uv,2);
 float fog=(explored-visible)*response.z;
 return float4((response.y*fog).xxx,1-explored+fog);
}
)";
            Ptr<ID3DBlob> vs,ps,error;
            if(FAILED(D3DCompile(shader,std::strlen(shader),"map visibility",nullptr,nullptr,"vs","vs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&vs,&error)) ||
               FAILED(D3DCompile(shader,std::strlen(shader),"map visibility",nullptr,nullptr,"ps","ps_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&ps,&error)))return false;
            Ptr<ID3D11VertexShader> next;
            if(FAILED(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&next)) ||
               FAILED(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel)))return false;
            D3D11_BUFFER_DESC cb={};cb.ByteWidth=32;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(device->CreateBuffer(&cb,nullptr,&settings)))return false;
            D3D11_BLEND_DESC b={};auto& rt=b.RenderTarget[0];rt.BlendEnable=TRUE;
            rt.SrcBlend=D3D11_BLEND_ONE;rt.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOp=D3D11_BLEND_OP_ADD;
            rt.SrcBlendAlpha=D3D11_BLEND_ZERO;rt.DestBlendAlpha=D3D11_BLEND_ONE;rt.BlendOpAlpha=D3D11_BLEND_OP_ADD;
            rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_RED|D3D11_COLOR_WRITE_ENABLE_GREEN|D3D11_COLOR_WRITE_ENABLE_BLUE;
            D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=TRUE;
            D3D11_DEPTH_STENCIL_DESC d={};d.DepthEnable=FALSE;
            if(FAILED(device->CreateBlendState(&b,&blend)) || FAILED(device->CreateRasterizerState(&r,&raster)) ||
               FAILED(device->CreateDepthStencilState(&d,&depth)))return false;
            vertex=std::move(next);
        }
        D3D11_TEXTURE2D_DESC d={};source->GetDesc(&d);
        if(d.Width!=unsigned(coverage.width)||d.Height!=unsigned(coverage.height)||d.SampleDesc.Count!=1)return false;
        if(target!=source){
            view.Reset();target=nullptr;width=height=0;
            if(FAILED(device->CreateRenderTargetView(source,nullptr,&view)))return false;
            target=source;width=d.Width;height=d.Height;
        }
        unsigned bytes=unsigned(coverage.tiles.size()*sizeof(render_core::VisibilityCoverage::Tile));
        if(bytes>capacity){
            records.Reset();buffer.Reset();capacity=0;D3D11_BUFFER_DESC b={};b.ByteWidth=bytes;b.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            b.Usage=D3D11_USAGE_DEFAULT;b.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;b.StructureByteStride=sizeof(render_core::VisibilityCoverage::Tile);
            if(FAILED(device->CreateBuffer(&b,nullptr,&buffer)) || FAILED(device->CreateShaderResourceView(buffer.Get(),nullptr,&records)))return false;capacity=bytes;
        }
        D3D11_BOX range={0,0,0,bytes,1,1};context->UpdateSubresource(buffer.Get(),0,&range,coverage.tiles.data(),0,0);
        float constants[]={float(width),float(height),float(coverage.tile_width),float(coverage.tile_height),
            render_core::VisibilityCoverage::feather,render_core::VisibilityCoverage::gray,render_core::VisibilityCoverage::fog_alpha,0};
        context->UpdateSubresource(settings.Get(),0,nullptr,constants,0,0);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        auto output=view.Get();context->OMSetRenderTargets(1,&output,nullptr);context->OMSetBlendState(blend.Get(),nullptr,~0u);
        context->OMSetDepthStencilState(depth.Get(),0);context->RSSetState(raster.Get());
        D3D11_VIEWPORT viewport={0,0,float(width),float(height),0,1};context->RSSetViewports(1,&viewport);
        auto cb=settings.Get();context->VSSetConstantBuffers(0,1,&cb);context->PSSetConstantBuffers(0,1,&cb);
        auto input=records.Get();context->VSSetShaderResources(0,1,&input);
        context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
        context->DrawInstanced(6,unsigned(coverage.tiles.size()),0,0);context->OMSetRenderTargets(0,nullptr,nullptr);
        input=nullptr;context->VSSetShaderResources(0,1,&input);
        return true;
    }
    std::size_t bytes()const{return capacity+(settings?32:0);}
};
}
