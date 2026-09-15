#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <array>
#include <cstdint>
#include <vector>
#include <cstring>
#include <stdexcept>
namespace c3x_renderer {
// One ordered pass over selected immutable caster records. Rectangles invoke
// the original pixel-center/barycentric test, so hardware triangle edge rules
// do not replace the native shadow contract. MAX is independent of draw order.
class GpuUnitShadow {
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    Ptr<ID3D11VertexShader> vertex;Ptr<ID3D11PixelShader> pixel;
    Ptr<ID3D11InputLayout> layout;Ptr<ID3D11Buffer> records,constants;
    Ptr<ID3D11BlendState> blend;Ptr<ID3D11RasterizerState> raster;
    Ptr<ID3D11DepthStencilState> depth;
    unsigned capacity=0;
    void check(HRESULT hr){if(FAILED(hr))throw std::runtime_error("GPU unit shadow pass failed");}
public:
    std::uint64_t passes=0,triangles=0,upload_bytes=0;
    void reset(){vertex.Reset();pixel.Reset();layout.Reset();records.Reset();constants.Reset();blend.Reset();raster.Reset();depth.Reset();capacity=0;}
    void draw(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11RenderTargetView* target,
              unsigned extent,std::vector<std::array<float,12>> const& selected){
        auto bytes=selected.size()*sizeof(selected[0]);
        if(!extent||extent>1536||bytes>16u*1024u*1024u)throw std::runtime_error("unit shadow pass budget");
        if(!vertex){char const* source=R"(
cbuffer Settings:register(b0){float extent;};
struct Input{float4 a:A;float4 b:B;float4 c:C;};
struct Output{float4 position:SV_Position;nointerpolation float4 a:A;nointerpolation float3 b:B;nointerpolation float3 c:C;};
Output VS(Input i,uint id:SV_VertexID){
 float2 low=max(0,floor(min(i.a.xy,min(i.b.xy,i.c.xy))));
 float2 high=min(extent-1,ceil(max(i.a.xy,max(i.b.xy,i.c.xy))))+1;
 const float2 corners[6]={float2(0,0),float2(1,0),float2(0,1),float2(0,1),float2(1,0),float2(1,1)};
 float2 p=lerp(low,high,corners[id]);Output o;o.position=float4(p.x*2/extent-1,1-p.y*2/extent,0,1);
 o.a=i.a;o.b=i.b.xyz;o.c=i.c.xyz;return o;
}
float PS(Output i):SV_Target{
 precise float px=i.position.x,py=i.position.y;
 precise float un=((i.b.x-px)*(i.c.y-py)-(i.b.y-py)*(i.c.x-px));
 precise float vn=((i.c.x-px)*(i.a.y-py)-(i.c.y-py)*(i.a.x-px));
 precise float u=un/i.a.w,v=vn/i.a.w;
 precise float w=1-u-v;precise float z=u*i.a.z+v*i.b.z+w*i.c.z;
 // Keep the last edge as a comparison: some drivers reassociate 1-u-v
 // into 1-(u+v), losing the CPU's rounded intermediate at shared edges.
 if(u<0||v<0||v>1-u||z<.002f)discard;return z;
})";
            Ptr<ID3DBlob> vs,ps,error;
            auto flags=D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_IEEE_STRICTNESS;
            check(D3DCompile(source,std::strlen(source),"unit shadow",nullptr,nullptr,"VS","vs_5_0",flags,0,&vs,&error));
            check(D3DCompile(source,std::strlen(source),"unit shadow",nullptr,nullptr,"PS","ps_5_0",flags,0,&ps,&error));
            check(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex));
            check(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel));
            D3D11_INPUT_ELEMENT_DESC elements[3]={{"A",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,0,D3D11_INPUT_PER_INSTANCE_DATA,1},
                {"B",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,16,D3D11_INPUT_PER_INSTANCE_DATA,1},{"C",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,32,D3D11_INPUT_PER_INSTANCE_DATA,1}};
            check(device->CreateInputLayout(elements,3,vs->GetBufferPointer(),vs->GetBufferSize(),&layout));
            D3D11_BUFFER_DESC d={};d.ByteWidth=16;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;check(device->CreateBuffer(&d,nullptr,&constants));
            D3D11_BLEND_DESC b={};auto& rt=b.RenderTarget[0];rt.BlendEnable=TRUE;rt.SrcBlend=rt.DestBlend=rt.SrcBlendAlpha=rt.DestBlendAlpha=D3D11_BLEND_ONE;
            rt.BlendOp=rt.BlendOpAlpha=D3D11_BLEND_OP_MAX;rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_RED;check(device->CreateBlendState(&b,&blend));
            D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=TRUE;check(device->CreateRasterizerState(&r,&raster));
            D3D11_DEPTH_STENCIL_DESC z={};check(device->CreateDepthStencilState(&z,&depth));
        }
        float clear[4]={-1,-1,-1,-1};context->ClearRenderTargetView(target,clear);++passes;
        if(selected.empty())return;
        if(bytes>capacity){records.Reset();capacity=0;D3D11_BUFFER_DESC d={};d.ByteWidth=UINT(bytes);d.Usage=D3D11_USAGE_DYNAMIC;
            d.BindFlags=D3D11_BIND_VERTEX_BUFFER;d.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;check(device->CreateBuffer(&d,nullptr,&records));capacity=UINT(bytes);}
        D3D11_MAPPED_SUBRESOURCE map={};check(context->Map(records.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&map));
        std::memcpy(map.pData,selected.data(),bytes);context->Unmap(records.Get(),0);upload_bytes+=bytes;triangles+=selected.size();
        float values[4]={float(extent)};context->UpdateSubresource(constants.Get(),0,nullptr,values,0,0);
        auto settings=constants.Get(),input=records.Get();UINT stride=48,offset=0;
        context->VSSetConstantBuffers(0,1,&settings);context->IASetInputLayout(layout.Get());context->IASetVertexBuffers(0,1,&input,&stride,&offset);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
        context->RSSetState(raster.Get());D3D11_VIEWPORT viewport={0,0,float(extent),float(extent),0,1};context->RSSetViewports(1,&viewport);
        context->OMSetDepthStencilState(depth.Get(),0);context->OMSetBlendState(blend.Get(),nullptr,~0u);context->OMSetRenderTargets(1,&target,nullptr);
        context->DrawInstanced(6,UINT(selected.size()),0,0);context->OMSetRenderTargets(0,nullptr,nullptr);
    }
};
}
