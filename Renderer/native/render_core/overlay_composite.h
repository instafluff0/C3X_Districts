#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cstring>
#include "linear_target.h"

namespace c3x_renderer { namespace render_core {
// Composites a retained premultiplied layer (color plus the depth it wrote)
// into the live scene at a whole-pixel offset, as one full-target draw.
//
// Static overlays that must cover animated water (roads, improvements,
// features, cliffs and their shadows near rivers and coasts) used to be
// redrawn after the water every frame. Premultiplied "over" is associative,
// so drawing them once into a cleared layer and compositing that layer over
// the live water is identical, provided the layer's depth was tested against
// the same static and water depth. The layer stores min(static, water,
// overlay) depth; writing it with LESS_EQUAL reproduces the overlay depth
// where an overlay drew and leaves the live depth elsewhere.
struct OverlayComposite {
    ID3D11VertexShader* vertex=nullptr;ID3D11PixelShader* pixel=nullptr;
    ID3D11Buffer* settings=nullptr;ID3D11BlendState* blend=nullptr;
    ID3D11DepthStencilState* depth=nullptr;ID3D11RasterizerState* rasterizer=nullptr;
    template<class T>void release(T*& p){if(p)p->Release();p=nullptr;}
    void reset(){release(vertex);release(pixel);release(settings);release(blend);release(depth);release(rasterizer);}
    ~OverlayComposite(){reset();}
    bool ensure(ID3D11Device* device){
        if(pixel)return true;
        static char const source[]=R"(
Texture2D<float4> layer:register(t0);
Texture2D<float> layer_depth:register(t1);
cbuffer Composite:register(b0){int4 move;int4 covered;float4 depth_adjust;};
float4 VS(uint id:SV_VertexID):SV_Position{float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
struct Output{float4 color:SV_Target;float depth:SV_Depth;};
Output PS(float4 position:SV_Position){
 int2 p=int2(position.xy)-move.xy;
 if(any(p<covered.xy)||any(p>=covered.zw))discard;
 float d=layer_depth.Load(int3(p,0));
 if(d>=1)discard;
 Output result;result.color=layer.Load(int3(p,0));result.depth=d+depth_adjust.x;return result;
})";
        auto compile=[&](char const* entry,char const* target,ID3DBlob** blob){
            ID3DBlob* errors=nullptr;HRESULT hr=D3DCompile(source,sizeof(source)-1,"overlay_composite",nullptr,nullptr,
                entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,blob,&errors);
            if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));errors->Release();}return hr;
        };
        ID3DBlob* blob=nullptr;HRESULT hr=compile("VS","vs_5_0",&blob);
        if(SUCCEEDED(hr))hr=device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vertex);release(blob);
        if(SUCCEEDED(hr))hr=compile("PS","ps_5_0",&blob);
        if(SUCCEEDED(hr))hr=device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&pixel);release(blob);
        D3D11_BUFFER_DESC b={};b.ByteWidth=48;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&settings);
        D3D11_BLEND_DESC blend_desc={};auto& rt=blend_desc.RenderTarget[0];
        rt.BlendEnable=TRUE;rt.SrcBlend=rt.SrcBlendAlpha=D3D11_BLEND_ONE;rt.DestBlend=rt.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;
        rt.BlendOp=rt.BlendOpAlpha=D3D11_BLEND_OP_ADD;rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
        if(SUCCEEDED(hr))hr=device->CreateBlendState(&blend_desc,&blend);
        D3D11_DEPTH_STENCIL_DESC d={};d.DepthEnable=TRUE;d.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;d.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
        if(SUCCEEDED(hr))hr=device->CreateDepthStencilState(&d,&depth);
        D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=TRUE;
        if(SUCCEEDED(hr))hr=device->CreateRasterizerState(&r,&rasterizer);
        if(FAILED(hr)){reset();return false;}
        return true;
    }
    // `move` maps a target pixel to layer pixel (target - move); `covered` is
    // the layer's valid rectangle; `depth_shift` rebases retained depth.
    bool draw(ID3D11DeviceContext* context,LinearTarget const& target,LinearTarget const& layer,
            int move_x,int move_y,D3D11_RECT covered,float depth_shift){
        if(!pixel||!target.target||!target.depth||target.sample_count!=1||!layer.samples||!layer.depth_samples)return false;
        struct Constants{int move[4],covered[4];float depth_adjust[4];} values={{move_x,move_y,0,0},
            {int(covered.left),int(covered.top),int(covered.right),int(covered.bottom)},{depth_shift,0,0,0}};
        context->UpdateSubresource(settings,0,nullptr,&values,0,0);
        context->OMSetRenderTargets(1,&target.target,target.depth);context->OMSetBlendState(blend,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(depth,0);context->RSSetState(rasterizer);
        D3D11_VIEWPORT viewport={0,0,float(target.width),float(target.height),0,1};context->RSSetViewports(1,&viewport);
        context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);
        context->PSSetConstantBuffers(0,1,&settings);
        ID3D11ShaderResourceView* views[]={layer.samples,layer.depth_samples};context->PSSetShaderResources(0,2,views);
        context->Draw(3,0);
        ID3D11ShaderResourceView* none[2]={};context->PSSetShaderResources(0,2,none);
        return true;
    }
};
}}
