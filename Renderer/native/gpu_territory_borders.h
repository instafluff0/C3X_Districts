#pragma once
#include "scene_projection.h"
#include <wrl/client.h>
#include <string>

namespace c3x_renderer {
// Reuses the immutable production terrain triangles and their native anchors.
// Color and depth stay on the GPU; ownership is captured by the game thread.
class TerritoryBorders {
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    Ptr<ID3D11VertexShader> vertex;
    Ptr<ID3D11PixelShader> pixel;
    Ptr<ID3D11InputLayout> natural_layout,ground_layout;
    Ptr<ID3D11Buffer> constants;
    Ptr<ID3D11BlendState> blend;
    Ptr<ID3D11RasterizerState> raster;
    Ptr<ID3D11DepthStencilState> no_depth;
    unsigned samples=0;
    bool ensure(ID3D11Device* device,unsigned count){
        if(pixel && samples==count)return true;
        *this=TerritoryBorders{};
        char const* source=R"(
#if SAMPLES==1
Texture2D<float> scene_depth:register(t0);
float depth_at(int2 p,uint sample){return scene_depth.Load(int3(p,0));}
#else
Texture2DMS<float,SAMPLES> scene_depth:register(t0);
float depth_at(int2 p,uint sample){return scene_depth.Load(p,sample);}
#endif
cbuffer Border:register(b0){
 float4 translation_extent; // translation xy, inverse logical extent zw
 float4 projection;         // native column, row, tile width, height metric
 float4 display_color;     // linear RGB, depth translation
 float4 metadata;          // native edge mask, width in world units, depth clearance, unused
};
struct V{float3 position:POSITION;float3 world:TEXCOORD0;};
struct P{float4 position:SV_Position;float2 local:TEXCOORD0;float2 world:TEXCOORD1;};
P VS(V v){P p;float2 q=v.world.xy-projection.xy;
 float h=v.world.z*112-2.5,base=(q.x-q.y+1)*projection.z*.25;
 float3 projected=float3((q.x+q.y)*projection.z*.5,
   base-h*(projection.z/224*.82),base+h*.0016*projection.w);
 p.position=float4((floor(projected.xy*256+.5)/256+translation_extent.xy)*translation_extent.zw*float2(2,-2)+float2(-1,1),
   clamp(.5-(floor(projected.z*256+.5)/256+display_color.w)/16384,.001,.999),1);
 p.local=q;p.world=v.world.xy;return p;}
float rounded_distance(float2 q,uint edges){
 float d=1000;
 if(edges&1)d=min(d,q.x);if(edges&8)d=min(d,1-q.x);
 if(edges&4)d=min(d,q.y);if(edges&2)d=min(d,1-q.y);
 const float r=.12;
 [unroll]for(uint x=0;x<2;x++)[unroll]for(uint y=0;y<2;y++){
  uint ux=x?8:1,vy=y?2:4;float2 c=float2(x?1-q.x:q.x,y?1-q.y:q.y);
  if((edges&ux)&&(edges&vy)&&all(c<r))d=min(d,r-length(c-r));
 }
 return d;
}
float4 PS(P p,uint sample:SV_SampleIndex):SV_Target{
 // Clip to this ownership tile even where a relief mesh extends beyond it.
 clip(min(min(p.local.x,p.local.y),min(1-p.local.x,1-p.local.y))+.00002);
 float d=rounded_distance(p.local,(uint)metadata.x),w=metadata.y;
 float noise=.93+.05*sin(dot(p.world,float2(7.7,11.3)))+.025*sin(dot(p.world,float2(19.1,-13.4)));
 float reach=5*w*noise,aa=max(fwidth(d),.0001);
 float band=pow(saturate(1-max(d,0)/reach),1.25)*.5686;
 float soft=exp(-pow(d/(w*.7),2))*.294;
 float main=saturate((w*.5-abs(d))/aa+.5)*.776;
 float core=saturate((w*.175-abs(d))/aa+.5)*.165;
 float alpha=1-(1-band)*(1-soft)*(1-main)*(1-core);
 alpha*=saturate(d/aa+.5);clip(alpha-.001);
 // Require coherent foreground depth, with a strong central difference.
 // This suppresses self-occlusion from terrain interpolation at steep edges.
 int2 at=int2(p.position.xy);float dz=metadata.z/16384;
 float center=depth_at(at,sample);float hidden=0;
 [unroll]for(int k=-2;k<=2;k++){
  float z=depth_at(at+int2(k,0),sample);
  hidden+=z<p.position.z-dz?1:0;
 }
 float occluded=(hidden>=3 && center<p.position.z-dz*2)?1:0;
 alpha*=lerp(1,.34,occluded);
 return float4(display_color.rgb*alpha,alpha);
})";
        std::string shader=source;
        std::string number=std::to_string(count);
        D3D_SHADER_MACRO defines[]={{"SAMPLES",number.c_str()},{nullptr,nullptr}};
        Ptr<ID3DBlob> vs,ps,error;
        auto compile=[&](char const* entry,char const* profile,Ptr<ID3DBlob>& blob){
            HRESULT hr=D3DCompile(shader.data(),shader.size(),"territory borders",defines,nullptr,entry,profile,
                D3DCOMPILE_ENABLE_STRICTNESS,0,&blob,&error);
            if(FAILED(hr)&&error)OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));return SUCCEEDED(hr);};
        if(!compile("VS","vs_5_0",vs)||!compile("PS","ps_5_0",ps))return false;
        if(FAILED(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex))||
           FAILED(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel)))return false;
        D3D11_INPUT_ELEMENT_DESC elements[]={
            {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TEXCOORD",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0}};
        if(FAILED(device->CreateInputLayout(elements,2,vs->GetBufferPointer(),vs->GetBufferSize(),&natural_layout)))return false;
        elements[1].AlignedByteOffset=120;
        if(FAILED(device->CreateInputLayout(elements,2,vs->GetBufferPointer(),vs->GetBufferSize(),&ground_layout)))return false;
        D3D11_BUFFER_DESC cb={};cb.ByteWidth=64;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(FAILED(device->CreateBuffer(&cb,nullptr,&constants)))return false;
        D3D11_BLEND_DESC bd={};auto& b=bd.RenderTarget[0];b.BlendEnable=TRUE;
        b.SrcBlend=b.SrcBlendAlpha=D3D11_BLEND_ONE;b.DestBlend=b.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;
        b.BlendOp=b.BlendOpAlpha=D3D11_BLEND_OP_ADD;b.RenderTargetWriteMask=15;
        if(FAILED(device->CreateBlendState(&bd,&blend)))return false;
        D3D11_RASTERIZER_DESC rd={};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;
        rd.DepthClipEnable=TRUE;rd.MultisampleEnable=TRUE;
        if(FAILED(device->CreateRasterizerState(&rd,&raster)))return false;
        D3D11_DEPTH_STENCIL_DESC ds={};ds.DepthEnable=FALSE;ds.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
        if(FAILED(device->CreateDepthStencilState(&ds,&no_depth)))return false;
        samples=count;return true;
    }
public:
    template<class Records,class Settings,class Visible>
    bool draw(ID3D11Device* device,ID3D11DeviceContext* context,Records const& records,
              Settings const& settings,render_core::LinearTarget& target,
              unsigned width,unsigned height,float zoom,float scale,Visible visible){
        bool any=false;for(auto const& r:records)any|=r.territory_edges!=0;
        if(!any)return true;
        if(!target.depth_samples || !ensure(device,target.sample_count))return false;
        auto rt=target.target;context->OMSetRenderTargets(1,&rt,nullptr);
        context->OMSetDepthStencilState(no_depth.Get(),0);context->OMSetBlendState(blend.Get(),nullptr,~0u);
        context->RSSetState(raster.Get());D3D11_VIEWPORT vp={0,0,float(target.width),float(target.height),0,1};
        SceneProjection(width,height,zoom).viewport(vp,4,0,0,scale);context->RSSetViewports(1,&vp);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
        auto cb=constants.Get();context->VSSetConstantBuffers(0,1,&cb);context->PSSetConstantBuffers(0,1,&cb);
        context->PSSetShaderResources(0,1,&target.depth_samples);
        for(auto const& r:records){
            if(!r.territory_edges || !visible(r))continue;
            auto const& chunk=r.content();
            float values[20]={settings.translation[0]+float(r.translation_x),settings.translation[1]+float(r.translation_y),
                settings.inverse_size[0],settings.inverse_size[1]};
            std::copy(std::begin(r.natural_projection),std::end(r.natural_projection),values+4);
            for(unsigned j=0;j<3;++j){float s=float((r.territory_rgb>>(16-j*8))&255)/255.f;
                values[8+j]=s<=.04045f?s/12.92f:std::pow((s+.055f)/1.055f,2.4f);}
            values[11]=settings.depth_translation+float(r.translation_y);
            values[12]=float(r.territory_edges);values[13]=4.2f*std::pow(values[6]/128.f,.65f)/(values[6]*.4472136f);
            values[14]=20.f*float(height)/800.f;
            context->UpdateSubresource(cb,0,nullptr,values,0,0);
            context->IASetInputLayout(chunk.vertex_stride==92?natural_layout.Get():ground_layout.Get());
            context->IASetVertexBuffers(0,1,&chunk.buffer,&chunk.vertex_stride,&chunk.vertex_offset);
            context->IASetIndexBuffer(chunk.indices,chunk.index_format,chunk.index_offset);
            context->DrawIndexed(chunk.index_count,0,0);
        }
        ID3D11ShaderResourceView* none=nullptr;context->PSSetShaderResources(0,1,&none);
        context->OMSetRenderTargets(0,nullptr,nullptr);return true;
    }
};
}
