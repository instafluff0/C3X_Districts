#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

namespace c3x_renderer { namespace sandbox {
// Two original completed maps, normally a wide view and the latest close view.
// The output is never captured back into either source. Scene identity is a
// caller-owned visibility/view stamp; it is separate from projection and time.
class CompletedMapPreview {
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    struct Source {
        Ptr<ID3D11Texture2D> color,depth;
        Ptr<ID3D11RenderTargetView> target;
        Ptr<ID3D11ShaderResourceView> color_view,depth_view;
        unsigned width=0,height=0;
        float zoom=1.f,coverage=1.f;
        std::uint64_t serial=0;
        bool valid=false;
    } sources[2];
    Ptr<ID3D11VertexShader> vertex;
    Ptr<ID3D11PixelShader> color_pixel,depth_pixel;
    Ptr<ID3D11Buffer> settings;
    Ptr<ID3D11RasterizerState> raster;
    Ptr<ID3D11SamplerState> sampler;
    Ptr<ID3D11DepthStencilState> depth_write;
    ID3D11Device* owner=nullptr;
    int selected=-1;
    float relative=1.f;

    bool ensure(ID3D11Device* device) {
        if(!device)return false;
        if(owner!=device){reset();owner=device;}
        if(color_pixel && depth_pixel && settings && raster && sampler && depth_write)return true;
        char const* code=R"(
Texture2D<float4> original_color:register(t0);
Texture2D<float> original_depth:register(t1);
SamplerState filtered:register(s0);
cbuffer Settings:register(b0) {
 float2 output_center;float2 source_center;
 float relative_scale;float2 source_extent;float padding;
};
float4 VS(uint id:SV_VertexID):SV_Position {
 float2 p=float2((id<<1)&2,id&2);
 return float4(p*float2(2,-2)+float2(-1,1),0,1);
}
float2 source_at(float2 at) {
 return (at-output_center)/relative_scale+source_center;
}
bool covered(float2 at) {return all(at>=0)&&all(at<source_extent);}
float4 color_at(float2 at) {
 // Only an identity transform may bypass filtering. A relative scale below
 // one is an ordinary zoom-out; absolute scene projection limits do not apply.
 if(relative_scale==1.f)return original_color.Load(int3(int2(at),0));
 return original_color.SampleLevel(filtered,at/source_extent,0);
}
float4 Color(float4 position:SV_Position):SV_Target {
 float2 at=source_at(position.xy);
 return covered(at)?color_at(at):float4(0,0,0,0);
}
struct Witness {float4 color:SV_Target;float depth:SV_Depth;};
Witness Depth(float4 position:SV_Position) {
 float2 at=source_at(position.xy);Witness result;
 result.color=covered(at)?color_at(at):float4(0,0,0,0);
 // This is transformed original depth, not freshly rendered geometry. The
 // snapshot includes the production scene's four-pixel guard on each side.
 result.depth=covered(at)?original_depth.Load(int3(int2(at)+int2(4,4),0)):1.f;
 return result;
}
)";
        auto compile=[&](char const* entry,char const* profile,Ptr<ID3DBlob>& blob) {
            Ptr<ID3DBlob> errors;
            return SUCCEEDED(D3DCompile(code,std::strlen(code),"completed_map_preview",
                nullptr,nullptr,entry,profile,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,
                &blob,&errors));
        };
        Ptr<ID3DBlob> vs,ps,ds;
        if(!compile("VS","vs_4_0",vs) || !compile("Color","ps_4_0",ps) ||
           !compile("Depth","ps_4_0",ds))return false;
        if(FAILED(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex)) ||
           FAILED(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&color_pixel)) ||
           FAILED(device->CreatePixelShader(ds->GetBufferPointer(),ds->GetBufferSize(),nullptr,&depth_pixel)))return false;
        D3D11_BUFFER_DESC bd={};bd.ByteWidth=32;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        D3D11_RASTERIZER_DESC rd={};rd.FillMode=D3D11_FILL_SOLID;
        rd.CullMode=D3D11_CULL_NONE;rd.DepthClipEnable=TRUE;
        D3D11_SAMPLER_DESC sd={};sd.Filter=D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
        sd.AddressU=sd.AddressV=sd.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.MaxLOD=D3D11_FLOAT32_MAX;
        D3D11_DEPTH_STENCIL_DESC dd={};dd.DepthEnable=TRUE;
        dd.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;dd.DepthFunc=D3D11_COMPARISON_ALWAYS;
        return SUCCEEDED(device->CreateBuffer(&bd,nullptr,&settings)) &&
            SUCCEEDED(device->CreateRasterizerState(&rd,&raster)) &&
            SUCCEEDED(device->CreateSamplerState(&sd,&sampler)) &&
            SUCCEEDED(device->CreateDepthStencilState(&dd,&depth_write));
    }
public:
    void invalidate() {
        for(auto& source:sources)source.valid=false;
        selected=-1;
    }
    void reset() {
        invalidate();for(auto& source:sources)source=Source{};
        vertex.Reset();color_pixel.Reset();depth_pixel.Reset();settings.Reset();
        raster.Reset();sampler.Reset();depth_write.Reset();owner=nullptr;relative=1.f;
    }
    bool prepare(ID3D11Device* device,unsigned slot,unsigned width,unsigned height) {
        if(slot>=2 || !width || !height || width>16376 || height>16376 || !ensure(device))return false;
        auto& source=sources[slot];source.valid=false;
        if(selected==int(slot))selected=-1;
        if(source.color && source.width==width && source.height==height)return true;
        Source next;
        D3D11_TEXTURE2D_DESC td={};td.Width=width;td.Height=height;
        td.MipLevels=td.ArraySize=td.SampleDesc.Count=1;
        td.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
        td.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        if(FAILED(device->CreateTexture2D(&td,nullptr,&next.color)) ||
           FAILED(device->CreateRenderTargetView(next.color.Get(),nullptr,&next.target)) ||
           FAILED(device->CreateShaderResourceView(next.color.Get(),nullptr,&next.color_view)))return false;
        td.Width=width+8;td.Height=height+8;td.Format=DXGI_FORMAT_R24G8_TYPELESS;
        td.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SHADER_RESOURCE_VIEW_DESC sd={};sd.Format=DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        sd.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D;sd.Texture2D.MipLevels=1;
        if(FAILED(device->CreateTexture2D(&td,nullptr,&next.depth)) ||
           FAILED(device->CreateShaderResourceView(next.depth.Get(),&sd,&next.depth_view)))return false;
        next.width=width;next.height=height;source=std::move(next);return true;
    }
    ID3D11RenderTargetView* target(unsigned slot)const {
        return slot<2?sources[slot].target.Get():nullptr;
    }
    bool commit(ID3D11DeviceContext* context,unsigned slot,ID3D11Texture2D* original_depth,
            float absolute_zoom,float coverage_min_zoom,std::uint64_t serial) {
        if(slot>=2 || !context || !original_depth || !std::isfinite(absolute_zoom) ||
           !std::isfinite(coverage_min_zoom) || absolute_zoom<=0 || coverage_min_zoom<=0 ||
           coverage_min_zoom>absolute_zoom)return false;
        auto& source=sources[slot];if(!source.color || !source.depth)return false;
        D3D11_TEXTURE2D_DESC d={};original_depth->GetDesc(&d);
        if(d.Width!=source.width+8 || d.Height!=source.height+8 || d.SampleDesc.Count!=1 ||
           d.MipLevels!=1 || d.ArraySize!=1 ||
           (d.Format!=DXGI_FORMAT_R24G8_TYPELESS && d.Format!=DXGI_FORMAT_D24_UNORM_S8_UINT))return false;
        // No readback or wait. The same immediate context orders the finished
        // map, depth copy and later display; this is not independent GPU work.
        context->OMSetRenderTargets(0,nullptr,nullptr);
        context->CopyResource(source.depth.Get(),original_depth);
        for(auto& other:sources)if(other.valid && other.serial!=serial)other.valid=false;
        source.zoom=absolute_zoom;source.coverage=coverage_min_zoom;
        source.serial=serial;source.valid=true;selected=-1;return true;
    }
    bool select(unsigned slot,float requested_absolute_zoom,std::uint64_t serial) {
        selected=-1;
        if(slot>=2 || !std::isfinite(requested_absolute_zoom) || requested_absolute_zoom<=0)return false;
        for(auto const& other:sources)if(other.valid && other.serial!=serial){invalidate();return false;}
        auto const& source=sources[slot];if(!source.valid)return false;
        if(requested_absolute_zoom<source.coverage)return false;
        float ratio=requested_absolute_zoom/source.zoom;
        if(!std::isfinite(ratio) || ratio<=0)return false;
        relative=ratio;selected=int(slot);return true;
    }
    bool draw(ID3D11Device* device,ID3D11DeviceContext* context,
            ID3D11RenderTargetView* destination,unsigned output_width,unsigned output_height,
            ID3D11DepthStencilView* output_depth=nullptr) {
        if(!context || !destination || !output_width || !output_height || selected<0 ||
           owner!=device || !color_pixel || !depth_pixel)return false;
        auto const& source=sources[selected];if(!source.valid)return false;
        struct Values {float output_center[2],source_center[2],scale,extent[2],padding;} values={
            {float(output_width/2),float(output_height/2)},
            {float(source.width/2),float(source.height/2)},relative,
            {float(source.width),float(source.height)},0};
        context->OMSetRenderTargets(1,&destination,output_depth);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(output_depth?depth_write.Get():nullptr,0);
        context->RSSetState(raster.Get());
        D3D11_VIEWPORT viewport={0,0,float(output_width),float(output_height),0,1};
        context->RSSetViewports(1,&viewport);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->GSSetShader(nullptr,nullptr,0);
        context->HSSetShader(nullptr,nullptr,0);context->DSSetShader(nullptr,nullptr,0);
        context->VSSetShader(vertex.Get(),nullptr,0);
        context->PSSetShader(output_depth?depth_pixel.Get():color_pixel.Get(),nullptr,0);
        context->UpdateSubresource(settings.Get(),0,nullptr,&values,0,0);
        auto constants=settings.Get();context->PSSetConstantBuffers(0,1,&constants);
        ID3D11ShaderResourceView* inputs[]={source.color_view.Get(),source.depth_view.Get()};
        context->PSSetShaderResources(0,output_depth?2u:1u,inputs);
        auto filter=sampler.Get();context->PSSetSamplers(0,1,&filter);
        context->Draw(3,0);
        ID3D11ShaderResourceView* empty[]={nullptr,nullptr};context->PSSetShaderResources(0,2,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);return true;
    }
    std::uint64_t bytes()const {
        std::uint64_t total=0;
        for(auto const& source:sources)if(source.color)
            total+=(std::uint64_t(source.width)*source.height+
                std::uint64_t(source.width+8)*(source.height+8))*4;
        return total;
    }
    unsigned width(unsigned slot)const{return slot<2?sources[slot].width:0;}
    unsigned height(unsigned slot)const{return slot<2?sources[slot].height:0;}
    float completed_zoom(unsigned slot)const{return slot<2?sources[slot].zoom:0.f;}
    float relative_zoom()const{return selected>=0?relative:0.f;}
};
} }
