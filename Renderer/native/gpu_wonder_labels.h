#pragma once
#include "render_core/wonder_labels.h"
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace c3x_renderer {
// Natural-wonder names drawn by the renderer (docs/natural_wonder_rendering.md, Labels):
// display-space text after tone mapping and before map fog, so fog darkens a fogged
// label and unexplored tiles hide it, as Civ III draws its fog last. Each name is
// rasterised once with GDI in C3X's font (Civ III get_font (10, FSF_NONE): Lucida Sans,
// lfHeight -10, normal weight) as white text over a black shadow offset 1,1, and drawn
// at its exact pixel size at every zoom.
class GpuWonderLabels {
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    struct Raster {unsigned x=0,y=0,width=0,height=0,advance=0;};
    struct Record {float x,y,width,height,u,v,pad0,pad1;};
    render_core::WonderLabels input;
    std::map<std::string,Raster> rasters;
    Ptr<ID3D11ShaderResourceView> atlas;
    Ptr<ID3D11VertexShader> vertex;
    Ptr<ID3D11PixelShader> pixel;
    Ptr<ID3D11Buffer> settings,buffer;
    Ptr<ID3D11ShaderResourceView> records;
    Ptr<ID3D11BlendState> blend;
    Ptr<ID3D11RasterizerState> raster;
    Ptr<ID3D11DepthStencilState> depth;
    Ptr<ID3D11RenderTargetView> view;
    ID3D11Texture2D* target=nullptr;
    unsigned capacity=0;
    // Lay out and rasterise every known name into one premultiplied BGRA atlas.
    bool rasterise(ID3D11Device* device){
        atlas.Reset();
        HDC dc=CreateCompatibleDC(nullptr);if(!dc)return false;
        HFONT font=CreateFontA(-10,0,0,0,FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_TT_ONLY_PRECIS,
            CLIP_DEFAULT_PRECIS,DEFAULT_QUALITY,DEFAULT_PITCH|FF_DONTCARE,"Lucida Sans");
        HGDIOBJ previous_font=font?SelectObject(dc,font):nullptr;
        constexpr unsigned row_width=1024;
        unsigned x=0,y=0,row=0;bool ok=font!=nullptr;
        for(auto& item:rasters){
            SIZE size={};
            if(!ok || !GetTextExtentPoint32A(dc,item.first.c_str(),int(item.first.size()),&size) ||
               size.cx<=0 || size.cy<=0 || unsigned(size.cx)+1>row_width){ok=false;break;}
            auto& r=item.second;r.advance=unsigned(size.cx);r.width=unsigned(size.cx)+1;r.height=unsigned(size.cy)+1;
            if(x+r.width>row_width){x=0;y+=row;row=0;}
            r.x=x;r.y=y;x+=r.width+1;row=(std::max)(row,r.height+1);
        }
        unsigned height=y+row;
        if(ok && height>0 && height<=4096){
            BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=LONG(row_width);
            info.bmiHeader.biHeight=-LONG(height);info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
            void* bits=nullptr;HBITMAP bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&bits,nullptr,0);
            HGDIOBJ previous_bitmap=bitmap?SelectObject(dc,bitmap):nullptr;
            ok=bitmap!=nullptr;
            std::vector<unsigned> shadow,text;
            auto render=[&](int offset,std::vector<unsigned>& out){
                std::memset(bits,0,std::size_t(row_width)*height*4);
                SetBkMode(dc,TRANSPARENT);SetTextColor(dc,RGB(255,255,255));SetTextAlign(dc,TA_LEFT|TA_TOP|TA_NOUPDATECP);
                for(auto const& item:rasters)
                    ok=ok && TextOutA(dc,int(item.second.x)+offset,int(item.second.y)+offset,item.first.c_str(),int(item.first.size()));
                GdiFlush();
                auto const* source=static_cast<unsigned const*>(bits);out.assign(source,source+std::size_t(row_width)*height);
            };
            if(ok){render(1,shadow);render(0,text);}
            if(ok){
                // White text over the black shadow, premultiplied: the text's own coverage
                // per channel, and its coverage plus the shadow's where the text is clear.
                std::vector<unsigned> pixels(text.size());
                for(std::size_t i=0;i<pixels.size();++i){
                    unsigned t=text[i],s=shadow[i];
                    unsigned tb=t&255u,tg=(t>>8)&255u,tr=(t>>16)&255u,ta=(std::max)({tb,tg,tr});
                    unsigned sa=(std::max)({s&255u,(s>>8)&255u,(s>>16)&255u});
                    unsigned alpha=ta+(255u-ta)*sa/255u;
                    pixels[i]=tb|(tg<<8)|(tr<<16)|(alpha<<24);
                }
                D3D11_TEXTURE2D_DESC desc={};desc.Width=row_width;desc.Height=height;desc.MipLevels=desc.ArraySize=1;
                desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.SampleDesc.Count=1;desc.Usage=D3D11_USAGE_IMMUTABLE;
                desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
                D3D11_SUBRESOURCE_DATA data={pixels.data(),row_width*4,0};Ptr<ID3D11Texture2D> texture;
                ok=SUCCEEDED(device->CreateTexture2D(&desc,&data,&texture)) &&
                   SUCCEEDED(device->CreateShaderResourceView(texture.Get(),nullptr,&atlas));
            }
            if(previous_bitmap)SelectObject(dc,previous_bitmap);
            if(bitmap)DeleteObject(bitmap);
        }else ok=false;
        if(previous_font)SelectObject(dc,previous_font);
        if(font)DeleteObject(font);
        DeleteDC(dc);
        if(!ok)atlas.Reset();
        return ok;
    }
public:
    void reset(){rasters.clear();atlas.Reset();vertex.Reset();pixel.Reset();settings.Reset();buffer.Reset();records.Reset();
        blend.Reset();raster.Reset();depth.Reset();view.Reset();target=nullptr;capacity=0;input.labels.clear();}
    bool draw(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* output,
              c3x_renderer_frame_v1 const& frame,float zoom){
        if(!input.capture(frame))return false;
        if(input.labels.empty())return true;
        if(!device||!context||!output||zoom<SceneProjection::minimum||zoom>SceneProjection::maximum)return false;
        D3D11_TEXTURE2D_DESC out_desc={};output->GetDesc(&out_desc);
        if(out_desc.Width!=unsigned(frame.target_width)||out_desc.Height!=unsigned(frame.target_height)||
           out_desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM||out_desc.SampleDesc.Count!=1)return false;
        bool fresh=false;
        for(auto const& label:input.labels)if(rasters.emplace(label.name,Raster{}).second)fresh=true;
        if((fresh || !atlas) && !rasterise(device))return false;
        if(!vertex){
            char const* shader=R"(
struct Record{float2 position,size,uv,pad;};StructuredBuffer<Record> records:register(t0);
Texture2D<float4> atlas:register(t1);
cbuffer Settings:register(b0){float4 size;};
struct V{float4 position:SV_Position;nointerpolation float2 origin:TEXCOORD0;nointerpolation float2 uv:TEXCOORD1;};
V vs(uint id:SV_VertexID,uint instance:SV_InstanceID){
 float2 corner[6]={float2(0,0),float2(1,0),float2(1,1),float2(0,0),float2(1,1),float2(0,1)};
 Record r=records[instance];V v;float2 p=r.position+corner[id]*r.size;
 v.origin=r.position;v.uv=r.uv;v.position=float4(p.x*2/size.x-1,1-p.y*2/size.y,0,1);return v;
}
float4 ps(V v):SV_Target{return atlas.Load(int3(int2(v.uv+floor(v.position.xy-v.origin)),0));}
)";
            Ptr<ID3DBlob> vs,ps,error;
            if(FAILED(D3DCompile(shader,std::strlen(shader),"wonder labels",nullptr,nullptr,"vs","vs_5_0",
                                  D3DCOMPILE_ENABLE_STRICTNESS,0,&vs,&error)) ||
               FAILED(D3DCompile(shader,std::strlen(shader),"wonder labels",nullptr,nullptr,"ps","ps_5_0",
                                  D3DCOMPILE_ENABLE_STRICTNESS,0,&ps,&error)) ||
               FAILED(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex)) ||
               FAILED(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel)))return false;
            D3D11_BUFFER_DESC cb={};cb.ByteWidth=16;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
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
        std::vector<Record> placed;placed.reserve(input.labels.size());
        for(auto const& label:input.labels){
            auto const& r=rasters[label.name];
            auto box=render_core::WonderLabels::box(frame,zoom,label.anchor_x,label.anchor_y);
            float left=float(box.left+(box.width-int(r.advance))/2);
            placed.push_back({left,float(box.top),float(r.width),float(r.height),float(r.x),float(r.y),0,0});
        }
        unsigned bytes=unsigned(placed.size()*sizeof(Record));
        if(bytes>capacity){records.Reset();buffer.Reset();capacity=0;
            D3D11_BUFFER_DESC b={};b.ByteWidth=bytes;b.BindFlags=D3D11_BIND_SHADER_RESOURCE;b.Usage=D3D11_USAGE_DEFAULT;
            b.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;b.StructureByteStride=sizeof(Record);
            if(FAILED(device->CreateBuffer(&b,nullptr,&buffer))||
               FAILED(device->CreateShaderResourceView(buffer.Get(),nullptr,&records)))return false;
            capacity=bytes;
        }
        D3D11_BOX range={0,0,0,bytes,1,1};context->UpdateSubresource(buffer.Get(),0,&range,placed.data(),0,0);
        float constants[]={float(out_desc.Width),float(out_desc.Height),0,0};
        context->UpdateSubresource(settings.Get(),0,nullptr,constants,0,0);
        context->OMSetRenderTargets(0,nullptr,nullptr);auto output_view=view.Get();
        context->OMSetRenderTargets(1,&output_view,nullptr);context->OMSetBlendState(blend.Get(),nullptr,~0u);
        context->OMSetDepthStencilState(depth.Get(),0);context->RSSetState(raster.Get());
        D3D11_VIEWPORT viewport={0,0,float(out_desc.Width),float(out_desc.Height),0,1};context->RSSetViewports(1,&viewport);
        auto cb=settings.Get();context->VSSetConstantBuffers(0,1,&cb);
        ID3D11ShaderResourceView* vs_views[]={records.Get()};context->VSSetShaderResources(0,1,vs_views);
        ID3D11ShaderResourceView* ps_views[]={nullptr,atlas.Get()};context->PSSetShaderResources(0,2,ps_views);
        context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
        context->DrawInstanced(6,unsigned(placed.size()),0,0);context->OMSetRenderTargets(0,nullptr,nullptr);
        vs_views[0]=nullptr;context->VSSetShaderResources(0,1,vs_views);
        ps_views[1]=nullptr;context->PSSetShaderResources(0,2,ps_views);
        return true;
    }
};
}
