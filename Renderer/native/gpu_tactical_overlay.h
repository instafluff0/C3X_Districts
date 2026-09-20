#pragma once
#include "tactical_overlay.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>

namespace c3x_renderer::tactical {
// One worker-owned, bounded RGBA scratch attachment. Native retained composition
// owns ordering/lifetime; this pass knows nothing about world or unit meshes.
class Gpu {
    template<class T>using Ptr=Microsoft::WRL::ComPtr<T>;
    Ptr<ID3D11VertexShader> vs;Ptr<ID3D11PixelShader> ps;
    Ptr<ID3D11Buffer> params,instances;Ptr<ID3D11ShaderResourceView> input,glyphs;
    Ptr<ID3D11BlendState> blend;Ptr<ID3D11RasterizerState> raster;Ptr<ID3D11SamplerState> sampler;
    Ptr<ID3D11Texture2D> texture;Ptr<ID3D11RenderTargetView> target;
    Ptr<ID3D11ComputeShader> pack_shader;Ptr<ID3D11Texture2D> packed_texture;
    Ptr<ID3D11ShaderResourceView> color_view;Ptr<ID3D11UnorderedAccessView> packed_view;
    unsigned width=0,height=0,capacity=0;
    void check(HRESULT hr){if(FAILED(hr))throw std::runtime_error("tactical GPU pass failed");}
    void initialize(ID3D11Device* d){
        if(vs)return;
        char const* source=R"(
struct Primitive {float4 bounds,shape,color,style;};
StructuredBuffer<Primitive> items:register(t0);Texture2D<float4> font:register(t1);SamplerState linearClamp:register(s0);
cbuffer Frame:register(b0){float4 viewport;float4 clock;};
struct V {float4 position:SV_Position;float2 location:TEXCOORD0;nointerpolation uint id:TEXCOORD1;};
V vertex(uint vertex:SV_VertexID,uint instance:SV_InstanceID){
 float2 corners[6]={float2(0,0),float2(1,0),float2(0,1),float2(0,1),float2(1,0),float2(1,1)};
 Primitive p=items[instance];V o;o.location=lerp(p.bounds.xy,p.bounds.zw,corners[vertex]);
 o.position=float4((o.location-viewport.xy)/viewport.zw*float2(2,-2)+float2(-1,1),0,1);o.id=instance;return o;
}
float segment(float2 q,float2 a,float2 b){float2 v=b-a;return length(q-a-v*saturate(dot(q-a,v)/max(dot(v,v),.001)));}
float coverage(float distance,float halfWidth){return saturate((halfWidth-distance)/max(fwidth(distance),.75)+.5);}
float4 over(float4 a,float4 b){return a+b*(1-a.a);}
float ringDistance(float2 q,Primitive p){float2 n=(q-p.shape.xy)/p.shape.zw;return abs(length(n)-1)/max(length(n/(p.shape.zw*max(length(n),.001))),.0001);}
float ring(float2 q,Primitive p){
 float2 n=(q-p.shape.xy)/p.shape.zw;
 float gap=min(abs(q.x-p.shape.x),abs(q.y-p.shape.y));
 return coverage(ringDistance(q,p),p.style.y*.5)*smoothstep(1.5,3.2,gap);
}
float arrow(float2 q,Primitive p,float phase){
 float2 n=(q-p.shape.xy)/p.shape.zw;float a=atan2(n.y,n.x)-phase;
 float sector=frac(a/6.2831853*4+.5)-.5;float tangent=sector*1.5707963;
 float radial=length(n);float h=.13,w=.105;
 // Four small inward-pointing triangular markers, rotating on the ellipse.
 float edge=max(abs(tangent)/w-(radial-.76)/h,(radial-.76)/h-1);
 return saturate(.5-edge/max(fwidth(edge),.15));
}
float glyph(float2 q,Primitive p){float2 uv=(q-p.shape.xy)/p.shape.zw;
 if(any(uv<0)||any(uv>1))return 0;uint g=(uint)p.style.y;
 return font.SampleLevel(linearClamp,(float2(g%16,g/16)+uv)/float2(16,6),0).r;}
float4 pixel(V i):SV_Target{
 Primitive p=items[i.id];float a=0,shadow=0;
 if(p.style.x<.5){float dist=segment(i.location,p.shape.xy,p.shape.zw);a=coverage(dist,p.style.y*.5);
  if(p.style.z>0)shadow=coverage(segment(i.location-float2(0,1.3),p.shape.xy,p.shape.zw),p.style.y*.5+1.25)*.65;
 }else if(p.style.x<1.5){float phase=p.style.z>0?clock.x*.9:0;
  a=max(ring(i.location,p),arrow(i.location,p,phase)*.88);
  shadow=max(ring(i.location-float2(0,1.4),p),arrow(i.location-float2(0,1.4),p,phase))*.5;
 }else if(p.style.x<2.5){a=glyph(i.location,p);shadow=max(max(glyph(i.location+float2(-1,0),p),glyph(i.location+float2(1,0),p)),max(glyph(i.location+float2(0,-1),p),glyph(i.location+float2(0,1.5),p)))*.9;
 }else {float2 v=p.shape.zw-p.shape.xy;float lengthV=max(length(v),.001);float along=dot(i.location-p.shape.xy,v)/lengthV;
  a=coverage(abs((i.location.x-p.shape.x)*v.y-(i.location.y-p.shape.y)*v.x)/lengthV,p.style.y*.5);
  if(along<0||along>=lengthV)a=0;
  if(p.style.z>.5&&p.style.z<1.5){float step=along*max(abs(v.x),abs(v.y))/lengthV;if(fmod(step,10)<5)a=0;}
  else if(p.style.z>1.5&&fmod(along,4*p.style.y)>=3*p.style.y)a=0;
 }
 float alpha=a*p.color.a;return over(float4(p.color.rgb*alpha,alpha),float4(.025,.035,.045,1)*shadow);
})";
        Ptr<ID3DBlob> v,p,error;
        auto compile=[&](char const* entry,char const* profile,Ptr<ID3DBlob>& out){
            HRESULT hr=D3DCompile(source,std::strlen(source),"tactical overlay",nullptr,nullptr,entry,profile,D3DCOMPILE_ENABLE_STRICTNESS,0,&out,&error);
            if(FAILED(hr)){if(error)throw std::runtime_error(std::string(static_cast<char*>(error->GetBufferPointer()),error->GetBufferSize()));check(hr);}};
        compile("vertex","vs_5_0",v);compile("pixel","ps_5_0",p);
        check(d->CreateVertexShader(v->GetBufferPointer(),v->GetBufferSize(),nullptr,&vs));
        check(d->CreatePixelShader(p->GetBufferPointer(),p->GetBufferSize(),nullptr,&ps));
        D3D11_BUFFER_DESC cb={};cb.ByteWidth=32;cb.Usage=D3D11_USAGE_DEFAULT;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;check(d->CreateBuffer(&cb,nullptr,&params));
        D3D11_BLEND_DESC bd={};auto& b=bd.RenderTarget[0];b.BlendEnable=TRUE;b.SrcBlend=D3D11_BLEND_ONE;b.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;b.BlendOp=D3D11_BLEND_OP_ADD;
        b.SrcBlendAlpha=D3D11_BLEND_ONE;b.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;b.BlendOpAlpha=D3D11_BLEND_OP_ADD;b.RenderTargetWriteMask=15;check(d->CreateBlendState(&bd,&blend));
        D3D11_RASTERIZER_DESC rd={};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;rd.DepthClipEnable=TRUE;check(d->CreateRasterizerState(&rd,&raster));
        D3D11_SAMPLER_DESC sd={};sd.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;sd.AddressU=sd.AddressV=sd.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;sd.MaxLOD=D3D11_FLOAT32_MAX;check(d->CreateSamplerState(&sd,&sampler));
        // Generic Windows font input, rasterized once at high resolution. No
        // copyrighted game bitmap or per-frame GDI drawing enters this pass.
        HDC dc=CreateCompatibleDC(nullptr);BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=1024;info.bmiHeader.biHeight=-384;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
        void* bits=nullptr;HBITMAP bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&bits,nullptr,0);
        if(!dc||!bitmap||!bits)throw std::runtime_error("tactical font allocation");
        auto old=SelectObject(dc,bitmap);std::memset(bits,0,1024*384*4);
        HFONT face=CreateFontW(-54,0,0,0,FW_SEMIBOLD,FALSE,FALSE,FALSE,ANSI_CHARSET,OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,ANTIALIASED_QUALITY,DEFAULT_PITCH,L"Segoe UI");
        auto oldfont=SelectObject(dc,face);SetTextColor(dc,RGB(255,255,255));SetBkMode(dc,TRANSPARENT);
        for(unsigned n=0;n<95;++n){wchar_t c=wchar_t(n+32);RECT r={LONG(n%16*64),LONG(n/16*64),LONG(n%16*64+64),LONG(n/16*64+64)};DrawTextW(dc,&c,1,&r,DT_CENTER|DT_VCENTER|DT_SINGLELINE|DT_NOPREFIX);}
        GdiFlush();D3D11_TEXTURE2D_DESC td={};td.Width=1024;td.Height=384;td.MipLevels=td.ArraySize=1;td.Format=DXGI_FORMAT_B8G8R8A8_UNORM;td.SampleDesc.Count=1;td.Usage=D3D11_USAGE_IMMUTABLE;td.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA data={bits,4096,0};Ptr<ID3D11Texture2D> atlas;HRESULT hr=d->CreateTexture2D(&td,&data,&atlas);
        SelectObject(dc,oldfont);DeleteObject(face);SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);check(hr);check(d->CreateShaderResourceView(atlas.Get(),nullptr,&glyphs));
    }
public:
    ID3D11Texture2D* draw(ID3D11Device* d,ID3D11DeviceContext* c,Input const& capture,std::array<int,4> area,double seconds){
        initialize(d);unsigned w=unsigned(area[2]-area[0]),h=unsigned(area[3]-area[1]);
        if(!w||!h||w>2240||h>1260||capture.primitives.empty())throw std::runtime_error("tactical extent");
        if(w!=width||h!=height){texture.Reset();target.Reset();color_view.Reset();packed_texture.Reset();packed_view.Reset();D3D11_TEXTURE2D_DESC td={};td.Width=w;td.Height=h;td.MipLevels=td.ArraySize=1;td.Format=DXGI_FORMAT_B8G8R8A8_UNORM;td.SampleDesc.Count=1;td.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
            check(d->CreateTexture2D(&td,nullptr,&texture));check(d->CreateRenderTargetView(texture.Get(),nullptr,&target));width=w;height=h;}
        unsigned count=unsigned(capture.primitives.size());
        if(count>capacity){input.Reset();instances.Reset();capacity=std::max(64u,count);D3D11_BUFFER_DESC b={};b.ByteWidth=capacity*sizeof(Primitive);b.Usage=D3D11_USAGE_DYNAMIC;b.BindFlags=D3D11_BIND_SHADER_RESOURCE;b.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;b.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;b.StructureByteStride=sizeof(Primitive);
            check(d->CreateBuffer(&b,nullptr,&instances));check(d->CreateShaderResourceView(instances.Get(),nullptr,&input));}
        D3D11_MAPPED_SUBRESOURCE mapped={};check(c->Map(instances.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped));std::memcpy(mapped.pData,capture.primitives.data(),count*sizeof(Primitive));c->Unmap(instances.Get(),0);
        float values[8]={float(area[0]),float(area[1]),float(w),float(h),float(std::fmod(seconds,6.981317007977318)),0,0,0};c->UpdateSubresource(params.Get(),0,nullptr,values,0,0);
        ID3D11RenderTargetView* rt=target.Get();float clear[4]={};c->ClearRenderTargetView(rt,clear);c->OMSetRenderTargets(1,&rt,nullptr);c->OMSetBlendState(blend.Get(),nullptr,~0u);c->OMSetDepthStencilState(nullptr,0);c->RSSetState(raster.Get());D3D11_VIEWPORT vp={0,0,float(w),float(h),0,1};c->RSSetViewports(1,&vp);
        c->IASetInputLayout(nullptr);c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);c->VSSetShader(vs.Get(),nullptr,0);c->PSSetShader(ps.Get(),nullptr,0);auto buffer=params.Get();c->VSSetConstantBuffers(0,1,&buffer);c->PSSetConstantBuffers(0,1,&buffer);
        ID3D11ShaderResourceView* views[]={input.Get(),glyphs.Get()};c->VSSetShaderResources(0,1,views);c->PSSetShaderResources(0,2,views);auto sam=sampler.Get();c->PSSetSamplers(0,1,&sam);c->DrawInstanced(6,count,0,0);
        views[0]=views[1]=nullptr;c->VSSetShaderResources(0,1,views);c->PSSetShaderResources(0,2,views);c->OMSetRenderTargets(0,nullptr,nullptr);return texture.Get();
    }
    ID3D11Texture2D* packed(ID3D11Device* d,ID3D11DeviceContext* c,Input const& capture,std::array<int,4> area,double seconds){
        draw(d,c,capture,area,seconds);
        // The existing native compositor consumes packed R32_UINT BGRA. Keep
        // conversion GPU-resident in reusable scratch rather than a CPU upload.
        if(!pack_shader){char const* source=R"(
Texture2D<float4> color:register(t0);RWTexture2D<uint> packed:register(u0);
[numthreads(8,8,1)]void main(uint3 at:SV_DispatchThreadID){uint w,h;packed.GetDimensions(w,h);if(at.x>=w||at.y>=h)return;
 uint4 c=uint4(saturate(color.Load(int3(at.xy,0)))*255+.5);packed[at.xy]=c.b|(c.g<<8)|(c.r<<16)|(c.a<<24);}
)";
            Ptr<ID3DBlob> code;check(D3DCompile(source,std::strlen(source),"tactical packing",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,nullptr));
            check(d->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&pack_shader));}
        if(!packed_texture){D3D11_TEXTURE2D_DESC desc={};texture->GetDesc(&desc);desc.Format=DXGI_FORMAT_R32_UINT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
            check(d->CreateTexture2D(&desc,nullptr,&packed_texture));check(d->CreateUnorderedAccessView(packed_texture.Get(),nullptr,&packed_view));check(d->CreateShaderResourceView(texture.Get(),nullptr,&color_view));}
        auto source=color_view.Get();auto destination=packed_view.Get();c->CSSetShader(pack_shader.Get(),nullptr,0);c->CSSetShaderResources(0,1,&source);c->CSSetUnorderedAccessViews(0,1,&destination,nullptr);c->Dispatch((width+7)/8,(height+7)/8,1);
        source=nullptr;destination=nullptr;c->CSSetShaderResources(0,1,&source);c->CSSetUnorderedAccessViews(0,1,&destination,nullptr);return packed_texture.Get();
    }

};
}
