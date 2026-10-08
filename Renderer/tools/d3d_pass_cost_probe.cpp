// Standalone D3D11 cost probe for render-target switches, copies and the
// completion check, used to separate per-pass from per-pixel GPU cost on a
// given driver (the Parallels VM translates D3D11 to Metal, where every
// render-target change may become a render pass with whole-attachment
// load/store). Build and run with d3d_pass_cost_probe.bat on Windows.
//
// Each case submits one "frame" of work, then waits for completion through a
// 1x1 staging copy and a blocking Map. It prints the median and p90 wall time
// from the first submission to completion over the repetitions.
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

char const* shader_source=R"(
cbuffer Rect:register(b0){float4 rect;float4 tint;};
Texture2D source:register(t0);SamplerState point_sampler:register(s0);
struct V{float4 position:SV_Position;float2 uv:TEXCOORD0;};
V vs(uint id:SV_VertexID){
    float2 corner=float2(id&1,id>>1);V v;
    v.position=float4(lerp(rect.xy,rect.zw,corner),0,1);v.uv=corner;return v;}
float4 ps_fill(V v):SV_Target{return tint;}
float4 ps_sample(V v):SV_Target{return source.Sample(point_sampler,v.uv)*tint;}
)";

char const* compute_source=R"(
cbuffer Area:register(b0){int4 area;};
Texture2D<uint> input_words:register(t0);RWTexture2D<uint> output_words:register(u0);
[numthreads(8,8,1)] void main(uint3 thread:SV_DispatchThreadID){
    int2 at=area.xy+int2(thread.xy);if(any(at>=area.zw))return;
    output_words[at]=input_words.Load(int3(at&63,0))+1;}
[numthreads(8,8,1)] void copy_rect(uint3 thread:SV_DispatchThreadID){
    int2 at=int2(thread.xy);if(any(at>=area.zw-area.xy))return;
    output_words[at]=input_words.Load(int3(area.xy+at,0));}
)";

struct Probe {
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> fill,sample;
    ComPtr<ID3D11Buffer> constants;ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11Texture2D> probe,staging;
    ComPtr<ID3D11ComputeShader> words,copier;ComPtr<ID3D11Buffer> default_constants,dynamic_constants;
    LARGE_INTEGER frequency{};

    struct Target {ComPtr<ID3D11Texture2D> texture;ComPtr<ID3D11RenderTargetView> rtv;ComPtr<ID3D11ShaderResourceView> srv;unsigned width=0,height=0;};

    bool init(){
        QueryPerformanceFrequency(&frequency);
        D3D_FEATURE_LEVEL level{};
        if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,
            D3D11_SDK_VERSION,&device,&level,&context)))return false;
        ComPtr<ID3DBlob> code,errors;
        auto compile=[&](char const* entry,char const* profile)->bool{
            code.Reset();errors.Reset();
            if(FAILED(D3DCompile(shader_source,std::strlen(shader_source),"probe",nullptr,nullptr,entry,profile,0,0,&code,&errors))){
                if(errors)std::fprintf(stderr,"%s\n",static_cast<char const*>(errors->GetBufferPointer()));return false;}
            return true;};
        if(!compile("vs","vs_5_0")||FAILED(device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vs)))return false;
        if(!compile("ps_fill","ps_5_0")||FAILED(device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&fill)))return false;
        if(!compile("ps_sample","ps_5_0")||FAILED(device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&sample)))return false;
        {
            ComPtr<ID3DBlob> c,e;
            if(FAILED(D3DCompile(compute_source,std::strlen(compute_source),"words",nullptr,nullptr,"main","cs_5_0",0,0,&c,&e))||
               FAILED(device->CreateComputeShader(c->GetBufferPointer(),c->GetBufferSize(),nullptr,&words)))return false;
            c.Reset();e.Reset();
            if(FAILED(D3DCompile(compute_source,std::strlen(compute_source),"copy",nullptr,nullptr,"copy_rect","cs_5_0",0,0,&c,&e))||
               FAILED(device->CreateComputeShader(c->GetBufferPointer(),c->GetBufferSize(),nullptr,&copier)))return false;
            D3D11_BUFFER_DESC d{};d.ByteWidth=16;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(device->CreateBuffer(&d,nullptr,&default_constants)))return false;
            d.Usage=D3D11_USAGE_DYNAMIC;d.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            if(FAILED(device->CreateBuffer(&d,nullptr,&dynamic_constants)))return false;
        }
        D3D11_BUFFER_DESC buffer{};buffer.ByteWidth=32;buffer.Usage=D3D11_USAGE_DYNAMIC;
        buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
        if(FAILED(device->CreateBuffer(&buffer,nullptr,&constants)))return false;
        D3D11_SAMPLER_DESC s{};s.Filter=D3D11_FILTER_MIN_MAG_MIP_POINT;
        s.AddressU=s.AddressV=s.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;s.MaxLOD=D3D11_FLOAT32_MAX;
        if(FAILED(device->CreateSamplerState(&s,&sampler)))return false;
        D3D11_TEXTURE2D_DESC t{};t.Width=t.Height=t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
        t.Format=DXGI_FORMAT_B8G8R8A8_UNORM;t.Usage=D3D11_USAGE_DEFAULT;
        if(FAILED(device->CreateTexture2D(&t,nullptr,&probe)))return false;
        t.Usage=D3D11_USAGE_STAGING;t.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        if(FAILED(device->CreateTexture2D(&t,nullptr,&staging)))return false;
        context->VSSetShader(vs.Get(),nullptr,0);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        context->VSSetConstantBuffers(0,1,constants.GetAddressOf());context->PSSetConstantBuffers(0,1,constants.GetAddressOf());
        context->PSSetSamplers(0,1,sampler.GetAddressOf());
        return true;
    }
    Target target(unsigned width,unsigned height){
        Target r;r.width=width;r.height=height;
        D3D11_TEXTURE2D_DESC t{};t.Width=width;t.Height=height;t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
        t.Format=DXGI_FORMAT_B8G8R8A8_UNORM;t.Usage=D3D11_USAGE_DEFAULT;
        t.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        device->CreateTexture2D(&t,nullptr,&r.texture);
        device->CreateRenderTargetView(r.texture.Get(),nullptr,&r.rtv);
        device->CreateShaderResourceView(r.texture.Get(),nullptr,&r.srv);
        float clear[4]={.2f,.3f,.4f,1};context->ClearRenderTargetView(r.rtv.Get(),clear);
        return r;
    }
    struct Words {ComPtr<ID3D11Texture2D> texture;ComPtr<ID3D11UnorderedAccessView> uav;ComPtr<ID3D11ShaderResourceView> srv;};
    Words words_target(unsigned width,unsigned height){
        Words r;D3D11_TEXTURE2D_DESC t{};t.Width=width;t.Height=height;t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
        t.Format=DXGI_FORMAT_R32_UINT;t.Usage=D3D11_USAGE_DEFAULT;t.BindFlags=D3D11_BIND_UNORDERED_ACCESS|D3D11_BIND_SHADER_RESOURCE;
        device->CreateTexture2D(&t,nullptr,&r.texture);device->CreateUnorderedAccessView(r.texture.Get(),nullptr,&r.uav);
        device->CreateShaderResourceView(r.texture.Get(),nullptr,&r.srv);return r;
    }
    // One interpreter-style command: constants, bind, dispatch over a 64x64
    // area. update=true writes constants with UpdateSubresource on a DEFAULT
    // buffer (as the native interpreter does), else Map(WRITE_DISCARD).
    void dispatch(Words const& out,Words const& in,int x,int y,bool update,bool unbind_after){
        int area[4]={x,y,x+64,y+64};
        ID3D11Buffer* cb=update?default_constants.Get():dynamic_constants.Get();
        if(update)context->UpdateSubresource(cb,0,nullptr,area,0,0);
        else{D3D11_MAPPED_SUBRESOURCE m{};context->Map(cb,0,D3D11_MAP_WRITE_DISCARD,0,&m);std::memcpy(m.pData,area,sizeof(area));context->Unmap(cb,0);}
        context->CSSetConstantBuffers(0,1,&cb);context->CSSetShader(words.Get(),nullptr,0);
        context->CSSetShaderResources(0,1,in.srv.GetAddressOf());context->CSSetUnorderedAccessViews(0,1,out.uav.GetAddressOf(),nullptr);
        context->Dispatch(8,8,1);
        if(unbind_after){ID3D11ShaderResourceView* s=nullptr;ID3D11UnorderedAccessView* u=nullptr;
            context->CSSetShaderResources(0,1,&s);context->CSSetUnorderedAccessViews(0,1,&u,nullptr);}
    }
    // Copies a 64x64 rectangle of `in` at (x,y) into `out` at the origin
    // with a compute shader instead of CopySubresourceRegion.
    void compute_copy(Words const& out,Words const& in,int x,int y){
        int area[4]={x,y,x+64,y+64};ID3D11Buffer* cb=dynamic_constants.Get();
        D3D11_MAPPED_SUBRESOURCE m{};context->Map(cb,0,D3D11_MAP_WRITE_DISCARD,0,&m);std::memcpy(m.pData,area,sizeof(area));context->Unmap(cb,0);
        context->CSSetConstantBuffers(0,1,&cb);context->CSSetShader(copier.Get(),nullptr,0);
        context->CSSetShaderResources(0,1,in.srv.GetAddressOf());context->CSSetUnorderedAccessViews(0,1,out.uav.GetAddressOf(),nullptr);
        context->Dispatch(8,8,1);
        ID3D11ShaderResourceView* sr=nullptr;ID3D11UnorderedAccessView* u=nullptr;
        context->CSSetShaderResources(0,1,&sr);context->CSSetUnorderedAccessViews(0,1,&u,nullptr);
    }
    void compute_copy_area(Words const& out,Words const& in,unsigned width,unsigned height){
        int area[4]={0,0,int(width),int(height)};ID3D11Buffer* cb=dynamic_constants.Get();
        D3D11_MAPPED_SUBRESOURCE m{};context->Map(cb,0,D3D11_MAP_WRITE_DISCARD,0,&m);std::memcpy(m.pData,area,sizeof(area));context->Unmap(cb,0);
        context->CSSetConstantBuffers(0,1,&cb);context->CSSetShader(copier.Get(),nullptr,0);
        context->CSSetShaderResources(0,1,in.srv.GetAddressOf());context->CSSetUnorderedAccessViews(0,1,out.uav.GetAddressOf(),nullptr);
        context->Dispatch((width+7)/8,(height+7)/8,1);
        ID3D11ShaderResourceView* sr=nullptr;ID3D11UnorderedAccessView* u=nullptr;
        context->CSSetShaderResources(0,1,&sr);context->CSSetUnorderedAccessViews(0,1,&u,nullptr);
    }
    void bind(Target const& t){
        ID3D11ShaderResourceView* none=nullptr;context->PSSetShaderResources(0,1,&none);
        context->OMSetRenderTargets(1,t.rtv.GetAddressOf(),nullptr);
        D3D11_VIEWPORT v{0,0,float(t.width),float(t.height),0,1};context->RSSetViewports(1,&v);
    }
    // Draws a w x h pixel quad at (x,y) of the bound target.
    void quad(Target const& t,int x,int y,int w,int h,ID3D11ShaderResourceView* source){
        D3D11_MAPPED_SUBRESOURCE m{};context->Map(constants.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&m);
        float data[8]={2.f*x/t.width-1,1-2.f*y/t.height,2.f*(x+w)/t.width-1,1-2.f*(y+h)/t.height,1,1,1,1};
        std::memcpy(m.pData,data,sizeof(data));context->Unmap(constants.Get(),0);
        context->PSSetShader(source?sample.Get():fill.Get(),nullptr,0);
        if(source)context->PSSetShaderResources(0,1,&source);
        context->Draw(4,0);
    }
    void finish(){
        context->CopyResource(staging.Get(),probe.Get());
        D3D11_MAPPED_SUBRESOURCE m{};
        if(SUCCEEDED(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&m)))context->Unmap(staging.Get(),0);
    }
    double now_ms(){LARGE_INTEGER t;QueryPerformanceCounter(&t);return 1000.*double(t.QuadPart)/double(frequency.QuadPart);}
    void run(char const* name,std::function<void()> const& frame,int repetitions=40){
        finish();frame();finish();
        std::vector<double> times;
        for(int i=0;i<repetitions;++i){double a=now_ms();frame();finish();times.push_back(now_ms()-a);}
        std::sort(times.begin(),times.end());
        std::printf("%-56s median %7.3f ms  p90 %7.3f ms\n",name,times[times.size()/2],times[times.size()*9/10]);
        std::fflush(stdout);
    }
};

}

int main(){
    Probe p;
    if(!p.init()){std::fprintf(stderr,"device init failed\n");return 1;}
    unsigned const W=2240,H=1260;
    auto a=p.target(W,H),b=p.target(W,H),small_a=p.target(256,256),small_b=p.target(256,256),tile=p.target(64,64);
    int const N=100;
    char name[128];
    p.run("empty frame (completion check only)",[]{});
    {
        // Does mapping an older fence (DO_NOT_WAIT) wait for work queued after it?
        ComPtr<ID3D11Texture2D> fence;D3D11_TEXTURE2D_DESC t{};t.Width=t.Height=t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
        t.Format=DXGI_FORMAT_B8G8R8A8_UNORM;t.Usage=D3D11_USAGE_STAGING;t.CPUAccessFlags=D3D11_CPU_ACCESS_READ;p.device->CreateTexture2D(&t,nullptr,&fence);
        auto heavy=[&]{for(int i=0;i<100;++i){auto& tt=i&1?b:a;p.bind(tt);p.quad(tt,(i*37)%(W-64),(i*53)%(H-64),64,64,nullptr);}};
        std::vector<double> early,late,whole;
        for(int r=0;r<20;++r){
            p.finish();
            p.context->CopyResource(fence.Get(),p.probe.Get());p.context->Flush();
            double t0=p.now_ms();while(p.now_ms()-t0<5){}   // let the fence copy complete
            heavy();p.context->Flush();
            double a0=p.now_ms();D3D11_MAPPED_SUBRESOURCE m{};
            HRESULT hr=p.context->Map(fence.Get(),0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&m);
            double a1=p.now_ms();if(SUCCEEDED(hr))p.context->Unmap(fence.Get(),0);
            early.push_back(a1-a0);
            double b0=p.now_ms();p.finish();whole.push_back(p.now_ms()-b0);
            late.push_back(hr==DXGI_ERROR_WAS_STILL_DRAWING?1.:0.);
        }
        std::sort(early.begin(),early.end());std::sort(whole.begin(),whole.end());
        std::printf("old fence Map(DO_NOT_WAIT) with 100 RT switches queued after: median %.3f ms (remaining drain after %.3f ms, still-drawing %d/20)\n",
            early[10],whole[10],int(std::count(late.begin(),late.end(),1.)));
        std::fflush(stdout);
    }
    for(int n:{25,N}){
        std::snprintf(name,sizeof(name),"%d fill quads 64x64, one full-screen target",n);
        p.run(name,[&]{p.bind(a);for(int i=0;i<n;++i)p.quad(a,(i*37)%(W-64),(i*53)%(H-64),64,64,nullptr);});
        std::snprintf(name,sizeof(name),"%d fill quads 64x64, alternating two full-screen targets",n);
        p.run(name,[&]{for(int i=0;i<n;++i){auto& t=i&1?b:a;p.bind(t);p.quad(t,(i*37)%(W-64),(i*53)%(H-64),64,64,nullptr);}});
        std::snprintf(name,sizeof(name),"%d fill quads 64x64, alternating two 256x256 targets",n);
        p.run(name,[&]{for(int i=0;i<n;++i){auto& t=i&1?small_b:small_a;p.bind(t);p.quad(t,(i*37)%192,(i*53)%192,64,64,nullptr);}});
        std::snprintf(name,sizeof(name),"%d sampled quads 64x64, ping-pong a<->b full-screen",n);
        p.run(name,[&]{for(int i=0;i<n;++i){auto& t=i&1?b:a;auto& s=i&1?a:b;p.bind(t);p.quad(t,(i*37)%(W-64),(i*53)%(H-64),64,64,s.srv.Get());}});
        std::snprintf(name,sizeof(name),"%d CopySubresourceRegion 64x64 into full-screen",n);
        p.run(name,[&]{for(int i=0;i<n;++i){D3D11_BOX box{0,0,0,64,64,1};
            p.context->CopySubresourceRegion(a.texture.Get(),0,(i*37)%(W-64),(i*53)%(H-64),0,tile.texture.Get(),0,&box);}});
        std::snprintf(name,sizeof(name),"%d copy 64x64 then draw, full-screen (copy/draw interleave)",n);
        p.run(name,[&]{for(int i=0;i<n;++i){D3D11_BOX box{0,0,0,64,64,1};
            p.context->CopySubresourceRegion(b.texture.Get(),0,(i*37)%(W-64),(i*53)%(H-64),0,tile.texture.Get(),0,&box);
            p.bind(a);p.quad(a,(i*41)%(W-64),(i*29)%(H-64),64,64,nullptr);}});
    }
    {
        auto wa=p.words_target(W,H),wb=p.words_target(W,H),wt=p.words_target(64,64);
        for(int n:{25,N}){
            auto at=[&](int i,int& x,int& y){x=(i*37)%(int(W)-64);y=(i*53)%(int(H)-64);};
            std::snprintf(name,sizeof(name),"%d dispatches, UpdateSubresource constants, unbind",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);p.dispatch(wa,wt,x,y,true,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches, Map-discard constants, unbind",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);p.dispatch(wa,wt,x,y,false,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches, Map-discard constants, no unbind",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);p.dispatch(wa,wt,x,y,false,false);}});
            std::snprintf(name,sizeof(name),"%d dispatches, Map-discard, alternating UAV targets",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);p.dispatch(i&1?wb:wa,wt,x,y,false,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches, Map-discard, ping-pong read/write",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);p.dispatch(i&1?wb:wa,i&1?wa:wb,x,y,false,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches, Map-discard, 64x64 snapshot copy before each",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);D3D11_BOX box{unsigned(x),unsigned(y),0,unsigned(x+64),unsigned(y+64),1};
                p.context->CopySubresourceRegion(wt.texture.Get(),0,0,0,0,wa.texture.Get(),0,&box);p.dispatch(wa,wt,x,y,false,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches, 64x64 compute copy before each",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);p.compute_copy(wt,wa,x,y);p.dispatch(wa,wt,x,y,false,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches, UpdateSubresource, 64x64 copy + clear before each",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);D3D11_BOX box{unsigned(x),unsigned(y),0,unsigned(x+64),unsigned(y+64),1};
                unsigned zero[4]={};p.context->ClearUnorderedAccessViewUint(wt.uav.Get(),zero);
                p.context->CopySubresourceRegion(wt.texture.Get(),0,0,0,0,wa.texture.Get(),0,&box);p.dispatch(wa,wt,x,y,true,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches, UpdateSubresource, 3 copies before each",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);D3D11_BOX box{unsigned(x),unsigned(y),0,unsigned(x+64),unsigned(y+64),1};
                for(int k=0;k<3;++k)p.context->CopySubresourceRegion(wt.texture.Get(),0,0,0,0,wa.texture.Get(),0,&box);p.dispatch(wa,wt,x,y,true,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches, Map-discard, 3 compute copies before each",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);for(int k=0;k<3;++k)p.compute_copy(wt,wa,x,y);p.dispatch(wa,wt,x,y,false,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches, new SRV+UAV views created for each",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);Probe::Words fresh;fresh.texture=wa.texture;
                p.device->CreateUnorderedAccessView(wa.texture.Get(),nullptr,&fresh.uav);p.device->CreateShaderResourceView(wa.texture.Get(),nullptr,&fresh.srv);
                p.dispatch(fresh,wt,x,y,false,true);}});
            std::snprintf(name,sizeof(name),"%d dispatches then one draw each (compute/render interleave)",n);
            p.run(name,[&]{for(int i=0;i<n;++i){int x,y;at(i,x,y);p.dispatch(wa,wt,x,y,false,true);p.bind(a);p.quad(a,x,y,64,64,nullptr);}});
        }
    }
    {
        auto wa=p.words_target(W,H),wb=p.words_target(W,H);
        for(int n:{1,4,16}){
            std::snprintf(name,sizeof(name),"%d full-screen word CopySubresourceRegion",n);
            p.run(name,[&]{for(int i=0;i<n;++i){D3D11_BOX box{0,0,0,W,H,1};p.context->CopySubresourceRegion((i&1?wa:wb).texture.Get(),0,0,0,0,(i&1?wb:wa).texture.Get(),0,&box);}});
            std::snprintf(name,sizeof(name),"%d full-screen word compute copies",n);
            p.run(name,[&]{for(int i=0;i<n;++i)p.compute_copy_area(i&1?wa:wb,i&1?wb:wa,W,H);});
        }
    }
    for(int n:{1,2,4,8}){
        std::snprintf(name,sizeof(name),"%d full-screen CopyResource",n);
        p.run(name,[&]{for(int i=0;i<n;++i)p.context->CopyResource((i&1?a:b).texture.Get(),(i&1?b:a).texture.Get());});
        std::snprintf(name,sizeof(name),"%d full-screen sampled draws, alternating targets",n);
        p.run(name,[&]{for(int i=0;i<n;++i){auto& t=i&1?b:a;auto& s=i&1?a:b;p.bind(t);p.quad(t,0,0,W,H,s.srv.Get());}});
    }
    return 0;
}
