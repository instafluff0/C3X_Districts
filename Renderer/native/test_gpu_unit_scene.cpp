#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cassert>
#include <cstdio>
#include <vector>
#include <array>
#include <cmath>
#include "Renderer/native/gpu_image_compositor.h"
#include "Renderer/native/gpu_unit_finish.h"
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using namespace c3x_gpu_images;
std::vector<unsigned> scene_read(ID3D11Device* d,ID3D11DeviceContext* c,ID3D11Texture2D* texture){
    D3D11_TEXTURE2D_DESC desc={};texture->GetDesc(&desc);desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;checked(d->CreateTexture2D(&desc,nullptr,&staging));c->CopyResource(staging.Get(),texture);
    D3D11_MAPPED_SUBRESOURCE mapped={};checked(c->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped));
    std::vector<unsigned> result(desc.Width*desc.Height);for(unsigned y=0;y<desc.Height;++y)std::memcpy(result.data()+y*desc.Width,static_cast<char*>(mapped.pData)+y*mapped.RowPitch,desc.Width*4);
    c->Unmap(staging.Get(),0);return result;
}
int test_gpu_unit_scene(){
    ComPtr<ID3D11Device> d;ComPtr<ID3D11DeviceContext> c;D3D_FEATURE_LEVEL level;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&d,&level,&c));
    char const* shader=R"(
float4 VS(uint id:SV_VertexID):SV_Position{float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),.25,1);}
float4 PS(float4 p:SV_Position,uint sample:SV_SampleIndex):SV_Target{
 uint n=uint(p.x)+uint(p.y)*1024;clip(float((n+sample)%5)-.5);
 return float4(float((n*7)%4096)/1024,float((n*11)%4096)/1024,float((n*31)%4096)/1024,1);
})";
    ComPtr<ID3DBlob> code;ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> ps;
    checked(D3DCompile(shader,std::strlen(shader),"scene unit oracle",nullptr,nullptr,"VS","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,nullptr));
    checked(d->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vs));code.Reset();
    checked(D3DCompile(shader,std::strlen(shader),"scene unit oracle",nullptr,nullptr,"PS","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,nullptr));
    checked(d->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&ps));
    D3D11_TEXTURE2D_DESC desc={};desc.Width=desc.Height=128;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
    desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
    ComPtr<ID3D11Texture2D> output;ComPtr<ID3D11RenderTargetView> target;checked(d->CreateTexture2D(&desc,nullptr,&output));checked(d->CreateRenderTargetView(output.Get(),nullptr,&target));
    desc.Width=desc.Height=1;desc.Format=DXGI_FORMAT_R32_FLOAT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    float zero=0;D3D11_SUBRESOURCE_DATA data={&zero,4,0};ComPtr<ID3D11Texture2D> heights;ComPtr<ID3D11ShaderResourceView> height_view;
    checked(d->CreateTexture2D(&desc,&data,&heights));checked(d->CreateShaderResourceView(heights.Get(),nullptr,&height_view));
    unsigned total=0,mismatches=0;
    for(int scale:{1,2,4})for(auto format:{Format::rgb555,Format::rgb565}){
        c3x_renderer::render_core::LinearTarget body,scene;
        assert(body.ensure(d.Get(),128*scale,128*scale,true));assert(scene.ensure(d.Get(),128*scale,128*scale,true));
        for(auto at:{&body,&scene}){
            float background[4]={at==&body?0.f:1.f,0,0,at==&body?0.f:1.f};c->ClearRenderTargetView(at->target,background);c->ClearDepthStencilView(at->depth,D3D11_CLEAR_DEPTH,1,0);
            c->OMSetRenderTargets(1,&at->target,at->depth);c->OMSetDepthStencilState(nullptr,0);c->OMSetBlendState(nullptr,nullptr,0xffffffffu);
            D3D11_VIEWPORT vp={0,0,float(128*scale),float(128*scale),0,1};c->RSSetViewports(1,&vp);c->RSSetState(nullptr);
            c->VSSetShader(vs.Get(),nullptr,0);c->PSSetShader(ps.Get(),nullptr,0);c->IASetInputLayout(nullptr);c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);c->Draw(3,0);
        }
        c->OMSetRenderTargets(0,nullptr,nullptr);
        c3x_renderer::render_core::LinearOutput transfer;assert(transfer.ensure(d.Get()));transfer.draw(c.Get(),body,target.Get(),1,scale);c->OMSetRenderTargets(0,nullptr,nullptr);
        c3x_renderer::GpuUnitFinish finish;std::array<float,16> ground={64,64,1,0,0,0,1,1,1};
        auto completed=finish.finish(d.Get(),c.Get(),output.Get(),128,128,height_view.Get(),ground);
        Compositor gpu(d.Get(),c.Get());auto source=gpu.attach_source(completed.Get());
        auto control=gpu.create(128,128,format),cd=gpu.create(128,128,Format::bgra32),actual=gpu.create(128,128,format),ad=gpu.create(128,128,Format::bgra32);
        Rect full={0,0,128,128};for(auto id:{control,actual,cd,ad}){Command fill={Kind::fill,id,0,full,full,0,0,id==cd||id==ad?0xff456789u:0x2192u};assert(gpu.submit(&fill,1));}
        Command a={Kind::unit_over,control,source,full,full,0,0,0,control,cd,cd};assert(gpu.submit(&a,1));
        // The production native boundary preserves samples before invoking
        // hardware resolve. This must be exact, including partial coverage.
        c3x_renderer::render_core::LinearTarget contribution;assert(contribution.ensure(d.Get(),128*scale,128*scale));
        c3x_renderer::render_core::LinearRestore resolve;assert(resolve.ensure(d.Get()));
        assert(resolve.draw(c.Get(),contribution,scene.samples,scene.depth_samples,0,0,{},nullptr,0,0,false,false,0,true));
        transfer.draw(c.Get(),contribution,target.Get(),1,scale);c->OMSetRenderTargets(0,nullptr,nullptr);
        ComPtr<ID3D11ShaderResourceView> body_view;checked(d->CreateShaderResourceView(output.Get(),nullptr,&body_view));
        c3x_renderer::UnitSceneSample sample={body_view.Get(),height_view.Get(),ground,128,128};
        a.destination=actual;a.source=0;a.background=actual;a.detail=a.background_detail=ad;assert(gpu.submit(&a,1,&sample));
        assert(scene_read(d.Get(),c.Get(),gpu.texture(control))==scene_read(d.Get(),c.Get(),gpu.texture(actual)));
        auto expected=scene_read(d.Get(),c.Get(),gpu.texture(cd)),observed=scene_read(d.Get(),c.Get(),gpu.texture(ad));
        for(unsigned i=0;i<expected.size();++i){++total;if(expected[i]!=observed[i]){if(mismatches<8)std::printf("RESOLVE scale=%d pixel=%u expected=%x actual=%x\n",scale,i,expected[i],observed[i]);++mismatches;}}
        if(scale==2){
            // Independently verify circular raw-map color AND depth transport.
            // A far red backdrop must disappear after masking, while every
            // covered HDR sample and its partial alpha retain exact bytes.
            auto reference=scene_read(d.Get(),c.Get(),output.Get());
            c3x_renderer::render_core::LinearTarget shifted;
            assert(shifted.ensure(d.Get(),256,256,true,false));
            assert(resolve.draw(c.Get(),shifted,scene.samples,scene.depth_samples,13,-9,{},nullptr,256,256,true));
            assert(resolve.draw(c.Get(),contribution,shifted.samples,shifted.depth_samples,0,0,{},nullptr,0,0,false,false,0,true));
            transfer.draw(c.Get(),contribution,target.Get(),1,scale);c->OMSetRenderTargets(0,nullptr,nullptr);
            auto moved=scene_read(d.Get(),c.Get(),output.Get());
            for(int y=0;y<128;++y)for(int x=0;x<128;++x)assert(moved[y*128+x]==reference[((y+9)%128)*128+(x-13+128)%128]);
        }
    }
    // Tight map regions restore at the native unit raster origin at every
    // authored sample scale; outside samples cannot wrap into unrelated map.
    for(int scale:{1,2,4})for(int offset:{-7,13}){
        c3x_renderer::render_core::LinearTarget region,work;
        assert(region.ensure(d.Get(),64,80,true,false));assert(work.ensure(d.Get(),128*scale,128*scale,true));
        float green[4]={0,1,0,1};c->ClearRenderTargetView(region.target,green);c->ClearDepthStencilView(region.depth,D3D11_CLEAR_DEPTH,.25f,0);
        c3x_renderer::render_core::LinearRestore restore;assert(restore.ensure(d.Get()));
        assert(restore.draw(c.Get(),work,region.samples,region.depth_samples,offset,offset,{},nullptr,64,80,false,true,scale));
        c3x_renderer::render_core::LinearOutput transfer;assert(transfer.ensure(d.Get()));transfer.draw(c.Get(),work,target.Get(),1,scale);c->OMSetRenderTargets(0,nullptr,nullptr);
        auto pixels=scene_read(d.Get(),c.Get(),output.Get());
        for(int y=0;y<128;++y)for(int x=0;x<128;++x)
            assert(bool(pixels[y*128+x]>>24)==(x>=offset&&x<offset+32&&y>=offset&&y<offset+40));
    }
    std::printf("SCENE_RESOLVE pixels=%u mismatches=%u\n",total,mismatches);std::fflush(stdout);assert(!mismatches);return 0;
}
