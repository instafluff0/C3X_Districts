"""Zero city-emission rejection and the exact incremental bind contract."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT=Path(__file__).resolve().parents[2]


class CitySubmission(unittest.TestCase):
    def test_emission_presence_and_incremental_bind(self):
        source=(ROOT/'Renderer/native/city_fidelity/gpu.h').read_text()
        methods='    bool emits('+source.split('    bool emits(',1)[1].split('    bool lights(',1)[0]
        run_cpp(r'''
#include <array>
#include <vector>
#include <cassert>
struct Context {int shader=0,blend=0,depth=0;unsigned calls=0;
 void PSSetShader(int v,void*,int){shader=v;++calls;}
 void OMSetBlendState(int v,void*,unsigned){blend=v;++calls;}
 void OMSetDepthStencilState(int v,int){depth=v;++calls;}
};
using ID3D11DeviceContext=Context;
struct Gpu {std::vector<std::array<void*,7>> materials;float night=0,emissive_scale=1;int ps[4]={1,2,3,4},emission=5,readonly_depth=6;
''' + methods + r'''
};
int main(){Gpu g;int texture;g.materials.resize(3);g.materials[1][1]=&texture;
 assert(!g.emits(100));assert(!g.emits(0));assert(!g.emits(1));
 for(float hour_weight:{.000001f,.2f,1.f}){g.night=hour_weight;assert(g.emits(1));assert(!g.emits(0));assert(!g.emits(2));}
 g.emissive_scale=0;assert(!g.emits(1));g.emissive_scale=2;assert(g.emits(1));
 for(bool reflected:{false,true}){Context c;g.bind_emission(&c,reflected);assert(c.calls==3);assert(c.shader==(reflected?4:2));assert(c.blend==5&&c.depth==6);}
}
''')

    def test_zero_emission_has_no_color_alpha_or_depth_effect_on_hardware(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cassert>
#include <vector>
#include <cstring>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
std::vector<unsigned char> read(ID3D11Device*d,ID3D11DeviceContext*c,ID3D11Texture2D*t){D3D11_TEXTURE2D_DESC td={};t->GetDesc(&td);td.BindFlags=td.MiscFlags=0;td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Texture2D>s;assert(SUCCEEDED(d->CreateTexture2D(&td,nullptr,&s)));c->CopyResource(s.Get(),t);D3D11_MAPPED_SUBRESOURCE m={};assert(SUCCEEDED(c->Map(s.Get(),0,D3D11_MAP_READ,0,&m)));unsigned stride=td.Format==DXGI_FORMAT_R16G16B16A16_FLOAT?8u:4u;std::vector<unsigned char>v(td.Width*td.Height*stride);for(unsigned y=0;y<td.Height;++y)std::memcpy(v.data()+y*td.Width*stride,static_cast<char*>(m.pData)+y*m.RowPitch,td.Width*stride);c->Unmap(s.Get(),0);return v;}
int main(){ComPtr<ID3D11Device>d;ComPtr<ID3D11DeviceContext>c;D3D_FEATURE_LEVEL fl;assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&d,&fl,&c)));
 const char*src=R"(Texture2D emission:register(t0);SamplerState wrap:register(s0);cbuffer C:register(b0){float night,scale;float2 unused;};struct P{float4 p:SV_POSITION;float2 uv:TEXCOORD0;};P VS(uint i:SV_VertexID){P o;o.p=float4(i==1?3:-1,i==2?-3:1,.4,1);o.uv=float2(i==1?2:0,i==2?2:0);return o;}float4 PS(P p):SV_Target{return float4(emission.Sample(wrap,p.uv).rgb*night*scale*1.45,1);})";
 ComPtr<ID3DBlob>v,p;assert(SUCCEEDED(D3DCompile(src,std::strlen(src),nullptr,nullptr,nullptr,"VS","vs_5_0",0,0,&v,nullptr)));assert(SUCCEEDED(D3DCompile(src,std::strlen(src),nullptr,nullptr,nullptr,"PS","ps_5_0",0,0,&p,nullptr)));ComPtr<ID3D11VertexShader>vs;ComPtr<ID3D11PixelShader>ps;assert(SUCCEEDED(d->CreateVertexShader(v->GetBufferPointer(),v->GetBufferSize(),nullptr,&vs)));assert(SUCCEEDED(d->CreatePixelShader(p->GetBufferPointer(),p->GetBufferSize(),nullptr,&ps)));
 D3D11_TEXTURE2D_DESC td={};td.Width=td.Height=16;td.ArraySize=td.MipLevels=td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;td.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D>t;ComPtr<ID3D11RenderTargetView>rt;assert(SUCCEEDED(d->CreateTexture2D(&td,nullptr,&t)));assert(SUCCEEDED(d->CreateRenderTargetView(t.Get(),nullptr,&rt)));auto target=rt.Get();
 ComPtr<ID3D11Texture2D>depth;ComPtr<ID3D11DepthStencilView>depth_view;td.Format=DXGI_FORMAT_R24G8_TYPELESS;td.BindFlags=D3D11_BIND_DEPTH_STENCIL;
 assert(SUCCEEDED(d->CreateTexture2D(&td,nullptr,&depth)));D3D11_DEPTH_STENCIL_VIEW_DESC dv={};dv.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;dv.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;assert(SUCCEEDED(d->CreateDepthStencilView(depth.Get(),&dv,&depth_view)));
 D3D11_DEPTH_STENCIL_DESC ds={};ds.DepthEnable=TRUE;ds.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;ds.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;ComPtr<ID3D11DepthStencilState>z;assert(SUCCEEDED(d->CreateDepthStencilState(&ds,&z)));c->OMSetDepthStencilState(z.Get(),0);
 c->OMSetRenderTargets(1,&target,depth_view.Get());
 D3D11_BLEND_DESC b={};auto&r=b.RenderTarget[0];r.BlendEnable=TRUE;r.SrcBlend=r.DestBlend=D3D11_BLEND_ONE;r.BlendOp=r.BlendOpAlpha=D3D11_BLEND_OP_ADD;r.SrcBlendAlpha=D3D11_BLEND_ZERO;r.DestBlendAlpha=D3D11_BLEND_ONE;r.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;ComPtr<ID3D11BlendState>blend;assert(SUCCEEDED(d->CreateBlendState(&b,&blend)));c->OMSetBlendState(blend.Get(),nullptr,~0u);
 D3D11_VIEWPORT vp={0,0,16,16,0,1};c->RSSetViewports(1,&vp);c->VSSetShader(vs.Get(),nullptr,0);c->PSSetShader(ps.Get(),nullptr,0);c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
 D3D11_BUFFER_DESC bd={};bd.ByteWidth=16;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;ComPtr<ID3D11Buffer>cb;assert(SUCCEEDED(d->CreateBuffer(&bd,nullptr,&cb)));auto buffer=cb.Get();c->PSSetConstantBuffers(0,1,&buffer);
 float clear[]={.31f,1.7f,2.8f,.43f};c->ClearRenderTargetView(target,clear);auto before=read(d.Get(),c.Get(),t.Get());c->ClearDepthStencilView(depth_view.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,.5f,9);auto before_depth=read(d.Get(),c.Get(),depth.Get());
 ComPtr<ID3D11Texture2D>emissive;ComPtr<ID3D11ShaderResourceView>srv;td.Width=td.Height=1;td.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;td.BindFlags=D3D11_BIND_SHADER_RESOURCE;float pixel[]={.7f,.2f,.3f,1};D3D11_SUBRESOURCE_DATA initial={pixel,16,0};assert(SUCCEEDED(d->CreateTexture2D(&td,&initial,&emissive)));assert(SUCCEEDED(d->CreateShaderResourceView(emissive.Get(),nullptr,&srv)));
 D3D11_SAMPLER_DESC sampler={};sampler.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;sampler.MaxLOD=D3D11_FLOAT32_MAX;ComPtr<ID3D11SamplerState>sample;assert(SUCCEEDED(d->CreateSamplerState(&sampler,&sample)));auto sampling=sample.Get();c->PSSetSamplers(0,1,&sampling);
 for(float activation:{0.f,.4f,1.f}){float values[]={activation,1.f,0,0};c->UpdateSubresource(buffer,0,nullptr,values,0,0);c->Draw(3,0);assert(read(d.Get(),c.Get(),t.Get())==before);assert(read(d.Get(),c.Get(),depth.Get())==before_depth);}
 auto texture=srv.Get();c->PSSetShaderResources(0,1,&texture);
 for(bool zero_scale:{false,true}){float values[]={zero_scale?1.f:0.f,zero_scale?0.f:1.f,0,0};c->UpdateSubresource(buffer,0,nullptr,values,0,0);c->Draw(3,0);assert(read(d.Get(),c.Get(),t.Get())==before);assert(read(d.Get(),c.Get(),depth.Get())==before_depth);}
 float active[]={1,1,0,0};c->UpdateSubresource(buffer,0,nullptr,active,0,0);c->Draw(3,0);assert(read(d.Get(),c.Get(),t.Get())!=before);assert(read(d.Get(),c.Get(),depth.Get())==before_depth);
 std::puts("CITY_ZERO_EMISSION_HARDWARE pass hdr_color_alpha_depth_exact=1 texture_absent=1 zero_activation=1 zero_scale=1");
}
''',timeout=60)


if __name__=='__main__':unittest.main()
