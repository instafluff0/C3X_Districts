"""Unit owner colour (Look A): a civ colour must read as paint, not glow.

The owner ramp follows the surface's luminance. For a dark owner colour (red,
blue) on a light surface that ratio pushed the colour past its own brightness
and clipped it to a pure hue; the look's extra saturation then crushed its
other channels. The ramp is now never brighter than the owner colour and the
extra saturation skips owner colour. This GPU oracle runs the generated unit
material shader (`environment_refresh/unit_shader.h`) on one fully owned pixel.
"""
import pathlib
import unittest

from Renderer.native.native_cpp_test import run_cpp

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHADER = ROOT / "Renderer/native/environment_refresh/unit_shader.h"


def program(header):
    source = header.split('R"C3XUNIT(', 1)[1].split(')C3XUNIT"', 1)[0]
    return r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
void checked(HRESULT hr){if(FAILED(hr))throw std::runtime_error("D3D call failed");}
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 char const* source=R"UNITLOOK(''' + source + r''')UNITLOOK";
 ComPtr<ID3DBlob> vs,ps,error;
 if(FAILED(D3DCompile(source,strlen(source),"unit",nullptr,nullptr,"VS","vs_5_0",0,0,&vs,&error))||
    FAILED(D3DCompile(source,strlen(source),"unit",nullptr,nullptr,"PS","ps_5_0",0,0,&ps,&error))){
  if(error)std::printf("%s\n",static_cast<char const*>(error->GetBufferPointer()));throw std::runtime_error("compile");}
 ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> pixel;
 checked(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex));
 checked(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel));
 D3D11_INPUT_ELEMENT_DESC desc[]={
  {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
  {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
  {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
  {"TEXCOORD",1,DXGI_FORMAT_R32G32B32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0},
  {"TANGENT",0,DXGI_FORMAT_R32G32B32_FLOAT,0,44,D3D11_INPUT_PER_VERTEX_DATA,0},
  {"BINORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,56,D3D11_INPUT_PER_VERTEX_DATA,0}};
 ComPtr<ID3D11InputLayout> layout;checked(device->CreateInputLayout(desc,6,vs->GetBufferPointer(),vs->GetBufferSize(),&layout));
 // One screen-covering triangle facing the light, above the ground plane.
 struct Vertex{float p[3],n[3],uv[2],shadow[3],t[3],b[3];};
 Vertex vertices[3]={{{-1,-1,.5f},{0,0,1},{.5f,.5f},{.1f,0,0},{1,0,0},{0,1,0}},
  {{-1,3,.5f},{0,0,1},{.5f,.5f},{.1f,0,0},{1,0,0},{0,1,0}},{{3,-1,.5f},{0,0,1},{.5f,.5f},{.1f,0,0},{1,0,0},{0,1,0}}};
 auto buffer=[&](void const* data,unsigned bytes,unsigned bind){
  D3D11_BUFFER_DESC d={};d.ByteWidth=bytes;d.BindFlags=bind;d.Usage=D3D11_USAGE_DEFAULT;
  D3D11_SUBRESOURCE_DATA s={data,0,0};ComPtr<ID3D11Buffer> b;checked(device->CreateBuffer(&d,&s,&b));return b;};
 auto vb=buffer(vertices,sizeof(vertices),D3D11_BIND_VERTEX_BUFFER);
 // Material: tint (owner mask mode 1), owner (red, strength .9), sun, sun colour,
 // moon, moon colour (.w 0: generic material), ambient (.w 0: no cutout), channels.
 float material[32]={1,1,1,1, .5f,.03f,.02f,.9f, 0,0,1,1, 1,.96f,.88f,1, 0,0,1,0, 0,0,0,0, .7f,.75f,.8f,0, 0,0,0,0};
 // Beauty frame: light, sun colour/exposure, ambient, view, Quality (no
 // self-shadow map, look gain .5, saturation .25, owner ramp 2).
 float beauty[20]={0,0,1,2.05f, 1,.72f,.56f,1, .7f,.75f,.8f,.62f, .490290f,-.735435f,.469979f,0, 0,.5f,.25f,2};
 auto mb=buffer(material,sizeof(material),D3D11_BIND_CONSTANT_BUFFER);
 auto bb=buffer(beauty,sizeof(beauty),D3D11_BIND_CONSTANT_BUFFER);
 D3D11_TEXTURE2D_DESC t={};t.Width=t.Height=1;t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
 t.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;t.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 ComPtr<ID3D11Texture2D> texel;checked(device->CreateTexture2D(&t,nullptr,&texel));
 ComPtr<ID3D11ShaderResourceView> view;checked(device->CreateShaderResourceView(texel.Get(),nullptr,&view));
 t.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D> color,read;checked(device->CreateTexture2D(&t,nullptr,&color));
 ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(color.Get(),nullptr,&target));
 t.BindFlags=0;t.Usage=D3D11_USAGE_STAGING;t.CPUAccessFlags=D3D11_CPU_ACCESS_READ;checked(device->CreateTexture2D(&t,nullptr,&read));
 D3D11_SAMPLER_DESC s={};s.Filter=D3D11_FILTER_MIN_MAG_MIP_POINT;s.AddressU=s.AddressV=s.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;s.MaxLOD=D3D11_FLOAT32_MAX;
 ComPtr<ID3D11SamplerState> sampler;checked(device->CreateSamplerState(&s,&sampler));
 D3D11_RASTERIZER_DESC raster={};raster.FillMode=D3D11_FILL_SOLID;raster.CullMode=D3D11_CULL_NONE;raster.DepthClipEnable=TRUE;
 ComPtr<ID3D11RasterizerState> rs;checked(device->CreateRasterizerState(&raster,&rs));
 auto shade=[&](float grey,float saturation,float* out){
  float rgba[4]={grey,grey,grey,0}; // alpha 0: fully owner-marked texel
  context->UpdateSubresource(texel.Get(),0,nullptr,rgba,16,16);
  beauty[18]=saturation;context->UpdateSubresource(bb.Get(),0,nullptr,beauty,0,0);
  float clear[4]={};context->ClearRenderTargetView(target.Get(),clear);
  D3D11_VIEWPORT viewport={0,0,1,1,0,1};context->RSSetViewports(1,&viewport);context->RSSetState(rs.Get());
  UINT stride=sizeof(Vertex),offset=0;auto v=vb.Get();
  context->IASetInputLayout(layout.Get());context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  context->IASetVertexBuffers(0,1,&v,&stride,&offset);context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
  ID3D11Buffer* cbs[2]={mb.Get(),bb.Get()};context->PSSetConstantBuffers(0,2,cbs);
  auto srv=view.Get();context->PSSetShaderResources(0,1,&srv);
  ID3D11SamplerState* samplers[2]={sampler.Get(),sampler.Get()};context->PSSetSamplers(0,2,samplers);
  auto rtv=target.Get();context->OMSetRenderTargets(1,&rtv,nullptr);
  context->Draw(3,0);context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(read.Get(),color.Get());
  D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(read.Get(),0,D3D11_MAP_READ,0,&m));
  std::memcpy(out,m.pData,16);context->Unmap(read.Get(),0);};
 float light[4],mid[4],plain[4];
 shade(.45f,.25f,light);shade(.30f,.25f,mid);shade(.45f,0,plain);
 std::printf("light=%.4f,%.4f,%.4f mid=%.4f,%.4f,%.4f unsaturated=%.4f,%.4f,%.4f\n",
  light[0],light[1],light[2],mid[0],mid[1],mid[2],plain[0],plain[1],plain[2]);
 if(!(light[0]>.05f&&light[1]>0))throw std::runtime_error("owner colour missing");
 // Both surfaces are lighter than the owner colour, so both show it at its
 // own brightness: the ramp no longer brightens it with the surface.
 for(int c=0;c<3;++c)if(std::fabs(light[c]-mid[c])>1e-3f*(1+light[c]))
  throw std::runtime_error("owner colour brighter than itself on a light surface");
 // The look's extra saturation leaves owner colour alone.
 for(int c=0;c<3;++c)if(std::fabs(light[c]-plain[c])>1e-3f*(1+light[c]))
  throw std::runtime_error("extra saturation applied to owner colour");
 std::puts("PASS unit owner look: owner colour capped at its own brightness, no extra saturation");
}catch(std::exception const& e){std::printf("FAIL unit owner look: %s\n",e.what());return 1;}}
'''


class UnitOwnerLookTests(unittest.TestCase):
    def test_gpu_owner_colour_reads_as_paint(self):
        run_cpp(program(SHADER.read_text()), timeout=90)


if __name__ == "__main__":
    unittest.main()
