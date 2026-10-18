"""City buildings cast their shadows through their own opacity masks.

Renderer64's shadow atlas (SandboxSceneShadow in sandbox/fresh_pipeline.h)
binds a masked city material's mask at t34 (bind_cutout). It compiled the
environment caster, which has no city branch: city materials (100 plus
channels) fell into its natural-opacity branch and were clipped by whatever
tree mask t33 last held, cutting lines through building shadows (the user's
city screenshot, October 17). The city-profile caster, which the production
SourceShadow already uses, clips them by their own mask.

This rasterizes a masked city quad with the caster the fresh pipeline selects
for the city profile, read from the runtime pack it compiles at run time, with
a tree mask that rejects everything at t33.
"""
from pathlib import Path
import re
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp

PACK = ROOT / "Renderer/packs/Renderer64ResidentRuntime/Renderer/native"


def fresh_city_caster():
    """The caster file SandboxSceneShadow::ensure compiles in the city profile."""
    source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
    line = re.search(r"auto caster=root\+(.*);", source).group(1)
    choice = re.search(r'renderer\.city_profile\?"([^"]+)":"([^"]+)"', line)
    return choice.group(1) if choice else re.fullmatch(r'"([^"]+)"', line).group(1)


class CityShadowCasterTests(unittest.TestCase):
    def test_fresh_pipeline_selects_the_city_caster(self):
        self.assertEqual(fresh_city_caster(), "city_fidelity/source_caster.hlsl")

    def test_masked_city_buildings_ignore_tree_masks(self):
        relative = fresh_city_caster()
        path = PACK / relative if (PACK / relative).is_file() else ROOT / "Renderer/native" / relative
        shader = path.read_text()
        self.assertNotIn(")SHADER\"", shader)
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
char const shader[]=R"SHADER(''' + shader + r''')SHADER";
void check(bool ok,char const* detail){if(!ok){std::printf("CITY_CASTER_FAIL %s\n",detail);std::exit(1);}}
struct Vertex{float uv[2],material,world[4],coverage,owner[4];};
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
 auto hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context);
 if(FAILED(hr))hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context);
 check(SUCCEEDED(hr),"device");
 ComPtr<ID3DBlob> vs,ps,errors;
 hr=D3DCompile(shader,sizeof(shader)-1,"caster",nullptr,nullptr,"VS","vs_5_0",0,0,&vs,&errors);check(SUCCEEDED(hr),"VS");
 hr=D3DCompile(shader,sizeof(shader)-1,"caster",nullptr,nullptr,"PSCutout","ps_5_0",0,0,&ps,&errors);check(SUCCEEDED(hr),"PSCutout");
 ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> pixel;ComPtr<ID3D11InputLayout> layout;
 check(SUCCEEDED(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex)),"vertex");
 check(SUCCEEDED(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel)),"pixel");
 D3D11_INPUT_ELEMENT_DESC elements[]={{"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
  {"TEXCOORD",1,DXGI_FORMAT_R32_FLOAT,0,8,D3D11_INPUT_PER_VERTEX_DATA,0},
  {"TEXCOORD",2,DXGI_FORMAT_R32G32B32A32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
  {"TEXCOORD",3,DXGI_FORMAT_R32_FLOAT,0,28,D3D11_INPUT_PER_VERTEX_DATA,0},
  {"TEXCOORD",4,DXGI_FORMAT_R32G32B32A32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0}};
 check(SUCCEEDED(device->CreateInputLayout(elements,5,vs->GetBufferPointer(),vs->GetBufferSize(),&layout)),"layout");
 // A city building quad over the page, material 100 (masked, clamped uv).
 Vertex quad[6]={};float corners[6][2]={{0,0},{1,0},{1,1},{0,0},{1,1},{0,1}};
 for(unsigned i=0;i<6;++i){quad[i].uv[0]=quad[i].world[0]=corners[i][0];quad[i].uv[1]=quad[i].world[1]=corners[i][1];
  quad[i].world[2]=.5f;quad[i].world[3]=1;quad[i].material=100;quad[i].coverage=1;}
 D3D11_BUFFER_DESC b{};b.ByteWidth=sizeof(quad);b.BindFlags=D3D11_BIND_VERTEX_BUFFER;D3D11_SUBRESOURCE_DATA data{quad,0,0};
 ComPtr<ID3D11Buffer> vertices,constants;check(SUCCEEDED(device->CreateBuffer(&b,&data,&vertices)),"vertices");
 float caster[20]={6,0,0,0, 0,6,0,0, 0,0,1,0};b={};b.ByteWidth=sizeof(caster);b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;data={caster,0,0};
 check(SUCCEEDED(device->CreateBuffer(&b,&data,&constants)),"constants");
 auto mask=[&](unsigned char const* rgba){D3D11_TEXTURE2D_DESC d{};d.Width=d.Height=2;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
  d.Format=DXGI_FORMAT_R8G8B8A8_UNORM;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;D3D11_SUBRESOURCE_DATA pixels{rgba,8,0};
  ComPtr<ID3D11Texture2D> texture;ComPtr<ID3D11ShaderResourceView> view;
  check(SUCCEEDED(device->CreateTexture2D(&d,&pixels,&texture)),"mask");
  check(SUCCEEDED(device->CreateShaderResourceView(texture.Get(),nullptr,&view)),"mask view");return view;};
 // A tree mask (red channel 0: every leaf texel cut) left at t33.
 unsigned char tree[16]={0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255};
 unsigned char solid[16]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
 unsigned char half[16]={255,255,255,0,255,255,255,255,255,255,255,0,255,255,255,255};
 D3D11_TEXTURE2D_DESC d{};d.Width=d.Height=64;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;d.Format=DXGI_FORMAT_R32_FLOAT;
 d.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D> field,read;ComPtr<ID3D11RenderTargetView> target;
 check(SUCCEEDED(device->CreateTexture2D(&d,nullptr,&field)) && SUCCEEDED(device->CreateRenderTargetView(field.Get(),nullptr,&target)),"field");
 d.BindFlags=0;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;check(SUCCEEDED(device->CreateTexture2D(&d,nullptr,&read)),"read");
 D3D11_BLEND_DESC blend{};auto& rt=blend.RenderTarget[0];rt.BlendEnable=TRUE;rt.SrcBlend=rt.DestBlend=rt.SrcBlendAlpha=rt.DestBlendAlpha=D3D11_BLEND_ONE;
 rt.BlendOp=rt.BlendOpAlpha=D3D11_BLEND_OP_MAX;rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;ComPtr<ID3D11BlendState> maximum;
 check(SUCCEEDED(device->CreateBlendState(&blend,&maximum)),"blend");
 D3D11_RASTERIZER_DESC r{};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;ComPtr<ID3D11RasterizerState> raster;
 check(SUCCEEDED(device->CreateRasterizerState(&r,&raster)),"raster");
 auto covered=[&](unsigned char const* city){
  auto trees=mask(tree),own=mask(city);
  float clear[4]={-1e6f,-1e6f,-1e6f,-1e6f};context->ClearRenderTargetView(target.Get(),clear);
  context->OMSetRenderTargets(1,target.GetAddressOf(),nullptr);context->OMSetBlendState(maximum.Get(),nullptr,~0u);
  context->RSSetState(raster.Get());D3D11_VIEWPORT viewport{0,0,64,64,0,1};context->RSSetViewports(1,&viewport);
  UINT stride=sizeof(Vertex),offset=0;context->IASetVertexBuffers(0,1,vertices.GetAddressOf(),&stride,&offset);
  context->IASetInputLayout(layout.Get());context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
  context->VSSetConstantBuffers(0,1,constants.GetAddressOf());context->PSSetConstantBuffers(0,1,constants.GetAddressOf());
  ID3D11ShaderResourceView* views[]={trees.Get(),own.Get()};context->PSSetShaderResources(33,2,views);
  context->Draw(6,0);context->CopyResource(read.Get(),field.Get());
  D3D11_MAPPED_SUBRESOURCE mapped{};check(SUCCEEDED(context->Map(read.Get(),0,D3D11_MAP_READ,0,&mapped)),"map");
  unsigned count=0,left=0;
  for(unsigned y=0;y<64;++y)for(unsigned x=0;x<64;++x){float value;
   std::memcpy(&value,static_cast<char const*>(mapped.pData)+y*mapped.RowPitch+x*4,4);
   if(value==.5f){++count;left+=x<32;}}
  context->Unmap(read.Get(),0);std::printf("covered=%u left=%u\n",count,left);return std::make_pair(count,left);};
 // An opaque building casts everywhere; a half-masked one only where its
 // own mask is opaque (texel column 1, the right half).
 auto opaque=covered(solid),masked=covered(half);
 check(opaque.first==64*64,"opaque city building cut by a tree mask");
 check(masked.first==64*32 && masked.second==0,"city building ignores its own mask");
 std::printf("PASS city caster masks\n");
}
''', timeout=120)


if __name__ == "__main__":
    unittest.main()
