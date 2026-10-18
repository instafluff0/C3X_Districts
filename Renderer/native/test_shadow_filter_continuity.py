"""Cast-shadow edges move continuously with the sample, not a texel at a time.

The paged shadow filter sampled a fixed window around the texel holding the
sample: four comparison taps at that texel's corners (shallow receivers) or a
3x3 box of exact per-texel comparisons (steep receivers, such as hill flanks).
Both jumped whenever the sample crossed a texel, so at close zoom a shadow edge
on a hill showed the shadow map's grid as dark, stair-stepped blotches. The
windows now slide with the sample's position inside its texel.

The production sampling source (fresh_pipeline.h) runs on a GPU (WARP if no
hardware) over a slanted blocker edge at ~3.2 screen pixels per shadow texel.
"""
import unittest
from pathlib import Path

from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]

PROGRAM = r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
static void check(bool ok,char const* what){if(!ok){std::printf("FAIL %s\n",what);std::exit(1);}}
static void checked(HRESULT hr,char const* what){check(SUCCEEDED(hr),what);}
int main(){
 // The binary runs in a scratch folder under the checkout; find the source upward.
 std::string path="Renderer/sandbox/fresh_pipeline.h";std::ifstream in(path);
 for(int up=0;up<8 && !in;++up){path="../"+path;in.open(path);}
 std::stringstream text;text<<in.rdbuf();auto header=text.str();
 auto start=header.find("SamplerComparisonState SandboxShadowComparison");auto end=header.find("\n)\");",start);
 check(start!=std::string::npos && end!=std::string::npos,"production sampling source");
 std::string body="cbuffer Table:register(b0){float4 pickup_pages[64];};cbuffer Query:register(b1){float4 U,V,L;float4 domain;float4 plane;};"
  "Texture2DArray field:register(t0);\n"+header.substr(start,end-start)+R"(
float4 ScreenVS(uint n:SV_VertexID):SV_POSITION {float2 p=float2((n<<1)&2,n&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
float SamplePS(float4 p:SV_POSITION):SV_TARGET {float2 xy=domain.xy+p.xy/128*domain.zw;
 float z=plane.x+plane.y*(xy.x-domain.x);
 return c3x_paged_visibility(field,float4(xy,z,1),float3(0,0,1),false,U,V,L,float4(1,0,0,0));}
)";
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
 auto hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context);
 if(FAILED(hr))hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context);checked(hr,"device");
 // One resident 1024x1024 page: a blocker 100 units up wherever i+.37j < 712.4.
 std::vector<float> depth(1024*1024);
 for(int j=0;j<1024;++j)for(int i=0;i<1024;++i)depth[j*1024+i]=i+.37f*j<712.4f?100.f:-1e6f;
 D3D11_TEXTURE2D_DESC d{};d.Width=d.Height=1024;d.ArraySize=d.MipLevels=d.SampleDesc.Count=1;d.Format=DXGI_FORMAT_R32_FLOAT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 D3D11_SUBRESOURCE_DATA initial{depth.data(),1024*4,0};ComPtr<ID3D11Texture2D> page;checked(device->CreateTexture2D(&d,&initial,&page),"page");
 D3D11_SHADER_RESOURCE_VIEW_DESC view{};view.Format=d.Format;view.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2DARRAY;view.Texture2DArray.MipLevels=1;view.Texture2DArray.ArraySize=1;
 ComPtr<ID3D11ShaderResourceView> field;checked(device->CreateShaderResourceView(page.Get(),&view,&field),"field");
 D3D11_SAMPLER_DESC sd={};sd.Filter=D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;sd.AddressU=sd.AddressV=sd.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;
 sd.ComparisonFunc=D3D11_COMPARISON_GREATER_EQUAL;sd.MaxLOD=D3D11_FLOAT32_MAX;ComPtr<ID3D11SamplerState> comparison;checked(device->CreateSamplerState(&sd,&comparison),"sampler");
 d.Width=d.Height=128;d.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D> output;checked(device->CreateTexture2D(&d,nullptr,&output),"output");
 ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(output.Get(),nullptr,&target),"target");
 d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Texture2D> readback;checked(device->CreateTexture2D(&d,nullptr,&readback),"readback");
 auto buffer=[&](unsigned bytes,void const* data){D3D11_BUFFER_DESC b{};b.ByteWidth=bytes;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA s{data,0,0};
  ComPtr<ID3D11Buffer> out;checked(device->CreateBuffer(&b,&s,&out),"constants");return out;};
 // Page range (0,0,1x1), no wrap, one texel per world unit, page 0 in layer 0.
 float table[64*4]{};table[2]=1;table[3]=1;table[8]=1;table[9]=1;table[12]=0;
 auto table_buffer=buffer(sizeof(table),table);
 float worst=0;
 for(int filtered=0;filtered<2;++filtered)for(int steep=0;steep<2;++steep){
  std::string source=filtered?"#define SANDBOX_FILTERED_SHADOW 1\n"+body:body;
  ComPtr<ID3DBlob> vs,ps,errors;
  checked(D3DCompile(source.data(),source.size(),nullptr,nullptr,nullptr,"ScreenVS","vs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&vs,&errors),"ScreenVS");
  hr=D3DCompile(source.data(),source.size(),nullptr,nullptr,nullptr,"SamplePS","ps_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&ps,&errors);
  if(FAILED(hr)&&errors)std::printf("%s\n",static_cast<char const*>(errors->GetBufferPointer()));checked(hr,"SamplePS");
  ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> pixel;
  checked(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex),"vs");
  checked(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel),"ps");
  // 40x40 texels over 128 pixels (~3.2 pixels per texel, a 3x close-up). A
  // steep receiver rises .5 per texel, forcing the exact per-texel path.
  float query[5*4]={1,0,0,0, 0,1,0,0, 0,0,1,0, 500,500,40,40, steep?-60.f:-2.f, steep?.5f:0.f,0,0};
  auto query_buffer=buffer(sizeof(query),query);
  ID3D11Buffer* cb[]={table_buffer.Get(),query_buffer.Get()};context->PSSetConstantBuffers(0,2,cb);
  ID3D11ShaderResourceView* srv=field.Get();context->PSSetShaderResources(0,1,&srv);
  ID3D11SamplerState* s=comparison.Get();context->PSSetSamplers(15,1,&s);
  ID3D11RenderTargetView* rt=target.Get();context->OMSetRenderTargets(1,&rt,nullptr);
  D3D11_VIEWPORT vp{0,0,128,128,0,1};context->RSSetViewports(1,&vp);
  context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);context->IASetInputLayout(nullptr);
  context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);context->Draw(3,0);
  context->CopyResource(readback.Get(),output.Get());D3D11_MAPPED_SUBRESOURCE mapped{};checked(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped),"map");
  std::vector<float> v(128*128);for(int y=0;y<128;++y)std::memcpy(&v[y*128],static_cast<char const*>(mapped.pData)+y*mapped.RowPitch,128*4);context->Unmap(readback.Get(),0);
  float step=0;unsigned lit=0,shadowed=0;
  for(int y=0;y<128;++y)for(int x=0;x<128;++x){float a=v[y*128+x];lit+=a>.95f;shadowed+=a<.05f;
   if(x<127)step=std::max(step,std::abs(a-v[y*128+x+1]));if(y<127)step=std::max(step,std::abs(a-v[(y+1)*128+x]));}
  std::printf("filtered=%d steep=%d largest pixel step %.3f lit %u shadowed %u\n",filtered,steep,step,lit,shadowed);
  check(lit>1000 && shadowed>1000,"edge in view");worst=std::max(worst,step);
 }
 // A texel holds ~3.2 pixels; a sliding 3-texel window changes under .2 per
 // pixel. A fixed window jumps by a whole texel column (1/3, or 1/2 of a tent).
 check(worst<.2f,"continuous shadow edge");
 std::printf("SHADOW_FILTER_CONTINUITY_PASS largest_step=%.3f\n",worst);
}
'''


class ShadowFilterContinuityTests(unittest.TestCase):
    def test_shadow_edge_slides_with_the_sample(self):
        run_cpp(PROGRAM, timeout=120)


if __name__ == "__main__":
    unittest.main()
