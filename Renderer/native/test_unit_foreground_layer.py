"""Execute the production unit depth boundary on D3D11."""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_preparation_cancellation import block_at


class UnitForegroundLayerTests(unittest.TestCase):
    def test_world_cannot_cut_body_but_body_keeps_self_depth(self):
        source = Path("Renderer/sandbox/direct_units.h").read_text()
        start = source.index("if(!reflected && layer==1){")
        boundary = block_at(source, start)
        pipeline = Path("Renderer/sandbox/fresh_pipeline.h").read_text()
        end = pipeline.index('return fail("real_units")')
        self.assertLess(pipeline.rindex('return fail("territory_borders")', 0, end), end)
        # Compile the actual depth reset, not a second implementation of it.
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cassert>
#include <cstring>
#include <initializer_list>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
struct Work {unsigned clears=0;void clear(ID3D11DepthStencilView*){++clears;}} work_owner,*work=&work_owner;
struct Scene {ID3D11DepthStencilView* depth=nullptr;} scene;
void begin_bodies(ID3D11DeviceContext* context,bool reflected,int layer){
''' + boundary + r'''
}
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,
  D3D11_SDK_VERSION,&device,&level,&context)));
 char const* source=R"(
 cbuffer Settings:register(b0){float4 color;float4 placement;};
 float4 VS(uint id:SV_VertexID):SV_Position{
  float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),placement.x,1);
 }
 float4 PS():SV_Target{return color;}
 )";
 ComPtr<ID3DBlob> code;ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> ps;
 assert(SUCCEEDED(D3DCompile(source,strlen(source),"unit layering",nullptr,nullptr,"VS","vs_5_0",0,0,&code,nullptr)));
 assert(SUCCEEDED(device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vs)));code.Reset();
 assert(SUCCEEDED(D3DCompile(source,strlen(source),"unit layering",nullptr,nullptr,"PS","ps_5_0",0,0,&code,nullptr)));
 assert(SUCCEEDED(device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&ps)));
 D3D11_BUFFER_DESC bd={};bd.ByteWidth=32;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
 ComPtr<ID3D11Buffer> buffer;assert(SUCCEEDED(device->CreateBuffer(&bd,nullptr,&buffer)));
 auto constants=buffer.Get();context->VSSetConstantBuffers(0,1,&constants);context->PSSetConstantBuffers(0,1,&constants);
 context->VSSetShader(vs.Get(),nullptr,0);context->PSSetShader(ps.Get(),nullptr,0);
 context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
 for(unsigned samples:{1u,4u}){
  D3D11_TEXTURE2D_DESC desc={};desc.Width=desc.Height=16;desc.ArraySize=desc.MipLevels=1;
  desc.SampleDesc.Count=samples;desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_RENDER_TARGET;
  ComPtr<ID3D11Texture2D> color,depth,resolved,read;ComPtr<ID3D11RenderTargetView> target;ComPtr<ID3D11DepthStencilView> dsv;
  assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&color)));
  assert(SUCCEEDED(device->CreateRenderTargetView(color.Get(),nullptr,&target)));
  desc.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;desc.BindFlags=D3D11_BIND_DEPTH_STENCIL;
  assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&depth)));
  assert(SUCCEEDED(device->CreateDepthStencilView(depth.Get(),nullptr,&dsv)));scene.depth=dsv.Get();
  desc.SampleDesc.Count=1;desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=0;
  assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&resolved)));
  desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
  assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&read)));
  auto rtv=target.Get();D3D11_VIEWPORT viewport={0,0,16,16,0,1};context->RSSetViewports(1,&viewport);
  context->OMSetDepthStencilState(nullptr,0);
  auto draw=[&](float r,float g,float b,float z){
   context->OMSetRenderTargets(1,&rtv,dsv.Get());float values[8]={r,g,b,1,z};
   context->UpdateSubresource(buffer.Get(),0,nullptr,values,0,0);context->Draw(3,0);
  };
  auto pixel=[&](){
   context->OMSetRenderTargets(0,nullptr,nullptr);
   if(samples>1)context->ResolveSubresource(resolved.Get(),0,color.Get(),0,DXGI_FORMAT_B8G8R8A8_UNORM);
   else context->CopyResource(resolved.Get(),color.Get());
   context->CopyResource(read.Get(),resolved.Get());D3D11_MAPPED_SUBRESOURCE mapped={};
   assert(SUCCEEDED(context->Map(read.Get(),0,D3D11_MAP_READ,0,&mapped)));
   auto result=*reinterpret_cast<unsigned*>(static_cast<char*>(mapped.pData)+8*mapped.RowPitch+8*4);
   context->Unmap(read.Get(),0);return result;
  };
  context->ClearDepthStencilView(dsv.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
  draw(1,0,0,.1f); // a river/city/world overlay in front of the unit's mesh depth
  begin_bodies(context.Get(),false,0);draw(0,0,1,.8f);assert(pixel()==0xffff0000u); // shadow stays occluded
  begin_bodies(context.Get(),false,1);draw(0,1,0,.7f);assert(pixel()==0xff00ff00u);
  draw(0,0,1,.8f);assert(pixel()==0xff00ff00u); // back-facing body part cannot overwrite the nearer part
  context->ClearDepthStencilView(dsv.Get(),D3D11_CLEAR_DEPTH,1,0);draw(1,0,0,.1f);
  begin_bodies(context.Get(),true,1);draw(0,1,0,.7f);assert(pixel()==0xffff0000u); // reflection retains world occlusion
 }
 assert(work_owner.clears==2);
}
''', timeout=90)
