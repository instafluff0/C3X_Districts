"""D3D stencil optimization must preserve partial coverage and coplanar depth."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class UnderlayOcclusionTests(unittest.TestCase):
    def test_partial_opaque_behind_equal_front_and_later_overlay(self):
        run_cpp(r'''
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cassert>
#include <cstring>
#include <vector>
#include "Renderer/tools/redraw_underlay_candidate.h"
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
 assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,
  D3D11_SDK_VERSION,&device,nullptr,&context)));
 char const* program=R"(
 cbuffer Values:register(b0){float4 color;float4 setup;};
 float4 VS(uint id:SV_VertexID):SV_Position {
  float2 p=id==0?float2(-1,-1):id==1?float2(-1,3):float2(3,-1);
  return float4(p,setup.x,1);
 }
 float4 PS(float4 p:SV_Position):SV_Target {
  if(setup.y>0)clip(setup.y-p.x);
  return float4(color.rgb*color.a,color.a);
 }
 void Mask(float4 p:SV_Position){
  if(setup.y>0)clip(setup.y-p.x);clip(color.a-1);
 }
 )";
 auto compile=[&](char const* entry,char const* profile){ComPtr<ID3DBlob> blob,error;
  assert(SUCCEEDED(D3DCompile(program,std::strlen(program),nullptr,nullptr,nullptr,entry,profile,
   D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&error)));return blob;};
 auto vb=compile("VS","vs_5_0"),pb=compile("PS","ps_5_0"),mb=compile("Mask","ps_5_0");
 ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> ps,mask;
 assert(SUCCEEDED(device->CreateVertexShader(vb->GetBufferPointer(),vb->GetBufferSize(),nullptr,&vs)));
 assert(SUCCEEDED(device->CreatePixelShader(pb->GetBufferPointer(),pb->GetBufferSize(),nullptr,&ps)));
 assert(SUCCEEDED(device->CreatePixelShader(mb->GetBufferPointer(),mb->GetBufferSize(),nullptr,&mask)));
 D3D11_TEXTURE2D_DESC td={};td.Width=33;td.Height=17;td.MipLevels=td.ArraySize=1;
 td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R8G8B8A8_UNORM;td.BindFlags=D3D11_BIND_RENDER_TARGET;
 ComPtr<ID3D11Texture2D> color,color_read,depth,depth_read;ComPtr<ID3D11RenderTargetView> target;
 assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&color)));
 assert(SUCCEEDED(device->CreateRenderTargetView(color.Get(),nullptr,&target)));
 td.BindFlags=0;td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&color_read)));
 td.Format=DXGI_FORMAT_R24G8_TYPELESS;td.BindFlags=D3D11_BIND_DEPTH_STENCIL;td.Usage=D3D11_USAGE_DEFAULT;td.CPUAccessFlags=0;
 assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&depth)));
 D3D11_DEPTH_STENCIL_VIEW_DESC dd={};dd.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;dd.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
 ComPtr<ID3D11DepthStencilView> depth_view;assert(SUCCEEDED(device->CreateDepthStencilView(depth.Get(),&dd,&depth_view)));
 td.BindFlags=0;td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&depth_read)));
 D3D11_BUFFER_DESC bd={};bd.ByteWidth=32;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
 ComPtr<ID3D11Buffer> values;assert(SUCCEEDED(device->CreateBuffer(&bd,nullptr,&values)));
 D3D11_DEPTH_STENCIL_DESC ds={};ds.DepthEnable=TRUE;ds.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;
 ds.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
 ComPtr<ID3D11DepthStencilState> original_depth;assert(SUCCEEDED(device->CreateDepthStencilState(&ds,&original_depth)));
 D3D11_BLEND_DESC blend={};auto& b=blend.RenderTarget[0];b.BlendEnable=TRUE;
 b.SrcBlend=b.SrcBlendAlpha=D3D11_BLEND_ONE;b.DestBlend=b.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;
 b.BlendOp=b.BlendOpAlpha=D3D11_BLEND_OP_ADD;b.RenderTargetWriteMask=15;
 ComPtr<ID3D11BlendState> original_blend;assert(SUCCEEDED(device->CreateBlendState(&blend,&original_blend)));
 D3D11_RASTERIZER_DESC rd={};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;
 rd.DepthClipEnable=TRUE;rd.ScissorEnable=TRUE;ComPtr<ID3D11RasterizerState> raster;
 assert(SUCCEEDED(device->CreateRasterizerState(&rd,&raster)));
 SandboxUnderlayOcclusion optimization;assert(optimization.ensure(device.Get()));
 auto draw=[&](float z,float alpha,float edge,float r,float g,float blue){float data[]={r,g,blue,alpha,z,edge,0,0};
  context->UpdateSubresource(values.Get(),0,nullptr,data,0,0);context->Draw(3,0);};
 auto render=[&](bool candidate,float z,float alpha,bool late){
  float clear[4]={};context->ClearRenderTargetView(target.Get(),clear);
  context->ClearDepthStencilView(depth_view.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
  auto* rt=target.Get();context->OMSetRenderTargets(1,&rt,depth_view.Get());
  context->VSSetShader(vs.Get(),nullptr,0);auto* cb=values.Get();context->VSSetConstantBuffers(0,1,&cb);context->PSSetConstantBuffers(0,1,&cb);
  context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);context->RSSetState(raster.Get());
  D3D11_VIEWPORT vp={0,0,33,17,0,1};context->RSSetViewports(1,&vp);D3D11_RECT rect={3,2,30,16};context->RSSetScissorRects(1,&rect);
  context->OMSetDepthStencilState(original_depth.Get(),0);
  if(candidate){
   context->OMSetBlendState(optimization.no_color.Get(),nullptr,~0u);context->PSSetShader(nullptr,nullptr,0);draw(.5f,1,0,.2f,.3f,.5f);
   context->OMSetDepthStencilState(optimization.mark.Get(),1);context->PSSetShader(mask.Get(),nullptr,0);draw(z,alpha,16,1,0,0);
   context->OMSetDepthStencilState(optimization.uncovered.Get(),0);
  }
  context->OMSetBlendState(original_blend.Get(),nullptr,~0u);context->PSSetShader(ps.Get(),nullptr,0);draw(.5f,1,0,.2f,.3f,.5f);
  context->OMSetDepthStencilState(original_depth.Get(),0);draw(z,alpha,16,1,0,0);
  if(late)draw(.5f,.25f,0,0,1,0);
  context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(color_read.Get(),color.Get());context->CopyResource(depth_read.Get(),depth.Get());
  std::vector<unsigned> result;D3D11_MAPPED_SUBRESOURCE mapped={};
  assert(SUCCEEDED(context->Map(color_read.Get(),0,D3D11_MAP_READ,0,&mapped)));
  for(unsigned y=0;y<17;++y)for(unsigned x=0;x<33;++x)result.push_back(reinterpret_cast<unsigned const*>(static_cast<unsigned char const*>(mapped.pData)+y*mapped.RowPitch)[x]);
  context->Unmap(color_read.Get(),0);assert(SUCCEEDED(context->Map(depth_read.Get(),0,D3D11_MAP_READ,0,&mapped)));
  unsigned marked=0;
  for(unsigned y=0;y<17;++y)for(unsigned x=0;x<33;++x){
   unsigned value=reinterpret_cast<unsigned const*>(static_cast<unsigned char const*>(mapped.pData)+y*mapped.RowPitch)[x];
   marked+=(value>>24)&1;result.push_back(value&0xffffffu);
  }
  // Thirteen opaque pixels per scissored row, fourteen rows. Partial alpha
  // and failed depth must never mark; this also rejects an inert-mask test.
  assert(marked==unsigned(candidate && z<=.5f && alpha==1.f?13*14:0));
  context->Unmap(depth_read.Get(),0);return result;
 };
 for(float z:{.45f,.5f,.55f})for(float alpha:{0.f,.5f,.99999994f,1.f})for(bool late:{false,true})
  assert(render(false,z,alpha,late)==render(true,z,alpha,late));
 // An empty picture must not accidentally satisfy the comparison.
 assert(render(false,.45f,1,false)!=render(false,.55f,1,false));
}
''', timeout=90)


if __name__ == '__main__':
    unittest.main()
