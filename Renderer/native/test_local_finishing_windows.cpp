// A localized change in MSAA scene color must be exactly reproducible by
// copying only its expanded damage from the production finishing shader.
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
static decltype(&D3DCompile) test_compile=nullptr;
static decltype(&D3DCompileFromFile) test_compile_file=nullptr;
namespace c3x_renderer { namespace render_core {
HRESULT compile_cached(wchar_t const* path,char const* entry,char const* target,ID3DBlob** blob,ID3DBlob** errors){
 return test_compile_file(path,nullptr,nullptr,entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,blob,errors);
}
} }
#define D3DCompile test_compile
#include "render_core/linear_target.h"
#undef D3DCompile
#include "city_fidelity/glow.h"
#include "render_core/scene_surface.h"
int main(){
 std::setvbuf(stdout,nullptr,_IONBF,0);assert(SetCurrentDirectoryW(L"../../../.."));
 auto create=reinterpret_cast<decltype(&D3D11CreateDevice)>(GetProcAddress(LoadLibraryA("d3d11.dll"),"D3D11CreateDevice"));
 auto compiler=LoadLibraryA("d3dcompiler_47.dll");
 test_compile=reinterpret_cast<decltype(test_compile)>(GetProcAddress(compiler,"D3DCompile"));
 test_compile_file=reinterpret_cast<decltype(test_compile_file)>(GetProcAddress(compiler,"D3DCompileFromFile"));
 ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;D3D_FEATURE_LEVEL level=D3D_FEATURE_LEVEL_11_0;
 assert(SUCCEEDED(create(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,&level,1,D3D11_SDK_VERSION,&device,nullptr,&context)));
 c3x_renderer::city_fidelity::Glow glow;assert(glow.ensure(device,"."));
 char const* code=R"(
 cbuffer Patch:register(b0){int4 changed;};
 float4 VS(uint id:SV_VertexID):SV_Position{float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
 float4 PS(float4 p:SV_Position,uint sample:SV_SampleIndex):SV_Target{
  int2 xy=int2(p.xy);float alpha=(sample+1)*.25;
  bool hot=all(xy>=changed.xy)&&all(xy<changed.zw);
  float3 rgb=hot?float3(5.0,2.0,1.2):float3(.12,.2,.08);
  rgb+=float3((xy.x*11+xy.y*17+sample*47)%1021,(xy.x*13+xy.y*19+sample*23)%1019,(xy.x*29+xy.y*7+sample*31)%1013)/127.0;
  return float4(rgb*alpha,alpha);
 })";
 auto compile=[&](char const* entry,char const* target){
  ID3DBlob *blob=nullptr,*errors=nullptr;auto hr=test_compile(code,std::strlen(code),"finishing-witness",nullptr,nullptr,entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&errors);
  if(errors){std::puts(static_cast<char*>(errors->GetBufferPointer()));errors->Release();}assert(SUCCEEDED(hr));return blob;
 };
 ID3D11VertexShader* vs=nullptr;ID3D11PixelShader* ps=nullptr;auto blob=compile("VS","vs_5_0");
 assert(SUCCEEDED(device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vs)));blob->Release();blob=compile("PS","ps_5_0");
 assert(SUCCEEDED(device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&ps)));blob->Release();
 D3D11_BUFFER_DESC bd={};bd.ByteWidth=16;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;ID3D11Buffer* settings=nullptr;
 assert(SUCCEEDED(device->CreateBuffer(&bd,nullptr,&settings)));
 D3D11_RASTERIZER_DESC rd={};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;rd.MultisampleEnable=true;rd.DepthClipEnable=true;
 ID3D11RasterizerState* rasterizer=nullptr;assert(SUCCEEDED(device->CreateRasterizerState(&rd,&rasterizer)));
 auto render=[&](D3D11_RECT patch){
  int box[]={int(patch.left*2),int(patch.top*2),int(patch.right*2),int(patch.bottom*2)};
  context->UpdateSubresource(settings,0,nullptr,box,0,0);context->PSSetConstantBuffers(0,1,&settings);
  context->OMSetRenderTargets(1,&glow.linear.target,nullptr);context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
  context->RSSetState(rasterizer);D3D11_VIEWPORT viewport={0,0,272,272,0,1};context->RSSetViewports(1,&viewport);
  context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  context->VSSetShader(vs,nullptr,0);context->PSSetShader(ps,nullptr,0);context->Draw(3,0);context->OMSetRenderTargets(0,nullptr,nullptr);
 };
 D3D11_TEXTURE2D_DESC td={};glow.color->GetDesc(&td);td.BindFlags=0;td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ID3D11Texture2D* staging=nullptr;assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&staging)));
 auto pixels=[&](){
  context->CopyResource(staging,glow.color);D3D11_MAPPED_SUBRESOURCE mapped={};assert(SUCCEEDED(context->Map(staging,0,D3D11_MAP_READ,0,&mapped)));
  std::vector<unsigned char> result(136*136*8);
  for(unsigned y=0;y<136;++y)std::memcpy(result.data()+y*136*8,static_cast<unsigned char*>(mapped.pData)+y*mapped.RowPitch,136*8);
  context->Unmap(staging,0);return result;
 };
 render({0,0,0,0});glow.reconstruct(context);auto base=pixels();
 unsigned checks=0,changed=0;
 for(auto patch:std::array<D3D11_RECT,9>{{{0,0,3,3},{3,3,5,5},{7,7,9,9},{63,63,64,64},{64,64,65,65},
      {100,30,104,32},{130,130,134,134},{133,50,136,54},{0,65,136,66}}}){
  render(patch);glow.reconstruct(context);auto full=pixels();
  D3D11_RECT damage={std::max<LONG>(4,patch.left-4),std::max<LONG>(4,patch.top-4),std::min<LONG>(132,patch.right+4),std::min<LONG>(132,patch.bottom+4)};
  assert(damage.left<damage.right && damage.top<damage.bottom);
  float sentinel[4]={0,0,0,0};context->ClearUnorderedAccessViewFloat(glow.output,sentinel);
  glow.reconstruct(context,&damage);auto partial=pixels();
  for(int y=4;y<132;++y)for(int x=4;x<132;++x){
   auto offset=(y*136+x)*8;bool affected=x>=damage.left && x<damage.right && y>=damage.top && y<damage.bottom;
   assert(std::memcmp(full.data()+offset,(affected?partial:base).data()+offset,8)==0);
   changed+=std::memcmp(full.data()+offset,base.data()+offset,8)!=0;
  }
  ++checks;
 }
 assert(changed>100);std::printf("PASS %u exact local HDR reconstruction checks: cell guards, workgroup edges, thin spans, MSAA coverage and bloom; changed_pixels=%u\n",checks,changed);
 // Retain resolved/finished pixels across local changes and both physical seams.
 assert(glow.ensure(device,".",136,136,true));
 unsigned circular_checks=0;
 for(auto patch:std::array<D3D11_RECT,9>{{{0,0,3,3},{3,3,5,5},{7,7,9,9},{63,63,64,64},{64,64,65,65},
      {100,30,104,32},{130,130,134,134},{133,50,136,54},{0,65,136,66}}}){
  render(patch);glow.reconstruct(context,nullptr,true,true);auto full=pixels();
  render({0,0,0,0});glow.reconstruct(context,nullptr,true,true);
  render(patch);
  auto damage=c3x_renderer::render_core::scene_filter_damage(136,136,std::vector<D3D11_RECT>{patch},4);
  bool first=true;for(auto r:damage){glow.reconstruct(context,&r,first,true);first=false;}
  auto partial=pixels();
  assert(full==partial);++circular_checks;
 }
 std::printf("PASS %u exact circular HDR/MSAA4 finishing checks: independent full reconstruction, filter seams and retained output\n",circular_checks);
 context->ClearState();staging->Release();rasterizer->Release();settings->Release();ps->Release();vs->Release();glow.reset();context->Release();device->Release();
}
