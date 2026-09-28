#define NOMINMAX
#include <windows.h>
#include "gpu_visibility.h"
#include <cassert>
#include <cstdio>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using namespace c3x_renderer;
using Microsoft::WRL::ComPtr;
int test_gpu_visibility(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context)));
 GpuVisibility pass;unsigned tested=0,max_error=0;
 // Real depth/stencil writes: discarded cutouts and depth-occluded fragments
 // must remain fogged; only surviving actor fragments bypass map coverage.
 char const* shader=R"(
 float4 vs(uint id:SV_VertexID):SV_Position{float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
 float ps(float4 p:SV_Position):SV_Depth{
  int2 xy=int2(p.xy)-4;
  if(xy.x<180||xy.x>=330||xy.y<20||xy.y>=200||(xy.x+xy.y)%4==0)discard;
  return xy.x<300?.25:.75;
 })";
 ComPtr<ID3DBlob> vs,ps;ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> pixel;
 assert(SUCCEEDED(D3DCompile(shader,std::strlen(shader),"actor coverage",nullptr,nullptr,"vs","vs_5_0",0,0,&vs,nullptr)));
 assert(SUCCEEDED(D3DCompile(shader,std::strlen(shader),"actor coverage",nullptr,nullptr,"ps","ps_5_0",0,0,&ps,nullptr)));
 assert(SUCCEEDED(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex)));
 assert(SUCCEEDED(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel)));
 D3D11_DEPTH_STENCIL_DESC state={};state.DepthEnable=TRUE;state.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;
 state.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;state.StencilEnable=TRUE;state.StencilReadMask=state.StencilWriteMask=0xff;
 state.FrontFace.StencilFunc=D3D11_COMPARISON_ALWAYS;state.FrontFace.StencilFailOp=state.FrontFace.StencilDepthFailOp=D3D11_STENCIL_OP_KEEP;
 state.FrontFace.StencilPassOp=D3D11_STENCIL_OP_REPLACE;state.BackFace=state.FrontFace;
 ComPtr<ID3D11DepthStencilState> marking;assert(SUCCEEDED(device->CreateDepthStencilState(&state,&marking)));
 for(int zoom:{64,128,160,192})for(int offset:{-37,0,23})for(int samples:{1,2}){
  c3x_renderer_frame_v1 f={};f.target_width=383;f.target_height=239;f.tile_width=zoom;f.tile_height=zoom/2;
  std::vector<c3x_renderer_tile_v1> tiles;
  for(int y=-16;y<20;++y)for(int x=-16;x<20;++x){if((x+y)&1)continue;
   c3x_renderer_tile_v1 t={};t.tile_x=x;t.tile_y=y;t.anchor_x=x*zoom/2+offset;t.anchor_y=y*zoom/4+offset;
   t.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
   if(x<5)t.tile_flags|=C3X_RENDERER_TILE_EXPLORED;if(x<2)t.tile_flags|=C3X_RENDERER_TILE_VISIBLE;tiles.push_back(t);
  }
  f.tiles=tiles.data();f.tile_count=unsigned(tiles.size());render_core::VisibilityCoverage coverage;assert(coverage.capture(f));
  std::vector<unsigned> source(f.target_width*f.target_height),expected;
  for(unsigned i=0;i<source.size();++i)source[i]=0xa0000000u|((i*73413u)&0xffffffu);
  coverage.apply(source.data(),expected);
  D3D11_TEXTURE2D_DESC desc={};desc.Width=f.target_width;desc.Height=f.target_height;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
  desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
  ComPtr<ID3D11Texture2D> output,read;assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&output)));
  desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
  assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&read)));
  D3D11_TEXTURE2D_DESC a={};a.Width=f.target_width+8;a.Height=f.target_height+8;a.MipLevels=a.ArraySize=1;
  a.SampleDesc.Count=samples;a.Format=DXGI_FORMAT_R24G8_TYPELESS;a.BindFlags=D3D11_BIND_DEPTH_STENCIL|D3D11_BIND_SHADER_RESOURCE;
  ComPtr<ID3D11Texture2D> actors;assert(SUCCEEDED(device->CreateTexture2D(&a,nullptr,&actors)));
  D3D11_DEPTH_STENCIL_VIEW_DESC dv={};dv.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;
  dv.ViewDimension=samples==1?D3D11_DSV_DIMENSION_TEXTURE2D:D3D11_DSV_DIMENSION_TEXTURE2DMS;
  ComPtr<ID3D11DepthStencilView> depth;assert(SUCCEEDED(device->CreateDepthStencilView(actors.Get(),&dv,&depth)));
  context->ClearDepthStencilView(depth.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,.5,0);
  context->OMSetRenderTargets(0,nullptr,depth.Get());context->OMSetDepthStencilState(marking.Get(),1);
  D3D11_VIEWPORT viewport={0,0,float(a.Width),float(a.Height),0,1};context->RSSetViewports(1,&viewport);
  context->RSSetState(nullptr);context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);context->Draw(3,0);
  context->OMSetRenderTargets(0,nullptr,nullptr);
  for(int repeat=0;repeat<4;++repeat){
   if(repeat==2)pass.reset();
   if(repeat==3)context->ClearDepthStencilView(depth.Get(),D3D11_CLEAR_STENCIL,1,0);
   context->UpdateSubresource(output.Get(),0,nullptr,source.data(),f.target_width*4,0);
   assert(pass.apply(device.Get(),context.Get(),output.Get(),coverage,repeat?actors.Get():nullptr,4));context->CopyResource(read.Get(),output.Get());
   D3D11_MAPPED_SUBRESOURCE m={};assert(SUCCEEDED(context->Map(read.Get(),0,D3D11_MAP_READ,0,&m)));
   for(int y=0;y<f.target_height;++y)for(int x=0;x<f.target_width;++x){unsigned got=((unsigned*)((char*)m.pData+y*m.RowPitch))[x],want=expected[y*f.target_width+x];
    if((repeat==1||repeat==2)&&x>=180&&x<300&&y>=20&&y<200&&(x+y)%4!=0)want=source[y*f.target_width+x];
    assert((got&0xff000000u)==(want&0xff000000u));
    for(int shift:{0,8,16}){unsigned error=unsigned(std::abs(int((got>>shift)&255u)-int((want>>shift)&255u)));
     max_error=std::max(max_error,error);if(error>1){std::printf("FOG_DIFF zoom=%d offset=%d xy=%d,%d got=%08x want=%08x error=%u\n",zoom,offset,x,y,got,want,error);return 1;}}
    ++tested;
   }
   context->Unmap(read.Get(),0);
  }
 }
 std::printf("VISIBILITY_GPU pixels=%u max_channel_error=%u reset_recovery=pass actor_stencil=pass occlusion=pass cutout=pass msaa=pass\n",tested,max_error);return 0;
}
