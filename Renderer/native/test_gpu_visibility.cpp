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
 for(int zoom:{64,128,160,192})for(int offset:{-37,0,23}){
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
  for(int repeat=0;repeat<3;++repeat){
   if(repeat==2)pass.reset();
   context->UpdateSubresource(output.Get(),0,nullptr,source.data(),f.target_width*4,0);
   assert(pass.apply(device.Get(),context.Get(),output.Get(),coverage));context->CopyResource(read.Get(),output.Get());
   D3D11_MAPPED_SUBRESOURCE m={};assert(SUCCEEDED(context->Map(read.Get(),0,D3D11_MAP_READ,0,&m)));
   for(int y=0;y<f.target_height;++y)for(int x=0;x<f.target_width;++x){unsigned got=((unsigned*)((char*)m.pData+y*m.RowPitch))[x],want=expected[y*f.target_width+x];
    assert((got&0xff000000u)==(want&0xff000000u));
    for(int shift:{0,8,16}){unsigned error=unsigned(std::abs(int((got>>shift)&255u)-int((want>>shift)&255u)));
     max_error=std::max(max_error,error);if(error>1){std::printf("FOG_DIFF zoom=%d offset=%d xy=%d,%d got=%08x want=%08x error=%u\n",zoom,offset,x,y,got,want,error);return 1;}}
    ++tested;
   }
   context->Unmap(read.Get(),0);
  }
 }
 std::printf("VISIBILITY_GPU pixels=%u max_channel_error=%u reset_recovery=pass\n",tested,max_error);return 0;
}
