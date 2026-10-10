"""Executable authoritative-occurrence and single-sample ownership contracts."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT=Path(__file__).resolve().parents[2]


class CameraRedrawWitness(unittest.TestCase):
    def test_capture_anchors_changed_strip_and_wrapped_occurrences(self):
        capture=(ROOT/'Renderer/sandbox/capture_model.h').read_text().replace('#pragma once','')
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cmath>
#include <vector>
#include <set>
#include <cassert>
#include <cstring>
''' + capture + r'''
int main(){
 c3x_renderer_frame_v1 f={};f.target_width=2240;f.target_height=1260;f.tile_width=128;f.tile_height=64;
 f.world_width_tiles=100;f.world_height_tiles=80;f.world_wrap_x=1;
 std::vector<c3x_renderer_tile_v1> tiles;
 for(int y=0;y<80;++y)for(int x=y%2;x<100;x+=2){
  c3x_renderer_tile_v1 t={};t.tile_x=x;t.tile_y=y;t.anchor_x=x*64-480;t.anchor_y=y*32-240;
  t.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;tiles.push_back(t);
 }
 f.tiles=tiles.data();f.tile_count=unsigned(tiles.size());SandboxCameraWitness w(f);
 std::set<std::pair<int,int>> last;
 for(auto v:std::vector<SandboxCameraWitness::View>{{0,0,1,"origin"},{96,48,1,"pan"},
     {288,144,1,"strip"},{6592,96,1,"wrap"},{-640,-256,1,"jump"},{96,48,1.25f,"zoom"}}){
  auto capture=w.capture(v);assert(!capture.empty());std::set<std::pair<int,int>> occurrences;
  unsigned rendered=0,prefetched=0,topology=0;
  for(auto const&t:capture){
   assert(t.anchor_x-t.tile_x*64==-480+v.x);assert(t.anchor_y-t.tile_y*32==-240+v.y);
   assert(occurrences.emplace(t.tile_x,t.tile_y).second);
   rendered+=(t.tile_flags&C3X_RENDERER_TILE_RENDER)!=0;prefetched+=(t.tile_flags&C3X_RENDERER_TILE_PREFETCH)!=0;
   topology+=(t.tile_flags&C3X_RENDERER_TILE_TOPOLOGY_HALO)!=0 && !(t.tile_flags&C3X_RENDERER_TILE_PREFETCH);
   if(t.tile_flags&C3X_RENDERER_TILE_RENDER)assert(t.anchor_x+128>=0 && t.anchor_x<=2240 && t.anchor_y+64>=0 && t.anchor_y<=1260);
   if(!(t.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))assert(t.city_id==-1 && t.resource_id==-1);
  }
  assert(rendered && prefetched && topology);
  if(v.x==288)assert(occurrences!=last);
  if(v.x==6592){bool wrapped=false;for(auto const&t:capture)wrapped|=t.tile_x<0;assert(wrapped);}
  last=occurrences;
 }
 assert(sandbox_capture_zoom(0)==1.f && sandbox_capture_zoom(45)==1.25f && sandbox_capture_zoom(90)==1.f);
 assert(sandbox_capture_zoom(135)==1.25f && sandbox_capture_zoom(180)==1.f);
 auto fixed=w.capture({0,0,1,"zoom_native"});auto projected=w.capture({0,0,1.25f,"zoom_projection"});
 assert(fixed.size()==projected.size());
 for(std::size_t i=0;i<fixed.size();++i)assert(std::memcmp(&fixed[i],&projected[i],sizeof(fixed[i]))==0);
 f.world_wrap_x=0;SandboxCameraWitness bounded(f);
 for(auto const&t:bounded.capture({288,144,1,"no_wrap"}))assert(t.tile_x>=0 && t.tile_x<100);
 f.world_wrap_x=1;SandboxCameraWitness resized(f);
 for(auto const&t:resized.capture({0,0,1,"native64",64})){
  assert(t.anchor_x-t.tile_x*32==320);assert(t.anchor_y-t.tile_y*16==195);
 }
 for(auto& t:tiles)t.tile_flags&=~C3X_RENDERER_TILE_EXPLORED;
 SandboxCameraWitness unexplored(f);
 for(auto const&t:unexplored.capture({0,0,1,"unexplored"}))assert(!(t.tile_flags&C3X_RENDERER_TILE_PREFETCH));
}
''')

    def test_alias_copy_reference_pixels_resize_swap_and_msaa(self):
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        # ea135a9b places configured_samples/prepare_assets between this method and ensure_targets.
        method='    bool ensure_linear_target('+source.split('    bool ensure_linear_target(',1)[1].split('    unsigned configured_samples() const {',1)[0]
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
#include "Renderer/native/render_core/linear_target.h"
using Microsoft::WRL::ComPtr;
using c3x_renderer::render_core::LinearTarget;
// e439ec74 memoizes switches for the process lifetime; this fixture toggles both modes in one process.
namespace c3x_renderer { namespace render_core {
DWORD cached_environment(char const* name,char* buffer,DWORD size){return GetEnvironmentVariableA(name,buffer,size);}
}}
struct Harness {struct {ID3D11Device* device;} renderer;
''' + method + r'''
};
std::vector<unsigned char> read(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* texture){
 D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);d.BindFlags=0;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ComPtr<ID3D11Texture2D> staging;assert(SUCCEEDED(device->CreateTexture2D(&d,nullptr,&staging)));
 context->CopyResource(staging.Get(),texture);D3D11_MAPPED_SUBRESOURCE m={};assert(SUCCEEDED(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&m)));
 std::vector<unsigned char> result(std::size_t(d.Width)*d.Height*8);
 for(unsigned y=0;y<d.Height;++y)std::memcpy(result.data()+std::size_t(y)*d.Width*8,static_cast<char*>(m.pData)+std::size_t(y)*m.RowPitch,d.Width*8);
 context->Unmap(staging.Get(),0);return result;
}
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context)));
 Harness p{{device.Get()}};LinearTarget alias,copy;
 for(unsigned size:{32u,48u,32u}){
  SetEnvironmentVariableA("C3X_SANDBOX_RESOLVE_COPY_REFERENCE",nullptr);assert(p.ensure_linear_target(alias,size,24,1,true));
  SetEnvironmentVariableA("C3X_SANDBOX_RESOLVE_COPY_REFERENCE","1");assert(p.ensure_linear_target(copy,size,24,1,true));
  assert(alias.resolved==alias.color && alias.view==alias.samples);assert(copy.resolved!=copy.color);
  assert(alias.bytes()==std::size_t(size)*24*12 && copy.bytes()==std::size_t(size)*24*20);
  for(unsigned frame=0;frame<16;++frame){
   float value[]={float(frame)/16,0.13f,float(15-frame)/9,1};
   context->ClearRenderTargetView(alias.target,value);context->ClearRenderTargetView(copy.target,value);
   context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(copy.resolved,copy.color);
   assert(read(device.Get(),context.Get(),alias.resolved)==read(device.Get(),context.Get(),copy.resolved));
  }
 }
 alias.swap(copy);assert(copy.resolved==copy.color && alias.resolved!=alias.color);
 alias.reset();copy.reset();
 SetEnvironmentVariableA("C3X_SANDBOX_RESOLVE_COPY_REFERENCE",nullptr);
 assert(p.ensure_linear_target(alias,32,24,2,true));assert(alias.resolved!=alias.color && alias.sample_count==2);
 alias.reset();assert(!alias.color&&!alias.resolved&&!alias.view&&!alias.samples);
}
''',timeout=60)


if __name__=='__main__':unittest.main()
