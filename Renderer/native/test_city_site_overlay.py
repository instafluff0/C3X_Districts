"""City-site grade publication and the independent GPU overlay boundary."""
import unittest

from Renderer.native.native_cpp_test import run_cpp


class CitySiteOverlayTests(unittest.TestCase):
    def test_copied_tiles_and_terrain_cache_isolation(self):
        run_cpp(r'''
#include "Renderer/native/render_core/city_site_overlay.h"
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_tile_v1 tiles[6]{};
 for(int i=0;i<6;++i){tiles[i].tile_x=2*i;tiles[i].tile_y=0;
  tiles[i].anchor_x=100*i;tiles[i].tile_flags=C3X_RENDERER_TILE_RENDER|
   C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
  tiles[i].city_site_grade=i+1;}
 tiles[1].city_site_grade=0; // C3X evaluation <= 0 has no drawn tile.
 tiles[2].tile_flags&=~C3X_RENDERER_TILE_RENDER;
 tiles[3].tile_flags&=~C3X_RENDERER_TILE_EXPLORED;
 tiles[4].tile_x=9; // Native parity excludes halfway tiles.
 tiles[5].city_site_grade=11;
 c3x_renderer_frame_v1 frame{};frame.tiles=tiles;frame.tile_count=6;
 frame.target_width=800;frame.target_height=600;frame.tile_width=128;frame.tile_height=64;
 CitySiteOverlay overlay;assert(overlay.capture(frame)&&overlay.tiles.size()==2);
 assert(overlay.tiles[0].x==0&&overlay.tiles[0].grade==0);
 assert(overlay.tiles[1].x==500&&overlay.tiles[1].grade==10);
 auto white=CitySiteOverlay::color(0),green=CitySiteOverlay::color(10);
 assert(white[0]==247.f/255.f&&white[1]==250.f/255.f&&white[2]==244.f/255.f);
 assert(green[0]==19.f/255.f&&green[1]==105.f/255.f&&green[2]==52.f/255.f);
 assert(CitySiteOverlay::fill_alpha>0.f&&CitySiteOverlay::inset_pixels==1.f);
 auto middle=CitySiteOverlay::color(5);
 for(int c=0;c<3;++c)assert(white[c]>middle[c]&&middle[c]>green[c]);
 assert(CitySiteOverlay::color(9)[0]-green[0]>55.f/255.f);
 auto terrain=CapturedScene::content(tiles[0]);tiles[0].city_site_grade=10;
 auto changed=CapturedScene::content(tiles[0]);
 assert(terrain.city_site_grade==0&&changed.city_site_grade==0);
 tiles[0].city_site_grade=12;assert(!overlay.capture(frame));
}
''')

    def test_gpu_inset_gap_zoom_and_unit_occlusion(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/gpu_city_site_overlay.h"
#include <cassert>
#include <vector>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,
     D3D11_SDK_VERSION,&device,&level,&context)));
 constexpr unsigned w=256,h=160;std::vector<unsigned> base(w*h,0xff505050u);
 D3D11_TEXTURE2D_DESC d{};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
 d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
 ComPtr<ID3D11Texture2D> target,read;assert(SUCCEEDED(device->CreateTexture2D(&d,nullptr,&target)));
 d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 assert(SUCCEEDED(device->CreateTexture2D(&d,nullptr,&read)));
 D3D11_TEXTURE2D_DESC a{};a.Width=w+8;a.Height=h+8;a.MipLevels=a.ArraySize=a.SampleDesc.Count=1;
 a.Format=DXGI_FORMAT_R24G8_TYPELESS;a.BindFlags=D3D11_BIND_DEPTH_STENCIL|D3D11_BIND_SHADER_RESOURCE;
 ComPtr<ID3D11Texture2D> actors;assert(SUCCEEDED(device->CreateTexture2D(&a,nullptr,&actors)));
 D3D11_DEPTH_STENCIL_VIEW_DESC dv{};dv.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;
 dv.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
 ComPtr<ID3D11DepthStencilView> depth;assert(SUCCEEDED(device->CreateDepthStencilView(actors.Get(),&dv,&depth)));
 c3x_renderer_tile_v1 tiles[2]{};auto& tile=tiles[0];tile.tile_x=tile.tile_y=0;tile.anchor_x=64;tile.anchor_y=48;
 tile.city_site_grade=11;tile.tile_flags=C3X_RENDERER_TILE_RENDER|
   C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
 tiles[1]=tile;tiles[1].tile_x=2;tiles[1].anchor_x=128;tiles[1].anchor_y=80;
 c3x_renderer_frame_v1 frame{};frame.tiles=tiles;frame.tile_count=2;
 frame.target_width=w;frame.target_height=h;frame.tile_width=128;frame.tile_height=64;
 c3x_renderer::GpuCitySiteOverlay pass;
 auto draw=[&](float zoom,unsigned stencil){
  context->UpdateSubresource(target.Get(),0,nullptr,base.data(),w*4,0);
  context->ClearDepthStencilView(depth.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1.f,stencil);
  assert(pass.draw(device.Get(),context.Get(),target.Get(),frame,actors.Get(),4,zoom));
  context->CopyResource(read.Get(),target.Get());D3D11_MAPPED_SUBRESOURCE m{};
  assert(SUCCEEDED(context->Map(read.Get(),0,D3D11_MAP_READ,0,&m)));
  std::vector<unsigned> pixels(w*h);
  for(unsigned y=0;y<h;++y)std::memcpy(pixels.data()+y*w,
      static_cast<char const*>(m.pData)+y*m.RowPitch,w*4);
  context->Unmap(read.Get(),0);return pixels;
 };
 auto normal=draw(1.f,0);
 assert(normal[80*w+128]!=base[0]); // filled center
 assert(normal[49*w+65]==base[0]); // outside diamond
 assert(normal[48*w+128]==base[0]); // inset leaves the top edge unpainted
 assert(normal[96*w+160]==base[0]); // adjacent diamonds leave a terrain seam
 assert(normal[80*w+48]==base[0]);
 auto zoomed=draw(1.5f,0);assert(zoomed[80*w+48]!=base[0]);
 auto protected_pixels=draw(1.f,1);assert(protected_pixels[80*w+128]==base[0]);
 tile.city_site_grade=10;auto second=draw(1.f,0);
 assert(((second[80*w+128]>>16)&255)>((normal[80*w+128]>>16)&255)+20);
 tile.city_site_grade=tiles[1].city_site_grade=0;
 auto cleared=draw(1.f,0);assert(cleared[80*w+128]==base[0]);
 pass.reset();tile.city_site_grade=1;auto white=draw(1.f,0);
 assert(white[80*w+128]!=normal[80*w+128]);
}
''', timeout=90)


if __name__ == "__main__":
    unittest.main()
