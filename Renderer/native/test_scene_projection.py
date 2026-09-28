import unittest
from Renderer.native.native_cpp_test import run_cpp


class SceneProjectionTests(unittest.TestCase):
    def test_guarded_projection_and_native_center(self):
        run_cpp(r'''
#include "Renderer/native/scene_projection.h"
#include <cassert>
struct Viewport {float TopLeftX,TopLeftY,Width,Height;};
int main(){
 for(unsigned width:{2240u,2239u})for(float scale:{1.f,1.125f,1.25f,1.5f}){
  c3x_renderer::SceneProjection p(width,1260,scale);
  assert(p.x(float(width/2))==float(width/2));
  for(float guard:{4.f,8.f})for(float margin:{0.f,320.f})for(float raster:{1.f,.375f}){
   float extent=float(width)+2*guard+2*margin;
   Viewport v{0,0,extent*raster,(1260+2*guard)*raster};p.viewport(v,guard,margin,0,raster);
   for(float x:{0.f,200.25f,float(width/2),float(width)}){
    float actual=v.TopLeftX+(x+guard+margin)/extent*v.Width;
    float expected=(p.x(x)+guard+margin)*raster;
    assert(std::abs(actual-expected)<.001f);
   }
  }
 }
}
''')

    def test_projected_geometry_survives_native_overlay_and_fixed_hud(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 constexpr unsigned w=64,h=48;Rect full={0,0,w,h};
 for(auto format:{Format::rgb555,Format::rgb565}){
  Compositor gpu(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
  auto world=gpu.create(w,h,Format::bgra32),words=gpu.create(w,h,format),detail=gpu.create(w,h,Format::bgra32);
  auto overlay=gpu.create(8,8,Format::bgra32),screen=gpu.create(w,h,Format::bgra32),view_words=gpu.create(w,h,format);
  std::vector<unsigned> canonical(w*h,0xff808080),ink(64,0x80000080);
  assert(gpu.upload(world,1,canonical.data(),canonical.size()));assert(gpu.upload(overlay,1,ink.data(),ink.size()));
  for(auto id:{world,detail,screen})retained.create(id,w,h,Format::bgra32);
  retained.create(words,w,h,format);retained.create(view_words,w,h,format);retained.create(overlay,8,8,Format::bgra32);
  // A new raster at display resolution contains 1-pixel detail which does not
  // exist in the canonical image. A subsequent image enlargement loses it.
  std::vector<unsigned> raster(w*h);for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
   raster[y*w+x]=x%2?0xffe0e0e0:0xff202020;
  D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
  d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
  D3D11_SUBRESOURCE_DATA data={raster.data(),w*4,0};ComPtr<ID3D11Texture2D> texture;
  checked(device->CreateTexture2D(&d,&data,&texture));
  unsigned normal_samples=0,view_samples=0;float requested=0;
  RetainedComposition::Sample sample=[&](long long,long long){++normal_samples;return RetainedComposition::SampledImage{};};
  sample.projected=[&](long long,long long,float scale){++view_samples;requested=scale;return RetainedComposition::SampledImage::bgra(texture.Get(),full);};
  retained.source(world,gpu.texture(world),sample,true,true);
  retained.source(words,gpu.texture(words));retained.source(overlay,gpu.texture(overlay));
  retained.record({Kind::copy,detail,world,full,full});
  Rect a={34,18,42,26};
  retained.record({Kind::unit_over,words,overlay,a,full,0,0,0,words,detail,detail});
  auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();
  retained.view(screen,detail,zoom,view_words);
  Rect panel={0,0,5,7};retained.record({Kind::fill,screen,0,panel,full,0,0,0xff16ab42});retained.commit(screen,full);
  for(float scale:{1.f,1.125f,1.25f,1.5f,1.25f,1.f}){
   zoom->reset(scale);auto output=retained_read(device.Get(),context.Get(),retained.sample(100+view_samples,1000).Get());
   assert(requested==scale&&normal_samples==0);
   for(unsigned x=6;x<w;++x)assert(output[x]==raster[x]);
   for(unsigned y=0;y<7;++y)for(unsigned x=0;x<5;++x)assert(output[y*w+x]==0xff16ab42);
   c3x_renderer::SceneProjection projection(w,h,scale);
   unsigned x=unsigned(projection.x(38)),y=unsigned(projection.y(22));
   unsigned below=raster[y*w+x]&255,expected=unsigned(std::round(128+below*(127.f/255)));
   assert(std::abs(int(output[y*w+x]&255)-int(expected))<=1);
  }
  assert(view_samples==6&&normal_samples==0);
  assert(retained_read(device.Get(),context.Get(),gpu.texture(world))==canonical);
  // A newer camera may supersede this publication before its first display.
  // It must retain the old complete pixels until the new world is adopted.
  RetainedComposition retired(device.Get(),context.Get());
  retired.create(world,w,h,Format::bgra32);retired.create(screen,w,h,Format::bgra32);
  RetainedComposition::Sample old=[](long long,long long){return RetainedComposition::SampledImage{};};
  old.projected=[](long long,long long,float){return RetainedComposition::SampledImage::frozen();};
  retired.source(world,gpu.texture(world),old,true,true);retired.view(screen,world,zoom);retired.commit(screen,full);
  for(float scale:{1.f,1.5f,1.25f}){zoom->reset(scale);
   auto preserved=retained_read(device.Get(),context.Get(),retired.sample(100,1000).Get());assert(preserved==canonical);
  }
  std::puts("PASS geometry projection: detail preserved, one scene sample, ordered overlay, fixed HUD, canonical image unchanged");
 }
 }catch(std::exception const& e){std::printf("FAIL projected composition: %s\n",e.what());return 1;}
}
''', timeout=90)


if __name__ == "__main__":
    unittest.main()
