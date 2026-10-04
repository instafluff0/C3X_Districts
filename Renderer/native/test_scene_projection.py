import unittest
from Renderer.native.native_cpp_test import run_cpp


class SceneProjectionTests(unittest.TestCase):
    def test_guarded_projection_and_native_center(self):
        run_cpp(r'''
#include "Renderer/native/scene_projection.h"
#include <cassert>
struct Viewport {float TopLeftX,TopLeftY,Width,Height;};
struct Rect {int left,top,right,bottom;};
int main(){
 for(unsigned width:{2240u,2239u})for(float scale:{.5f,.625f,.75f,.875f,1.f,1.125f,1.25f,1.5f,1.75f,2.f,2.5f,3.f}){
  c3x_renderer::SceneProjection p(width,1260,scale);
  assert(p.x(float(width/2))==float(width/2));
  for(float guard:{4.f,8.f})for(float margin:{0.f,320.f}){
   Rect display{int(margin),0,int(width+2*guard+margin),int(1260+2*guard)};
   auto source=p.source_rect(display,guard,margin);
   // Every canonical pixel whose transformed footprint touches the target
   // remains selected, including odd viewport centers and guarded edges.
   for(int x=-2;x<int(width+2*margin)+18;++x){
    float left=p.x(float(x)-guard-margin)+guard+margin,right=p.x(float(x+1)-guard-margin)+guard+margin;
    if(right>display.left&&left<display.right)assert(x>=source.left&&x<source.right);
   }
   for(int y=-2;y<1278;++y){
    float top=p.y(float(y)-guard)+guard,bottom=p.y(float(y+1)-guard)+guard;
    if(bottom>display.top&&top<display.bottom)assert(y>=source.top&&y<source.bottom);
   }
   assert(scale<=1.f||source.right-source.left<int(width));
   assert(scale>=1.f||source.right-source.left>int(width));
  }
  for(float guard:{4.f,8.f})for(float margin:{0.f,320.f})for(float raster:{1.f,.375f}){
   float extent=float(width)+2*guard+2*margin;
   Viewport v{0,0,extent*raster,(1260+2*guard)*raster};p.viewport(v,guard,margin,0,raster);
   for(float x:{0.f,200.25f,float(width/2),float(width)}){
    float translation[2]={guard+margin,guard},inverse[2]={1.f/extent,1.f/(1260+2*guard)};
    p.clip_transform(translation,inverse,guard,margin);
    float actual=v.TopLeftX+(x+translation[0])*inverse[0]*v.Width;
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
  for(float scale:{.5f,.625f,.75f,.875f,1.f,1.125f,1.25f,1.5f,1.75f,2.f,2.5f,3.f,1.25f,1.f}){
   zoom->reset(scale);auto output=retained_read(device.Get(),context.Get(),retained.sample(100+view_samples,1000).Get());
   assert(requested==scale&&normal_samples==0);
   for(unsigned x=6;x<w;++x)assert(output[x]==raster[x]);
   for(unsigned y=0;y<7;++y)for(unsigned x=0;x<5;++x)assert(output[y*w+x]==0xff16ab42);
   c3x_renderer::SceneProjection projection(w,h,scale);
   unsigned x=unsigned(projection.x(38)),y=unsigned(projection.y(22));
   unsigned below=raster[y*w+x]&255,expected=unsigned(std::round(128+below*(127.f/255)));
   assert(std::abs(int(output[y*w+x]&255)-int(expected))<=1);
  }
  assert(view_samples==14&&normal_samples==0);
  assert(retained_read(device.Get(),context.Get(),gpu.texture(world))==canonical);
  // A newer camera may supersede this publication before its first display.
  // It must retain the old complete pixels until the new world is adopted.
  RetainedComposition retired(device.Get(),context.Get());
  retired.create(world,w,h,Format::bgra32);retired.create(screen,w,h,Format::bgra32);
  RetainedComposition::Sample old=[](long long,long long){return RetainedComposition::SampledImage{};};
  old.projected=[](long long,long long,float){return RetainedComposition::SampledImage::frozen();};
  retired.source(world,gpu.texture(world),old,true,true);retired.view(screen,world,zoom);retired.commit(screen,full);
  for(float scale:{.5f,.625f,.75f,.875f,1.f,1.5f,1.25f}){zoom->reset(scale);
   auto preserved=retained_read(device.Get(),context.Get(),retired.sample(100,1000).Get());assert(preserved==canonical);
  }
  std::puts("PASS geometry projection: detail preserved, one scene sample, ordered overlay, fixed HUD, canonical image unchanged");
 }
 }catch(std::exception const& e){std::printf("FAIL projected composition: %s\n",e.what());return 1;}
}
''', timeout=90)

    def test_keyed_pathfinder_keeps_projected_map_detail(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 constexpr unsigned w=64,h=48;Rect full={0,0,w,h},mark={36,22,40,26};
 for(auto format:{Format::rgb555,Format::rgb565}){
  Compositor gpu(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
  auto world=gpu.create(w,h,Format::bgra32),words=gpu.create(w,h,format),detail=gpu.create(w,h,Format::bgra32);
  auto route=gpu.create(w,h,format),route_detail=gpu.create(w,h,Format::bgra32);
  auto screen=gpu.create(w,h,Format::bgra32);
  unsigned key=format==Format::rgb555?0x7c1f:0xf81f,ink=format==Format::rgb555?0x03e0:0x07e0;
  std::vector<unsigned> canonical(w*h,0xff808080),raster(w*h),route_words(w*h,key),route_pixels(w*h,0xffff00ff);
  for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)raster[y*w+x]=x%2?0xffe0e0e0:0xff202020;
  for(int y=mark.top;y<mark.bottom;++y)for(int x=mark.left;x<mark.right;++x){route_words[y*w+x]=ink;route_pixels[y*w+x]=0xff00ff00;}
  assert(gpu.upload(world,1,canonical.data(),canonical.size()));
  assert(gpu.upload(route,1,route_words.data(),route_words.size()));
  assert(gpu.upload(route_detail,1,route_pixels.data(),route_pixels.size()));
  for(auto id:{world,detail,route_detail,screen})retained.create(id,w,h,Format::bgra32);
  for(auto id:{words,route})retained.create(id,w,h,format);
  D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
  d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
  D3D11_SUBRESOURCE_DATA data={raster.data(),w*4,0};ComPtr<ID3D11Texture2D> texture;
  checked(device->CreateTexture2D(&d,&data,&texture));
  unsigned projected=0,canonical_samples=0;
  RetainedComposition::Sample sample=[&](long long,long long){++canonical_samples;return RetainedComposition::SampledImage{};};
  sample.projected=[&](long long,long long,float){++projected;return RetainedComposition::SampledImage::bgra(texture.Get(),full);};
  retained.source(world,gpu.texture(world),sample,true,true);
  retained.source(route,gpu.texture(route));retained.source(route_detail,gpu.texture(route_detail));
  retained.record({Kind::copy,detail,world,full,full});
  retained.record({Kind::quantize,words,world,full,full});
  retained.record({Kind::native_image,words,route,full,full,0,0,key,0,detail,route_detail,int(w),int(h)});
  auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();retained.view(screen,detail,zoom);
  retained.commit(screen,full);
  for(float scale:{.5f,.625f,.75f,.875f,1.f,1.5f,2.f}){
   zoom->reset(scale);auto output=retained_read(device.Get(),context.Get(),retained.sample(100+projected,1000).Get());
   assert(output[4*w+8]==raster[4*w+8]); // map keeps one-pixel detail through the route
   c3x_renderer::SceneProjection p(w,h,scale);
   unsigned x=unsigned(p.x(38)),y=unsigned(p.y(24));
   assert((output[y*w+x]&0x00ffffff)==0x0000ff00);
  }
  assert(projected==7&&canonical_samples==0);
 }
 }catch(std::exception const& e){std::printf("FAIL keyed route projection: %s\n",e.what());return 1;}
}
''', timeout=90)


if __name__ == "__main__":
    unittest.main()
