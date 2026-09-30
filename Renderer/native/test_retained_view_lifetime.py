"""Retired cameras release history retained by native HUD copies."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class RetainedViewLifetimeTests(unittest.TestCase):
    def test_retired_hud_history_is_bounded(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 constexpr unsigned w=128,h=96;Rect full={0,0,w,h},area={29,41,93,73};
 for(bool projected:{false,true}){
 Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
 auto create=[&](unsigned x,unsigned y,Format f){auto id=live.create(x,y,f);assert(id);retained.create(id,x,y,f);return id;};
 auto screen=create(w,h,Format::rgb555),detail=create(w,h,Format::bgra32),hud=create(w,h,Format::rgb555),sprite=create(64,32,Format::bgra32);
 std::vector<unsigned> pixels(w*h),program(64*32);
 for(unsigned i=0;i<program.size();++i)program[i]=i%3==0?0xff000000u:i%3==1?0x8000001fu:0x00000400u;
 assert(live.upload(sprite,1,program.data(),program.size()));retained.source(sprite,live.texture(sprite));
 retained.record({Kind::fill,hud,0,full,full});
 unsigned current=0;auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();
 for(unsigned view=1;view<=80;++view){
  current=view;auto map=create(w,h,Format::bgra32);
  std::fill(pixels.begin(),pixels.end(),0xff234567u+view);assert(live.upload(map,1,pixels.data(),pixels.size()));
  D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
  d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
  D3D11_SUBRESOURCE_DATA data={pixels.data(),w*4,0};ComPtr<ID3D11Texture2D> texture;checked(device->CreateTexture2D(&d,&data,&texture));
  RetainedComposition::Sample sample=[&,view,texture](long long,long long){return current==view?RetainedComposition::SampledImage::bgra(texture.Get(),full):RetainedComposition::SampledImage::frozen();};
  if(projected)sample.projected=[&,view,texture](long long,long long,float){return current==view?RetainedComposition::SampledImage::bgra(texture.Get(),full):RetainedComposition::SampledImage::frozen();};
  retained.source(map,live.texture(map),sample,true,true);retained.view(detail,map,zoom,screen);
  Command native_map={Kind::quantize,screen,map,full,full};assert(live.submit(&native_map,1));
  for(auto c:{Command{Kind::native_blend,hud,sprite,area,full,0,0,0,screen},Command{Kind::copy,screen,hud,area,full,area.left,area.top}}){assert(live.submit(&c,1));retained.record(c);}
  retained.commit(screen,full);auto result=retained.sample(view,1000);
  assert(retained_read(device.Get(),context.Get(),result.Get())==retained_read(device.Get(),context.Get(),live.texture(screen)));
  assert(retained.bytes()<w*h*4*12);
  if(view%10==0)std::printf("HISTORY projected=%u views=%u bytes=%llu nodes=%zu\n",unsigned(projected),view,retained.bytes(),retained.node_count());
  retained.destroy(map);live.destroy(map);
 }
 retained.clear();assert(!retained.bytes()&&!retained.node_count());
 }
 }catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}
}
''', timeout=90)
