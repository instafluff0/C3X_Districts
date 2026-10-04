"""Retired cameras release history retained by native HUD copies."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class RetainedViewLifetimeTests(unittest.TestCase):
    def test_completed_intermediates_reclaim_within_one_frame(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 constexpr unsigned w=2240,h=1260,key=31775;Rect full={0,0,w,h};
 Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
 std::vector<unsigned> pixels(w*h,0x400),expected(w*h,0);
 auto base=live.create(w,h,Format::rgb555),source=live.create(w,h,Format::rgb555),glyph=live.create(32,32,Format::rgb555);
 for(auto id:{base,source})retained.create(id,w,h,Format::rgb555);
 retained.create(glyph,32,32,Format::rgb555);
 assert(live.upload(base,1,pixels.data(),pixels.size()));retained.source(base,live.texture(base));
 for(unsigned i=0;i<pixels.size();++i)pixels[i]=31744+i%32;
 assert(live.upload(source,1,pixels.data(),pixels.size()));
 retained.source(source,live.texture(source),[](long long,long long){return RetainedComposition::SampledImage{};},true);
 std::vector<unsigned> transparent(32*32,key);assert(live.upload(glyph,1,transparent.data(),transparent.size()));retained.source(glyph,live.texture(glyph));
 Id branch=100,screen=101;retained.create(screen,w,h,Format::rgb555);retained.record({Kind::fill,screen,0,full,full});
 for(unsigned n=0;n<32;++n){
  Rect part={int(n%16*32),int(n/16*32),int(n%16*32+32),int(n/16*32+32)};
  retained.create(branch,w,h,Format::rgb555);retained.record({Kind::copy,branch,base,full,full});
  retained.record({Kind::native_image,branch,source,full,full,0,0,31744+n,0,0,0,w,h});
  retained.record({Kind::native_image,branch,glyph,part,part,0,0,key,0,0,0,32,32});
  retained.record({Kind::copy,screen,branch,part,full,part.left,part.top});
  for(int y=part.top;y<part.bottom;++y)for(int x=part.left;x<part.right;++x){unsigned i=y*w+x;expected[i]=pixels[i]==31744+n?0x400:pixels[i];}
 }
 retained.commit(screen,full);
 for(unsigned tick=1;tick<=3;++tick){
  assert(retained_read(device.Get(),context.Get(),retained.sample(tick,1000).Get())==expected);
  assert(retained.bytes()<=256u*1024u*1024u);
 }
 std::printf("PASS 32 fullscreen intermediates, three exact bounded frames\n");
 }catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}
}
''', timeout=120)

    def test_uploads_reclaim_completed_frame_outputs(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 constexpr unsigned w=2240,h=1260,key=31775;Rect full={0,0,w,h};
 Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
 auto source=live.create(w,h,Format::rgb555);retained.create(source,w,h,Format::rgb555);
 std::vector<unsigned> pixels(w*h,key);for(unsigned n=0;n<pixels.size();n+=17)pixels[n]=0x3e0;
 assert(live.upload(source,1,pixels.data(),pixels.size()));
 retained.source(source,live.texture(source),[](long long,long long){return RetainedComposition::SampledImage{};},true);
 Id screen=100;retained.create(screen,w,h,Format::rgb555);
 retained.record({Kind::fill,screen,0,full,full,0,0,0x400});
 for(unsigned n=0;n<3;++n)retained.record({Kind::native_image,screen,source,full,full,0,0,key,0,0,0,w,h});
 retained.commit(screen,full);
 auto expected=retained_read(device.Get(),context.Get(),retained.sample(1,1000).Get());
 // City uploads arrive between visual samples. Last frame's evaluated
 // operation results are reclaimable, while these immutable uploads are not.
 for(Id id=200;id<220;++id){retained.create(id,w,h,Format::rgb555);retained.source(id,live.texture(source));}
 assert(retained.ready()&&retained.bytes()<=256u*1024u*1024u);
 for(Id id=200;id<220;++id)retained.destroy(id);
 assert(retained_read(device.Get(),context.Get(),retained.sample(2,1000).Get())==expected);
 std::printf("PASS uploads between frames reclaim results and restore exact pixels\n");
 }catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}
}
''', timeout=120)

    def test_cold_form_results_evict_and_restore_exactly(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 constexpr unsigned w=2240,h=1260,key=31775;Rect full={0,0,w,h};
 Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
 auto create=[&](Format f){auto id=live.create(w,h,f);assert(id);retained.create(id,w,h,f);return id;};
 auto source=create(Format::rgb555);std::vector<unsigned> pixels(w*h,key);
 for(unsigned n=0;n<pixels.size();n+=17)pixels[n]=0x3e0;
 assert(live.upload(source,1,pixels.data(),pixels.size()));
 retained.source(source,live.texture(source),[](long long,long long){return RetainedComposition::SampledImage{};},true);
 std::vector<Id> forms;std::vector<unsigned> colors;unsigned tick=0;
 auto check=[&](unsigned index){
  retained.commit(forms[index],full);auto image=retained_read(device.Get(),context.Get(),retained.sample(++tick,1000).Get());
  for(unsigned n=0;n<image.size();++n)assert(image[n]==(n%17==0?0x3e0:colors[index]));
  assert(retained.bytes()<=256u*1024u*1024u);
 };
 for(unsigned n=0;n<16;++n){
  auto target=Id(100+n);retained.create(target,w,h,Format::rgb555);forms.push_back(target);colors.push_back(0x400+n);
  retained.record({Kind::fill,target,0,full,full,0,0,colors.back()});
  retained.record({Kind::native_image,target,source,full,full,0,0,key,0,0,0,w,h});check(n);
 }
 // Revisit every saved form, in both directions. Evicted GPU results must
 // rebuild from their immutable recipes without borrowing a later form.
 for(unsigned n=16;n>0;--n)check(n-1);
 for(unsigned n=0;n<16;++n)check(n);
 std::printf("PASS cold fullscreen forms: 48 exact restores, bytes=%llu nodes=%zu\n",retained.bytes(),retained.node_count());
 }catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}
}
''', timeout=120)

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
