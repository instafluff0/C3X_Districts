"""Form input is independent of GPU progress and native config-off dispatch."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class NativeHitSceneTests(unittest.TestCase):
    def test_original_function_dispatch(self):
        source = (ROOT / 'injected_code.c').read_text()
        hook = source.split('#ifdef PCX_Image_get_pixel\n', 1)[1].split('\n#endif', 1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstddef>
#define __fastcall
struct PCX_Image {struct {void* Image;} JGL;};
struct State {
 int custom_renderer_native_operation=79;
 struct {bool enable_custom_rendering=false;} current_config;
 c3x_renderer_native_image_fn custom_renderer_native_image=nullptr;
} state;State* is=&state;
unsigned original_calls=0,queries=0,probes=0;int response=1;bool observing=true;
PCX_Image image={{&state}};
unsigned PCX_Image_get_pixel(PCX_Image* p,int edx,int x,int y){
 assert(edx==79 && x==-3 && y==55);
 assert(state.custom_renderer_native_operation==(state.current_config.enable_custom_rendering?C3X_NATIVE_HIT_PIXEL:79));
 ++original_calls;return 0x12345678;
}
bool custom_renderer_native_probe_on(){++probes;return observing;}
int query(int op,void* p,void*,void const* from,void const* to,unsigned){
 assert(op==C3X_NATIVE_HIT_PIXEL&&p==&state);auto point=static_cast<int const*>(from);
 assert(point[0]==-3&&point[1]==55);++queries;*static_cast<unsigned*>(const_cast<void*>(to))=0x7c1f;return response;
}
''' + hook + r'''
int main(){
 state.custom_renderer_native_image=query;
 // Deliberately invalid PCX pointer: the disabled branch must delegate before
 // dereferencing it, checking the probe, or sending any renderer operation.
 assert(patch_PCX_Image_get_form_hit_pixel(nullptr,79,-3,55)==0x12345678);
 assert(original_calls==1&&queries==0&&probes==0);
 state.current_config.enable_custom_rendering=true;
 assert(patch_PCX_Image_get_form_hit_pixel(&image,79,-3,55)==0x7c1f);
 assert(original_calls==1&&queries==1);
 response=0;assert(patch_PCX_Image_get_form_hit_pixel(&image,79,-3,55)==0x12345678);
 assert(original_calls==2&&queries==2);
 response=-1;assert(patch_PCX_Image_get_form_hit_pixel(&image,79,-3,55)==0);
 assert(original_calls==2&&queries==3); // Failed ownership never reads stale native bits.
 observing=false;assert(patch_PCX_Image_get_form_hit_pixel(&image,79,-3,55)==0x12345678);
 assert(original_calls==3&&queries==3);
 state.custom_renderer_native_image=nullptr;observing=true;
 assert(patch_PCX_Image_get_form_hit_pixel(&image,79,-3,55)==0x12345678);
 assert(original_calls==4&&queries==3);
 assert(state.custom_renderer_native_operation==79);
}
''')

    def test_private_cpu_hit_read_preserves_only_audited_bits_lease(self):
        run_cpp(r'''
#include "Renderer/native/native_lifetime_registry.h"
#include <cassert>
int main(){c3x_native_images::Lifetimes lifetimes;int image=0;
 for(int operation:{C3X_NATIVE_BITS,C3X_NATIVE_DC,C3X_NATIVE_PIXEL}){
  assert(lifetimes.observe(C3X_NATIVE_INIT,&image,0,1));
  bool retained=lifetimes.observe(operation,&image,C3X_NATIVE_HIT_PIXEL,1);
  assert(retained==(operation==C3X_NATIVE_BITS));
 }
 assert(lifetimes.observe(C3X_NATIVE_INIT,&image,0,1));
 assert(!lifetimes.observe(C3X_NATIVE_BITS,&image,0,1));
 assert(!lifetimes.observe(C3X_NATIVE_MAP,&image,C3X_NATIVE_HIT_PIXEL,1));
 assert(lifetimes.observe(C3X_NATIVE_INIT,&image,0,1));
 assert(!lifetimes.observe(C3X_NATIVE_BITS,&image,C3X_NATIVE_HIT_PIXEL,2));
}
''')

    def test_retained_input_lifetimes_and_long_redraw_sequence(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <vector>
using namespace c3x_gpu_images;
using namespace c3x_native_hit;
int main(){
 for(auto format:{Format::rgb555,Format::rgb565}){
  Scene scene;Rect all{0,0,32,32};unsigned value=0;
  scene.create(1,32,32,format,opaque_map);scene.create(2,32,32,format);
  scene.create(3,32,32,format);scene.create(4,32,32,Format::bgra32);
  std::vector<unsigned> sprite(1024,0);sprite[9*32+7]=65536|0x1234;
  scene.upload(4,sprite.data(),sprite.size());
  for(unsigned frame=0;frame<20000;++frame){
   scene.submit({Kind::copy,2,1,all,all});
   scene.submit({Kind::fill,2,0,{2,3,28,29},all,0,0,format==Format::rgb555?0x7c1fu:0xf81fu});
   scene.submit({Kind::native_sprite,2,4,all,all});
   scene.submit({Kind::fill,2,0,{10,10,12,12},all,0,0,0});
   scene.submit({Kind::copy,3,2,all,all});
   scene.submit({Kind::copy,3,3,all,all});
   assert(scene.pixel(3,0,0,value)&&value==opaque_map);
   assert(scene.pixel(3,7,9,value)&&value==0x1234);
   assert(scene.pixel(3,10,10,value)&&value==0);
   assert(scene.pixel(3,2,3,value)&&value==(format==Format::rgb555?0x7c1f:0xf81f));
   assert(scene.nodes()<=8&&scene.bytes()==4096);
  }
  scene.destroy(1);scene.destroy(2);scene.destroy(4);
  assert(scene.pixel(3,7,9,value)&&value==0x1234); // copied old content survives source retirement
  scene.create(4,32,32,Format::bgra32);sprite[9*32+7]=65536|0x4567;
  scene.upload(4,sprite.data(),sprite.size());
  assert(scene.pixel(3,7,9,value)&&value==0x1234); // address/id reuse cannot change a snapshot
  assert(scene.pixel(3,-1,0,value)&&value==0);
  assert(scene.pixel(3,32,31,value)&&value==0);
  assert(!scene.pixel(4,0,0,value));
  scene.destroy(3);scene.destroy(4);assert(scene.nodes()==0&&scene.bytes()==0);
 }
}
''')

    def test_partial_hud_copies_do_not_retain_overwritten_frames(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
using namespace c3x_native_hit;
int main(){
 Scene scene;Rect all{0,0,2240,1260};unsigned value=0;
 scene.create(1,2240,1260,Format::rgb555,opaque_map);
 scene.create(2,2240,1260,Format::rgb555);scene.create(3,2240,1260,Format::rgb555);
 scene.create(4,32,32,Format::bgra32,65536|0x1234);
 for(unsigned frame=0;frame<20000;++frame){
  scene.submit({Kind::copy,2,1,all,all});
  for(int n=0;n<6;++n){Rect area{n*100,100,n*100+32,132};
   scene.submit({Kind::copy,3,2,area,all,area.left,area.top});
   scene.submit({Kind::native_sprite,3,4,area,all});
   scene.submit({Kind::copy,2,3,area,all,area.left,area.top});
  }
  for(int n=0;n<6;++n){assert(scene.pixel(2,n*100+7,109,value)&&value==0x1234);
   assert(scene.pixel(3,n*100+7,109,value)&&value==0x1234);}
  assert(scene.pixel(2,1000,100,value)&&value==opaque_map);
  assert(scene.pixel(3,1000,100,value)&&value==0);
  assert(scene.nodes()<256);
 }
}
''')

    def test_dense_map_overlays_have_bounded_regional_work(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <chrono>
#include <cstdio>
using namespace c3x_native_hit;
int main(){Scene scene;Rect all{0,0,2240,1260};unsigned value=0;
 scene.create(1,2240,1260,Format::rgb555,opaque_map);
 scene.create(2,128,64,Format::bgra32,65536|0x1234);
 scene.create(3,2240,1260,Format::rgb555);
 std::vector<unsigned> expected(2240*1260,opaque_map);
 auto begin=std::chrono::steady_clock::now();std::size_t peak=0;
 for(int frame=0;frame<8;++frame){
  // Preserve a changing source snapshot across copy/overwrite operations.
  scene.submit({Kind::quantize,1,0,all,all});std::fill(expected.begin(),expected.end(),opaque_map);
  for(int n=0;n<2600;++n){int x=(n*64+frame*7)%2240,y=((n/35)*32+frame*3)%1260;
   Rect area{x,y,x+128,y+64};scene.submit({Kind::native_sprite,1,2,area,all});
   for(int yy=y;yy<std::min(y+64,1260);++yy)for(int xx=x;xx<std::min(x+128,2240);++xx)expected[yy*2240+xx]=0x1234;
  }
  scene.submit({Kind::copy,3,1,all,all});
  scene.submit({Kind::fill,1,0,all,all,0,0,0});
  assert(scene.pixel(1,0,0,value)&&value==0);
  for(int y=0;y<1260;y+=13)for(int x=0;x<2240;x+=17)
   assert(scene.pixel(3,x,y,value)&&value==expected[y*2240+x]);
  peak=std::max(peak,scene.nodes());assert(scene.nodes()<16000);
 }
 auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
 std::printf("INPUT_DENSE_MAP draws=20800 peak_nodes=%zu total_ms=%.1f\n",peak,ms);
 assert(ms<5000); // Regression guard: old code stalls and exceeds history depth.
}
''')

    def test_full_screen_keyed_ui_transfers_skip_uniform_regions(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <chrono>
#include <cstdio>
using namespace c3x_native_hit;
int main(){Scene scene;Rect all{0,0,2240,1260};unsigned value=0;
 scene.create(1,2240,1260,Format::rgb555,opaque_map);
 scene.create(2,2240,1260,Format::rgb555);
 scene.create(3,2240,1260,Format::rgb555);
 std::vector<unsigned> hud(2240*1260,0x7c1f);
 for(int y=1170;y<1260;++y)for(int x=1920;x<2240;++x)
  hud[y*2240+x]=(x+y)%3?0x1234:0x7c1f;
 scene.upload(2,hud.data(),hud.size());
 Command overlay{Kind::native_image,1,2,all,all,0,0,0x7c1f};
 overlay.source_width=2240;overlay.source_height=1260;
 auto begin=std::chrono::steady_clock::now();
 for(unsigned n=0;n<400;++n){
  scene.submit(overlay);
  scene.submit({Kind::copy,3,1,all,all});
  assert(scene.pixel(3,1135,600,value)&&value==opaque_map);
  assert(scene.pixel(3,1931,1201,value)&&value==opaque_map);
  assert(scene.pixel(3,1932,1201,value)&&value==0x1234);
  assert(scene.nodes()<2000&&scene.bytes()<2u*1024u*1024u);
 }
 auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
 std::printf("INPUT_KEYED_UI transfers=400 total_ms=%.1f\n",ms);
 assert(ms<3000);
 // Uniform collapse must respect clipping, translation and source bounds.
 scene.submit({Kind::fill,2,0,all,all,0,0,0x1234});
 scene.submit({Kind::color_key,1,2,{-10,-10,22,22},{0,0,20,20},0,0,0x7c1f});
 assert(scene.pixel(1,19,19,value)&&value==0x1234);
 assert(scene.pixel(1,20,20,value)&&value==opaque_map);
 scene.destroy(1);scene.destroy(2);scene.destroy(3);assert(scene.bytes()==0&&scene.nodes()==0);
}
''')

    def test_shared_blend_inputs_do_not_repeat_graph_traversal(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
using namespace c3x_native_hit;
int main(){Scene scene;Rect all{0,0,32,32};unsigned value=0;
 scene.create(1,32,32,Format::rgb555,0x1234);
 scene.create(2,32,32,Format::bgra32,0x80000000);
 // A native blend's prior and background can be the same retained image.
 // Both references are necessary but pruning must visit their shared history once.
 for(unsigned n=0;n<150;++n){scene.submit({Kind::native_blend,1,2,all,all,0,0,0,1});}
 assert(scene.pixel(1,10,10,value)&&value==0);
 assert(scene.nodes()<30);
}
''')

    def test_changing_sprite_uploads_bound_bytes_and_preserve_copied_input(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <chrono>
#include <cstdio>
using namespace c3x_native_hit;
int main(){Scene scene;Rect all{0,0,2240,1260};unsigned value=0;
 scene.create(1,2240,1260,Format::rgb555,opaque_map);
 scene.create(2,128,64,Format::bgra32);
 scene.create(3,2240,1260,Format::rgb555);
 std::vector<unsigned> source(128*64),expected(2240*1260,opaque_map),saved;
 auto begin=std::chrono::steady_clock::now();std::size_t peak=0;
 // Real sprite preparation replaces the uploaded source between draws. A
 // constant-source fixture cannot expose retained payload growth during a
 // sequence of redraws while a replacement camera is still pending.
 for(unsigned n=0;n<10400;++n){
  for(unsigned p=0;p<source.size();++p)source[p]=(p+n)%5?65536|((p+n)%32767+1):0;
  scene.upload(2,source.data(),source.size());
  int x=(n*64)%2240,y=((n/35)*32)%1260;Rect area{x,y,x+128,y+64};
  scene.submit({Kind::native_sprite,1,2,area,all});
  for(int yy=y;yy<std::min(y+64,1260);++yy)for(int xx=x;xx<std::min(x+128,2240);++xx){
   auto word=source[(yy-y)*128+xx-x];if(word&65536)expected[yy*2240+xx]=word&65535;
  }
  if(n==2599){scene.submit({Kind::copy,3,1,all,all});saved=expected;}
  peak=std::max(peak,scene.bytes());assert(scene.bytes()<32u*1024u*1024u);
 }
 for(int y=0;y<1260;y+=13)for(int x=0;x<2240;x+=17){
  assert(scene.pixel(1,x,y,value)&&value==expected[y*2240+x]);
  assert(scene.pixel(3,x,y,value)&&value==saved[y*2240+x]);
 }
 auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
 std::printf("INPUT_CHANGING_SPRITES draws=10400 peak_bytes=%zu total_ms=%.1f\n",peak,ms);
 assert(ms<5000);
 scene.destroy(1);scene.destroy(2);scene.destroy(3);assert(scene.bytes()==0&&scene.nodes()==0);
}
''')

    def test_transparent_hud_redraws_preserve_holes_without_retaining_old_maps(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <array>
using namespace c3x_native_hit;
int main(){Scene scene;Rect all{0,0,144,96};unsigned value=0;
 scene.create(1,144,96,Format::rgb555);scene.create(2,144,96,Format::rgb555);scene.create(3,144,96,Format::rgb555);
 std::array<unsigned,144*96> screen{},hud{};std::array<unsigned,16*16> sprite{};
 for(unsigned n=0;n<sprite.size();++n)sprite[n]=n%3==0?0xff000000:n%3==1?0x80000400:0x0000001f;
 scene.create(4,16,16,Format::bgra32);scene.upload(4,sprite.data(),sprite.size());
 for(unsigned frame=0;frame<2000;++frame){
  auto background=0x1234+(frame%32);scene.create(1,144,96,Format::rgb555,background);screen.fill(background);
  scene.submit({Kind::copy,2,1,all,all});
  for(int n=0;n<6;++n){Rect area{13+n*19,53,13+n*19+16,69};
   scene.submit({Kind::native_blend,3,4,area,all,0,0,0,2});
   for(int y=53;y<69;++y)for(int x=13+n*19;x<13+n*19+16;++x){auto p=sprite[(y-53)*16+x-13-n*19],weight=p>>24,word=p&65535,below=screen[y*144+x];
    if(weight==255)continue;
    unsigned result=0;for(unsigned shift:{0,5,10}){
     unsigned foreground=((word>>shift)&31)*8,base=((below>>shift)&31)*8;
     result|=((foreground+(base*weight>>8))>>3)<<shift;
    }
    hud[y*144+x]=result&65535;
   }
   scene.submit({Kind::copy,2,3,area,all,area.left,area.top});
   for(int y=53;y<69;++y)for(int x=13+n*19;x<13+n*19+16;++x)screen[y*144+x]=hud[y*144+x];
  }
  assert(scene.nodes()<768);
  if(frame%100==0)for(int y=0;y<96;++y)for(int x=0;x<144;++x){
   assert(scene.pixel(2,x,y,value)&&value==screen[y*144+x]);
   assert(scene.pixel(3,x,y,value)&&value==hud[y*144+x]);
  }
 }
}
''')

    def test_repeated_destination_key_misses_visit_prior_input_once(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <chrono>
using namespace c3x_native_hit;
int main(){Scene scene;Rect all{0,0,64,64};unsigned value=0;
 scene.create(1,64,64,Format::rgb555,opaque_map);
 scene.create(2,64,64,Format::bgra32);
 std::vector<unsigned> keys(64*64,0x123403e0);scene.upload(2,keys.data(),keys.size());
 auto start=std::chrono::steady_clock::now();
 for(unsigned n=0;n<1024;++n){
  scene.submit({Kind::native_blend,1,2,all,all,0,0,4,1});
  assert(scene.pixel(1,17,29,value)&&value==opaque_map);
  assert(scene.nodes()<32);
 }
 assert(std::chrono::steady_clock::now()-start<std::chrono::seconds(2));
 // Matching destination replaces the packed word. With a distinct background,
 // a failed comparison must still retain the actual destination's prior value.
 scene.create(1,64,64,Format::rgb555,0x1234);
 scene.submit({Kind::native_blend,1,2,all,all,0,0,4,1});
 assert(scene.pixel(1,17,29,value)&&value==0x3e0);
 scene.create(3,64,64,Format::rgb555,0x5678);
 scene.submit({Kind::native_blend,1,2,all,all,0,0,4,3});
 assert(scene.pixel(1,17,29,value)&&value==0x3e0);
}
''')


if __name__ == '__main__':
    unittest.main()
