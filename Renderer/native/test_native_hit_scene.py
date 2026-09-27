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
 struct {bool enable_custom_rendering=false;} current_config;
 c3x_renderer_native_image_fn custom_renderer_native_image=nullptr;
} state;State* is=&state;
unsigned original_calls=0,queries=0,probes=0;int response=1;bool observing=true;
PCX_Image image={{&state}};
unsigned PCX_Image_get_pixel(PCX_Image* p,int edx,int x,int y){
 assert(edx==79 && x==-3 && y==55);++original_calls;return 0x12345678;
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
  assert(scene.nodes()<64);
 }
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
 assert(scene.nodes()==152);
}
''')


if __name__ == '__main__':
    unittest.main()
