"""Execute the loading barrier and config-off menu boundary with controlled readiness."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
ROOT = Path(__file__).resolve().parents[2]

class RendererLifecycleTests(unittest.TestCase):
    def test_retired_viewer_keeps_completed_pixels(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('auto draw=[this,weak,capture,selected,x,y,w,h,sharpness]')
        draw=source[start:source.index('\n            c3x_gpu_images::RetainedComposition::Sample sample=',start)]
        run_cpp(r'''
#include <cassert>
#include <memory>
struct Rect {int left,top,right,bottom;};
int samples=0;
struct Sampled {
 int kind=0;
 static Sampled frozen(){return {2};}
 static Sampled held(){return {3};}
 static Sampled bgra(void* texture,Rect rect,float sharpness,unsigned long long generation){
  assert(texture&&rect.left==4&&rect.top==9&&rect.right==24&&rect.bottom==39&&sharpness==.75f&&generation==31);
  ++samples;return {1};
 }
};
struct Capture {bool current=true;bool valid(){return current;}};
struct Selected {bool current=true;bool valid(){return current;}};
struct Texture {void* Get(){return this;}};
struct Prepared {
 bool ready=true;float zoom=1.f;int device_generation=5,serial=7;Texture front;
 unsigned long long source_generation=31;long long pending_since=99;
};
struct Harness {
 bool camera_active=false,camera_scene_complete=true,completed_usable=false;
 bool completed_scene_usable()const{return completed_usable;}
 struct {int device_generation=5,gpu_serial=7;} renderer_state;
 Capture captured;Selected selection;
 std::shared_ptr<Prepared> prepared=std::make_shared<Prepared>();
 auto make_draw(){
  std::weak_ptr<Prepared> weak=prepared;
  auto* capture=&captured;auto* selected=&selection;
  int x=4,y=9,w=20,h=30;float sharpness=.75f;
''' + draw + r'''
  return draw;
 }
};
int main(){
 Harness owner;auto draw=owner.make_draw();
 assert(draw(1,1000,1.f).kind==1&&samples==1);
 owner.captured.current=false;assert(draw(2,1000,1.f).kind==2&&samples==1);
 owner.captured.current=true;owner.selection.current=false;
 assert(draw(3,1000,1.f).kind==2&&samples==1);
 owner.selection.current=true;assert(draw(4,1000,1.f).kind==1&&samples==2);
 // A camera job now holds the displayed pose instead of retiring the sampler (25221b37).
 owner.camera_active=true;assert(draw(5,1000,1.f).kind==3&&samples==2);
 owner.camera_active=false;owner.renderer_state.device_generation=6;
 assert(draw(6,1000,1.f).kind==2&&samples==2);
 owner.renderer_state.device_generation=5;owner.renderer_state.gpu_serial=8;
 assert(draw(7,1000,1.f).kind==2&&samples==2);
 owner.renderer_state.gpu_serial=7;owner.prepared->ready=false;
 assert(draw(8,1000,1.f).kind==3&&samples==2);
 owner.prepared->pending_since=0;assert(draw(8,1000,1.f).kind==0&&samples==2);
 owner.prepared->pending_since=99;
 owner.prepared->ready=true;assert(draw(9,1000,.5f).kind==3&&samples==2);
 assert(draw(10,1000,1.f).kind==1&&samples==3);
 owner.prepared.reset();assert(draw(11,1000,1.f).kind==2&&samples==3);
 Harness camera;auto moving=camera.make_draw();camera.camera_scene_complete=false;
 assert(moving(12,1000,1.f).kind==3&&samples==3);
 camera.completed_usable=true;assert(moving(13,1000,1.f).kind==1&&samples==4);
}
''')

    def test_menu_unloads_only_custom_scene_before_native_background(self):
        source=(ROOT/'injected_code.c').read_text()
        start=source.index('int __fastcall\npatch_Sprite_draw_main_menu_background')
        wrapper=source[start:source.index('\n}\n',start)+3].replace('this','self')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#define __fastcall
constexpr int __=0;
struct RECT{int left,top,right,bottom;};
struct JGL_Image{RECT Clip_Rect{0,0,10,10},Image_Rect{0,0,100,100};};
JGL_Image native_image;
struct Sprite{};struct PCX_Image{struct {JGL_Image* Image=&native_image;}JGL;};struct PCX_Color_Table{};
struct {struct {bool enable_custom_rendering=false;}current_config;void*custom_renderer_module=nullptr;}state,*is=&state;
int calls=0,unloads=0,clears=0;Sprite sprite;PCX_Image canvas;PCX_Color_Table palette;
void unload_custom_renderer(){++unloads;state.custom_renderer_module=nullptr;}
int PCX_Image_fill_area(PCX_Image* c,int,RECT* r,int color){
 assert(!state.custom_renderer_module && c==&canvas && r->right==100 &&
        c->JGL.Image->Clip_Rect.right==100 && unsigned(color)==0x80000000u);++clears;return 0;
}
int Sprite_draw(Sprite* s,int,PCX_Image* c,int x,int y,PCX_Color_Table* p){
 assert(s==&sprite&&c==&canvas&&x==4&&y==9&&p==&palette&&c->JGL.Image->Clip_Rect.right==10);++calls;
 if(state.current_config.enable_custom_rendering)assert(!state.custom_renderer_module);
 return 73;
}
'''+wrapper+r'''
int main(){
 state.custom_renderer_module=&sprite;
 assert(patch_Sprite_draw_main_menu_background(&sprite,0,&canvas,4,9,&palette)==73&&unloads==0&&calls==1&&clears==0);
 state.current_config.enable_custom_rendering=true;
 assert(patch_Sprite_draw_main_menu_background(&sprite,0,&canvas,4,9,&palette)==73&&unloads==1&&calls==2&&clears==1);
 assert(patch_Sprite_draw_main_menu_background(&sprite,0,&canvas,4,9,&palette)==73&&unloads==1&&calls==3&&clears==1);
}
''')
