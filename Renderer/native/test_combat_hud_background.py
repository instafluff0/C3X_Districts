"""Exercise the injected HUD background without reading GPU-owned map pixels."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_zoom_integration import function

ROOT = Path(__file__).resolve().parents[2]


class CombatHudBackgroundTests(unittest.TestCase):
    def test_gpu_rectangle_and_native_pixel_paths(self):
        source = (ROOT / 'injected_code.c').read_text()
        bodies = '\n'.join(result + ' ' + function(source, name) for result, name in [
            ('void', 'discard_combat_odds_hud_background'),
            ('void', 'restore_combat_odds_hud_background'),
            ('bool', 'save_combat_odds_hud_background')])
        bodies = bodies.replace('saved = malloc (sizeof *saved)', 'saved = (PCX_Image*)malloc (sizeof *saved)')
        bodies = bodies.replace('unsigned short * larger = realloc (', 'unsigned short * larger = (unsigned short*)realloc (')
        run_cpp(r'''
#include <cassert>
#include <cstdlib>
#include <cstring>
constexpr int __=0;
#define PCX_Image_create_and_init_jgl_image create_image
struct RECT {int left,top,right,bottom;};
struct JGL_Image;
unsigned short* get_pixel(JGL_Image*,int,int,int);
struct VTable {unsigned short*(*m07_m05_Get_Pixel)(JGL_Image*,int,int,int)=get_pixel;} vtable;
struct JGL_Image {VTable* vtable;RECT Image_Rect;unsigned short pixels[200];};
struct PCX_Image {struct {JGL_Image* Image;} JGL;};
struct {struct {bool enable_custom_rendering=true;}current_config;
 void* custom_renderer_native_image=&vtable;
 PCX_Image* custom_renderer_combat_odds_background=nullptr;
 bool combat_odds_hud_rect_drawn=false;
 JGL_Image* combat_odds_hud_background_canvas=nullptr;
 unsigned short* combat_odds_hud_background_pixels=nullptr;
 int combat_odds_hud_background_pixel_capacity=0;
 int combat_odds_hud_drawn_left=0,combat_odds_hud_drawn_top=0,combat_odds_hud_drawn_w=0,combat_odds_hud_drawn_h=0;
} state,*is=&state;
int reads=0,creates=0,copies=0;bool fail_create=false;
unsigned short* get_pixel(JGL_Image* image,int,int x,int y){++reads;return &image->pixels[y*20+x];}
void PCX_Image_construct(PCX_Image* image){image->JGL.Image=nullptr;}
int create_image(PCX_Image* image,int,int w,int h,int depth,int mode,int p5,int p6){
 assert(depth==0&&mode==1&&p5==0&&p6==0);++creates;
 delete image->JGL.Image;image->JGL.Image=fail_create?nullptr:new JGL_Image{&vtable,{0,0,w,h},{}};return 0;
}
void patch_JGL_Image_copy(JGL_Image* source,int,JGL_Image* destination,RECT* from,RECT* to){
 ++copies;assert(from->right-from->left==to->right-to->left&&from->bottom-from->top==to->bottom-to->top);
 for(int y=0;y<from->bottom-from->top;++y)for(int x=0;x<from->right-from->left;++x)
  destination->pixels[(to->top+y)*20+to->left+x]=source->pixels[(from->top+y)*20+from->left+x];
}
'''+bodies+r'''
int main(){
 JGL_Image original{&vtable,{0,0,20,10},{}};PCX_Image canvas{{&original}},other{{nullptr}};
 for(int i=0;i<200;++i)original.pixels[i]=i;
 assert(!save_combat_odds_hud_background(nullptr,0,0,4,2));
 assert(!save_combat_odds_hud_background(&canvas,0,0,0,2));
 assert(save_combat_odds_hud_background(&canvas,3,4,4,2));
 assert(creates==1&&copies==1&&!reads&&state.combat_odds_hud_rect_drawn);
 std::memset(original.pixels,0,sizeof original.pixels);
 restore_combat_odds_hud_background(&canvas);
 assert(copies==2&&!reads&&!state.combat_odds_hud_rect_drawn&&!state.combat_odds_hud_background_canvas);
 for(int y=0;y<10;++y)for(int x=0;x<20;++x)
  assert(original.pixels[y*20+x]==((y>=4&&y<6&&x>=3&&x<7)?y*20+x:0));
 assert(save_combat_odds_hud_background(&canvas,3,4,4,2));assert(creates==1);
 restore_combat_odds_hud_background(&other);assert(copies==3&&!state.combat_odds_hud_rect_drawn);
 assert(save_combat_odds_hud_background(&canvas,3,4,5,2));assert(creates==2);
 restore_combat_odds_hud_background(&canvas);
 fail_create=true;assert(!save_combat_odds_hud_background(&canvas,3,4,6,2));
 assert(creates==3&&!state.combat_odds_hud_rect_drawn&&!reads);
 // Custom-off keeps the original pixel buffer path and exact saved rectangle.
 state.current_config.enable_custom_rendering=false;int before=copies;
 assert(save_combat_odds_hud_background(&canvas,3,4,4,2));assert(reads==8&&copies==before);
 original.pixels[83]=999;restore_combat_odds_hud_background(&canvas);
 assert(reads==16&&copies==before&&original.pixels[83]==83);
 std::free(state.custom_renderer_combat_odds_background);std::free(state.combat_odds_hud_background_pixels);
}
''')


if __name__ == '__main__':
    unittest.main()
