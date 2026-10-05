"""Native unit ink follows completed body placements, including wrapped views."""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp


class UnitHudTests(unittest.TestCase):
    def test_completed_placements_zoom_wrap_and_retirement(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_hud_anchors.h"
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Pose {c3x_renderer_unit_v1 draw{};};
int main(){
 UnitHudAnchors hud;Pose p{};p.draw.unit_id=7;p.draw.body_x=5;p.draw.body_y=15;
 p.draw.sprite_width=p.draw.sprite_height=191;p.draw.projection_scale_milli=1000;
 std::vector<Pose> poses{p};hud.publish(poses); // completed centre 100,110
 for(double scale:{.5,1.,1.5,3.}){
  auto offset=hud.offset(7,90,100,800,600,scale);
  assert(offset.visible&&90+offset.x==int(std::lround((100-400)*scale+400)));
  assert(100+offset.y==int(std::lround((110-300)*scale+300)));
 }
 // A pending pose does not move the HUD until the corresponding image completes.
 poses[0].draw.body_x+=45;
 assert(hud.offset(7,90,100,800,600,1.).x==10);
 hud.publish(poses);assert(hud.offset(7,90,100,800,600,1.).x==55);
 poses.push_back(poses[0]);poses.back().draw.body_x+=768;
 hud.publish(poses);assert(hud.offset(7,858,100,800,600,1.).x==55);
 hud.publish(poses,20,30);assert(hud.offset(7,90,100,800,600,1.).x==35);
 hud.publish(std::vector<Pose>{});assert(!hud.offset(7,90,100,800,600,1.).visible);
}
''')

    def test_status_copies_stable_id_and_preserves_config_off(self):
        source = (ROOT / 'injected_code.c').read_text()
        start = source.index('void __fastcall\npatch_Unit_draw_map_status')
        wrapper = source[start:source.index('\n}\n', start)+3].replace('this', 'self')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#define __fastcall
struct Unit{struct{int ID=7;}Body;};
struct PCX_Image{struct{void* Image;}JGL;};
struct{struct{bool enable_custom_rendering=false;}current_config;
 bool custom_renderer_unit_bootstrap=false;int custom_renderer_zoom_native_tile_width=128;}state,*is=&state;
bool custom_renderer_zoom_transform_active(){return false;}
void custom_renderer_zoom_transform_point(int*,int*){}
void custom_renderer_hud_layout_offset(int,int,int*x,int*y){*x=*y=0;}
int scopes=0,natives=0;Unit* expected=nullptr;PCX_Image* target=nullptr;
int custom_renderer_hud_scope(void* image,int x,int y,unsigned,int id){
 if(image){assert(id==7&&x==132&&y==232);++scopes;return 1;}
 assert(id==-1);--scopes;return 1;
}
void Unit_draw_status(Unit* u,int edx,PCX_Image* c,int x,int y,bool marks){
 assert(u==expected&&edx==123&&c==target&&x==100&&y==200&&marks);
 assert(scopes==int(state.current_config.enable_custom_rendering));++natives;
}
'''+wrapper+r'''
int main(){Unit u;PCX_Image canvas{{&u}};expected=&u;target=&canvas;
 patch_Unit_draw_map_status(&u,123,&canvas,100,200,true);assert(natives==1&&scopes==0);
 state.current_config.enable_custom_rendering=true;
 patch_Unit_draw_map_status(&u,123,&canvas,100,200,true);assert(natives==2&&scopes==0);
 state.custom_renderer_unit_bootstrap=true;
 patch_Unit_draw_map_status(&u,123,&canvas,100,200,true);assert(natives==2&&scopes==0);
}
''')


if __name__ == '__main__':
    unittest.main()
