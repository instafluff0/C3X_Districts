"""The resident GPU unit ground point follows Civ III's native sprite center."""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp


class DirectUnitPlacementTests(unittest.TestCase):
    def test_native_center_survives_canvas_size_zoom_and_wrap(self):
        source = (Path(__file__).parents[1] / "sandbox/direct_units.h").read_text()
        real = source.split("template<class Target>bool draw_real", 1)[1]
        preparation = real.split("auto draw=instance.draw;", 1)[1].split(
            "if(!renderer.prepare_unit_action(action))", 1)[0]
        placement = real.split("float scale=pose.projection_scale*scene_scale;", 1)[1].split(
            "float values[32]", 1)[0]
        shader = source.split("float2 local=", 1)[1].split("Output o;", 1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/unit_animation_runtime.h"
#include <cassert>
#include <cmath>
#include <vector>
struct float2 {
 float x,y;
 float2(float a,float b):x(a),y(b){}
 float2 operator+(float2 b)const{return {x+b.x,y+b.y};}
 float2 operator*(float s)const{return {x*s,y*s};}
};
struct Instance {c3x_renderer_unit_v1 draw{};};
struct Unit {int minimum_canvas;};
struct Action {bool loop=true;};
struct Mesh {unsigned bones=1;};
struct Part {float cutout=0;};
float2 ground(Instance instance,int minimum,c3x_renderer_frame_v1 frame,
              float scene_scale,bool reflected,float x,float y,float z){
 Unit unit{minimum};Action action;Mesh mesh;auto* source=&mesh;Part part;
 float ground_depth=0;unsigned frame_number=0;float angle=0;bool blended=false;
 struct {int width=2400,height=1400;} scene;
 for(auto const& unused:std::vector<int>{0}){
  (void)unused;auto draw=instance.draw;
''' + preparation.replace("return false;", "return {-99999,-99999};") + r'''
  float scale=pose.projection_scale*scene_scale;
''' + placement.replace("unit.scale,unit.offset_z", "1,0") + r'''
  float2 origin(placement_values[0],placement_values[1]);
  struct {float x;} pass_control{reflected?2.f:0.f};
  float2 local=''' + shader + r'''
  return pixel;
 }
 return {-99999,-99999};
}
int main(){
 c3x_renderer_frame_v1 frame{};frame.target_width=2240;frame.target_height=1260;
 frame.world_width_tiles=80;
 unsigned cases=0;
 for(int width:{120,191,240,256,511})for(int height:{121,192,240,320})
 for(int projection:{500,1000,1250,1500})for(int minimum:{0,256,512})
 for(float resolution:{.5f,1.f,2.f})for(bool reflected:{false,true})
 for(int wrap:{-1,0,1}){
  frame.tile_width=128*projection/1000;frame.world_wrap_x=wrap!=0;
  int span=frame.world_width_tiles*frame.tile_width/2;
  Instance instance;auto& d=instance.draw;
  d.unit_id=7;d.action=1;d.direction=1;d.frame_count=16;
  d.sprite_width=width;d.sprite_height=height;d.projection_scale_milli=projection;
  d.body_x=1120-width*projection/2000+wrap*span;
  d.body_y=630-height*projection/2000;
  for(float z:{0.f,.75f}){
   auto pixel=ground(instance,minimum,frame,resolution,reflected,.25f,-.5f,z);
   float guard=reflected?8.f:4.f,scale=projection/1000.f;
   float ex=(1120+guard+(.25f+.5f)*64*scale)*resolution;
   float ey=(630+guard+(-.25f*32+(reflected?1:-1)*z*(150.f*128/224))*scale)*resolution;
   assert(std::abs(pixel.x-ex)<.001f&&std::abs(pixel.y-ey)<.001f);
   ++cases;
  }
 }
 assert(cases==8640);
}
''')


if __name__ == "__main__":
    unittest.main()
