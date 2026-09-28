"""Authored low ground: bounded data, deterministic wrap and shared surface height."""
import unittest
from pathlib import Path
from Renderer.native.native_cpp_test import run_cpp


class LowReliefTests(unittest.TestCase):
    def test_unit_ground_uses_captured_anchor_and_continuous_travel(self):
        source=(Path(__file__).resolve().parents[2]/'Renderer/sandbox/direct_units.h').read_text()
        methods=source[source.index('    float low_ground('):source.index('    template<class T>static void drop')]
        run_cpp(r'''
#include "Renderer/lab/shared/natural/queries.h"
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
struct Renderer {c3x_renderer::fidelity::NaturalData natural;c3x_renderer::render_core::WorldCoast world_coast;} renderer;
struct UnitGround {
''' + methods + r'''
};
int main(){
 for(auto& f:renderer.natural.low_relief.fields){f.width=f.height=16;f.amplitude=64;f.span=96;
  for(unsigned y=0;y<16;++y)for(unsigned x=0;x<16;++x)f.pixels.push_back((x*11+y*7)%256);
 }
 std::vector<unsigned> tiles(48*64/2,2|(2<<8));
 renderer.world_coast.update({48,64,true,true},tiles.data(),tiles.size(),1);
 UnitGround unit;c3x_renderer_tile_v1 tile{};tile.tile_x=tile.tile_y=12;
 c3x_renderer_frame_v1 frame{};frame.tiles=&tile;frame.tile_count=1;
 for(int width:{64,128,256})for(int camera:{-431,0,395}){
  frame.tile_width=width;frame.tile_height=width/2;tile.anchor_x=100+camera;tile.anchor_y=200-camera;
  for(float travel:{0.f,.1f,.25f,.5f,.75f,1.f}){
   float x=tile.anchor_x+width*.5f+travel*width*.5f;
   float y=tile.anchor_y+width*.25f+travel*width*.25f;
   float expected=unit.low_ground(12.5f+travel,.5f);
   assert(std::abs(unit.unit_low_ground(frame,x,y)-expected)<.0001f);
   assert(std::abs(unit.unit_low_ground(frame,x+48*width*.5f,y)-expected)<.0001f);
  }
 }
 frame.tile_count=0;assert(unit.unit_low_ground(frame,100,100)==0);
}
''')

    def test_wrapping_continuity_biomes_rivers_and_surface_agreement(self):
        run_cpp(r'''
#include "Renderer/lab/shared/natural/queries.h"
#include <cassert>
#include <limits>
using namespace c3x_renderer;
int main(){
 fidelity::NaturalData natural;
 std::vector<unsigned char> data={'C','3','X','L','O','W','1',0};
 auto append=[&](auto v){auto p=reinterpret_cast<unsigned char*>(&v);data.insert(data.end(),p,p+sizeof(v));};
 for(unsigned biome=0;biome<2;++biome){append(16u);append(16u);append(28.f);append(96.f);
  for(unsigned y=0;y<16;++y)for(unsigned x=0;x<16;++x)data.push_back((x*11+y*7+biome*31)%256);
 }
 assert(natural.low_relief.load(data));
 render_core::World dims{48,64,true,true};
 for(float x:{-1.21f,.1f,5.3f,24.73f})for(float y:{-7.19f,3.2f,18.91f})for(unsigned b:{0u,1u}){
  float h=natural.low_relief.sample(b,x,y,dims);assert(h>=0&&h<=28);
  assert(std::abs(h-natural.low_relief.sample(b,x+24,y+24,dims))<.0001f);
  assert(std::abs(h-natural.low_relief.sample(b,x+32,y-32,dims))<.0001f);
  assert(h==natural.low_relief.sample(b,x,y,dims));
 }
 auto empty=[](auto,auto){};
 auto check=[&](unsigned tile,bool positive){
  std::vector<unsigned> tiles(48*64/2,tile);render_core::WorldCoast coast;
  coast.update(dims,tiles.data(),tiles.size(),1);
  render_core::ExactPointCache<render_core::ShoreSample> scratch;
  fidelity::SurfaceQueries query(coast,scratch,12,12,empty,empty,true);
  for(float x:{11.9f,12.f,12.001f,12.4f}){
   float h=query.low_height(natural,x,.6f);
   assert(positive?h>0:h==0);
   float support=-1;
   assert(std::abs(query.height(natural,[](float,float){return 0.f;},x,.6f,&support)-(2.5f+h))<.0001f);
   float next=query.low_height(natural,x+.0001f,.6f);assert(std::abs(next-h)<.005f);
  }
 };
 check(2|(2<<8),true);check(1|(1<<8),true); // both supported biomes
 check(0,false);check(11|(11<<8),false); // desert and water unchanged
 check(2|(2<<8)|(2<<16),false); // a river bed cannot be raised
 for(unsigned real:{4u,5u,6u,10u})check(2|(real<<8),false); // lowland and authored relief remain intact
 auto bad=data;bad.pop_back();assert(!natural.low_relief.load(bad));
 bad=data;float huge=100;std::memcpy(bad.data()+16,&huge,4);assert(!natural.low_relief.load(bad));
 bad=data;float nan=std::numeric_limits<float>::quiet_NaN();std::memcpy(bad.data()+20,&nan,4);assert(!natural.low_relief.load(bad));
 assert(natural.low_relief.load({}));assert(natural.low_relief.sample(0,2,3,dims)==0);
}
''')


if __name__ == '__main__':
    unittest.main()
