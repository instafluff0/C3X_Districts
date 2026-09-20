"""Visual flow follows connectivity, wraps and remote outlet invalidation."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class WaterMotionTests(unittest.TestCase):
    def test_bounded_motion_and_authoritative_camera_basis(self):
        run_cpp(r'''
#include <cassert>
#include "Renderer/native/render_core/water_material_frame.h"
int main(){
 using namespace c3x_renderer::render_core;
 static_assert(sizeof(WaterMaterialFrame)==32,"two constant-buffer vectors");
 c3x_renderer_tile_v1 tile={};tile.tile_x=10;tile.tile_y=12;
 tile.anchor_x=100;tile.anchor_y=200;tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 c3x_renderer_frame_v1 frame={};frame.target_width=640;frame.target_height=480;
 frame.tile_width=128;frame.tile_height=64;frame.tiles=&tile;frame.tile_count=1;
 frame.presentation_frequency=1000;frame.world_width_tiles=100;frame.world_height_tiles=80;
 frame.world_wrap_x=frame.world_wrap_y=1;
 auto initial=water_material_frame(frame);
 for(float v:initial.drift)assert(v==0);
 assert(initial.camera[2]==100 && initial.camera[3]==80);
 // An equivalent anchor must not move the optical eye or form a tile seam.
 tile.tile_x+=2;tile.anchor_x+=128;auto equivalent=water_material_frame(frame);
 for(unsigned i=0;i<4;++i)assert(initial.camera[i]==equivalent.camera[i]);
 tile.anchor_x-=64;auto pan=water_material_frame(frame);
 assert(pan.camera[0]==initial.camera[0]+.5f && pan.camera[1]==initial.camera[1]+.5f);
 // Bounded phases return/reverse rather than accumulating a sheet translation.
 bool positive=false,negative=false;float last=0;
 for(int second=1;second<=240;++second){frame.presentation_time_ticks=second*1000;
  auto sample=water_material_frame(frame);
  assert(std::abs(sample.drift[0])<=.11f && std::abs(sample.drift[1])<=.18f && std::abs(sample.drift[2])<=.20f);
  positive=positive || sample.drift[0]>last;negative=negative || sample.drift[0]<last;last=sample.drift[0];
 }
 assert(positive && negative);
}
''')

    def test_drainage_and_curve_inputs(self):
        run_cpp(r'''
#include <cassert>
#include "Renderer/native/render_core/world_topology.h"
#include "Renderer/lab/shared/natural/world.h"
int main(){
 using namespace c3x_renderer::render_core;
 WorldTopology world;World dims{32,32,true,false};
 std::vector<std::uint32_t> tiles(512,2u|(2u<<8));world.update(dims,tiles.data(),tiles.size());
 for(int c=20;c<=30;++c)tiles[world.index(c,8)]|=32u<<16;
 auto mouth=world.index(31,7);tiles[mouth]=12u|(12u<<8);
 world.update(dims,tiles.data(),tiles.size());
 for(int c=20;c<=30;++c){
  assert(((world.river_flow(world.index(c,8))>>4)&3)==1);
  assert(world.river_flow(world.index(c,8))==world.river_flow(world.index(c-16,-8)));
 }
 hydro::Field field;field.map_width=32;field.map_height=32;field.wraps=true;
 for(int r=5;r<11;++r)for(int c=18;c<33;++c){auto i=world.index(c,r);auto v=world.at(i);
  field.tiles[{c,r}]={c,r,c+r,c-r,int(v&255),int((v>>8)&255),unsigned((v>>16)&255),world.river_flow(i)};
 }
 river::Corridor corridor;corridor.build(field,[](double,double){return 0.;});
 for(int c=21;c<30;++c){auto sample=corridor.sample({c+.5,8});assert(sample.distance<12);assert(sample.flow.x>.8);}
 // A far-away outlet changes the direction of unchanged upstream terrain.
 auto upstream=world.index(28,8);auto before=world.at(upstream);auto direction=world.river_flow(upstream);
 c3x_renderer::fidelity::NaturalWorld::PageInputs proof;
 proof.values.push_back({upstream,before});proof.flow.push_back(static_cast<unsigned char>(direction));
 assert(proof.valid(world,1));
 tiles[mouth]=2u|(2u<<8);world.update(dims,tiles.data(),tiles.size());
 assert(!proof.valid(world,2));
 assert(world.at(upstream)==before);assert(world.river_flow(upstream)!=direction);
 // A second outlet creates a deterministic watershed, independent of view.
 tiles[world.index(19,7)]=12u|(12u<<8);world.update(dims,tiles.data(),tiles.size());
 assert(((world.river_flow(upstream)>>4)&3)==2);
 assert(world.river_flow(world.index(10,10))==0);
}
''')


if __name__ == '__main__':
    unittest.main()
