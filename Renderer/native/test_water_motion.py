"""Visual flow follows connectivity, wraps and remote outlet invalidation."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class WaterMotionTests(unittest.TestCase):
    def test_drainage_and_curve_inputs(self):
        run_cpp(r'''
#include <cassert>
#include "Renderer/native/render_core/world_topology.h"
#include "Renderer/native/source_fidelity/river_corridor.h"
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
 tiles[mouth]=2u|(2u<<8);world.update(dims,tiles.data(),tiles.size());
 assert(world.at(upstream)==before);assert(world.river_flow(upstream)!=direction);
 // A second outlet creates a deterministic watershed, independent of view.
 tiles[world.index(19,7)]=12u|(12u<<8);world.update(dims,tiles.data(),tiles.size());
 assert(((world.river_flow(upstream)>>4)&3)==2);
 assert(world.river_flow(world.index(10,10))==0);
}
''')


if __name__ == '__main__':
    unittest.main()
