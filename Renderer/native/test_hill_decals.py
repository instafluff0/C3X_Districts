"""Rock patches must reuse receiver positions, triangulation and normals."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class HillDecalTests(unittest.TestCase):
    def test_steep_surface_indexed_and_expanded_agree_without_intersections(self):
        run_cpp(r'''
#include "Renderer/lab/shared/natural/hill_decals.h"
#include <cassert>
#include <cstring>
using namespace c3x_renderer::fidelity;
int main(){
 GroundProjection project{4,-2,128,64,.8f,800};
 auto height=[](float x,float y,float*s){if(s)*s=1;return 2.5f+50*std::sin(x*5)*std::cos(y*7);};
 struct Shore{float distance=100,beach_width=0,rocky=0;};
 auto shore=[](float,float){return Shore{};};
 auto weights=[](float,float){return std::array<float,5>{1,0,0,0,0};};
 auto sample=[&](float u,float v){return ground_surface(project,u,v,height,shore,weights);};
 unsigned emitted=0;
 for(unsigned divisions:{16u,32u,64u}){
  std::vector<MapVertex> ground,expanded;std::vector<unsigned> indices;PatchLayouts layouts;
  assert(emit_ground_grid(ground,sample,[]{return false;},divisions,&indices,&layouts.get(divisions),false));
  for(auto i:indices)expanded.push_back(ground[i]);
  for(int seed=0;seed<20;++seed){
   Tile owner{seed*2,seed*4,4,-2,5};std::vector<MapVertex> a,b;
   emit_hill_decals(owner,4,-2,ground,&indices,a);
   emit_hill_decals(owner,4,-2,expanded,nullptr,b);
   assert(a.size()==b.size());
   if(!a.empty())assert(!std::memcmp(a.data(),b.data(),a.size()*sizeof(MapVertex)));
   for(std::size_t i=0;i<a.size();i+=3){
    bool receiver=false;
    for(std::size_t j=0;j<indices.size();j+=3){
     bool match=true;
     for(unsigned k=0;k<3;++k){auto const&v=a[i+k];auto const&g=ground[indices[j+k]];
      match&=v.x==g.x && v.y==g.y && v.z==g.z && v.world_z==g.world_z &&
       v.normal_x==g.normal_x && v.normal_y==g.normal_y && v.normal_z==g.normal_z;
      assert(v.material_plains==2 && v.material_desert>=0 && v.material_desert<=2);
     }
     receiver|=match;
    }
    assert(receiver);++emitted;
   }
  }
 }
 assert(emitted>0);
}
''')


if __name__ == '__main__':
    unittest.main()
