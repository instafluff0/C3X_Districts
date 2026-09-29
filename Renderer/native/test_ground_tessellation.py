"""Close terrain must retain curved detail and share watertight patch edges."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class GroundTessellationTests(unittest.TestCase):
    def test_curvature_edges_coverage_and_cancellation(self):
        run_cpp(r'''
#include "Renderer/lab/shared/natural/ground.h"
#include <cassert>
#include <cstring>
#include <map>
using namespace c3x_renderer::fidelity;
int main(){
 auto sample=[](float u,float v){MapVertex p={};p.u=u;p.v=v;
  p.world_x=u;p.world_y=v;p.world_z=.2f*std::sin(9*u)*std::cos(7*v);
  p.normal_x=-1.8f*std::cos(9*u)*std::cos(7*v);
  p.normal_y=1.4f*std::sin(9*u)*std::sin(7*v);p.normal_z=1;
  p.base_terrain=-9;return p;};
 float coarse_error=0,dense_error=0;
 for(unsigned cells:{16u,64u}){
  std::vector<MapVertex> mesh;std::vector<unsigned> indices;
  assert(emit_ground_grid(mesh,sample,[]{return false;},cells,&indices,nullptr,false,64));
  std::vector<MapVertex> expanded;
  assert(emit_ground_grid(expanded,sample,[]{return false;},cells,nullptr,nullptr,false,64));
  assert(expanded.size()==indices.size());
  std::map<std::pair<unsigned,unsigned>,unsigned> edges;
  double area=0;float error=0;
  for(unsigned t=0;t<indices.size();t+=3){
   auto a=mesh[indices[t]],b=mesh[indices[t+1]],c=mesh[indices[t+2]];
   double twice=(b.u-a.u)*(c.v-a.v)-(b.v-a.v)*(c.u-a.u);
   assert(twice>0);area+=twice*.5;
   float truth=sample((a.u+b.u+c.u)/3,(a.v+b.v+c.v)/3).world_z;
   error=std::max(error,std::abs(truth-(a.world_z+b.world_z+c.world_z)/3));
   for(unsigned k=0;k<3;++k){
    assert(!std::memcmp(&mesh[indices[t+k]],&expanded[t+k],sizeof(MapVertex)));
    unsigned i=indices[t+k],j=indices[t+(k+1)%3];if(i>j)std::swap(i,j);
    ++edges[{i,j}];
   }
  }
  assert(std::abs(area-1)<1e-9);
  unsigned boundary=0;
  for(auto const&edge:edges){
   assert(edge.second==1||edge.second==2);
   if(edge.second!=1)continue;
   auto a=mesh[edge.first.first],b=mesh[edge.first.second];
   assert((a.u==b.u&&(a.u==0||a.u==1))||(a.v==b.v&&(a.v==0||a.v==1)));
   assert(std::abs(a.u-b.u)+std::abs(a.v-b.v)==1.f/64);++boundary;
  }
  assert(boundary==256);
  for(unsigned side=0;side<4;++side)for(unsigned i=0;i<=64;++i){
   float t=i/64.f,u=side==0?0:side==1?1:t,v=side==2?0:side==3?1:t;
   auto exact=sample(u,v);unsigned matches=0;
   for(auto const&p:mesh)if(p.u==u&&p.v==v){
    assert(!std::memcmp(&p,&exact,sizeof(p)));++matches;
   }
   assert(matches==1); // Coarse and dense sides sample the same boundary.
  }
  if(cells==16){coarse_error=error;assert(mesh.size()<600);}
  else dense_error=error;
  auto before=mesh;auto before_indices=indices;unsigned calls=0;
  assert(!emit_ground_grid(mesh,sample,[&]{return ++calls==20;},16,&indices,nullptr,false,64));
  assert(indices==before_indices && mesh.size()==before.size());
  assert(!std::memcmp(mesh.data(),before.data(),mesh.size()*sizeof(MapVertex)));
 }
 assert(dense_error<coarse_error*.1f);
}
''')


if __name__ == '__main__':
    unittest.main()
