"""Candidate refinement must bound interpolation and keep complete owner edges."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class SurfaceRefinementTests(unittest.TestCase):
    def test_bounds_watertight_edges_receiver_values_and_atomic_cancellation(self):
        run_cpp(r'''
#include "Renderer/tools/redraw_density_candidate.h"
#include <cassert>
#include <map>
using namespace c3x_renderer::fidelity;
int main(){
 std::vector<MapVertex> grid;
 for(unsigned y=0;y<=64;++y)for(unsigned x=0;x<=64;++x){
  MapVertex v={};v.u=v.world_x=x/64.f;v.v=v.world_y=y/64.f;
  v.x=v.u*128;v.y=v.v*64;v.normal_z=1;v.base_terrain=-9;
  // Smooth relief plus a sharp material/coverage transition through the grid.
  v.world_z=.006f*std::sin(v.u*6)*std::cos(v.v*4);v.material_plains=1;
  v.authored_relief_blend=v.u>.49f && v.u<.52f?.7f:0;
  grid.push_back(v);
 }
 std::vector<MapVertex> mesh,expanded;std::vector<unsigned> indices;
 assert(append_refined_surface_grid(mesh,grid,64,[]{return false;},&indices));
 assert(append_refined_surface_grid(expanded,grid,64,[]{return false;},nullptr));
 assert(indices.size()==expanded.size() && indices.size()<64*64*6);
 std::map<std::pair<unsigned,unsigned>,unsigned> edges;double area=0;
 for(unsigned t=0;t<indices.size();t+=3){
  auto a=mesh[indices[t]],b=mesh[indices[t+1]],c=mesh[indices[t+2]];
  double twice=(b.u-a.u)*(c.v-a.v)-(b.v-a.v)*(c.u-a.u);
  assert(twice>0);area+=twice*.5;
  for(unsigned k=0;k<3;++k){
   auto const&v=mesh[indices[t+k]];
   assert(!std::memcmp(&v,&expanded[t+k],sizeof(v)));
   auto const&fine=grid[unsigned(v.v*64)*65+unsigned(v.u*64)];
   assert(!std::memcmp(&v,&fine,sizeof(v))); // Receiver attributes stay exact.
   unsigned i=indices[t+k],j=indices[t+(k+1)%3];if(i>j)std::swap(i,j);++edges[{i,j}];
  }
  // Probe interior intersections with the original diagonals, including
  // stitched fans whose radial edges are not aligned with the fine grid.
  for(unsigned p=1;p<8;++p)for(unsigned q=1;q<8-p;++q){
   float wa=p/8.f,wb=q/8.f,wc=1-wa-wb;
   float x=(a.u*wa+b.u*wb+c.u*wc)*64,y=(a.v*wa+b.v*wb+c.v*wc)*64;
   unsigned ix=std::min(unsigned(x),63u),iy=std::min(unsigned(y),63u);
   float u=x-ix,v=y-iy;unsigned fa=iy*65+ix,fb=fa+1,fd=fa+65,fc=fd+1;
   auto av=surface_refinement_value(a),bv=surface_refinement_value(b),cv=surface_refinement_value(c);
   auto first=surface_refinement_value(grid[fa]);
   auto second=surface_refinement_value(grid[u>=v?fb:fc]);
   auto third=surface_refinement_value(grid[u>=v?fc:fd]);
   for(unsigned f=0;f<av.size();++f){
    float truth=u>=v?first[f]*(1-u)+second[f]*(u-v)+third[f]*v:
                       first[f]*(1-v)+second[f]*u+third[f]*(v-u);
    float reduced=av[f]*wa+bv[f]*wb+cv[f]*wc;
    assert(std::abs(truth-reduced)<=surface_refinement_limit(f)+.000001f);
   }
  }
 }
 assert(std::abs(area-1)<1e-9);unsigned boundary=0;
 for(auto const&e:edges){assert(e.second==1 || e.second==2);if(e.second==2)continue;
  auto a=mesh[e.first.first],b=mesh[e.first.second];
  assert((a.u==b.u && (a.u==0 || a.u==1)) || (a.v==b.v && (a.v==0 || a.v==1)));
  assert(std::abs(a.u-b.u)+std::abs(a.v-b.v)==1.f/64);++boundary;
 }
 assert(boundary==256);
 auto saved=mesh;auto saved_indices=indices;unsigned calls=0;
 assert(!append_refined_surface_grid(mesh,grid,64,[&]{return ++calls==17;},&indices));
 assert(indices==saved_indices && mesh.size()==saved.size());
 assert(!std::memcmp(mesh.data(),saved.data(),mesh.size()*sizeof(MapVertex)));
 assert(!append_refined_surface_grid(mesh,grid,32,[]{return false;},&indices));
}
''')


if __name__ == '__main__':
    unittest.main()
