"""Unexplored map edges must hide cut terrain instead of exposing a fringe."""
import unittest
from Renderer.native.native_cpp_test import run_cpp

class UnseenBoundaryTests(unittest.TestCase):
    def test_unseen_is_opaque_and_explored_edge_fades_inward(self):
        run_cpp(r'''
#include "Renderer/native/render_core/visibility_coverage.h"
#include <cassert>
#include <cmath>
using Coverage=c3x_renderer::render_core::VisibilityCoverage;
int main(){
 // Every neighbor combination, including diagonal corners: an unknown
 // center cannot reveal underlay, cliff skirts or protruding terrain triangles.
 for(unsigned pattern=0;pattern<6561;++pattern){
  unsigned cells=0,n=pattern;
  for(unsigned i=0;i<9;++i)if(i!=4){cells|=(n%3)<<(i*2);n/=3;}
  for(float u:{0.f,.01f,.09f,.5f,.91f,.99f,1.f})
   for(float v:{0.f,.01f,.09f,.5f,.91f,.99f,1.f})
    assert(Coverage::coverage(cells,u,v,1)==0.f);
  // Every shared edge beside an unseen neighbor is black along its entire
  // length, even when the diagonal beyond that neighbor is explored.
  cells|=2u<<8;
  for(int edge=0;edge<4;++edge){
   int neighbor[]={3,5,1,7};
   if((cells>>(neighbor[edge]*2))&3u)continue;
   for(float t:{0.f,.01f,.09f,.5f,.91f,.99f,1.f}){
    float u=edge<2?float(edge):t,v=edge<2?t:float(edge-2);
    assert(Coverage::coverage(cells,u,v,1)==0.f);
   }
  }
 }
 unsigned known=0;for(unsigned i=0;i<9;++i)known|=2u<<(i*2);
 for(int edge=0;edge<4;++edge){
  unsigned cells=known;int neighbor[]={3,5,1,7};cells&=~(3u<<(neighbor[edge]*2));
  float prior=-1;
  for(int step=0;step<=100;++step){
   float d=Coverage::feather*step/100.f,u=.5f,v=.5f;
   if(edge==0)u=d;else if(edge==1)u=1-d;else if(edge==2)v=d;else v=1-d;
   float value=Coverage::coverage(cells,u,v,1);
   assert(value>=prior-1e-6f&&value>=0&&value<=1);
   if(!step)assert(value==0);if(step==100)assert(value==1);prior=value;
  }
 }
 // Explored fog still blends between known tiles; fully explored interiors
 // retain their complete terrain footprint, independent of visible state.
 unsigned explored=0;for(unsigned i=0;i<9;++i)explored|=1u<<(i*2);
 for(float u:{0.f,.1f,.5f,.9f,1.f})for(float v:{0.f,.1f,.5f,.9f,1.f})
  assert(Coverage::coverage(explored,u,v,1)==1.f);
 assert(Coverage::coverage(known&~(3u<<6),0,.5f,2)==.5f);
}
''')

if __name__=='__main__':unittest.main()
