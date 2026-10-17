"""Fog boundaries blend across tile edges like native Civ III.

The user's reference (October 17): unexplored black and fog of war blend
across the tile edge, half at the edge and complete a fifth (unexplored) or
a quarter (fog of war) of a tile into either tile, straight along edges and
rounded at corners. Unexplored tiles used to stay opaque with the whole fade
on the explored side, because nothing was drawn under them; explored tiles
now extend flat ground into that rim (test_unexplored_apron).
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class UnseenBoundaryTests(unittest.TestCase):
    def test_native_blend_across_the_edge(self):
        run_cpp(r'''
#include "Renderer/native/render_core/visibility_coverage.h"
#include <cassert>
#include <cmath>
#include <initializer_list>
using Coverage=c3x_renderer::render_core::VisibilityCoverage;
// Cells index (v+1)*3+(u+1): 3 is the u-1 side, 1 the v-1 side, 0 their diagonal.
// Cells not listed are unexplored.
unsigned cells(unsigned center,std::initializer_list<int> others,unsigned other){
 unsigned c=center<<8;for(int i:others)c|=other<<(2*i);return c;
}
bool near(float a,float b){return std::fabs(a-b)<1e-5f;}
int main(){
 float const s=Coverage::shroud;assert(near(s,.2f) && near(Coverage::feather,.25f));
 // Deep inside an unexplored region: black everywhere.
 unsigned dark=cells(0,{0,1,2,3,5,6,7,8},0);
 for(float u:{0.f,.1f,.5f,.9f,.999f})for(float v:{0.f,.1f,.5f,.9f,.999f})assert(Coverage::coverage(dark,u,v,1)==0);
 // A straight boundary: the u-1 column unexplored, the rest explored.
 unsigned explored_side=cells(1,{1,2,5,7,8},1);
 float prior=-1;
 for(int k=0;k<=40;++k){float u=k/100.f;
  float c=Coverage::coverage(explored_side,u,.5f,1);
  for(float v:{.3f,.5f,.7f})assert(near(Coverage::coverage(explored_side,u,v,1),c)); // straight
  assert(c>=prior);prior=c;
 }
 assert(near(Coverage::coverage(explored_side,0,.5f,1),.5f) && near(Coverage::coverage(explored_side,s,.5f,1),1));
 // The unexplored tile across that edge mirrors it: half at the edge, black
 // a fifth of a tile in. Its rim shows the ground explored tiles extend.
 unsigned mirrored=cells(0,{5,8,2},1);
 assert(near(Coverage::coverage(mirrored,.999999f,.5f,1),.5f) && Coverage::coverage(mirrored,1-s,.5f,1)<1e-6f);
 assert(near(Coverage::coverage(mirrored,1-s/2,.5f,1),1-Coverage::coverage(explored_side,s/2,.5f,1)));
 // Corners are rounded: explored with three unexplored neighbours around a
 // corner is a quarter covered there, not a half like the straight edge.
 unsigned concave=cells(1,{2,5,6,7,8},1); // diagonal and both sides unexplored
 assert(near(Coverage::coverage(concave,0,0,1),.25f) && near(Coverage::coverage(concave,0,.5f,1),.5f));
 unsigned convex=cells(1,{1,2,3,5,6,7,8},1); // only the diagonal unexplored
 assert(near(Coverage::coverage(convex,0,0,1),.75f) && Coverage::coverage(convex,0,.5f,1)==1);
 // Fog of war blends the same way over a quarter tile.
 unsigned fogged=cells(1,{3,0,6},2),visible=cells(2,{5,2,8},1);
 assert(near(Coverage::coverage(fogged,0,.5f,2),.5f) && near(Coverage::coverage(visible,.999999f,.5f,2),.5f));
 assert(Coverage::coverage(fogged,.25f,.5f,2)<1e-6f && Coverage::coverage(visible,.75f,.5f,2)>1-1e-6f);
}
''')


if __name__=='__main__':unittest.main()
