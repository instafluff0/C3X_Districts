"""Fog boundaries fade like native Civ III's FogOfWar.pcx.

Civ III draws 81 fog sprites, one per state of the four tiles around a tile
vertex, on the dual grid whose corners are tile centres: clear ground keeps
its light, fog of war 0.6 of it, unexplored none. The user (October 17) found
our fade a semi-opaque band: two half steps met at a flat, half-covered shelf
on the tile edge. VisibilityCoverage::light now interpolates the seen and
explored flags between tile centres, each through one smooth step, and these
tests hold it to curves measured from the sprites (dither averaged along the
edge), with no shelf at the edge.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class UnseenBoundaryTests(unittest.TestCase):
    def test_native_fade_from_fog_of_war_sprites(self):
        run_cpp(r'''
#include "Renderer/native/render_core/visibility_coverage.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
using Coverage=c3x_renderer::render_core::VisibilityCoverage;
// Cells index (v+1)*3+(u+1); state 0 unexplored, 1 fog of war, 2 seen.
unsigned cells(unsigned center,std::initializer_list<int> others,unsigned other){
 unsigned c=center<<8;for(int i:others)c|=other<<(2*i);return c;
}
// Light across a straight edge, t tiles from the darker tile's centre (0)
// to the lighter tile's centre (1): the darker tile owns t<.5.
float across(unsigned dark,unsigned lit,float t){
 return t<.5f?Coverage::light(cells(dark,{2,5,8},lit),.5f+t,.5f):Coverage::light(cells(lit,{0,3,6},dark),t-.5f,.5f);
}
int main(){
 assert(std::fabs(Coverage::fog_light-.6f)<1e-6f);
 // Uniform states: black, fog light and clear.
 for(float u:{.01f,.3f,.5f,.8f,.99f})for(float v:{.01f,.5f,.99f}){
  assert(Coverage::light(cells(0,{0,1,2,3,5,6,7,8},0),u,v)==0);
  assert(std::fabs(Coverage::light(cells(1,{0,1,2,3,5,6,7,8},1),u,v)-.6f)<1e-6f);
  assert(Coverage::light(cells(2,{0,1,2,3,5,6,7,8},2),u,v)==1);
 }
 // FogOfWar.pcx along the dual cell (t, light): unexplored to fog (UFFU),
 // unexplored to clear (USSU) and fog to clear (FSSF).
 struct Sample {float t,light;};
 Sample fogged[]={{.25f,.04f},{.3f,.07f},{.35f,.13f},{.4f,.20f},{.45f,.31f},{.5f,.40f},
  {.55f,.47f},{.6f,.53f},{.65f,.56f},{.7f,.59f},{.75f,.60f}};
 Sample clear[]={{.1f,.07f},{.2f,.27f},{.3f,.59f},{.4f,.86f},{.5f,.98f},{.55f,1}};
 Sample lifted[]={{.1f,.64f},{.2f,.72f},{.3f,.82f},{.4f,.93f},{.5f,.99f},{.55f,1}};
 float worst=0;
 for(auto s:fogged)worst=std::max(worst,std::fabs(across(0,1,s.t)-s.light));
 for(auto s:clear)worst=std::max(worst,std::fabs(across(0,2,s.t)-s.light));
 for(auto s:lifted)worst=std::max(worst,std::fabs(across(1,2,s.t)-s.light));
 std::printf("largest difference from the sprites: %.3f\n",worst);
 assert(worst<.05f);
 // One continuous, monotone step with no shelf: the light still changes
 // quickly where the two tiles meet (two half steps had zero slope there).
 for(auto pair:{std::make_pair(0u,1u),std::make_pair(0u,2u),std::make_pair(1u,2u)}){
  float prior=-1;
  for(int k=0;k<=100;++k){float l=across(pair.first,pair.second,k/100.f);assert(l>=prior-1e-6f);prior=l;}
  assert(std::fabs(across(pair.first,pair.second,.4999f)-across(pair.first,pair.second,.5001f))<1e-3f);
  float range=pair.second==1?.6f:pair.first==1?.4f:1.f;
  float slope=(across(pair.first,pair.second,.51f)-across(pair.first,pair.second,.49f))/.02f/range;
  std::printf("states %u|%u: slope at the edge %.2f of the step per tile\n",pair.first,pair.second,slope);
  assert(slope>.5f);
 }
 // Straight along edges, rounded at corners: a lone seen diagonal corner
 // lights a rounded patch, as along the sprites' dual diagonal (SUUU).
 for(float v:{.2f,.5f,.8f})assert(std::fabs(Coverage::light(cells(0,{5,2,8},1),.9f,v)-Coverage::light(cells(0,{5,2,8},1),.9f,.5f))<1e-6f);
 float diagonal[][2]={{.5f,.5f},{.6f,.18f}}; // distance from the seen centre, light
 for(auto d:diagonal){float a=1-d[0];float light=Coverage::light(cells(0,{0},2),.5f-a,.5f-a);
  std::printf("lone seen corner t=%.1f light=%.2f sprite=%.2f\n",d[0],light,d[1]);assert(std::fabs(light-d[1])<.08f);}
 assert(Coverage::light(cells(0,{0},2),.001f,.001f)<Coverage::light(cells(0,{0,1,3},2),.001f,.001f));
}
''')


if __name__=='__main__':unittest.main()
