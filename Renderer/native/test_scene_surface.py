"""Circular scene addresses and damage must preserve exact pixel ownership."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class SceneSurfaceTests(unittest.TestCase):
    def test_wrapping_damage_bounds_and_stationary_world_addresses(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scene_surface.h"
#include <cassert>
struct Rect {int left,top,right,bottom;};
using namespace c3x_renderer::render_core;
int main(){
 assert(scene_surface_extent(8,8) && scene_surface_extent(2240,1192));
 assert(!scene_surface_extent(2241,1192) && !scene_surface_extent(2240,1193));
 assert(!scene_surface_extent(0,480) && !scene_surface_extent(640,7));
 for(int width:{16,37,128})for(int height:{19,64})
 for(int ox:{-1000,-1,0,1,900})for(int oy:{-999,0,800}){
  auto spans=scene_spans<Rect>(width,height,ox,oy);
  assert(spans.size()>=1 && spans.size()<=4);
  std::vector<unsigned> physical(width*height),logical(width*height);
  for(auto s:spans)for(int y=s.rect.top;y<s.rect.bottom;++y)for(int x=s.rect.left;x<s.rect.right;++x){
   int lx=x-s.x,ly=y-s.y;
   assert(lx>=0 && lx<width && ly>=0 && ly<height);
   ++physical[y*width+x];++logical[ly*width+lx];
   assert(((lx-ox)%width+width)%width==x);
   assert(((ly-oy)%height+height)%height==y);
  }
  for(auto v:physical)assert(v==1);for(auto v:logical)assert(v==1);
 }
 std::vector<Rect> input={{-4,-3,11,17},{7,8,26,24},{15,16,17,18},{0,0,0,0},{99,99,101,102}};
 auto result=scene_damage_union(37,27,input);std::vector<unsigned> coverage(37*27);
 for(auto r:result){
  assert(r.left>=0 && r.top>=0 && r.right<=37 && r.bottom<=27);
  for(int y=r.top;y<r.bottom;++y)for(int x=r.left;x<r.right;++x)assert(++coverage[y*37+x]==1);
 }
 for(auto r:input)for(int y=std::max(0,r.top);y<std::min(27,r.bottom);++y)
  for(int x=std::max(0,r.left);x<std::min(37,r.right);++x)assert(coverage[y*37+x]);
 assert(scene_damage_union(0,27,input).empty());
 assert(scene_damage_union(2249,1200,input).empty());
 input.clear();for(int y=0;y<1200;y+=8)for(int x=(y/8)%2*8;x<2248;x+=16)input.push_back({x,y,x+1,y+1});
 result=scene_damage_union(2248,1200,input);assert(result.size()*sizeof(Rect)<340*1024);
}
''')


if __name__ == "__main__":
    unittest.main()
