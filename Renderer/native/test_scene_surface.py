"""Circular scene addresses and damage must preserve exact pixel ownership."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class SceneSurfaceTests(unittest.TestCase):
    def test_abandoned_view_cannot_prepare_or_certify_pixels(self):
        source = (Path(__file__).with_name("c3x_renderer.cpp")).read_text()
        def method(signature):
            return "    " + signature + source.split(signature, 1)[1].split("\n    }", 1)[0] + "\n    }\n"
        run_cpp(r'''
#include "Renderer/native/render_core/scene_guard.h"
#include <cassert>
struct Rect {int left,top,right,bottom;};
struct State {
 bool world_preparation=true,shared_scene_surface=true,scene_guard_failed=false,cache_valid=true;
 int scene_guard_pad=8;unsigned contributors=12;
 struct {bool valid=true;void clear(){valid=false;}} geometry_cache;
 struct {bool color=true;} scene_scratch;
 c3x_renderer::render_core::SceneGuard<Rect> scene_guard;
 std::uint64_t scene_static_signature=42,resource_pixel_signature=42;
 void clear_geometry_vertex_buffers(){contributors=0;}
''' + method("bool scene_guard_pending() const {") + method("void discard_scene_view() {") + r'''
};
int main(){
 State s;assert(s.scene_guard.configure(64,48));
 s.scene_guard.commit(s.scene_guard.select({{8,8,56,40}}));
 assert(s.scene_guard_pending());s.discard_scene_view();
 assert(!s.cache_valid && !s.resource_pixel_signature); // no cache-hit path into discarded inputs
 assert(!s.contributors && !s.geometry_cache.valid && !s.scene_static_signature);
 assert(!s.scene_guard_pending()); // idle worker cannot commit an empty draw
 assert(s.scene_guard.pending==s.scene_guard.dirty.size());
 s.geometry_cache.valid=true;s.cache_valid=true;s.contributors=12;
 assert(s.scene_guard_pending() && !s.scene_guard.select({{8,8,56,40}}).empty());
}
''')

    def test_static_guard_coverage_and_sample_transfers(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scene_guard.h"
#include <cassert>
struct Rect {int left,top,right,bottom;};
using namespace c3x_renderer::render_core;
int main(){
 SceneGuard<Rect> guard;assert(!guard.configure(5000,8));assert(guard.configure(53,37));
 auto original=guard.pending;auto visible=guard.select({{16,8,35,29}});assert(!visible.empty());
 guard.commit(visible);assert(guard.pending<original && guard.pending>0);
 assert(guard.select({{16,8,35,29}}).empty());
 while(guard.pending){auto batch=guard.select({},128);assert(!batch.empty());
  unsigned area=0;for(auto r:batch)area+=(r.right-r.left)*(r.bottom-r.top);assert(area<=128);guard.commit(batch);}
 guard.invalidate({{0,0,1,1},{52,36,53,37}});assert(guard.pending==2);
 guard.invalidate_all();assert(guard.pending==original);guard.reset();assert(guard.pending==0);
 assert(guard.configure(1024,1024));auto bounded=guard.select({},128u*1024u);
 assert(bounded.size()==1 && bounded[0].bottom==128);guard.reset();
 for(int w:{16,37,128})for(int h:{19,64})for(int pad:{8,24})
 for(int x:{-123,0,1,999})for(int y:{-111,0,25}){
  std::vector<unsigned> visits(w*h);
  for(auto t:scene_guard_transfers<Rect>(w,h,pad,x,y,{{0,0,w,h}}))
   for(int row=t.rect.top;row<t.rect.bottom;++row)for(int col=t.rect.left;col<t.rect.right;++col){
    assert(++visits[row*w+col]==1);int sx=col-t.x,sy=row-t.y;
    assert(sx>=0 && sx<w+pad*2 && sy>=0 && sy<h+pad*2);
    auto mod=[](int a,int n){return (a%n+n)%n;};
    int lx=mod(col+x,w),ly=mod(row+y,h);
    assert(sx==mod(lx+pad-x,w+pad*2));assert(sy==mod(ly+pad-y,h+pad*2));
   }
  for(auto v:visits)assert(v==1);
 }
}
''')

    def test_incremental_filter_matches_full_circular_convolution(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scene_surface.h"
#include <cassert>
struct Rect {int left,top,right,bottom;};
using namespace c3x_renderer::render_core;
int main(){
 for(int w:{16,37,128})for(int h:{19,64}){
  std::vector<int> scene(w*h),retained(w*h),expected(w*h);
  auto filter=[&](int x,int y){int sum=0;
   for(int dy=-4;dy<=4;++dy)for(int dx=-4;dx<=4;++dx)
    sum+=scene[((y+dy+h)%h)*w+(x+dx+w)%w]*(5-std::abs(dx))*(5-std::abs(dy));
   return sum;
  };
  for(int step=0;step<24;++step){
   // Moving exposed strips, opposite seams, and a one-pixel local edit.
   int x=step*7%w,y=step*11%h;
   std::vector<Rect> changed={{x,0,std::min(w,x+3),h},{0,y,w,std::min(h,y+2)},
                             {w-1,h-1,w,h},{0,0,1,1}};
   for(auto r:changed)for(int j=r.top;j<r.bottom;++j)for(int i=r.left;i<r.right;++i)
    scene[j*w+i]=(step+i*3+j*7)%17;
   auto damage=scene_filter_damage(w,h,changed,4);
   std::vector<unsigned> visits(w*h);
   for(auto r:damage)for(int j=r.top;j<r.bottom;++j)for(int i=r.left;i<r.right;++i){
    assert(++visits[j*w+i]==1);retained[j*w+i]=filter(i,j);
   }
   for(int j=0;j<h;++j)for(int i=0;i<w;++i)expected[j*w+i]=filter(i,j);
   assert(retained==expected);
  }
 }
 assert(scene_filter_damage<Rect>(37,19,{},4).empty());
}
''')

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
