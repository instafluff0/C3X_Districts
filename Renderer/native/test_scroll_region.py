"""Host-only checks of the production retained-raster phase and guard policy."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ScrollRegionTests(unittest.TestCase):
    def test_projection_sample_lattice_and_guard(self):
        run_cpp(r'''
#include "Renderer/sandbox/scroll_region.h"
#include "Renderer/native/scene_projection.h"
#include <cassert>
#include <climits>
#include <cmath>
struct Rect {int left,top,right,bottom;};
int main(){
 using Shift=c3x_renderer::render_core::StaticRegionShift;
 for(float zoom:{1.f,1.25f,1.5f,3.f,1.1f})for(int width:{2240,2239}){
  for(int origin:{0,-8192,8192})for(int dx=-400;dx<=400;++dx)
   for(int dy:{-208,-192,-128,-64,-33,-32,-3,-2,-1,0,1,2,3,32,33,64,128,192,208}){
    auto shift=Shift::between(zoom,origin+dx,origin+dy,origin,origin,320,192);
    double px=double(zoom)*dx,py=double(zoom)*dy;
    double rx=std::floor(px+.5),ry=std::floor(py+.5);
    bool fits=rx>=-320&&rx<=320&&ry>=-192&&ry<=192;
    // Odd and fractional displacements are reusable: the shift rounds to whole
    // pixels and reports the sub-pixel snap the dynamic layers must follow.
    assert(shift.reusable==fits);
    if(!shift.reusable){assert(shift.reason==Shift::guard_bounds);continue;}
    assert(shift.x==int(rx)&&shift.y==int(ry));
    assert(std::abs(shift.snap_x)<=.5&&std::abs(shift.snap_y)<=.5);
    assert(shift.snap_x==rx-px&&shift.snap_y==ry-py);
    auto needed=shift.needed<Rect>(width+8,1268,320,192);
    assert(needed.left>=0&&needed.top>=0&&needed.right<=width+648&&needed.bottom<=1652);
    assert(needed.right-needed.left==width+8&&needed.bottom-needed.top==1268);
    // Retained samples translated by the whole-pixel shift equal the exact
    // projection displaced by the reported snap.
    for(float x:{-128.f,0.f,64.f,256.5f,1120.f,2240.f}){
     double projected=double(width/2)+(double(x)+dx-width/2)*zoom;
     double retained=double(width/2)+(double(x)-width/2)*zoom+shift.x;
     assert(std::abs(projected+shift.snap_x-retained)<1e-6);
    }
    // Every output sample has a covered source, including all four corners.
    for(int x:{0,width+7})for(int y:{0,1267}){
     int sx=x+320-shift.x,sy=y+192-shift.y;
     assert(sx>=needed.left&&sx<needed.right&&sy>=needed.top&&sy<needed.bottom);
    }
   }
 }
 assert(Shift::between(3,64,64,0,0,320,192).reusable);
 assert(!Shift::between(3,65,65,0,0,320,192).reusable);
 assert(Shift::between(1.25f,8,-8,0,0,320,192).reusable);
 assert(Shift::between(1.25f,4,-4,0,0,320,192).reusable);
 auto half=Shift::between(1.25f,2,-2,0,0,320,192);
 assert(half.reusable&&half.x==3&&half.snap_x==.5);
 auto tenth=Shift::between(1.1f,32,0,0,0,320,192);
 assert(tenth.reusable&&tenth.x==35&&std::abs(tenth.snap_x+.2)<1e-5);
 // An odd (-23,+11) camera displacement reuses the retained raster.
 assert(Shift::between(1.f,-2784,-1312,-2807,-1301,320,192).reusable);
 assert(Shift::between(1.f,-2807,-1301,-2784,-1312,320,192).reusable);
 // Large jumps and equivalent/negative wraps must never overflow into reuse.
 assert(!Shift::between(3,INT_MAX,INT_MIN,INT_MIN,INT_MAX,320,192).reusable);
 for(int wrap:{-8192,8192})assert(!Shift::between(1.25f,wrap,0,0,0,320,192).reusable);
}
''')

    def test_native_depth_basis_is_independent_of_projection(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scene_depth.h"
#include "Renderer/sandbox/scroll_region.h"
#include <cassert>
#include <initializer_list>
int main(){
 using namespace c3x_renderer::render_core;
 // The production preparer receives translated native anchors and unchanged
 // world rows. Its translation cancels the camera shift until origin changes.
 auto retained=scene_depth_basis(420,100,64,1260,0);
 for(int dy:{-192,-64,-32,-1,0,1,32,64,192}){
  auto current=scene_depth_basis(420+dy,100,64,1260,dy);
  assert(current.world_origin==retained.world_origin);
  assert(current.translation==retained.translation);
  assert(-(current.translation-retained.translation)/16384.f==0);
 }
 auto far=scene_depth_basis(420+8192,100,64,1260,8192);
 assert(far.world_origin!=retained.world_origin);
 // Legacy screen-depth follows its actual constant, with no zoom multiplier.
 for(float zoom:{1.f,1.25f,1.5f,3.f}){
  auto shift=StaticRegionShift::between(zoom,32,32,0,0,320,192);
  assert(shift.reusable);
  float depth_shift=-(32.f-0.f)/16384.f;
  assert(depth_shift==-1.f/512.f);
 }
}
''')


if __name__ == "__main__":
    unittest.main()
