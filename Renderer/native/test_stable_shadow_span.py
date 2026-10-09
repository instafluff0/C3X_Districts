"""Scrolling never changes the shadow sampling span.

The span (shadow texel density) followed the receivers the view contained,
with a 1.15x-1.45x hysteresis band. Passing coasts and unexplored land shrinks
and grows that extent, so busy scrolls refitted: 11 of 58 shadow builds, 8 of
them redrawing all 25 pages, and 30 of 53 step frames showing a stale static
layer while it refined (performance review, section 25). The span now comes
from the receiver region's size and the light, which do not change with the
camera.
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp


class StableShadowSpanTests(unittest.TestCase):
    def test_span_holds_while_the_receiver_extent_changes_under_a_scroll(self):
        run_cpp(r'''
#include "Renderer/native/render_core/shadow_sampling_grid.h"
#include <cassert>
#include <cmath>
#include <cstdio>
using c3x_renderer::render_core::StableShadowSpan;
using c3x_renderer::render_core::ShadowSamplingGrid;
// The previous rule: refit to 1.15x the receivers' extent whenever it left
// the [extent, 1.45 x extent] band.
struct Hysteresis {std::array<float,2> span{};unsigned refits=0;
 void update(float const* needed,float guard){for(unsigned a=0;a<2;++a){float want=needed[a+2]-needed[a]+2*guard;
  if(!(span[a]>=want&&span[a]<=want*1.45f)){span[a]=std::ceil(want*1.15f/2.f)*2.f;++refits;}}}};
int main(){
 // A tilted light: u mixes x, y and height; v mostly follows x-y.
 std::array<float,12> light={.70f,.70f,-.45f,0, .55f,-.55f,.62f,0, 0,0,1,0};
 std::array<float,2> region={64,96};float height=2.5f;float guard=ShadowSamplingGrid::guard;
 StableShadowSpan stable;Hysteresis old;
 std::array<float,2> first{};
 for(int step=0;step<200;++step){
  // The receivers fill 55-100% of the region as the view passes coasts and
  // unexplored land, and the region moves with the camera.
  float fill=.55f+.45f*float((step*37)%100)/100.f,shift=float(step)*1.5f;
  float u=StableShadowSpan::region_extent(light,0,region,height)*fill,v=StableShadowSpan::region_extent(light,1,region,height)*fill;
  float needed[4]={shift,shift*.5f,shift+u,shift*.5f+v};
  assert(stable.update(needed,light,region,height,guard));old.update(needed,guard);
  if(step==0)first=stable.span;
  assert(stable.span==first);
  // Every receiver still fits, and the span configures a valid page window.
  float bounds[4]={needed[0],needed[1],needed[0]+stable.span[0]-2*guard,needed[1]+stable.span[1]-2*guard};
  ShadowSamplingGrid grid;assert(grid.configure(bounds,light)&&grid.quality_span==stable.span);
 }
 assert(stable.refits==2); // one per axis, at the first build
 assert(old.refits>10);    // the previous rule refitted while scrolling
 // A receiver that reaches past the region grows the span once; a new region
 // (zoom) or light starts again.
 float tall[4]={0,0,first[0]+20,10};assert(stable.update(tall,light,region,height,guard));
 assert(stable.span[0]>first[0]&&stable.refits==3);
 auto grown=stable.span;float small[4]={0,0,10,10};assert(stable.update(small,light,region,height,guard)&&stable.span==grown);
 assert(stable.update(small,light,{32,48},height,guard)&&stable.span[0]<first[0]);
 // Returning to a zoom keeps its grown span: no second growth or refit.
 auto before=stable.refits;assert(stable.update(small,light,region,height,guard)&&stable.span==grown&&stable.refits==before);
 assert(stable.update(small,light,{32,48},height,guard)&&stable.refits==before);
 light[0]+=.01f;auto refits=stable.refits;assert(stable.update(small,light,{32,48},height,guard)&&stable.refits==refits+2);
 float broken[4]={0,0,NAN,1};assert(!stable.update(broken,light,{32,48},height,guard));
 std::printf("PASS stable shadow span: scroll_refits=0 previous_rule_refits=%u\n",old.refits);
}
''')


if __name__ == '__main__':
    unittest.main()
