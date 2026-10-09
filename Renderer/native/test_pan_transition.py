"""Camera steps slide into place, cruise while scrolling and ease to rest.

A published step is first shown at the previous camera's position and slides
to rest; the previous world (underlay) keeps its position relative to the new
one. An isolated step (a recentre) eases in and out. Steps that follow within
the restart window cruise at a constant speed that carries across steps, and
the last one glides to rest on a cubic ease-out instead of stopping dead.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class PanTransitionTests(unittest.TestCase):
    def test_shift_eases_scroll_cruises_and_glides_to_rest(self):
        run_cpp(r'''
#include "Renderer/native/pan_transition.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
using c3x_renderer::PanTransition;
int main(){
 const long long f=1000; // ticks per second
 PanTransition pan;assert(!pan.sample(0,f).active);
 // An isolated step eases in and out: full offset at first, barely moving
 // after 10 ms, half way at half of its .28 s + .35 ms/px duration.
 pan.begin(160,-80,1000,f);auto o=pan.sample(1000,f);
 assert(o.active&&o.x==160&&o.y==-80&&o.under_x==0&&o.under_y==0);
 o=pan.sample(1010,f);assert(o.x>=159&&o.under_x==o.x-160&&o.under_y==o.y+80);
 o=pan.sample(1171,f);assert(std::abs(o.x-80)<=2&&std::abs(o.y+40)<=1);
 o=pan.sample(1330,f);assert(std::abs(o.x)<=2);
 assert(!pan.sample(1344,f).active&&!pan.moving());
 // Scrolling: 100 px every 400 ms after a first, isolated step.
 long long t=5000;pan.begin(100,0,t,f);
 for(int step=1;step<10;++step){
  t+=400;
  // Speed just before and just after the step arrives stays continuous.
  int before=pan.sample(t-20,f).x-pan.sample(t,f).x;
  int previous=pan.sample(t,f).x;pan.begin(100,0,t,f);o=pan.sample(t,f);
  assert(o.x==previous+100&&o.under_x==o.x-100);
  int after=o.x-pan.sample(t+20,f).x;
  if(step>=6)assert(std::abs(before-after)<=1&&after>=4&&after<=6); // 100 px / 400 ms = 5 px per 20 ms
 }
 // No further step: the last one glides to rest. Speed falls steadily and
 // the offset reaches zero only after the tail.
 int last=pan.sample(t,f).x,speed=1000;bool slowed=false;
 for(long long dt=20;dt<1000;dt+=20){
  o=pan.sample(t+dt,f);if(!o.active)break;
  int moved=last-o.x;assert(moved>=0&&moved<=speed+1);
  if(moved<speed)slowed=true;speed=moved;last=o.x;
 }
 // 400 ms cruise, then a 400 ms tail (reserve .25 at this interval).
 assert(slowed&&pan.sample(t+790,f).x<=1&&!pan.sample(t+801,f).active&&!pan.moving());
 // A long pause restarts with an eased shift.
 pan.begin(50,0,20000,f);o=pan.sample(20010,f);assert(o.x>=49);
 // Zero steps and cancellation leave nothing active.
 pan.begin(0,0,30000,f);assert(!pan.moving());pan.begin(40,0,31000,f);pan.cancel();assert(!pan.sample(31001,f).active);
 std::printf("PASS pan transition: shift=ease-in-out scroll=cruise speed_continuous=1 glide=cubic restart=1\n");
}
''')


class GlideTrailingStripTests(unittest.TestCase):
    def test_trailing_strip_comes_from_the_previous_world_not_the_slid_output(self):
        # While a slide runs, the selected world's output is the slid
        # composite, whose trailing strip is itself an older world. Copying
        # that output for the next step nested the strips: during continuous
        # scrolling the trailing edge showed repeated vertical slices or black
        # (October 8, v1 and v3 frames, most visible at 2x and 3x).
        from Renderer.lab.platform import ROOT
        session = (ROOT / "Renderer/native/gpu_composition_session.h").read_text()
        world = session[session.index("    void world(Command const& input){"):session.index("public:\n    Session(")]
        begin = world[world.index("if(c.kind==Kind::world_begin){"):world.index("}else if(world_destination!=c.destination")]
        self.assertIn("layers.copy_world_inputs(pan_under)", begin)
        self.assertIn("!resized", begin)
        self.assertIn("pan_under_scale=layers.view_scale();", begin)
        # A slide whose trailing world was copied before a zoom step would
        # show it at the wrong scale (a mismatched seam after zooming to 3x).
        start = session[session.index("    void start_pan(){"):session.index("pan.begin(pan_step_x,pan_step_y")]
        self.assertIn("std::abs(layers.view_scale()-pan_under_scale)>1e-6", start)
        self.assertNotIn("copy_selected_world", session)
        layers = (ROOT / "Renderer/native/retained_composition.h").read_text()
        copy = layers[layers.index("    bool copy_world_inputs(Texture out[2]){"):layers.index("    void select_world(")]
        self.assertIn("assemble(n->inputs[i],drawn_ticks,drawn_frequency", copy)
        self.assertIn("exact_plane(n->inputs[i],n->area)", copy)
        self.assertNotIn("CopyResource(out[i].Get(),n->output[i].Get())", copy)
        # Steps stay in screen pixels at the presented zoom (the selected world
        # is the zoomed view); Civ III's native steps are 128, 128 and 84 px at
        # 1x, 2x and 3x.
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        glide = source[source.index("// A small camera step slides into place"):source.index("session.camera_step(step?dx:0,step?dy:0);")]
        self.assertIn("*presented/65536.", glide)
        # A zoom re-anchors Civ III's camera; sliding that move showed seams
        # at the screen edge during zoom-in notches (October 8, v5 against v6).
        self.assertIn("bool same_zoom=presented==pan_origin_zoom;", glide)
        self.assertIn("&&settled&&same_zoom&&", glide)
        self.assertIn("pan_origin_zoom=presented;", source[source.index("session.camera_step(step?dx:0,step?dy:0);"):][:200])

    def test_glide_is_on_by_default(self):
        # The user chose the glide as the default display (October 8);
        # C3X_RENDERER_GLIDE=0 opts out for A/B runs.
        from Renderer.lab.platform import ROOT
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        glide = source[source.index("// A small camera step slides into place"):source.index("session.camera_step(step?dx:0,step?dy:0);")]
        self.assertIn('bool enabled=!(c3x_renderer::render_core::cached_environment("C3X_RENDERER_GLIDE",glide,sizeof(glide))&&glide[0]==\'0\');', glide)

if __name__ == '__main__':
    unittest.main()
