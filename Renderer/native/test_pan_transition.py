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
    def test_every_pixel_shows_the_presented_camera_when_steps_overlap(self):
        # One screen row composed as RetainedComposition does: the trailing
        # image is shifted by under_x, then the current world by x; pixels
        # neither copy writes keep the previous frame (stale). On the busy
        # save steps arrive every 230-750 ms, often before the previous slide
        # has finished; the trailing strip then needs the older world.
        # Rules for the trailing image copied at the next step's world_begin:
        #  0 the previous camera's resting world at x - step: a strip as wide
        #    as the unfinished offset stayed stale (October 8, a0);
        #  1 the last drawn frame at its offset: when no frame ran between a
        #    commit and the next step it is a step behind (one-step seams, g1);
        #  2 the world composed as the next frame would show it (production).
        run_cpp(r'''
#include "Renderer/native/pan_transition.h"
#include <cassert>
#include <cstdio>
#include <vector>
using c3x_renderer::PanTransition;
constexpr int W=640,stale=-1000000;
// A world image at camera c shows world position s+c at screen pixel s.
std::vector<int> world(int c){std::vector<int> r(W);for(int s=0;s<W;++s)r[s]=s+c;return r;}
void shifted(std::vector<int>& to,std::vector<int> const& from,int x){
 for(int s=0;s<W;++s){int t=s-x;if(t>=0&&t<W)to[s]=from[t];}}
std::vector<int> compose(PanTransition::Offset o,std::vector<int> const& under,std::vector<int> const& current){
 if(!o.active)return current;std::vector<int> frame(W,stale);shifted(frame,under,o.under_x);shifted(frame,current,o.x);return frame;}
// Wrong pixels over a scroll: steps after `gaps` ms, a frame every `frame` ms.
int run(std::vector<int> const& gaps,int step,int rule,int frame_ms){
 const long long f=1000;PanTransition pan;
 int camera=0,bad=0,out_offset=0,copied=0;auto current=world(0),under=world(0),out=world(0);std::vector<int> copy;
 long long next=1000+gaps[0],commit=0,last_frame=0;std::size_t arrived=0;bool pending=false;
 for(long long t=1000;t<next+3000||arrived<gaps.size();++t){
  if(arrived<gaps.size()&&t>=next){
   // world_begin: the trailing image is taken before the new world records.
   auto shown=pan.sample(t,f);
   if(rule==0){copy=current;copied=0;}
   else if(rule==1){copy=out;copied=out_offset;}
   else{copy=compose(shown,under,current);copied=shown.active?shown.x:0;}
   pending=true;commit=t+30;++arrived;next=arrived<gaps.size()?t+gaps[arrived]:t+1000000;
  }
  if(pending&&t>=commit){
   // commit_display: the new world arrives and the slide starts.
   pending=false;camera+=step;current=world(camera);under=copy;
   pan.begin(step,0,t,f,rule==0?0:copied,0);
  }
  if(t-last_frame<frame_ms)continue;
  last_frame=t;auto o=pan.sample(t,f);auto drawn=compose(o,under,current);
  for(int s=0;s<W;++s)if(drawn[s]!=s+camera-o.x)++bad;
  out=drawn;out_offset=o.active?o.x:0;
 }
 return bad;
}
int main(){
 std::vector<int> busy={300,230,750,310,260,600,240,420,330,280,700,250};
 std::vector<int> native(24,78);
 for(int step:{128,-128,84}){
  for(int frame_ms:{16,61}){
   assert(run(busy,step,2,frame_ms)==0);assert(run(native,step,2,frame_ms)==0);
   assert(run(busy,step,0,frame_ms)>0);assert(run(native,step,0,frame_ms)>0);
  }
  // Sparse frames (a commit with no frame before the next step) misplace
  // the last drawn frame by a step.
  assert(run(native,step,1,61)>0);
 }
 std::printf("PASS glide trailing strip: composed=exact resting_world=stale last_frame=misplaced\n");
}
''')

    def test_trailing_image_is_composed_as_the_next_frame_would_show_it(self):
        from Renderer.lab.platform import ROOT
        session = (ROOT / "Renderer/native/gpu_composition_session.h").read_text()
        world = session[session.index("    void world(Command const& input){"):session.index("public:\n    Session(")]
        begin = world[world.index("if(c.kind==Kind::world_begin){"):world.index("}else if(world_destination!=c.destination")]
        self.assertIn("auto shown=pan.sample(now.QuadPart,frequency.QuadPart);if(zoom->moving())shown={};", begin)
        self.assertIn("layers.copy_world_output(pan_next,pan_under_offset_x,pan_under_offset_y,\n"
                      "                    shown.under_x,shown.under_y,shown.active?pan_under:nullptr)", begin)
        self.assertIn("!resized", begin)
        self.assertIn("pan_under_scale=layers.view_scale();", begin)
        # A slide whose trailing world was copied before a zoom step would
        # show it at the wrong scale (a mismatched seam after zooming to 3x).
        start = session[session.index("    void start_pan(){"):session.index("    // Private retained IDs")]
        self.assertIn("std::abs(layers.view_scale()-pan_under_scale)>1e-6", start)
        # The running slide keeps its own trailing image until the next starts.
        self.assertLess(start.index("pan.begin(pan_step_x,pan_step_y,now.QuadPart,frequency.QuadPart,pan_under_offset_x,pan_under_offset_y);"),
                        start.index("std::swap(pan_under[0],pan_next[0]);std::swap(pan_under[1],pan_next[1]);"))
        layers = (ROOT / "Renderer/native/retained_composition.h").read_text()
        copy = layers[layers.index("    bool copy_world_output(Texture out[2],int x,int y,int under_x,int under_y,Texture const* under){"):layers.index("    void select_world(")]
        self.assertIn("if(slid){shifted(out[i].Get(),under[i].Get(),under_x,under_y);shifted(out[i].Get(),plane,x,y);}", copy)
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
