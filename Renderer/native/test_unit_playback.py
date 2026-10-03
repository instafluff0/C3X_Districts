"""Caller eligibility and authored clip time, before any preparation admission."""
import json
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class UnitPlaybackTests(unittest.TestCase):
    def test_authored_time_freeze_resume_interruption_and_prediction(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_playback.h"
#include "Renderer/native/render_core/unit_frame_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Clip {bool ambient=true,loop=true;double duration=3;unsigned frames=91;};
int main() {
 UnitPlayback playback;UnitFramePreparation queue;Clip clip;
 c3x_renderer_unit_v1 r{};r.unit_id=1;r.action=13;r.frame_count=15;
 r.presentation_frequency=1000000;std::strcpy(r.unit_key,"worker");unsigned step;
 auto draw=[&](long long t,bool selected=false) {
  r.presentation_time_ticks=t;
  bool advance=playback.resolve(r,clip,selected,step);queue.observe(r,true,advance && step,step);return advance;
 };
 assert(draw(0) && r.frame_count==90 && r.action_cursor==0 && step==2);int origin=r.action_cursor;
 for(int frame=1;frame<90;++frame) {
  // Native cursor wraps every 15 frames; source playback must not restart.
  r.action_cursor=frame%15;r.frame_count=15;
  assert(draw((frame*1000000LL+29)/30) && r.action_cursor==(origin+frame)%90);
 }
 assert(draw(3000000) && r.action_cursor==origin); // Exact authored three-second cycle.
 r.action=1;assert(!draw(3100000));assert(r.action_cursor==0 && queue.empty());
 for(int i=1;i<=40;++i) {r.action_cursor=i%15;assert(!draw(3100000+i*66000));assert(r.action_cursor==0 && queue.empty());}
 assert(draw(5800000,true) && r.action_cursor==0); // Selection starts here, no catch-up.
 assert(draw(5866000,true) && r.action_cursor==1 && step==2);
 assert(!draw(5932000) && r.action_cursor==1 && queue.empty());
 assert(draw(10000000,true) && r.action_cursor==1); // Resuming an unselected idle stays frozen until selected.
 r.unit_id=2;assert(!draw(10000000) && r.action_cursor==0); // Independent instance.
 r.unit_id=1;assert(draw(10100000,true) && r.action_cursor==4);
 r.body_x=900;r.direction=5;r.hour=20;
 assert(draw(10100000,true) && r.action_cursor==4); // Repeated render isn't elapsed time.
 Clip directed;directed.ambient=false;directed.loop=false;r.action=2;r.action_cursor=7;r.frame_count=15;
 assert(playback.resolve(r,directed,false,step) && r.action_cursor==7 && r.frame_count==15);
 r.action=1;assert(draw(10166000,true) && r.action_cursor==0); // New action lifecycle.
 playback.clear();assert(draw(10232000,true) && r.action_cursor==0);
 // A visible selected/work loop advances even across a slow native transaction.
 assert(draw(10732000,true) && r.action_cursor==15);
 r.action=13;r.action_cursor=0;r.frame_count=15;assert(draw(11000000)&&r.action_cursor==0);origin=r.action_cursor;
 assert(draw(11700000) && r.action_cursor==(origin+21)%90);
 assert(draw(15700000) && r.action_cursor==(origin+51)%90); // modulo authored duration, no catch-up draws
 playback.forget(r.unit_id);r.action_cursor=5;r.frame_count=15;
 assert(draw(20000000) && r.action_cursor==30); // new observation uses native phase

}
''')

    def test_all_worker_actions_preserve_native_start_and_existing_phases(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_playback.h"
#include <cassert>
#include <set>
using namespace c3x_renderer::render_core;
struct Clip {bool ambient=true,loop=true;double duration=3;unsigned frames=91;};
int main(){
 Clip clip;
 for(int action:{11,13,14,15,16,17,18}){
  UnitPlayback visible,reordered;std::set<int> phases;int first[15]={};
  c3x_renderer_unit_v1 request{};request.action=action;request.presentation_frequency=1000000;
  request.presentation_time_ticks=3500000;std::strcpy(request.unit_key,"worker");unsigned step;
  // Native off-screen/load setup supplies independent cursors. Preserve each
  // normalized phase even though the custom clip has a different frame count.
  for(int id=0;id<15;++id){request.unit_id=id;request.action_cursor=id;request.frame_count=15;
   assert(visible.resolve(request,clip,false,step));
   first[id]=request.action_cursor;assert(first[id]==id*6);phases.insert(first[id]);}
  assert(phases.size()==15);
  for(int id=14;id>=0;--id){request.unit_id=id;request.body_x=id*300;request.direction=5;
   request.action_cursor=id;request.frame_count=15;
   request.projection_scale_milli=1500;assert(reordered.resolve(request,clip,false,step));
   assert(request.action_cursor==first[id]);} // viewport order/position/zoom are irrelevant
  request.presentation_time_ticks+=900000;
  for(int id=0;id<15;++id){request.unit_id=id;request.action_cursor=0;request.frame_count=15;
   assert(visible.resolve(request,clip,false,step));
   int wanted=(first[id]+27)%90;assert(request.action_cursor==wanted);
   assert(visible.resolve(request,clip,false,step)&&request.action_cursor==wanted);
  }
  // A camera revisit retains the authored cycle even if native cursors wrapped.
  request.presentation_time_ticks+=3000000;
  for(int id=0;id<15;++id){request.unit_id=id;request.action_cursor=0;request.frame_count=15;
   assert(visible.resolve(request,clip,false,step)&&request.action_cursor==(first[id]+27)%90);}
  // Fresh visible jobs start at zero, regardless of ID or absolute clock.
  for(int id:{100,201,302}){request.unit_id=id;request.action_cursor=0;request.frame_count=15;
   assert(visible.resolve(request,clip,false,step)&&request.action_cursor==0);
   request.presentation_time_ticks+=500000;}
  // Actual retirement and reset accept a new authoritative native phase.
  visible.forget(7);request.unit_id=7;request.action_cursor=5;request.frame_count=15;
  assert(visible.resolve(request,clip,false,step)&&request.action_cursor==30);
  visible.clear();request.action_cursor=0;request.frame_count=15;
  assert(visible.resolve(request,clip,false,step)&&request.action_cursor==0);
  // Invalid initial native phase must not synthesize a new work performance.
  visible.clear();request.frame_count=0;assert(!visible.resolve(request,clip,false,step));
  request.frame_count=15;request.action_cursor=-1;assert(!visible.resolve(request,clip,false,step));
  request.frame_count=65537;request.action_cursor=0;assert(!visible.resolve(request,clip,false,step));
 }
 // Even looped native combat and generic BUILD keep the supplied action cursor.
 UnitPlayback playback;c3x_renderer_unit_v1 request{};unsigned step;
 request.presentation_frequency=1000000;request.presentation_time_ticks=3500000;
 for(int action:{2,3,4,5,6,7,8,9,10,12})for(int id:{1,2,3}){
  request.unit_id=id;request.action=action;request.action_cursor=7;request.frame_count=15;
  assert(playback.resolve(request,clip,false,step)&&request.action_cursor==7&&request.frame_count==15);
 }
 clip.ambient=false;request.action=13;request.action_cursor=7;request.frame_count=15;
 assert(playback.resolve(request,clip,false,step)&&request.action_cursor==7&&request.frame_count==15);
}
''')

    def test_builder_pack_retains_source_sample_rate(self):
        bindings=json.loads((ROOT/'Renderer/packs/UnitAnimationFidelity/bindings.json').read_text())
        unit=next(u for u in bindings.values() if isinstance(u,dict) and u.get('key0')=='PRTO_Worker')
        for action in ['road','mine','irrigate','forest','jungle','plant','fortress']:
            clip=unit[action]
            self.assertEqual(clip['ambient'],1)
            self.assertAlmostEqual((clip['frames']-1)/clip['duration'],30,places=4)

if __name__=='__main__':unittest.main()
