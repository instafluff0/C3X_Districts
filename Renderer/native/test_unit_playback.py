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
 assert(draw(0) && r.frame_count==90 && r.action_cursor==0 && step==2);
 for(int frame=1;frame<90;++frame) {
  // Native cursor wraps every 15 frames; source playback must not restart.
  r.action_cursor=frame%15;r.frame_count=15;
  assert(draw((frame*1000000LL+29)/30) && r.action_cursor==frame);
 }
 assert(draw(3000000) && r.action_cursor==0); // Exact authored three-second cycle.
 r.action=1;assert(!draw(3100000));assert(r.action_cursor==0 && queue.empty());
 for(int i=1;i<=40;++i) {r.action_cursor=i%15;assert(!draw(3100000+i*66000));assert(r.action_cursor==0 && queue.empty());}
 assert(draw(5800000,true) && r.action_cursor==0); // Selection starts here, no catch-up.
 assert(draw(5866000,true) && r.action_cursor==1 && step==2);
 assert(!draw(5932000) && r.action_cursor==1 && queue.empty());
 assert(draw(10000000,true) && r.action_cursor==1);
 r.unit_id=2;assert(!draw(10000000) && r.action_cursor==0); // Independent instance.
 r.unit_id=1;assert(draw(10100000,true) && r.action_cursor==4);
 r.body_x=900;r.direction=5;r.hour=20;
 assert(draw(10100000,true) && r.action_cursor==4); // Repeated render isn't elapsed time.
 Clip directed;directed.ambient=false;directed.loop=false;r.action=2;r.action_cursor=7;r.frame_count=15;
 assert(playback.resolve(r,directed,false,step) && r.action_cursor==7 && r.frame_count==15);
 r.action=1;assert(draw(10166000,true) && r.action_cursor==0); // New action lifecycle.
 playback.clear();assert(draw(10232000,true) && r.action_cursor==0);
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
