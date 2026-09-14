"""Native observations authorize bounded, exact future-pose work."""
import unittest
from Renderer.native.native_cpp_test import run_cpp

class UnitFramePreparationTests(unittest.TestCase):
    def test_finite_observation_replacement_and_lifecycle(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_frame_preparation.h"
#include <cassert>
using c3x_renderer::render_core::UnitFramePreparation;
int main() {
 UnitFramePreparation queue;c3x_renderer_unit_v1 r{};r.struct_size=sizeof(r);
 std::strcpy(r.unit_key,"unit");r.frame_count=15;r.unit_id=1;
 queue.observe(r,true);c3x_renderer_unit_v1 out[2]{};
 assert(queue.take(out,2)==1 && out[0].action_cursor==1);
 for(int n=0;n<100;++n){r.body_x=n;r.presentation_time_ticks=n*66000;r.presentation_frequency=1000000;queue.observe(r,true);}
 assert(queue.empty()); // Placement/frozen cursors cannot start an autonomous loop.
 r.action_cursor=1;queue.observe(r,true);r.action=2;r.direction=4;queue.observe(r,false);
 assert(queue.take(out,2)==1 && out[0].action==2 && out[0].direction==4 && out[0].action_cursor==2);
 r.action_cursor=14;queue.observe(r,false);assert(queue.empty());
 queue.observe(r,true); // Same observed terminal cursor offers nothing twice.
 r.action_cursor=13;queue.observe(r,true);r.action_cursor=14;queue.observe(r,true);
 assert(queue.take(out,2)==1 && out[0].action_cursor==0);
 r.hour=1;r.season=3;r.display_color_rgb=0x123456;queue.observe(r,true);
 assert(queue.take(out,2)==1 && out[0].hour==1 && out[0].season==3 && out[0].display_color_rgb==0x123456);
 for(int id=0;id<100;++id){r.unit_id=id;queue.observe(r,true);}
 unsigned total=0;while(!queue.empty())total+=queue.take(out,2);assert(total==32);
 queue.clear();queue.observe(r,true);assert(!queue.empty());
 queue.observe(r,true,false);assert(queue.empty()); // Deselection retracts even a queued prediction.
 queue.observe(r,true,true,2);assert(queue.take(out,2)==1 && out[0].action_cursor==1);
 queue.clear();assert(queue.empty());
}
''')
