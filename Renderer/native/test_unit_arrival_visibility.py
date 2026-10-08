"""Copied native reveals follow visual arrival without delaying loss of sight."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class UnitArrivalVisibilityTests(unittest.TestCase):
    def test_arrival_steps_camera_scope_and_visibility_loss(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_arrival_visibility.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 UnitArrivalVisibility gate;
 c3x_renderer_tile_v1 tiles[3]{};
 for(int i=0;i<3;++i){tiles[i].tile_x=i*2;tiles[i].tile_y=4;
  tiles[i].tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_RENDER;}
 constexpr unsigned explored=C3X_RENDERER_TILE_EXPLORED,visible=C3X_RENDERER_TILE_VISIBLE;
 tiles[0].tile_flags|=explored|visible;
 c3x_renderer_frame_v1 frame{};frame.tiles=tiles;frame.tile_count=3;
 auto sample=[&]{gate.capture(frame,11);return gate.sample(frame,11);};
 assert(sample().tiles[0].tile_flags&visible);
 gate.pending={{7,10}};tiles[1].tile_flags|=explored|visible;
 assert(!(sample().tiles[1].tile_flags&explored)); // ready terrain stays covered during travel
 tiles[1].anchor_x=300; // scrolling reprojects the same pending reveal
 assert(!(sample().tiles[1].tile_flags&explored));
 gate.pending.push_back({7,20});tiles[2].tile_flags|=explored|visible;
 assert(!(sample().tiles[2].tile_flags&explored));
 gate.pending.erase(gate.pending.begin());
 auto first=sample();assert(first.tiles[1].tile_flags&visible);
 assert(!(first.tiles[2].tile_flags&explored)); // first arrival cannot reveal the second step
 gate.pending.clear();assert(sample().tiles[2].tile_flags&visible);
 gate.pending={{8,30}};tiles[0].tile_flags&=~visible;
 assert(!(sample().tiles[0].tile_flags&visible)); // native loss never waits
 tiles[0].tile_flags|=visible;assert(!(sample().tiles[0].tile_flags&visible));
 tiles[0].tile_flags&=~(visible|explored);assert(!(sample().tiles[0].tile_flags&explored));
 gate.pending.clear();assert(!(sample().tiles[0].tile_flags&explored));
 tiles[0].tile_flags|=visible|explored;gate.pending={{9,40}};sample();
 assert(gate.sample(frame,12).tiles[0].tile_flags&visible); // viewer/map scope discards old gates
 // No native input is changed and no unexplored tile can become visible.
 tiles[2].tile_flags&=~(visible|explored);assert(!(sample().tiles[2].tile_flags&explored));
 assert(tiles[1].tile_flags&visible);
}
''')

    def test_late_render_does_not_attach_an_earlier_reveal_to_a_future_step(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_arrival_visibility.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 UnitArrivalVisibility gate;c3x_renderer_tile_v1 tile{};
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
 c3x_renderer_frame_v1 frame{};frame.tiles=&tile;frame.tile_count=1;
 frame.presentation_frequency=1000;frame.presentation_time_ticks=1;
 gate.capture(frame,1);gate.sample(frame,1);
 auto old=tile;gate.pending={{7,10},{7,20}};
 tile.tile_flags|=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_EXPLORED;
 frame.presentation_time_ticks=15;gate.capture(frame,1);
 assert(!(gate.sample(frame,1).tiles[0].tile_flags&C3X_RENDERER_TILE_EXPLORED));
 // Rendering an older completed view must not erase the admitted reveal.
 auto prior=frame;prior.tiles=&old;prior.presentation_time_ticks=100;
 gate.sample(prior,1);
 gate.pending={{7,20}};
 assert(gate.sample(frame,1).tiles[0].tile_flags&C3X_RENDERER_TILE_VISIBLE);
 // A genuinely newer native loss still applies to an older displayed view.
 auto loss=frame;loss.tiles=&old;loss.presentation_time_ticks=25;gate.capture(loss,1);
 assert(!(gate.sample(frame,1).tiles[0].tile_flags&C3X_RENDERER_TILE_EXPLORED));
 // A delayed older capture cannot re-reveal the lost cell.
 gate.capture(frame,1);
 assert(!(gate.sample(frame,1).tiles[0].tile_flags&C3X_RENDERER_TILE_EXPLORED));
}
''')

    def test_reveal_count_reports_newly_shown_sight_once(self):
        # Diagnostic for reveal timing (`reveal-shown`): each newly displayed
        # cell counts once, with the native capture time that revealed it.
        run_cpp(r'''
#include "Renderer/native/render_core/unit_arrival_visibility.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 UnitArrivalVisibility gate;c3x_renderer_tile_v1 tiles[2]{};
 for(int i=0;i<2;++i){tiles[i].tile_x=i*2;tiles[i].tile_y=4;tiles[i].tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_RENDER;}
 tiles[0].tile_flags|=C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
 c3x_renderer_frame_v1 frame{};frame.tiles=tiles;frame.tile_count=2;frame.presentation_frequency=1000;frame.presentation_time_ticks=10;
 gate.capture(frame,1);gate.sample(frame,1);assert(gate.revealed==0); // first sight is not a reveal
 tiles[1].tile_flags|=C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;frame.presentation_time_ticks=50;
 gate.capture(frame,1);gate.sample(frame,1);assert(gate.revealed==1&&gate.revealed_capture==50);
 gate.sample(frame,1);assert(gate.revealed==0);
}
''')

    def test_acceleration_and_deceleration_keep_native_duration(self):
        # The facing turn runs during travel (unit_pose_transition), so a step
        # takes exactly vanilla's constant-speed time.
        run_cpp(r'''
#include "Renderer/native/render_core/unit_locomotion.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 double end=UnitLocomotion::duration(128,225); // vanilla constant-speed travel time
 assert(std::abs(end-128/225.)<1e-12&&UnitLocomotion::duration(128,450)==end/2);
 double previous=0,peak=0;
 for(int i=0;i<=1000;++i){double x=UnitLocomotion::sample(end*i/1000.,128,225);
  assert(x>=previous&&x<=128);peak=std::max(peak,(x-previous)/(end/1000.));previous=x;}
 assert(peak>225*1.18&&peak<225*1.2); // ramps are repaid by a slightly faster cruise
 auto x=[](double t){return UnitLocomotion::sample(t,128,225);};
 assert(x(.04)<x(.08)-x(.04)); // successive early intervals accelerate
 assert(x(end)-x(end-.04)<x(end-.04)-x(end-.08)); // successive late intervals slow
 assert(x(-1)==0&&x(end+1)==128);
}
''')


if __name__ == '__main__':
    unittest.main()
