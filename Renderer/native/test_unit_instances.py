"""Retained content validity and selection sampling without native callbacks."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class UnitInstanceTests(unittest.TestCase):
    def test_retained_instances_selected_occurrences_and_lifecycle(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_instances.h"
#include "Renderer/native/render_core/unit_frame_preparation.h"
#include <cassert>
#include <string>
#include <vector>
using namespace c3x_renderer::render_core;
struct Clip {std::string name;bool ambient=true,loop=true;double duration=3;unsigned frames=91;};
struct Unit {std::vector<std::string> keys;std::vector<Clip> actions;};
int main(){
 std::vector<Unit> catalog={{{"worker"},{{"idle"},{"road"},{"move",false,false,1,16}}}};
 auto name=[](int a)->char const*{return a==1?"idle":a==13?"road":a==2?"move":nullptr;};
 UnitInstances world(2);UnitInstances::Selection a,b,old;unsigned step=0;
 c3x_renderer_unit_v1 input{};input.struct_size=sizeof(input);input.unit_id=10;
 input.action=13;input.frame_count=15;input.sprite_width=191;input.sprite_height=191;
 input.presentation_frequency=1000000;input.projection_scale_milli=1000;
 std::strcpy(input.unit_key,"worker");c3x_renderer_unit_v1 out{};
 auto capture=[&](UnitInstances::Selection& selection,unsigned flags=1){return world.capture(input,flags,catalog,name,selection);};
 assert(capture(a));assert(world.bindings==1 && world.size()==1);
 assert(world.sample(a,0,1000000,catalog,out,step) && out.action_cursor==0);
 // Sixty renderer frames, one native capture: exact authored 30 Hz source.
 for(int n=1;n<=60;++n){assert(world.sample(a,(n*1000000LL+59)/60,1000000,catalog,out,step));assert(out.action_cursor==n/2);}
 assert(input.action_cursor==0 && input.frame_count==15); // immutable native input
 auto revision=a.revision;
 input.body_x=800;input.projection_scale_milli=1500;input.action_cursor=7;input.presentation_time_ticks=1000000;
 assert(capture(b) && b.revision==revision && world.bindings==1);
 assert(world.sample(a,1000000,1000000,catalog,out,step) && out.body_x==0 && out.projection_scale_milli==1000);
 assert(world.sample(b,1000000,1000000,catalog,out,step) && out.body_x==800 && out.projection_scale_milli==1500);
 // Independent content changes invalidate every old occurrence, not other IDs.
 old=b;input.direction=3;assert(capture(b) && b.revision!=revision);
 assert(!world.sample(old,1000000,1000000,catalog,out,step));
 input.unit_id=11;assert(capture(a));input.display_color_rgb=0x123456;assert(capture(a));
 assert(world.sample(b,1000000,1000000,catalog,out,step));
 // Native one-shots never extrapolate movement/cursor without native evidence.
 input.unit_id=10;input.action=2;input.action_cursor=7;input.frame_count=15;assert(capture(b));
 assert(world.sample(b,5000000,1000000,catalog,out,step) && out.action_cursor==7 && out.body_x==800);
 input.action=1;assert(capture(b));
 assert(world.sample(b,5100000,1000000,catalog,out,step) && out.action_cursor==0 && step==0);
 assert(world.sample(b,5200000,1000000,catalog,out,step) && out.action_cursor==0 && step==0);
 assert(capture(b,3));assert(world.sample(b,5300000,1000000,catalog,out,step));
 assert(world.sample(b,5400000,1000000,catalog,out,step) && out.action_cursor==3);
 // Hidden authoritative state retires the unit, including its old animation selection.
 old=b;assert(!capture(a,C3X_RENDERER_UNIT_STATE_CAPTURED|C3X_RENDERER_UNIT_HIDDEN));
 assert(!world.sample(old,5400000,1000000,catalog,out,step));assert(capture(b,3));
 // Despawn, eviction, catalog reset and unsupported captures invalidate tokens.
 old=b;world.forget(10);assert(!world.sample(old,5500000,1000000,catalog,out,step));
 assert(capture(b,3) && b.revision!=old.revision);
 assert(world.sample(b,5500000,1000000,catalog,out,step) && out.action_cursor==0);
 old=a;input.unit_id=12;assert(capture(a));assert(world.size()==2 && world.evictions==1);
 assert(!world.sample(old,0,1000000,catalog,out,step));
 old=b;world.clear();assert(capture(b));assert(!world.sample(old,0,1000000,catalog,out,step));
 old=b;std::strcpy(input.unit_key,"unknown");assert(!capture(a));assert(!world.sample(old,0,1000000,catalog,out,step));
 // Copied native movement samples refine only visible screen position. A
 // newer sample corrects travel; interruption and fog retire the segment.
 UnitInstances motion;UnitInstances::Selection first,second;
 c3x_renderer_unit_v1 moving{};moving.struct_size=sizeof(moving);moving.unit_id=55;
 moving.action=2;moving.frame_count=16;moving.body_x=1000;moving.body_y=200;
 moving.projection_scale_milli=1000;moving.presentation_frequency=1000000;
 moving.presentation_time_ticks=1000000;std::strcpy(moving.unit_key,"worker");
 c3x_renderer_unit_visual_v1 visual{};visual.struct_size=sizeof(visual);
 visual.unit_id=55;visual.action=2;visual.pixel_x=100;visual.pixel_y=50;
 visual.target_x=200;visual.target_y=50;visual.body_x=1000;visual.body_y=200;
 visual.max_hp=4;
 visual.projection_scale_milli=1000;visual.flags=C3X_RENDERER_UNIT_STATE_CAPTURED;
 visual.presentation_time_ticks=1000000;visual.presentation_frequency=1000000;
 assert(motion.observe(visual));
 assert(motion.capture(moving,1,catalog,name,first));
 assert(motion.animated(first,catalog));
 assert(motion.sample(first,1016000,1000000,catalog,out,step)&&out.body_x==1002);
 visual.pixel_x=106;visual.body_x=1006;visual.presentation_time_ticks=1066000;
 moving.body_x=1006;moving.presentation_time_ticks=1066000;
 assert(motion.observe(visual)&&motion.capture(moving,1,catalog,name,second));
 assert(motion.sample(second,1099000,1000000,catalog,out,step)&&out.body_x==1009);
 assert(motion.sample(second,2000000,1000000,catalog,out,step)&&out.body_x<=1015); // bounded extrapolation
 auto stale=visual;stale.presentation_time_ticks=1000000;assert(!motion.observe(stale));
 c3x_renderer_unit_move_v1 accepted{};accepted.struct_size=sizeof(accepted);
 accepted.unit_id=55;accepted.old_x=4;accepted.old_y=4;accepted.new_x=5;accepted.new_y=5;
 accepted.action=2;accepted.source_visible=0;accepted.target_visible=1;
 accepted.presentation_frequency=1000000;accepted.presentation_time_ticks=1100000;
 assert(motion.move(accepted));assert(!motion.observe(visual));
 moving.presentation_time_ticks=1100000;assert(motion.capture(moving,1,catalog,name,first));
 assert(!first.has_visual); // the old pixel segment cannot cross the accepted move
 visual.presentation_time_ticks=1100000;assert(motion.observe(visual));
 assert(motion.capture(moving,1,catalog,name,first)&&first.has_visual);
 accepted.old_x=5;accepted.old_y=5;accepted.new_x=6;accepted.new_y=6;
 accepted.source_visible=1;accepted.target_visible=0;accepted.presentation_time_ticks=1200000;
 assert(motion.move(accepted));assert(motion.size()==0);
 accepted.target_visible=1;accepted.presentation_time_ticks=1100000;
 assert(!motion.move(accepted)); // a late reveal cannot undo newer fog loss
 moving.presentation_time_ticks=1100000;assert(!motion.capture(moving,1,catalog,name,first));
 accepted.presentation_time_ticks=1250000;
 assert(motion.move(accepted)); // a later reveal may reuse the stable ID
 accepted.action=-1;accepted.presentation_time_ticks=1260000;
 assert(motion.move(accepted)); // Native fast moves may have no active FLC action.
 accepted.action=2;
 c3x_renderer_unit_spawn_v1 born{};born.struct_size=sizeof(born);
 born.unit_id=55;born.tile_x=6;born.tile_y=6;born.unit_type_id=1;born.owner_id=2;
 born.visible=1;born.presentation_frequency=1000000;born.presentation_time_ticks=1150000;
 assert(!motion.spawn(born)); // an old birth cannot retire the newer accepted move
 born.visible=1;born.presentation_frequency=1000000;born.presentation_time_ticks=1300000;
 assert(motion.spawn(born));assert(motion.size()==0);
 visual.flags=C3X_RENDERER_UNIT_STATE_CAPTURED;
 visual.presentation_time_ticks=1200000;assert(!motion.observe(visual));
 moving.presentation_time_ticks=1200000;assert(!motion.capture(moving,1,catalog,name,first));
 visual.presentation_time_ticks=1300000;assert(motion.observe(visual));
 moving.presentation_time_ticks=1300000;
 moving.action=1;assert(motion.capture(moving,1,catalog,name,first));
 assert(!motion.sample(second,1100000,1000000,catalog,out,step));
 visual.flags=C3X_RENDERER_UNIT_STATE_CAPTURED|C3X_RENDERER_UNIT_HIDDEN;
 visual.presentation_time_ticks=1400000;assert(motion.observe(visual));
 assert(!motion.sample(first,1100000,1000000,catalog,out,step));
 visual.flags=C3X_RENDERER_UNIT_STATE_CAPTURED;
 visual.presentation_time_ticks=1350000;assert(!motion.observe(visual));
 moving.presentation_time_ticks=1350000;assert(!motion.capture(moving,1,catalog,name,first));
 born.presentation_time_ticks=1350000;assert(!motion.spawn(born));
 accepted.presentation_time_ticks=1350000;assert(!motion.move(accepted));
 moving.presentation_time_ticks=1500000;
 assert(!motion.capture(moving,C3X_RENDERER_UNIT_STATE_CAPTURED|C3X_RENDERER_UNIT_HIDDEN,
                        catalog,name,first));
 visual.presentation_time_ticks=1450000;assert(!motion.observe(visual));
 c3x_renderer_unit_state_v1 state{};state.struct_size=sizeof(state);
 state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;state.unit_id=55;
 state.tile_x=6;state.tile_y=6;state.unit_type_id=1;state.owner_id=2;
 state.action=13;state.damage=1;state.max_hp=4;state.visible=1;
 state.presentation_frequency=1000000;state.presentation_time_ticks=1600000;
 assert(motion.state(state)&&motion.state_of(55)->damage==1);
 state.damage=2;state.presentation_time_ticks=1550000;
 assert(!motion.state(state)&&motion.state_of(55)->damage==1);
 moving.presentation_time_ticks=1550000;assert(!motion.capture(moving,1,catalog,name,first));
 state.damage=2;state.presentation_time_ticks=1700000;
 assert(motion.state(state)&&motion.state_of(55)->damage==2);
 state.kind=C3X_RENDERER_UNIT_STATE_RETIRE;state.presentation_time_ticks=1800000;
 assert(motion.state(state)&&motion.state_of(55)->kind==C3X_RENDERER_UNIT_STATE_RETIRE);
 state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;state.presentation_time_ticks=1750000;
 assert(!motion.state(state));
 state.presentation_time_ticks=1900000;
 assert(!motion.state(state)); // Only a new accepted birth can reuse a retired ID.
 visual.presentation_time_ticks=1900000;assert(!motion.observe(visual));
 moving.presentation_time_ticks=1900000;assert(!motion.capture(moving,1,catalog,name,first));
 accepted.presentation_time_ticks=1900000;assert(!motion.move(accepted));
 born.presentation_time_ticks=2000000;assert(motion.spawn(born));
 state.presentation_time_ticks=2000000;assert(motion.state(state));
 UnitFramePreparation queue;input.unit_key[0]=0;input.unit_id=12;input.action_cursor=1;input.frame_count=16;
 queue.observe(input,true);assert(!queue.empty());queue.forget(12);assert(queue.empty());
}
''')


if __name__ == '__main__':
    unittest.main()
