"""Accepted combat target placement and atomic body handoffs in the scene owner."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class UnitCombatTests(unittest.TestCase):
    def test_native_cadence_continues_between_captures_and_clamps_death(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_playback.h"
#include "Renderer/native/input_recording/codec.h"
#include "Renderer/native/unit_animation_runtime.h"
#include <cassert>
#include <limits>
using namespace c3x_renderer::render_core;
struct Clip {bool ambient=false,loop=false;double duration=1.633;unsigned frames=50;};
int main(){
 UnitPlayback player;Clip clip;c3x_renderer_unit_animation_v1 fact{};
 fact.struct_size=sizeof(fact);fact.visual.struct_size=sizeof(fact.visual);
 fact.visual.unit_id=7;fact.visual.action=3;fact.visual.presentation_frequency=1000000;
 fact.frames=10;fact.frame_seconds=.125f;fact.display_unit_id=7;assert(player.observe(fact));
 c3x_inputs::Writer writer;c3x_inputs::unit_animation_fields(writer,fact);
 c3x_renderer_unit_animation_v1 decoded{};c3x_inputs::Reader reader{writer.bytes};
 c3x_inputs::unit_animation_fields(reader,decoded);reader.done();
 assert(decoded.visual.unit_id==7&&decoded.frames==10&&decoded.frame_seconds==.125f&&decoded.display_unit_id==7);
 c3x_renderer_unit_v1 body{};body.unit_id=7;body.action=3;body.presentation_frequency=1000000;
 unsigned predict=0;
 for(int n=0;n<120;++n){
  body.presentation_time_ticks=n*25000;
  if(n%4==0){fact.cursor=(n/5)%10;fact.visual.presentation_time_ticks=body.presentation_time_ticks;assert(player.observe(fact));}
  assert(player.resolve(body,clip,false,predict));
  assert(std::abs(body.action_cursor-int((n%50)*65536./50))<=1&&predict==1);
  c3x_renderer::NativeUnitDraw draw{};draw.unit_id=7;draw.action=body.action;draw.direction=2;
  draw.sprite=draw.expected_sprite=draw.canvas=draw.expected_canvas=1;
  draw.sprite_width=draw.sprite_height=191;draw.action_cursor=body.action_cursor;draw.frame_count=body.frame_count;
  c3x_renderer::UnitAnimationPose pose;assert(c3x_renderer::prepare_native_unit_pose(draw,false,pose));

 }
 player.forget(7);fact.visual.action=6;fact.visual.presentation_time_ticks=3000000;
 fact.frames=8;fact.cursor=0;assert(player.observe(fact));body.action=6;
 body.presentation_time_ticks=3500000;assert(player.resolve(body,clip,false,predict));assert(body.action_cursor==32768);
 body.presentation_time_ticks=5000000;assert(player.resolve(body,clip,false,predict));assert(body.action_cursor==65535&&predict==0);
 fact.cursor=1;fact.visual.presentation_time_ticks=5000000;assert(player.observe(fact));
 assert(player.resolve(body,clip,false,predict)&&body.action_cursor==65535); // no corpse restart
 player.forget(7);assert(!player.directed(7,6));
 fact.frame_seconds=std::numeric_limits<float>::quiet_NaN();assert(!player.observe(fact));
 fact.frame_seconds=0;assert(!player.observe(fact));
 fact.frame_seconds=.125f;fact.frames=0;assert(!player.observe(fact));
}
''')

    def test_same_combat_action_restarts_after_native_idle(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_playback.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Clip {bool ambient=false,loop=false;double duration=1;unsigned frames=31;};
int main(){
 for(int action:{3,6,7,8,9,10,12}){
  UnitPlayback player;Clip clip;c3x_renderer_unit_animation_v1 fact{};
  fact.visual.unit_id=7;fact.visual.action=action;fact.visual.presentation_frequency=1000;
  fact.frames=10;fact.frame_seconds=.1f;assert(player.observe(fact));
  c3x_renderer_unit_v1 body{};body.unit_id=7;body.action=action;body.presentation_frequency=1000;
  unsigned next=0;body.presentation_time_ticks=800;
  assert(player.resolve(body,clip,false,next)&&body.action_cursor>50000);
  fact.visual.action=1;fact.visual.presentation_time_ticks=1100;assert(player.observe(fact));
  assert(!player.directed(7,action));
  fact.visual.action=action;fact.visual.presentation_time_ticks=2000;fact.cursor=0;
  assert(player.observe(fact));body.presentation_time_ticks=2000;
  assert(player.resolve(body,clip,false,next)&&body.action_cursor==0&&next==1);
 }
}
''')

    def test_all_one_shot_handoffs_keep_native_duration(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_playback.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Clip {bool ambient=false,loop=false;double duration=2;unsigned frames=61;};
int main(){
 for(int action:{6,7,8,9,10,12}){
  UnitPlayback player;Clip clip;c3x_renderer_unit_animation_v1 fact{};
  fact.visual.unit_id=9;fact.visual.action=action;fact.visual.presentation_frequency=1000;
  fact.frames=12;fact.frame_seconds=.1f;fact.cursor=2;fact.visual.presentation_time_ticks=100;
  assert(player.observe(fact));
  c3x_renderer_unit_v1 body{};body.unit_id=9;body.action=action;body.presentation_frequency=1000;
  unsigned next=0;body.presentation_time_ticks=500;assert(player.resolve(body,clip,true,next));
  assert(std::abs(body.action_cursor-32768)<=1&&next==1); // .2 source + .4 elapsed / 1.2 native duration
  fact.cursor=0;fact.visual.presentation_time_ticks=500;assert(player.observe(fact));
  assert(player.resolve(body,clip,true,next)&&std::abs(body.action_cursor-32768)<=1); // sparse recapture never restarts
  body.presentation_time_ticks=1500;assert(player.resolve(body,clip,true,next));
  assert(body.action_cursor==65535&&next==0);
  player.forget(9);fact.visual.presentation_time_ticks=1500;assert(player.observe(fact));
  assert(player.resolve(body,clip,true,next)&&body.action_cursor==0&&next==1); // actual next lifecycle may restart
 }
}
''')

    def test_approach_hold_return_wrap_and_retirement(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_instances.h"
#include <cassert>
#include <string>
using namespace c3x_renderer::render_core;
struct Clip {std::string name;bool ambient=false,loop=false;double duration=1;unsigned frames=31;};
struct Unit {std::vector<std::string> keys={"warrior"};std::vector<Clip> actions={{"idle",true,true},{"move"},{"fortify"},{"attack"},{"death"}};};
int main(){
 UnitInstances world;std::vector<Unit> catalog{Unit{}};UnitInstances::Selection selected;
 c3x_renderer_unit_state_v1 state{};state.struct_size=sizeof(state);state.unit_id=7;
 state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;state.tile_x=state.tile_y=4;
 state.action=1;state.visible=1;state.max_hp=3;state.presentation_frequency=1000000;
 c3x_renderer_unit_v1 body{};body.struct_size=sizeof(body);body.unit_id=7;body.action=1;
 body.frame_count=15;body.sprite_width=body.sprite_height=191;
 body.projection_scale_milli=1000;body.presentation_frequency=1000000;std::strcpy(body.unit_key,"warrior");
 c3x_renderer_unit_visual_v1 visual{};visual.struct_size=sizeof(visual);visual.unit_id=7;
 visual.flags=1;visual.max_hp=3;visual.projection_scale_milli=1000;
 visual.presentation_frequency=1000000;visual.target_x=320;visual.target_y=160;
 auto name=[](int a){return a==2?"move":a==7?"fortify":a==6?"death":a==3||a==4?"attack":"idle";};
 auto capture=[&](int action,long long ticks){
  state.action=body.action=visual.action=action;
  state.presentation_time_ticks=body.presentation_time_ticks=visual.presentation_time_ticks=ticks;
  assert(world.state(state));assert(world.observe(visual));assert(world.capture(body,1,catalog,name,selected));
 };
 c3x_renderer_tile_v1 tile{};tile.tile_x=tile.tile_y=4;tile.anchor_x=1031;tile.anchor_y=563;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE;
 c3x_renderer_frame_v1 frame{};frame.tiles=&tile;frame.tile_count=1;frame.tile_width=128;frame.tile_height=64;
 frame.world_width_tiles=frame.world_height_tiles=12;frame.world_wrap_x=frame.world_wrap_y=1;
 auto pose=[&](long long ticks){auto p=world.scene_poses(frame,ticks,1000000,catalog);assert(p.size()==1);return p[0];};
 capture(1,0);assert(pose(0).draw.body_x==1000);
 visual.target_x=384;capture(2,100);assert(pose(100).draw.body_x==1000);
 // Sparse or contradictory intermediate native pixels never steer travel.
 visual.pixel_x=9000;capture(2,100100);assert(pose(100100).draw.body_x==1023);
 assert(pose(200100).draw.body_x==1045);assert(pose(300100).draw.body_x==1064);
 auto old=selected;
 state.action=7;state.presentation_time_ticks=400100;assert(world.state(state));
 c3x_renderer_unit_v1 sampled{};unsigned step=0;
 assert(!world.sample(old,400100,1000000,catalog,sampled,step));
 assert(pose(400100).draw.body_x==1064); // no missing actor before new body arrives
 capture(7,400101);assert(pose(500100).draw.body_x==1064);
 capture(3,600100);assert(pose(800100).draw.body_x==1064);
 capture(4,900100);assert(pose(1000100).draw.body_x==1064);
 // Zoom transforms the same world-space stance.
 frame.tile_width=192;frame.tile_height=96;auto zoom=pose(1100100);
 assert(zoom.draw.body_x==1080&&zoom.draw.projection_scale_milli==1500);
 frame.tile_width=128;frame.tile_height=64;
 visual.target_x=320;capture(2,1200100);assert(pose(1200100).draw.body_x==1064);
 assert(pose(1300100).draw.body_x==1042);
 world.pause_motion(1300100);assert(pose(2000100).draw.body_x==1042);
 world.resume_motion(2100100,1000000);assert(pose(2200100).draw.body_x==1019);
 capture(1,2300100);assert(pose(2300100).draw.body_x==1000);
 // A native wrapped endpoint selects the adjacent half-tile occurrence.
 state.tile_x=tile.tile_x=10;visual.target_x=0;capture(2,2400100);
 assert(pose(2400100).draw.body_x==1000);assert(pose(2800100).draw.body_x==1064);
 state.visible=0;state.presentation_time_ticks=2900100;assert(world.state(state));
 assert(world.scene_poses(frame,2900100,1000000,catalog).empty());
 world.clear();state.visible=1;state.tile_x=tile.tile_x=4;visual.target_x=384;
 capture(6,3000100);pose(3000100);assert(pose(3400100).draw.body_x==1064);
 state.kind=C3X_RENDERER_UNIT_STATE_RETIRE;state.presentation_time_ticks=3500100;assert(world.state(state));
 assert(world.scene_poses(frame,3600100,1000000,catalog).empty());
 // Late body captures cannot resurrect a dead identity.
 body.presentation_time_ticks=3600100;assert(!world.capture(body,1,catalog,name,selected));
}
''')

    def test_victory_advance_keeps_displayed_combat_stance(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_instances.h"
#include <cassert>
#include <string>
using namespace c3x_renderer::render_core;
struct Clip {std::string name;bool ambient=true,loop=true;double duration=1;unsigned frames=31;};
struct Unit {std::vector<std::string> keys={"warrior"};std::vector<Clip> actions={{"idle"},{"move",false,true},{"attack",false,true}};};
int main(){
 for(bool diagonal:{false,true}){
  UnitInstances world;std::vector<Unit> catalog{Unit{}};
  c3x_renderer_unit_state_v1 state{};state.struct_size=sizeof(state);state.unit_id=7;
  state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;state.tile_x=state.tile_y=4;
  state.action=1;state.visible=1;state.max_hp=3;state.presentation_frequency=1000000;
  c3x_renderer_unit_v1 body{};body.struct_size=sizeof(body);body.unit_id=7;body.action=1;
  body.frame_count=15;body.sprite_width=body.sprite_height=191;body.direction=diagonal?4:3;
  body.projection_scale_milli=1000;body.presentation_frequency=1000000;std::strcpy(body.unit_key,"warrior");
  c3x_renderer_unit_visual_v1 visual{};visual.struct_size=sizeof(visual);visual.unit_id=7;
  visual.flags=1;visual.max_hp=3;visual.projection_scale_milli=1000;
  visual.presentation_frequency=1000000;visual.target_x=320;visual.target_y=160;
  auto capture=[&](long long ticks){
   state.presentation_time_ticks=body.presentation_time_ticks=visual.presentation_time_ticks=ticks;
   assert(world.state(state));assert(world.observe(visual));UnitInstances::Selection selected;
   assert(world.capture(body,1,catalog,[](int a){return a==2?"move":a==3?"attack":"idle";},selected));
  };
  c3x_renderer_tile_v1 tiles[2]{};
  int dx=diagonal?1:2,dy=diagonal?1:0;
  for(int i=0;i<2;++i){auto& t=tiles[i];t.tile_x=4+i*dx;t.tile_y=4+i*dy;
   t.anchor_x=1031+i*dx*64;t.anchor_y=563+i*dy*32;
   t.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE;}
  c3x_renderer_frame_v1 frame{};frame.tiles=tiles;frame.tile_count=2;frame.tile_width=128;frame.tile_height=64;
  frame.world_width_tiles=frame.world_height_tiles=12;
  auto pose=[&](long long t){auto p=world.scene_poses(frame,t,1000000,catalog);assert(p.size()==1);return p[0].draw;};
  capture(0);auto origin=pose(0);
  visual.target_x+=dx*32;visual.target_y+=dy*16;capture(100);pose(100);
  auto stance=pose(400100);assert(stance.body_x==origin.body_x+dx*32);
  // Native victory targets the next tile before the accepted move arrives.
  visual.target_x+=dx*32;visual.target_y+=dy*16;capture(500100);
  auto held=pose(600100);assert(held.body_x==stance.body_x&&held.body_y==stance.body_y);
  c3x_renderer_unit_move_v1 move{};move.struct_size=sizeof(move);move.unit_id=7;move.action=2;
  move.old_x=move.old_y=4;move.new_x=4+dx;move.new_y=4+dy;
  move.source_visible=move.target_visible=1;move.presentation_frequency=1000000;move.presentation_time_ticks=600101;
  assert(world.begin_motion(move,12,12,false,false));
  auto first=pose(600101);assert(first.body_x==held.body_x&&first.body_y==held.body_y);
  move.presentation_time_ticks=600102;assert(world.move(move));
  state.tile_x=4+dx;state.tile_y=4+dy;capture(600103);
  auto previous=first;
  for(long long t=610101;t<=2000101;t+=10000){
   auto next=pose(t);assert(next.body_x>=previous.body_x&&next.body_y>=previous.body_y);
   assert(next.body_x-previous.body_x<=3&&next.body_y-previous.body_y<=3);previous=next;
  }
  assert(previous.body_x==origin.body_x+dx*64&&previous.body_y==origin.body_y+dy*32);
  // A native correction has no accepted travel and discards the old stance.
  visual.target_x=(state.tile_x+1)*64+32;visual.target_y=(state.tile_y+1)*32;
  capture(2100101);pose(2100101);pose(2500101);
  state.tile_x=state.tile_y=4;visual.target_x=320;visual.target_y=160;capture(2600101);
  auto corrected=pose(2600101);assert(corrected.body_x==origin.body_x&&corrected.body_y==origin.body_y);
 }
}
''')

    def test_native_stack_groups_replace_stale_bodies_but_preserve_travel(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_instances.h"
#include <cassert>
#include <string>
using namespace c3x_renderer::render_core;
struct Clip {std::string name;bool ambient=true,loop=true;double duration=1;unsigned frames=31;};
struct Unit {std::vector<std::string> keys={"warrior"};std::vector<Clip> actions={{"idle"},{"move",false,true}};};
int main(){
 UnitInstances world;std::vector<Unit> catalog{Unit{}};
 c3x_renderer_tile_v1 tiles[2]{};
 for(int i=0;i<2;++i){tiles[i].tile_x=4+2*i;tiles[i].tile_y=4;tiles[i].anchor_x=100+128*i;
  tiles[i].anchor_y=100;tiles[i].tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE;}
 c3x_renderer_frame_v1 frame{};frame.tiles=tiles;frame.tile_count=2;frame.tile_width=128;frame.tile_height=64;
 frame.world_width_tiles=frame.world_height_tiles=12;
 auto capture=[&](int id,int group,int tile,long long ticks){
  c3x_renderer_unit_state_v1 state{};state.struct_size=sizeof(state);state.unit_id=id;
  state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;state.tile_x=tile;state.tile_y=4;state.action=1;
  state.visible=1;state.max_hp=3;state.presentation_frequency=1000000;state.presentation_time_ticks=ticks;
  assert(world.state(state));
  c3x_renderer_unit_animation_v1 event{};event.struct_size=sizeof(event);event.display_unit_id=group;
  auto& v=event.visual;v.struct_size=sizeof(v);v.unit_id=id;v.action=1;v.max_hp=3;v.flags=1;
  v.projection_scale_milli=1000;v.target_x=(tile+1)*64;v.target_y=160;
  v.presentation_frequency=1000000;v.presentation_time_ticks=ticks;
  assert(world.observe_animation(event));
  c3x_renderer_unit_v1 body{};body.struct_size=sizeof(body);body.unit_id=id;body.action=1;
  body.frame_count=15;body.sprite_width=body.sprite_height=191;body.projection_scale_milli=1000;
  body.presentation_frequency=1000000;body.presentation_time_ticks=ticks;std::strcpy(body.unit_key,"warrior");
  UnitInstances::Selection selected;
  assert(world.capture(body,1,catalog,[](int a){return a==2?"move":"idle";},selected));
 };
 auto poses=[&](long long ticks){return world.scene_poses(frame,ticks,1000000,catalog);};
 capture(0,0,4,0);assert(poses(0).size()==1);
 capture(17,17,4,1);auto p=poses(1);assert(p.size()==1&&p[0].draw.unit_id==17);
 capture(84,17,4,2);assert(poses(2).size()==2); // army commander + member share selection
 capture(0,0,4,3);p=poses(3);assert(p.size()==1&&p[0].draw.unit_id==0);
 world.forget(84);capture(17,17,4,4);
 c3x_renderer_unit_move_v1 move{};move.struct_size=sizeof(move);move.unit_id=17;move.action=2;
 move.old_x=4;move.old_y=4;move.new_x=6;move.new_y=4;move.source_visible=move.target_visible=1;
 move.presentation_frequency=1000000;move.presentation_time_ticks=5;
 assert(world.begin_motion(move,12,12,false,false));
 capture(0,0,4,6);p=poses(6);assert(p.size()==2); // native chooses background while actor departs
 move.presentation_time_ticks=7;assert(world.move(move));capture(17,17,6,8);
 p=poses(200006);assert(p.size()==2);
 auto moving=std::find_if(p.begin(),p.end(),[](auto const& pose){return pose.draw.unit_id==17;});
 assert(moving!=p.end()&&moving->draw.action==2);
 p=poses(600006);assert(p.size()==2); // independent source + destination selections
 capture(0,0,6,900007);p=poses(900007);assert(p.size()==1&&p[0].draw.unit_id==0);
 world.clear();assert(poses(700000).empty());
}
''')



if __name__ == '__main__':
    unittest.main()
