"""Renderer-owned tile travel, independent of sparse or late native FLC samples."""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp

class UnitMotionTests(unittest.TestCase):
    def test_tile_travel_clock_queue_wrap_zoom_and_retirement(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_instances.h"
#include <cassert>
#include <string>
using namespace c3x_renderer::render_core;
struct Clip {std::string name;bool ambient=true,loop=true;double duration=1;unsigned frames=31;};
struct Unit {std::vector<std::string> keys={"settler"};std::vector<Clip> actions={{"idle"},{"move",false,true}};};
struct Fixture {
 UnitInstances world;std::vector<Unit> catalog{Unit{}};
 c3x_renderer_unit_state_v1 state{};c3x_renderer_unit_v1 body{};
 c3x_renderer_unit_move_v1 event{};c3x_renderer_tile_v1 tiles[2]{};
 c3x_renderer_frame_v1 frame{};
 Fixture(int x=4,int y=4){
  state.struct_size=sizeof(state);state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;
  state.unit_id=7;state.tile_x=x;state.tile_y=y;state.max_hp=3;
  state.visible=1;state.action=1;state.presentation_frequency=1000000;
  body.struct_size=sizeof(body);body.unit_id=7;body.action=1;body.frame_count=15;
  body.body_x=1000;body.body_y=500;body.sprite_width=body.sprite_height=191;
  body.projection_scale_milli=1000;body.presentation_frequency=1000000;
  std::strcpy(body.unit_key,"settler");
  frame.tile_count=2;frame.tiles=tiles;frame.tile_width=128;frame.tile_height=64;
  frame.world_width_tiles=frame.world_height_tiles=12;frame.world_wrap_x=frame.world_wrap_y=1;
  tiles[0].tile_x=x;tiles[0].tile_y=y;tiles[0].anchor_x=1031;tiles[0].anchor_y=563;
  tiles[0].tile_flags=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_RENDER;
  tiles[1]=tiles[0];tiles[1].tile_x=(x+2)%12;tiles[1].anchor_x+=128;
  capture();
  event.struct_size=sizeof(event);event.unit_id=7;event.old_x=x;event.old_y=y;
  event.new_x=(x+2)%12;event.new_y=y;event.action=2;
  event.source_visible=event.target_visible=1;event.presentation_time_ticks=1;
  event.presentation_frequency=1000000;
 }
 void capture(){
  assert(world.state(state));UnitInstances::Selection selection;
  assert(world.capture(body,11,catalog,[](int a){return a==2?"move":"idle";},selection));
 }
 void begin(){assert(world.begin_motion(event,12,12,true,true));}
 UnitInstances::ScenePose pose(long long ticks){auto p=world.scene_poses(frame,ticks,1000000,catalog);assert(p.size()==1);return p[0];}
 void commit(){event.presentation_time_ticks+=100;assert(world.move(event));}
};
int main(){
 Fixture a;a.begin();
 assert(a.world.travel_seconds_remaining()==0.); // unsampled travel cannot defer a reveal
 // Even a 10-second transport delay must display the source first.
 auto p=a.pose(10000000);assert(p.draw.body_x==1000&&p.draw.action==2&&p.cursor);
 assert(a.world.travel_seconds_remaining()>.56&&a.world.travel_seconds_remaining()<.57); // vanilla 128/225 s
 p=a.pose(10200000);assert(p.draw.body_x==1044&&p.draw.action_cursor==197);
 // An early commit and destination idle capture cannot truncate travel.
 a.commit();a.state.tile_x=6;a.state.presentation_time_ticks=200;
 a.body.body_x=1000; // Native camera already recentered; scene camera has not.
 a.body.presentation_time_ticks=200;a.capture();
 p=a.pose(10300000);assert(p.draw.body_x==1071&&p.draw.action==2&&p.draw.action_cursor==316);
 p=a.pose(10800000);assert(p.draw.body_x==1128&&p.draw.action==1);
 assert(a.world.travel_seconds_remaining()==0.);
 // Capture can select another body before the mover ever gets an idle draw.
 // Confirmed arrival ends the segment and yields ownership to that new stack.
 Fixture sparse;sparse.begin();sparse.pose(0);
 sparse.state.action=sparse.body.action=2;
 sparse.state.presentation_time_ticks=sparse.body.presentation_time_ticks=50;sparse.capture();
 sparse.commit();sparse.state.tile_x=6;sparse.state.presentation_time_ticks=200;
 assert(sparse.world.state(sparse.state));
 p=sparse.pose(800000);assert(p.draw.body_x==1128&&p.draw.action==1&&!p.travelling);
 assert(sparse.world.motion_count(7)==0);
 sparse.state.unit_id=sparse.body.unit_id=8;sparse.state.action=sparse.body.action=1;
 sparse.state.presentation_time_ticks=sparse.body.presentation_time_ticks=300;sparse.capture();
 p=sparse.pose(900000);assert(p.draw.unit_id==8);
 // Keep the body, cursor and marker anchor through arrival in a copied
 // terrain view that predates native sight. Exercise both occurrence paths.
 for(int extras:{0,5}){
  Fixture fog;fog.frame.presentation_frequency=1000000;
  for(int i=0;i<extras;++i){fog.state.unit_id=fog.body.unit_id=20+i;fog.capture();}
  fog.state.unit_id=fog.body.unit_id=7;fog.capture();
  fog.tiles[1].tile_flags=C3X_RENDERER_TILE_RENDER;
  fog.begin();fog.world.scene_poses(fog.frame,0,1000000,fog.catalog);fog.commit();
  fog.state.tile_x=6;fog.state.presentation_time_ticks=200;assert(fog.world.state(fog.state));
  auto check=[&](long long ticks,int x){
   auto poses=fog.world.scene_poses(fog.frame,ticks,1000000,fog.catalog);
   auto scout=std::find_if(poses.begin(),poses.end(),[](auto const& pose){return pose.draw.unit_id==7;});
   assert(scout!=poses.end()&&scout->draw.body_x==x&&scout->cursor);
  };
  check(800000,1128);check(900000,1128);
  assert(fog.world.motion_count(7)==0&&fog.world.pending_arrivals().empty());
  // A newer terrain capture can revoke sight even before another body/state
  // callback. An old accepted move must not bypass that current fog.
  auto current=fog.frame;current.presentation_time_ticks=201;
  auto lost=fog.world.scene_poses(current,900001,1000000,fog.catalog);
  assert(std::none_of(lost.begin(),lost.end(),[](auto const& pose){return pose.draw.unit_id==7;}));
  // The next leg must not disappear just because its source is still fogged
  // in the retained terrain. It starts at the previous rendered endpoint.
  fog.event.old_x=6;fog.event.new_x=6;fog.event.new_y=2;
  fog.event.presentation_time_ticks=300;fog.begin();check(1000000,1128);check(1100000,1128);
  fog.state.visible=0;fog.state.presentation_time_ticks=400;assert(fog.world.state(fog.state));
  auto hidden=fog.world.scene_poses(fog.frame,1200000,1000000,fog.catalog);
  assert(std::none_of(hidden.begin(),hidden.end(),[](auto const& pose){return pose.draw.unit_id==7;}));
 }
 // A replacement camera can remain prepared while native composition catches
 // up. Hidden elapsed time cannot consume the rest of an accepted move.
 Fixture held;held.begin();held.pose(10000000);held.world.pause_motion(10300000);
 assert(held.world.travel_seconds_remaining()==0.); // a frozen scene must be rebuilt immediately
 p=held.pose(10900000);assert(p.draw.body_x==1071&&p.draw.action_cursor==316);
 assert(p.pose_ticks==10300000); // heading/joint blends hold with travel
 held.world.pause_motion(11000000); // supersession preserves the original pause
 held.commit();held.state.tile_x=6;held.state.presentation_time_ticks=200;
 held.body.presentation_time_ticks=200;held.capture();
 held.world.resume_motion(12000000,1000000);
 p=held.pose(12100000);assert(p.draw.body_x==1098&&p.draw.action==2&&p.draw.action_cursor==435);
 assert(p.pose_ticks==10400000); // adoption does not consume the blend
 held.world.resume_motion(12200000,1000000); // later native import cannot pause/rebase it again
 p=held.pose(12500000);assert(p.draw.body_x==1128&&p.draw.action==1);
 // Native intermediate body samples do not restart or accelerate playback.
 Fixture b;b.begin();b.pose(0);
 b.state.action=b.body.action=2;b.state.presentation_time_ticks=b.body.presentation_time_ticks=100;
 b.body.body_x=1110;b.body.action_cursor=14;b.capture();
 p=b.pose(200000);assert(p.draw.body_x==1044&&p.draw.action_cursor==197);
 p=b.pose(300000);assert(p.draw.body_x==1071&&p.draw.action_cursor==316);
 // A delayed camera snapshot cannot rewind the live scene's clock.
 p=b.pose(100000);assert(p.draw.body_x==1071&&p.draw.action_cursor==316);
 // Camera pan and zoom transform the same interpolated body and cursor anchor.
 b.tiles[0].anchor_x+=50;b.tiles[0].anchor_y+=30;
 b.frame.tile_width=192;b.frame.tile_height=96;
 p=b.pose(300000);assert(p.draw.body_x==1141&&p.draw.body_y==498&&p.draw.projection_scale_milli==1500&&p.cursor);
 // Consecutive accepted steps preserve run phase across tile boundaries.
 constexpr long long origin=10250000;
 Fixture c;c.begin();c.pose(origin);c.commit();
 c.event.old_x=6;c.event.new_x=8;c.event.presentation_time_ticks=200;c.begin();c.commit();c.pose(origin+300000);
 auto arrivals=c.world.pending_arrivals();
 assert(arrivals.size()==2&&arrivals[0].second==101&&arrivals[1].second==300); // native commits, not movement starts

 p=c.pose(origin+500000);assert(p.draw.body_x==1122&&p.draw.action_cursor==544);
 p=c.pose(origin+900000);assert(p.draw.body_x==1208&&p.draw.action_cursor==922);
 // A next step arriving after a long native confirmation wait starts at
 // the shared endpoint, faces its own direction and displays its full travel.
 // It moves at once: the facing turn runs during travel, as natively.
 Fixture late;late.begin();late.pose(origin);
 p=late.pose(origin+800000);assert(p.draw.body_x==1128&&p.draw.action==1&&p.draw.direction==2);
 late.commit();late.event.old_x=6;late.event.new_x=6;late.event.new_y=2;
 late.event.presentation_time_ticks=200;late.begin();late.commit();
 p=late.pose(origin+1600000);assert(p.draw.body_x==1128&&p.draw.body_y==500&&p.draw.direction==8&&p.draw.action==2);
 p=late.pose(origin+1980000);assert(p.draw.body_x==1128&&p.draw.body_y==454&&p.draw.direction==8);
 // A visibility refresh freezes the same camera. A turn accepted during
 // that refresh must retain its entire visible travel through adoption.
 Fixture reveal;reveal.begin();reveal.pose(0);reveal.pose(800000);
 reveal.world.pause_motion(800000);
 reveal.commit();reveal.event.old_x=6;reveal.event.new_x=6;reveal.event.new_y=2;
 reveal.event.presentation_time_ticks=200;reveal.begin();reveal.commit();
 p=reveal.pose(1600000);assert(p.draw.body_x==1128&&p.draw.body_y==500&&p.draw.direction==8);
 reveal.world.resume_motion(1800000,1000000);
 p=reveal.pose(2180000);assert(p.draw.body_y==454&&p.draw.action==2);
 // Horizontal and vertical seam crossings choose a neighboring copy.
 Fixture w(10,4);w.begin();w.pose(0);p=w.pose(300000);assert(p.draw.body_x==1071);
 w.commit();w.event.old_x=0;w.event.new_x=2;w.event.presentation_time_ticks=200;w.begin();w.pose(400000);
 p=w.pose(800000);assert(p.draw.body_x==1181);
 Fixture v(4,10);v.event.new_x=4;v.event.new_y=0;v.begin();v.pose(0);
 p=v.pose(300000);assert(p.draw.body_x==1000&&p.draw.body_y==536);
 // All eight accepted directions override a stale SE body observation.
 int offsets[8][2]={{1,-1},{2,0},{1,1},{0,2},{-1,1},{-2,0},{-1,-1},{0,-2}};
 for(int direction=1;direction<=8;++direction){
  Fixture turn;turn.body.direction=3;turn.capture();
  turn.event.new_x=4+offsets[direction-1][0];turn.event.new_y=4+offsets[direction-1][1];
  turn.begin();assert(turn.pose(0).draw.direction==direction);
  assert(turn.pose(100000).draw.direction==direction);
 }
 // Hidden/retired actors and unrelated corrections end travel immediately.
 c.state.visible=0;c.state.presentation_time_ticks=1000;assert(c.world.state(c.state));
 assert(c.world.scene_poses(c.frame,910000,1000000,c.catalog).empty());
 Fixture r;r.begin();r.pose(0);r.state.kind=C3X_RENDERER_UNIT_STATE_RETIRE;r.state.presentation_time_ticks=100;
 assert(r.world.state(r.state));assert(r.world.scene_poses(r.frame,100000,1000000,r.catalog).empty());
 Fixture correction;correction.begin();correction.pose(0);
 correction.event.new_x=8;correction.commit();
 assert(correction.pose(300000).draw.body_x==1000); // no stale travel after teleport
 Fixture invalid;invalid.event.new_x=12;assert(!invalid.world.begin_motion(invalid.event,12,12,true,true));
 invalid.event.new_x=8;assert(!invalid.world.begin_motion(invalid.event,12,12,true,true));
}
''')

    def test_travel_keeps_native_start_and_duration(self):
        # Moves must take exactly native Civ III's time: a step starts at its
        # move event (QPC), not at the first displayed sample after transport.
        run_cpp(r'''
#include "Renderer/native/render_core/unit_instances.h"
#include <cassert>
#include <string>
using namespace c3x_renderer::render_core;
struct Clip {std::string name;bool ambient=true,loop=true;double duration=1;unsigned frames=31;};
struct Unit {std::vector<std::string> keys={"settler"};std::vector<Clip> actions={{"idle"},{"move",false,true}};};
struct Fixture {
 UnitInstances world;std::vector<Unit> catalog{Unit{}};
 c3x_renderer_unit_state_v1 state{};c3x_renderer_unit_v1 body{};
 c3x_renderer_unit_move_v1 event{};c3x_renderer_tile_v1 tiles[2]{};
 c3x_renderer_frame_v1 frame{};
 Fixture(int x=4,int y=4){
  state.struct_size=sizeof(state);state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;
  state.unit_id=7;state.tile_x=x;state.tile_y=y;state.max_hp=3;
  state.visible=1;state.action=1;state.presentation_frequency=1000000;
  body.struct_size=sizeof(body);body.unit_id=7;body.action=1;body.frame_count=15;
  body.body_x=1000;body.body_y=500;body.sprite_width=body.sprite_height=191;
  body.projection_scale_milli=1000;body.presentation_frequency=1000000;
  std::strcpy(body.unit_key,"settler");
  frame.tile_count=2;frame.tiles=tiles;frame.tile_width=128;frame.tile_height=64;
  frame.world_width_tiles=frame.world_height_tiles=12;frame.world_wrap_x=frame.world_wrap_y=1;
  tiles[0].tile_x=x;tiles[0].tile_y=y;tiles[0].anchor_x=1031;tiles[0].anchor_y=563;
  tiles[0].tile_flags=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_RENDER;
  tiles[1]=tiles[0];tiles[1].tile_x=(x+2)%12;tiles[1].anchor_x+=128;
  capture();
  event.struct_size=sizeof(event);event.unit_id=7;event.old_x=x;event.old_y=y;
  event.new_x=(x+2)%12;event.new_y=y;event.action=2;
  event.source_visible=event.target_visible=1;event.presentation_time_ticks=1;
  event.presentation_frequency=1000000;
 }
 void capture(){
  assert(world.state(state));UnitInstances::Selection selection;
  assert(world.capture(body,11,catalog,[](int a){return a==2?"move":"idle";},selection));
 }
 void begin(){assert(world.begin_motion(event,12,12,true,true));}
 UnitInstances::ScenePose pose(long long ticks){auto p=world.scene_poses(frame,ticks,1000000,catalog);assert(p.size()==1);return p[0];}
 void commit(){event.presentation_time_ticks+=100;assert(world.move(event));}
};
int main(){
 constexpr long long q=1000000,offset=5*q; // scene clock = QPC - offset
 double duration=128/225.;long long ticks=static_cast<long long>(duration*q);
 // Native start 80 ms before the first sample: arrival stays native.
 Fixture a;a.world.native_clock(100*q,100*q-offset);
 a.event.presentation_time_ticks=10*q+offset;a.begin();
 auto p=a.pose(10*q+80000);assert(p.draw.action==2&&p.draw.body_x>1000);
 a.commit();a.state.tile_x=6;a.state.presentation_time_ticks=a.event.presentation_time_ticks;
 a.body.presentation_time_ticks=a.event.presentation_time_ticks;a.capture();
 p=a.pose(10*q+ticks-50000);assert(p.draw.action==2&&p.draw.body_x<1128); // decelerating, not yet there
 p=a.pose(10*q+ticks+5000);assert(p.draw.action==1&&p.draw.body_x==1128);
 assert(a.world.arrivals.size()==1);auto r=a.world.arrivals[0];
 assert(r.native_start==10*q&&r.visual_start==10*q&&r.visual_end==10*q+ticks);
 // A very late first sample skips at most a quarter of the travel.
 Fixture b;b.world.native_clock(100*q,100*q-offset);
 b.event.presentation_time_ticks=10*q+offset;b.begin();
 p=b.pose(20*q);assert(p.draw.action==2&&p.draw.body_x>1000&&p.draw.body_x<1128);
 p=b.pose(20*q+static_cast<long long>(.74*ticks));assert(p.draw.action==2);
 // A queued step keeps native time: it starts where the previous one ended.
 Fixture c;c.world.native_clock(100*q,100*q-offset);
 c.event.presentation_time_ticks=10*q+offset;c.begin();c.pose(10*q);c.commit();
 c.event.old_x=6;c.event.new_x=8;c.event.presentation_time_ticks=10*q+offset+ticks;c.begin();c.commit();
 c.pose(10*q+ticks+30000);
 c.world.scene_poses(c.frame,10*q+2*ticks+5000,q,c.catalog); // its endpoint is outside this frame
 assert(c.world.arrivals.size()==2&&c.world.arrivals[1].visual_start==10*q+ticks&&
        c.world.arrivals[1].visual_end==10*q+2*ticks);
 // A frozen view holds travel (nothing unseen is skipped); afterwards the step
 // runs at double speed until the held time is recovered, then arrives natively.
 Fixture h;h.world.native_clock(100*q,100*q-offset);
 h.event.presentation_time_ticks=10*q+offset;h.begin();h.pose(10*q);
 p=h.pose(10*q+100000);int before=p.draw.body_x;h.world.pause_motion(10*q+100000);
 h.world.resume_motion(10*q+250000,q); // 150 ms frozen
 p=h.pose(10*q+250000);assert(p.draw.body_x==before); // no jump at resume
 h.commit();h.state.tile_x=6;h.state.presentation_time_ticks=h.event.presentation_time_ticks;
 h.body.presentation_time_ticks=h.event.presentation_time_ticks;h.capture();
 for(long long t=10*q+260000;t<=10*q+ticks+5000;t+=10000)h.pose(t);
 p=h.pose(10*q+ticks+5000);assert(p.draw.action==1&&p.draw.body_x==1128); // native arrival despite the hold
 assert(h.world.arrivals.size()==1&&h.world.arrivals[0].visual_end==10*q+ticks);
 // Civ III confirms a step one or two animator ticks after the travel time.
 // The measured overhead stretches later steps, so they arrive when Civ III
 // confirms them and a path never stops between tiles.
 Fixture e;e.world.native_clock(100*q,100*q-offset);
 e.event.presentation_time_ticks=10*q+offset;e.begin();e.pose(10*q);
 e.event.presentation_time_ticks=10*q+offset+ticks+130000;assert(e.world.move(e.event)); // confirmed 130 ms late
 e.state.tile_x=6;e.state.presentation_time_ticks=e.event.presentation_time_ticks;assert(e.world.state(e.state));
 e.pose(10*q+ticks+5000);
 long long second=10*q+ticks+130000;
 e.event.old_x=6;e.event.new_x=8;e.event.presentation_time_ticks=second+offset;e.begin();
 e.world.scene_poses(e.frame,second,q,e.catalog);
 e.event.presentation_time_ticks=second+offset+ticks+130000;assert(e.world.move(e.event));
 e.world.scene_poses(e.frame,second+ticks+60000,q,e.catalog);assert(e.world.arrivals.size()==1); // still travelling
 e.world.scene_poses(e.frame,second+ticks+135000,q,e.catalog);
 assert(e.world.arrivals.size()==2&&e.world.arrivals[1].visual_end==second+ticks+130000&&e.world.arrivals[1].committed==second+ticks+130000);
 // Without the native clock (replay), the first displayed sample starts travel.
 Fixture d;d.event.presentation_time_ticks=10*q;d.begin();d.pose(12*q);
 p=d.pose(12*q+ticks-5000);assert(p.draw.action==2);
}
''')

    def test_native_target_hook_delegates_when_disabled_and_copies_tiles(self):
        source=(ROOT/'injected_code.c').read_text()
        start=source.index('void __fastcall\npatch_FLC_Animation_set_move_target')
        wrapper=source[start:source.index('\n}\n',start)+3].replace('this','self')
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstdio>
#define __fastcall
constexpr int __=0,AT_RUN=2;
struct Unit{struct{int ID=7,X=10,Y=4,CivID=2;}Body;};
struct Animation_Info{int field_98[41];};
struct FLC_Animation{struct Unit* Unit;struct Animation_Info* Animation_Info;float field_104;};
struct LARGE_INTEGER{long long QuadPart;};
struct {int Map;} bic,*p_bic_data=&bic;
int natives=0,events=0;FLC_Animation* original=nullptr;
void FLC_Animation_set_pixel_target_with_offset(FLC_Animation* a,int edx,int x,int y){assert(edx==123&&x==768&&y==128);original=a;++natives;}
int receive(c3x_renderer_unit_move_v1 const* v){assert(natives>events);assert(v->old_x==10&&v->old_y==4&&v->new_x==0&&v->new_y==4&&v->unit_id==7&&v->action==2&&v->speed==250);++events;return 1;}
struct {struct{bool enable_custom_rendering=false;}current_config;c3x_renderer_unit_move_fn custom_renderer_unit_motion=receive;
 int custom_renderer_viewer_civ_id=2;unsigned custom_renderer_map_epoch=2,custom_renderer_viewer_epoch=3;LARGE_INTEGER custom_renderer_qpc_frequency{1000};char custom_renderer_test_save[1]={};}state,*is=&state;
void wrap_tile_coords(int*,int* x,int* y){*x=(*x+12)%12;*y=(*y+12)%12;}
bool Map_in_range(int*,int,int x,int y){return x>=0&&x<12&&y>=0&&y<12;}
bool visible=true,target_visible=true;bool custom_renderer_tile_visible_at(int x,int){return visible&&(x==10||target_visible);}
bool QueryPerformanceCounter(LARGE_INTEGER* now){now->QuadPart=100;return true;}
void debug(char const*){}auto p_OutputDebugStringA=debug;
'''+wrapper+r'''
int main(){
 Unit unit;Animation_Info art{};art.field_98[38]=250; // INI Fast Speed at +0x130
 FLC_Animation animation{&unit,&art,1.f};
 patch_FLC_Animation_set_move_target(&animation,123,768,128);assert(natives==1&&!events&&original==&animation);
 state.current_config.enable_custom_rendering=true;
 patch_FLC_Animation_set_move_target(&animation,123,768,128);assert(natives==2&&events==1);
 // Own exploration begins before native terrain sight catches up.
 target_visible=false;patch_FLC_Animation_set_move_target(&animation,123,768,128);assert(natives==3&&events==2);
 // The same step by a foreign unit cannot animate into hidden terrain.
 unit.Body.CivID=3;patch_FLC_Animation_set_move_target(&animation,123,768,128);assert(natives==4&&events==2);
 unit.Body.CivID=2;
 visible=false;patch_FLC_Animation_set_move_target(&animation,123,768,128);assert(natives==5&&events==2);

}
''')

if __name__=='__main__':unittest.main()
