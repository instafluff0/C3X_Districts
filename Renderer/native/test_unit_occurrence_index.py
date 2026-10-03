"""Native ordered occurrence lookup for busy retained-unit scenes."""
from pathlib import Path
import unittest
from Renderer.native.test_unit_contribution_plan import host_cpp


class UnitOccurrenceIndexTests(unittest.TestCase):
    def test_index_preserves_native_order_duplicate_wraps_stacks_motion_and_authority(self):
        host_cpp(r'''
#include "Renderer/native/render_core/unit_instances.h"
#include <cassert>
#include <string>
using namespace c3x_renderer::render_core;
struct Clip {std::string name;bool ambient=true,loop=true;double duration=1;unsigned frames=31;};
struct Unit {std::vector<std::string> keys={"unit"};std::vector<Clip> actions={{"idle"},{"move",false,true}};};
int main(){
 UnitInstances world;std::vector<Unit> catalog{Unit{}};
 auto capture=[&](int id,int x,int y){
  c3x_renderer_unit_state_v1 state{};state.struct_size=sizeof(state);state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;
  state.unit_id=id;state.tile_x=x;state.tile_y=y;state.visible=1;state.action=1;state.max_hp=3;state.presentation_frequency=1000000;
  assert(world.state(state));c3x_renderer_unit_v1 body{};body.struct_size=sizeof(body);body.unit_id=id;
  body.action=1;body.direction=2;body.frame_count=15;body.sprite_width=body.sprite_height=128;
  body.projection_scale_milli=1000;body.presentation_frequency=1000000;std::strcpy(body.unit_key,"unit");
  UnitInstances::Selection selection;assert(world.capture(body,C3X_RENDERER_UNIT_STATE_CAPTURED,catalog,[](int){return "idle";},selection));
 };
 capture(1,4,4);capture(2,8,8);capture(3,10,10);capture(4,2,2);capture(5,4,4); // indexed path; newest stack owner is 5
 c3x_renderer_tile_v1 tiles[8]{};
 auto tile=[&](unsigned i,int x,int y,int ax,int ay,unsigned flags){tiles[i].tile_x=x;tiles[i].tile_y=y;
  tiles[i].anchor_x=ax;tiles[i].anchor_y=ay;tiles[i].tile_flags=flags;};
 unsigned visible=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_RENDER;
 tile(0,16,16,300,300,visible); // Native order, not sorted canonical/world order.
 tile(1,-8,-8,100,100,visible);
 tile(2,4,4,200,200,visible);
 tile(3,4,4,200,200,visible); // duplicate anchor must not create an extra body
 tile(4,28,4,400,200,C3X_RENDERER_TILE_VISIBLE); // visibility alone grants no draw authority
 tile(5,4,28,200,400,C3X_RENDERER_TILE_PREFETCH); // draw authority without current visibility grants none
 tile(6,6,4,264,200,visible);
 tile(7,40,4,500,200,C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_PREFETCH);
 c3x_renderer_frame_v1 frame{};frame.tiles=tiles;frame.tile_count=8;frame.tile_width=128;frame.tile_height=64;
 frame.world_width_tiles=frame.world_height_tiles=12;frame.world_wrap_x=frame.world_wrap_y=1;
 auto poses=world.scene_poses(frame,0,1000000,catalog);assert(poses.size()==4);
 for(unsigned i=0;i<4;++i)assert(poses[i].draw.unit_id==5);
 assert(poses[0].tile_x==16&&poses[1].tile_x==-8&&poses[2].tile_x==4&&poses[3].tile_x==40);
 assert(poses[0].draw.body_x==300&&poses[1].draw.body_x==100&&poses[2].draw.body_x==200&&poses[3].draw.body_x==500);
 c3x_renderer_unit_move_v1 move{};move.struct_size=sizeof(move);move.unit_id=1;move.action=2;
 move.old_x=4;move.old_y=4;move.new_x=6;move.new_y=4;move.source_visible=move.target_visible=1;move.presentation_frequency=1000000;
 assert(world.begin_motion(move,12,12,true,true));
 poses=world.scene_poses(frame,0,1000000,catalog);assert(poses.size()==8);
 for(unsigned i=0;i<4;++i)assert(poses[i].draw.unit_id==1&&poses[i].travelling&&poses[i+4].draw.unit_id==5&&!poses[i+4].travelling);
 auto start_x=poses[0].draw.body_x;poses=world.scene_poses(frame,100000,1000000,catalog);
 assert(poses[0].draw.body_x>start_x&&poses.size()==8);
 // A new capture can change anchors/zoom or hide every occurrence without any persistent index invalidation.
 for(auto& t:tiles)t.anchor_x+=17;
 frame.tile_width=160;frame.tile_height=80;poses=world.scene_poses(frame,100000,1000000,catalog);
 assert(poses.size()==8&&poses[0].draw.projection_scale_milli==1250);
 for(auto& t:tiles)t.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;
 assert(world.scene_poses(frame,100000,1000000,catalog).empty());
}
''')

    def test_actual_index_equals_linear_authority_for_hash_collisions_extreme_coordinates_and_bounds(self):
        source=(Path(__file__).resolve().parents[2]/"Renderer/native/render_core/unit_instances.h").read_text()
        begin=source.index("        struct Occurrences {")
        end=source.index("        for(auto& pair:instances){",begin)
        block=source[begin:end].replace("} occurrences(frame,instances.size()>4);","};")
        host_cpp(r'''
#include "Renderer/native/render_core/unit_instances.h"
#include <cassert>
#include <climits>
'''+block+r'''
std::size_t fail_bytes=0;
void* operator new(std::size_t bytes){if(bytes==fail_bytes)throw std::bad_alloc();if(auto* p=std::malloc(bytes))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
int main(){
 std::vector<c3x_renderer_tile_v1> tiles(257);unsigned random=11;
 auto next=[&](){random=random*1664525u+1013904223u;return random;};
 for(auto& t:tiles){t.tile_x=int(next()%257)-128;t.tile_y=int(next()%257)-128;
  t.tile_flags=(next()%2?C3X_RENDERER_TILE_VISIBLE:0)|(next()%3?C3X_RENDERER_TILE_RENDER:C3X_RENDERER_TILE_PREFETCH);}
 tiles[0].tile_x=INT_MIN;tiles[0].tile_y=INT_MAX;tiles[0].tile_flags=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_RENDER;
 tiles[1]=tiles[0];tiles[1].tile_flags=C3X_RENDERER_TILE_VISIBLE; // no draw authority
 c3x_renderer_frame_v1 frame{};frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
 for(unsigned wrap=0;wrap<4;++wrap)for(int extent:{0,-1,1,7,130}){
  frame.world_wrap_x=wrap&1;frame.world_wrap_y=wrap&2;frame.world_width_tiles=extent;frame.world_height_tiles=extent;
  Occurrences indexed(frame,true),linear(frame,false);
  for(unsigned query=0;query<300;++query){int x=query<tiles.size()?tiles[query].tile_x:int(next()%257)-128;
   int y=query<tiles.size()?tiles[query].tile_y:int(next()%257)-128;
   auto key=indexed.key(x,y);std::vector<unsigned> a,b,expected;
   for(unsigned i=indexed.next(key);i!=UINT_MAX;i=indexed.next(key,i))a.push_back(i);
   for(unsigned i=linear.next(key);i!=UINT_MAX;i=linear.next(key,i))b.push_back(i);
   for(unsigned i=0;i<tiles.size();++i){auto const& t=tiles[i];
    bool sx=frame.world_wrap_x&&extent>0?(std::int64_t(t.tile_x)-x)%extent==0:t.tile_x==x;
    bool sy=frame.world_wrap_y&&extent>0?(std::int64_t(t.tile_y)-y)%extent==0:t.tile_y==y;
    if(sx&&sy&&(t.tile_flags&C3X_RENDERER_TILE_VISIBLE)&&(t.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))expected.push_back(i);}
   assert(a==b&&a==expected);
  }
 }
 // Failure to admit either optional allocation retains the native linear iterator.
 for(auto bytes:{1024*sizeof(Occurrences::Bucket),tiles.size()*sizeof(unsigned)}){
  fail_bytes=bytes;Occurrences fallback(frame,true);fail_bytes=0;
  assert(fallback.table.empty()&&fallback.links.empty());
  assert(fallback.next(fallback.key(INT_MIN,INT_MAX))==0);
 }
 // Oversized metadata cannot expand the transient index budget or overflow its power-of-two sizing.
 tiles.resize(8193);frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
 Occurrences bounded(frame,true);assert(bounded.table.empty()&&bounded.links.empty());
 assert(bounded.next(bounded.key(INT_MIN,INT_MAX))==0);
}
''')

if __name__=='__main__':unittest.main()
