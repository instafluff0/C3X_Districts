"""Loading pages and native representative copies precede the first camera."""
import unittest
from pathlib import Path
from Renderer.native.native_cpp_test import run_cpp

class WorldBootstrapTests(unittest.TestCase):
    def test_changed_world_pages_preserve_completed_regions_and_exact_required_marker(self):
        source=(Path(__file__).parent/'c3x_renderer.cpp').read_text()
        methods='    bool required_world_changes_for'+source.split('    bool required_world_changes_for',1)[1].split('    int reconcile_world()',1)[0]
        arm=source.split('} else if (command == Command::require_world_changes) {',1)[1].split('} else if (command == Command::reset)',1)[0]
        run_cpp(r'''
#include "Renderer/native/render_core/world_input_capture.h"
#include "Renderer/native/render_core/world_preparation_region.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Worker {
 ScenePublication scene_changes;WorldInputCapture world_input;bool scene_changes_ok=true,required_world_changes=false;
 c3x_renderer_camera_identity_v1 required_world_changes_identity={},job_required_world_identity={};
'''+methods+r'''
 int arm(c3x_renderer_camera_identity_v1 identity){job_required_world_identity=identity;int result=0;
'''+arm+r'''
 return result;}
};
int main(){
 Worker w;CapturedScene scene;WorldPreparationSchedule schedule;
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=64;f.tile_width=128;f.tile_height=64;
 f.target_width=2240;f.target_height=1260;std::vector<unsigned> topology(2048,2|(2<<8));
 f.world_topology=topology.data();f.world_topology_count=unsigned(topology.size());
 c3x_renderer_camera_identity_v1 identity{1,1,1,1};assert(w.scene_changes.capture(f,identity));
 bool changed=false;assert(w.scene_changes.apply(scene,changed));
 assert(w.arm(identity)==C3X_RENDERER_RESULT_SUPERSEDED);
 auto copy=[&](bool changed_city){
  unsigned pages=0;while(w.world_input.needs_snapshot(*w.scene_changes.state())){
   auto page=w.world_input.page(*w.scene_changes.state());page.count=std::min(128u,2048-page.first);
   for(unsigned n=0;n<page.count;++n){auto index=page.first+n;auto& tile=page.tiles[n];tile={};
    tile.tile_y=index/32;tile.tile_x=2*(index%32)+(tile.tile_y&1);tile.terrain_type=tile.real_terrain_type=2;
    tile.city_id=changed_city && tile.tile_x==30 && tile.tile_y==30?7:-1;
    tile.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE|
      C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO;
   }
   assert(w.world_input.accept(page,w.scene_changes));std::vector<std::pair<int,int>> dirty;
   assert(w.scene_changes.apply(scene,changed,&dirty));
   for(auto const& tile:dirty)schedule.invalidate(f,tile.first,tile.second);
   ++pages;if(pages==1)assert(w.arm(identity)==C3X_RENDERER_RESULT_SUPERSEDED);
  }assert(pages==16 && w.world_input.passes==1 && !w.world_input.cursor);
 };
 copy(false);schedule.configure(f,scene.scope_sequence(),2,3,true,4);
 while(!schedule.empty())schedule.finish(true);assert(schedule.completed==64);
 assert(w.arm(identity)==C3X_RENDERER_RESULT_OK && w.required_world_changes_for(identity));
 assert(!w.consume_required_world_changes(identity,false) && w.required_world_changes_for(identity));
 auto wrong=identity;++wrong.scene_epoch;
 assert(!w.consume_required_world_changes(wrong,true) && w.required_world_changes_for(identity));
 assert(w.consume_required_world_changes(identity,true) && !w.required_world_changes_for(identity));
 ++identity.scene_epoch;assert(w.scene_changes.capture(f,identity));assert(w.scene_changes.apply(scene,changed));
 w.world_input.reset();copy(true);auto retained=schedule.completed;
 assert(retained>0 && retained<64);schedule.configure(f,scene.scope_sequence(),2,3,true,4);
 assert(schedule.completed==retained);auto pending=64-retained;unsigned rebuilt=0;
 while(!schedule.empty()){schedule.finish(true);++rebuilt;}
 assert(rebuilt==pending && schedule.completed==64 && !schedule.unavailable);
 assert(w.arm(identity)==C3X_RENDERER_RESULT_OK && w.required_world_changes_for(identity));
 assert(!w.consume_required_world_changes(identity,false) && w.required_world_changes_for(identity));
 assert(w.consume_required_world_changes(identity,true));
}
''')

    def test_real_ordered_client_bootstraps_owned_pages_and_bodies_on_game_boundary(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <thread>
using namespace c3x_remote_scene;
struct State {unsigned cursor=0,units=0;std::thread::id game,transport;std::vector<int> order;};
struct Fake {
 State& state;c3x_renderer_camera_identity_v1 identity{};c3x_renderer_frame_v1 frame{};
 std::vector<unsigned> topology;std::vector<c3x_renderer_tile_v1> records;
 explicit Fake(State& s):state(s){}
 bool alive()const{return true;}void publication_pressure(std::size_t){}void supersede_pending_camera(){}
 void worker(){assert(std::this_thread::get_id()!=state.game);if(state.transport==std::thread::id{})state.transport=std::this_thread::get_id();assert(state.transport==std::this_thread::get_id());}
 int seed_world_scope(c3x_renderer_camera_request_v1 const& r){worker();identity=r.identity;frame=*r.frame;
  assert(frame.tile_count==1&&frame.tiles[0].city_id==17);topology.assign(frame.world_topology,frame.world_topology+frame.world_topology_count);
  assert(topology.size()==3200&&topology[0]==7);records.resize(topology.size());state.order.push_back(1);return 1;}
 int world_query(c3x_renderer_world_page_v1& p,bool required=false){assert(required);worker();p={};p.first=state.cursor;p.capacity=128;p.identity=identity;p.frame=frame;
  p.frame.tiles=nullptr;p.frame.tile_count=0;p.frame.world_topology=nullptr;p.frame.world_topology_count=0;return 1;}
 int world_submit(c3x_renderer_world_page_v1 const& p,int code){worker();if(code!=1)return code;
  assert(p.first==state.cursor&&p.count==128&&p.identity.map_epoch==identity.map_epoch);
  for(unsigned i=0;i<p.count;++i){assert(p.tiles[i].tile_x==int(p.first+i));records[p.first+i]=p.tiles[i];}
  state.cursor+=p.count;state.order.push_back(2);return 1;}
 int unit(c3x_renderer_unit_v1 const& u,c3x_renderer_gpu_unit_v1 const& target,int*){worker();
  assert(state.cursor==3200&&u.unit_id==71&&!target.ticket&&!target.destination);
  assert(std::string(u.unit_key)=="native-representative");++state.units;state.order.push_back(3);return 1;}
 int camera_begin(c3x_renderer_camera_request_v1 const& r,long long& ticket){worker();
  assert(state.cursor==3200&&state.units==1&&r.identity.map_epoch==identity.map_epoch);
  assert(records[3199].city_id==3199&&topology[0]==7);state.order.push_back(4);ticket=1;return C3X_RENDERER_RESULT_PENDING;}
 unsigned stats(){worker();return state.cursor;}
 int arm_world_changes(c3x_renderer_camera_identity_v1 const& value){worker();
  if(std::memcmp(&value,&identity,sizeof(value)))return C3X_RENDERER_RESULT_SUPERSEDED;
  assert(state.cursor==3200 && state.units==1);state.order.push_back(5);return C3X_RENDERER_RESULT_OK;}
};
int main(){
 State state;state.game=std::this_thread::get_id();AsyncSceneClient<Fake> client(true,[](char const*){assert(false);},state);
 c3x_renderer_tile_v1 tile{};tile.city_id=17;
 std::vector<unsigned> topology(3200,7);c3x_renderer_frame_v1 frame{};
 frame.tiles=&tile;frame.tile_count=1;frame.world_topology=topology.data();frame.world_topology_count=unsigned(topology.size());
 frame.world_width_tiles=frame.world_height_tiles=80;
 c3x_renderer_camera_request_v1 r{C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(r),&frame,{1,1,1,1}};
 assert(client.seed_world_scope(r)==1);tile.city_id=99;topology[0]=99;
 while(state.cursor<3200){c3x_renderer_world_page_v1 p{};assert(client.world_seed_query(p)==1);
  // Copied native pages are read on the game thread, never the transport worker.
  assert(std::this_thread::get_id()==state.game);c3x_renderer_tile_v1 values[128]{};p.tiles=values;p.count=128;
  for(unsigned i=0;i<p.count;++i){values[i].tile_x=int(p.first+i);values[i].city_id=int(p.first+i);}
  assert(client.world_seed_submit(p,1)==1);values[0].city_id=-100;}
 c3x_renderer_unit_v1 unit{};unit.struct_size=sizeof(unit);unit.unit_id=71;std::strcpy(unit.unit_key,"native-representative");
 c3x_renderer_gpu_unit_v1 target{sizeof(target)};int bounds[4]{};
 assert(client.unit(unit,target,bounds)==1);std::strcpy(unit.unit_key,"reused-storage");
 long long ticket=0;assert(client.camera_begin(r,ticket)==C3X_RENDERER_RESULT_PENDING);assert(client.stats()==3200);
 assert(state.order.front()==1&&state.order.size()==28&&state.order[26]==3&&state.order.back()==4);
 assert(client.arm_world_changes(r.identity)==C3X_RENDERER_RESULT_OK && state.units==1 && state.order.back()==5);
 auto stale=r.identity;++stale.viewer_epoch;assert(client.arm_world_changes(stale)==C3X_RENDERER_RESULT_SUPERSEDED);
 // Native callback refusal never masquerades as completed initial capture.
 c3x_renderer_world_page_v1 p{};p.capacity=128;assert(client.world_seed_submit(p,C3X_RENDERER_RESULT_PENDING)==C3X_RENDERER_RESULT_PENDING);
}
''')
