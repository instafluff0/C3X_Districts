"""Loading pages and native representative copies precede the first camera."""
import unittest
from Renderer.native.native_cpp_test import run_cpp

class WorldBootstrapTests(unittest.TestCase):
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
 int world_query(c3x_renderer_world_page_v1& p){worker();p={};p.first=state.cursor;p.capacity=128;p.identity=identity;p.frame=frame;
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
 // Native callback refusal never masquerades as completed initial capture.
 c3x_renderer_world_page_v1 p{};p.capacity=128;assert(client.world_seed_submit(p,C3X_RENDERER_RESULT_PENDING)==C3X_RENDERER_RESULT_PENDING);
}
''')
