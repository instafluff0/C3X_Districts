"""Required world paging survives a busy camera and visual-worker lock."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class RequiredWorldQueryTests(unittest.TestCase):
    def test_required_pages_wait_and_preserve_authority_while_background_yields(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        start = source.index('    int trial_world_query(')
        methods = source[start:source.index('\n#endif', start)]
        run_cpp(r'''
#include "Renderer/native/render_core/world_input_capture.h"
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
struct Worker {
 std::mutex call_mutex,state_mutex;
 bool has_job=false,camera_pending=false,camera_active=false,camera_paused=false;
 c3x_renderer::render_core::ScenePublication scene_changes;
 c3x_renderer::render_core::WorldInputCapture world_input;
 enum class Command {publish_world};
 int submit_locked(std::unique_lock<std::mutex>& lock,Command){assert(lock.owns_lock());return C3X_RENDERER_RESULT_OK;}
''' + methods + r'''
};
struct Transport {
 Worker& worker;
 explicit Transport(Worker& value):worker(value){}
 bool alive()const{return true;}
 void publication_pressure(std::size_t){}
 int world_query(c3x_renderer_world_page_v1& page,bool required=false){return worker.trial_world_query(page,required);}
 int world_submit(c3x_renderer_world_page_v1 const& page,int code){return worker.trial_world_submit(page,code);}
};
int main(){
 Worker w;c3x_renderer_world_page_v1 page{};
 assert(w.trial_world_query(page)==C3X_RENDERER_RESULT_PENDING);
 assert(w.trial_world_query(page,true)==C3X_RENDERER_RESULT_SUPERSEDED);
 std::vector<unsigned> topology(256,2|(2<<8));c3x_renderer_frame_v1 frame{};
 frame.world_width_tiles=16;frame.world_height_tiles=32;
 frame.world_topology=topology.data();frame.world_topology_count=256;frame.world_topology_revision=7;
 c3x_renderer_camera_identity_v1 identity{1,1,3,7};assert(w.scene_changes.capture(frame,identity));
 for(auto flag:{&w.has_job,&w.camera_pending,&w.camera_active,&w.camera_paused}){
  *flag=true;assert(w.trial_world_query(page)==C3X_RENDERER_RESULT_PENDING);
  assert(w.trial_world_query(page,true)==C3X_RENDERER_RESULT_OK);
  assert(page.first==0&&page.capacity==128&&!page.tiles&&!page.frame.world_topology);*flag=false;
 }
 // A real contender holds each lock. Required queries wait without confusing
 // transient contention with a permanent initialization error.
 for(auto gate:{&w.call_mutex,&w.state_mutex}){
  std::unique_lock<std::mutex> held(*gate);
  auto background=std::async(std::launch::async,[&]{c3x_renderer_world_page_v1 p{};return w.trial_world_query(p);});
  assert(background.get()==C3X_RENDERER_RESULT_PENDING);
  std::promise<void> entered;
  auto required=std::async(std::launch::async,[&]{entered.set_value();return w.trial_world_query(page,true);});
  entered.get_future().wait();assert(required.wait_for(20ms)==std::future_status::timeout);
  held.unlock();assert(required.get()==C3X_RENDERER_RESULT_OK);
 }
 // Exercise the actual ordered client path for both asynchronous and direct
 // admission, with a camera still pending throughout the complete world pass.
 for(bool async:{false,true}){
  w.world_input.reset();w.camera_pending=true;
  c3x_remote_scene::AsyncSceneClient<Transport> client(async,[](char const*){assert(false);},w);
  for(unsigned first:{0u,128u}){
   assert(client.world_seed_query(page)==C3X_RENDERER_RESULT_OK&&page.first==first);
   assert(page.identity.visibility_epoch==3&&page.identity.scene_epoch==7);
   c3x_renderer_tile_v1 records[128]{};page.tiles=records;page.count=128;
   for(unsigned i=0;i<128;++i){auto n=first+i;auto& tile=records[i];
    tile.tile_y=n/8;tile.tile_x=2*(n%8)+(tile.tile_y&1);
    tile.terrain_type=tile.real_terrain_type=2;
    tile.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|
     C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO;
   }
   auto stale=page;++stale.identity.visibility_epoch;
   assert(client.world_seed_submit(stale,C3X_RENDERER_RESULT_OK)==C3X_RENDERER_RESULT_SUPERSEDED);
   assert(client.world_seed_submit(page,C3X_RENDERER_RESULT_PENDING)==C3X_RENDERER_RESULT_PENDING);
   assert(w.world_input.cursor==first&&!w.world_input.passes);
   assert(client.world_seed_submit(page,C3X_RENDERER_RESULT_OK)==C3X_RENDERER_RESULT_OK);
  }
  assert(w.world_input.passes==1&&w.world_input.cursor==0&&w.world_input.records==256);
 }
}
''')


if __name__ == '__main__':
    unittest.main()
