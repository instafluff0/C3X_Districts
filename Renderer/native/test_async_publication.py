"""Exercise the real bridge with a stalled consumer and recycled caller arrays."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class AsyncPublicationTests(unittest.TestCase):
    def test_consumer_failure_releases_an_already_queued_setup_waiter(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
int main(){
 std::promise<void> entered,release;auto held=release.get_future();std::atomic<int> configured{0};
 c3x_async::Publication queue;
 assert(queue.post(1,[&]{entered.set_value();held.wait();throw std::runtime_error("helper lost");}));
 entered.get_future().get();
 auto result=std::async(std::launch::async,[&]{
  try{queue.setup([&]{++configured;return 1;});return false;}catch(std::exception const&){return true;}
 });
 auto deadline=std::chrono::steady_clock::now()+2s;
 while(queue.accepted()!=2&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
 assert(queue.accepted()==2);release.set_value();
 assert(result.wait_for(2s)==std::future_status::ready&&result.get());
 queue.stop();assert(!queue.healthy()&&configured==0);
}
''')

    def test_injected_unit_publication_has_no_cpu_fallback(self):
        source=(ROOT/'injected_code.c').read_text()
        start=source.index('\t// The resident path consumes native image identities')
        body=source[start:source.index('\n}\n',start)]
        run_cpp(r'''
#include <cassert>
using RECT=int;
constexpr int C3X_NATIVE_UNIT_DRAW=1,C3X_RENDERER_RESULT_OK=1,C3X_RENDERER_RESULT_ERROR=0;
int reply=0,errors=0;
int translate_custom_renderer_native(int,int*,int*,RECT*,RECT*,unsigned){return reply;}
void log_custom_renderer_event(char const*,int){++errors;}
struct Unit {struct {struct {int left=0,top=0,right=0,bottom=0;}Rect;}Body;}unit;
bool forward(){
 int draw=0,image=0,underlay=0,body_bounds[4]={-1,-2,3,4};unsigned flags=0;
 auto display_unit=&unit;
''' + body.replace('C3X_NATIVE_UNIT_DRAW, image, underlay,','C3X_NATIVE_UNIT_DRAW, &image, &underlay,') + r'''
}
int main(){
 reply=0;assert(!forward()&&errors==1&&unit.Body.Rect.right==0);
 reply=-1;assert(!forward()&&errors==2&&unit.Body.Rect.right==0);
 reply=1;assert(forward()&&errors==2&&unit.Body.Rect.left==-1&&unit.Body.Rect.bottom==4);
}
''')

    def test_latest_camera_replaces_pending_payload_and_preserves_reliable_order(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <vector>
int main(){
 std::promise<void> entered,release;auto held=release.get_future();std::vector<int> seen;
 c3x_async::Publication queue({},128,8);
 assert(queue.post(32,[&]{entered.set_value();held.wait();}));entered.get_future().get();
 assert(queue.post(32,[&]{seen.push_back(-1);}));
 for(int n=0;n<10000;++n)assert(queue.post(32,[&,n]{seen.push_back(n);},1));
 assert(queue.post(32,[&]{seen.push_back(-2);}));
 assert(queue.post(32,[&]{seen.push_back(10000);},1));
 release.set_value();queue.stop();
 assert(queue.healthy());assert(seen==std::vector<int>({-1,-2,10000}));
 assert(queue.accepted()==queue.completed());
}
''')

    def test_ordered_ids_camera_adoption_and_owned_payloads(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <chrono>
#include <cstring>
using namespace std::chrono_literals;
struct State {
 std::mutex mutex;std::condition_variable wake;bool held=false,entered=false;
 int creates=0,adoptions=0,reads=0,observations=0,pages=0,animations=0;long long current=0;
 std::vector<unsigned> uploaded;std::vector<long long> sources;
 void barrier(){std::unique_lock<std::mutex> lock(mutex);entered=true;wake.notify_all();wake.wait(lock,[&]{return !held;});}
 void hold(){std::lock_guard<std::mutex> lock(mutex);held=true;entered=false;}
 void release(){std::lock_guard<std::mutex> lock(mutex);held=false;wake.notify_all();}
 void wait(){std::unique_lock<std::mutex> lock(mutex);assert(wake.wait_for(lock,2s,[&]{return entered;}));}
};
struct Fake {
 State& state;long long next=0; c3x_renderer_frame_v1 frame={};c3x_renderer_tile_v1 tile={};
 explicit Fake(State& value):state(value){}
 bool alive()const{return true;}
 int stats(){return state.creates;}
 int camera_begin(c3x_renderer_camera_request_v1 const& value,long long& ticket){
  state.barrier();frame=*value.frame;tile=frame.tiles[0];frame.tiles=&tile;ticket=++next;return C3X_RENDERER_RESULT_PENDING;
 }
 int camera_ready(long long ticket,c3x_renderer_gpu_camera_view_v1& value){
  value={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(value)};value.camera.frame=frame;value.camera.ticket=ticket;
  value.image={sizeof(value.image)};value.image.width=640;value.image.height=480;
  value.camera.output={C3X_RENDERER_API_VERSION,sizeof(value.camera.output)};
  value.camera.output.width=640;value.camera.output.height=480;return C3X_RENDERER_RESULT_OK;
 }
 int camera_poll(long long ticket,c3x_renderer_gpu_camera_view_v1& value){
  ++state.adoptions;state.current=ticket;camera_ready(ticket,value);value.image.ticket=100+ticket;value.image.map_image=1000+ticket;return C3X_RENDERER_RESULT_OK;
 }
 int camera_cancel(long long){return C3X_RENDERER_RESULT_OK;}
 int world_submit(c3x_renderer_world_page_v1 const& page,int code){
  if(code!=C3X_RENDERER_RESULT_OK)return code;
  assert(page.count==1&&page.tiles[0].anchor_x==17);++state.pages;return C3X_RENDERER_RESULT_OK;
 }
 int world_delta_submit(c3x_renderer_world_page_v1 const& page,int code){return world_submit(page,code);}
 int images(c3x_renderer_gpu_images_v1 const& value,c3x_renderer_gpu_result_v1& result,unsigned*,unsigned){
  state.barrier();assert(value.ticket==100+state.current);
  for(unsigned n=0;n<value.command_count;++n){state.sources.push_back(value.commands[n].source);assert(value.commands[n].source==1000+state.current);}
  if(value.action==C3X_GPU_CREATE)result.image=2000+(++state.creates);
  else if(value.action==C3X_GPU_UPLOAD){assert(value.image>=2001);state.uploaded.assign(value.pixels,value.pixels+value.pixel_count);}
  else if(value.action==C3X_GPU_READBACK)++state.reads;
  return C3X_RENDERER_RESULT_OK;
 }
 int unit_animation(c3x_renderer_unit_animation_v1 const& value){
  assert(value.visual.unit_id==7&&value.visual.target_x==384&&value.frames==10&&value.frame_seconds==.125f&&value.display_unit_id==7);
  ++state.animations;return C3X_RENDERER_RESULT_OK;
 }
 int unit(c3x_renderer_unit_v1 const& value,c3x_renderer_gpu_unit_v1 const& target,int*){
  assert(!target.ticket);assert(std::string(value.unit_key)=="copied");++state.observations;return C3X_RENDERER_RESULT_OK;
 }
};
int main(){
 State state;state.hold();
 c3x_remote_scene::AsyncSceneClient<Fake> client(true,[](char const*){assert(false);},state);
 c3x_renderer_tile_v1 tile={};tile.anchor_x=17;
 c3x_renderer_frame_v1 frame={};frame.tiles=&tile;frame.tile_count=1;
 c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,{}};
 long long camera=0;auto start=std::chrono::steady_clock::now();
 assert(client.camera_begin(request,camera)==C3X_RENDERER_RESULT_PENDING);
 assert(std::chrono::steady_clock::now()-start<100ms);state.wait();tile.anchor_x=999;
 c3x_renderer_gpu_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
 for(int n=0;n<1000;++n)assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_PENDING);
 c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);std::strcpy(unit.unit_key,"copied");
 c3x_renderer_gpu_unit_v1 target={sizeof(target)};int bounds[4]={};
 assert(client.unit(unit,target,bounds)==C3X_RENDERER_RESULT_OK);std::strcpy(unit.unit_key,"reused");
 c3x_renderer_unit_animation_v1 animation{};animation.visual.unit_id=7;animation.visual.target_x=384;
 animation.frames=10;animation.frame_seconds=.125f;animation.display_unit_id=7;
 auto posted=std::chrono::steady_clock::now();assert(client.unit_animation(animation)==C3X_RENDERER_RESULT_OK);
 assert(std::chrono::steady_clock::now()-posted<100ms);animation.visual.target_x=-1;animation.frames=999;animation.display_unit_id=999;
 state.release();client.stats();assert(state.adoptions==0&&state.observations==1&&state.animations==1);
 // Civ III refuses background world capture until the first map is displayed.
 // This is a deferred capture, not an empty snapshot or a renderer failure.
 c3x_renderer_world_page_v1 page={};page.struct_size=sizeof(page);page.capacity=128;
 for(int deferred:{C3X_RENDERER_RESULT_PENDING,C3X_RENDERER_RESULT_SUPERSEDED,C3X_RENDERER_RESULT_ERROR}){
  assert(client.world_submit(page,deferred)==deferred);
  assert(client.world_delta_submit(page,deferred)==deferred);
 }
 client.stats();assert(client.alive()&&state.pages==0);
 assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_OK);
 assert(view.camera.frame.tiles[0].anchor_x==17&&view.camera.ticket==camera);
 auto first=view.image;client.stats();assert(state.adoptions==1);
 // Once capture is available, its copied records still enter in order.
 tile.anchor_x=17;page.count=1;page.tiles=&tile;
 assert(client.world_submit(page,C3X_RENDERER_RESULT_OK)==C3X_RENDERER_RESULT_OK);
 assert(client.world_delta_submit(page,C3X_RENDERER_RESULT_OK)==C3X_RENDERER_RESULT_OK);
 tile.anchor_x=999;client.stats();assert(state.pages==2&&client.alive());
 // Pause the actual consumer for two seconds. Creates still return distinct
 // usable identities; caller memory can immediately be overwritten or freed.
 state.hold();c3x_renderer_gpu_images_v1 image={};image.struct_size=sizeof(image);image.ticket=first.ticket;
 image.action=C3X_GPU_CREATE;image.width=image.height=2;
 c3x_renderer_gpu_result_v1 made={sizeof(made)};
 assert(client.images(image,made,nullptr,0)==C3X_RENDERER_RESULT_OK);state.wait();
 auto reserved=made.image;assert(reserved>0&&reserved!=first.map_image);
 unsigned pixels[4]={11,22,33,44};image.action=C3X_GPU_UPLOAD;image.image=reserved;image.pixels=pixels;image.pixel_count=4;
 auto before=std::chrono::steady_clock::now();assert(client.images(image,made,nullptr,0)==C3X_RENDERER_RESULT_OK);
 assert(std::chrono::steady_clock::now()-before<100ms);pixels[0]=999;
 std::this_thread::sleep_for(2s);state.release();client.stats();assert(state.uploaded[0]==11);
 // Inspecting a new camera leaves the old ticket usable until adoption is
 // explicitly queued. Old draws precede adoption; new draws use its new map.
 assert(client.camera_begin(request,camera)==C3X_RENDERER_RESULT_PENDING);client.stats();
 assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_PENDING);client.stats();assert(state.adoptions==1);
 c3x_renderer_gpu_command_v1 draw={};draw.source=first.map_image;draw.destination=reserved;
 image={};image.struct_size=sizeof(image);image.action=C3X_GPU_SUBMIT;image.ticket=first.ticket;image.commands=&draw;image.command_count=1;
 assert(client.images(image,made,nullptr,0)==C3X_RENDERER_RESULT_OK);
 assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_OK);
 image.ticket=view.image.ticket;draw.source=view.image.map_image;
 assert(client.images(image,made,nullptr,0)==C3X_RENDERER_RESULT_OK);draw.source=-1;
 client.stats();assert(state.adoptions==2&&state.sources==std::vector<long long>({1001,1002}));
 image.action=C3X_GPU_READBACK;assert(client.images(image,made,nullptr,0)==C3X_RENDERER_RESULT_BAD_ARGUMENT);assert(state.reads==0);
 assert(client.camera_cancel(camera)==C3X_RENDERER_RESULT_OK);
 assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_SUPERSEDED);
}
''')

    def test_bounded_overflow_stops_publication_without_wait_or_partial_frame(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
int main(){
 std::promise<void> entered,release;auto held=release.get_future();std::atomic<int> errors{0},frames{0};
 c3x_async::Publication queue([&](char const*){++errors;},64,8);
 assert(queue.post(32,[&]{entered.set_value();held.wait();}));entered.get_future().get();
 assert(queue.post(32,[&]{++frames;}));auto start=std::chrono::steady_clock::now();
 assert(!queue.post(1,[&]{++frames;}));assert(std::chrono::steady_clock::now()-start<100ms);
 assert(!queue.healthy()&&errors==1);release.set_value();queue.stop();assert(frames==0);
}
''')


if __name__ == "__main__":
    unittest.main()
