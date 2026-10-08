"""Exercise the real bridge with a stalled consumer and recycled caller arrays."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class AsyncPublicationTests(unittest.TestCase):
    def test_world_preparation_retires_camera_reuse_but_preserves_displayed_images(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
struct State {
 int begins=0,adoptions=0,preparations=0,draws=0;
 int preparation_result=C3X_RENDERER_RESULT_OK;
 bool animating=false;long long current=0;
};
struct Fake {
 State& state;c3x_renderer_frame_v1 frame={};
 explicit Fake(State& value):state(value){}
 bool alive()const{return true;}
 void supersede_pending_camera(){}
 void publication_pressure(std::size_t){}
 int stats(){return 0;}
 int camera_begin(c3x_renderer_camera_request_v1 const& request,long long& ticket){
  frame=*request.frame;ticket=++state.begins;return C3X_RENDERER_RESULT_PENDING;
 }
 int camera_ready(long long ticket,c3x_renderer_gpu_camera_view_v1& view){
  view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
  view.camera.frame=frame;view.camera.ticket=ticket;
  view.camera.output={C3X_RENDERER_API_VERSION,sizeof(view.camera.output)};
  view.image={sizeof(view.image)};return C3X_RENDERER_RESULT_OK;
 }
 int camera_poll(long long ticket,c3x_renderer_gpu_camera_view_v1& view){
  camera_ready(ticket,view);view.image.ticket=100+ticket;view.image.map_image=1000+ticket;
  state.current=ticket;++state.adoptions;state.animating=true;return C3X_RENDERER_RESULT_OK;
 }
 int prepare_world_loading(c3x_renderer_camera_identity_v1 const&){
  ++state.preparations;state.animating=false;return state.preparation_result;
 }
 int images(c3x_renderer_gpu_images_v1 const& request,c3x_renderer_gpu_result_v1&,unsigned*,unsigned){
  assert(request.ticket==100+state.current&&request.image==1000+state.current);
  ++state.draws;return C3X_RENDERER_RESULT_OK;
 }
 int images_batch(std::vector<c3x_remote_scene::ImageBatch::Operation>& operations,
                  std::vector<c3x_remote_scene::ImageBatch::Reply>& replies){
  replies=c3x_remote_scene::ImageBatch::execute(operations,[&](auto const& request,auto& result){
   return images(request,result,nullptr,0);
  });return C3X_RENDERER_RESULT_OK;
 }
};
int main(){
 State state;c3x_remote_scene::AsyncSceneClient<Fake> client(true,[](char const*){assert(false);},state);
 c3x_renderer_frame_v1 frame={C3X_RENDERER_API_VERSION,sizeof(frame)};frame.presentation_time_ticks=1;
 c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,{}};
 c3x_renderer_gpu_camera_view_v1 view={};long long camera=0;
 auto adopt=[&]{
  client.stats();assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_PENDING);
  client.stats();assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_OK);client.stats();
 };
 auto draw=[&](c3x_renderer_gpu_frame_v1 const& image){
  c3x_renderer_gpu_images_v1 request={};request.struct_size=sizeof(request);
  request.action=C3X_GPU_SUBMIT;request.ticket=image.ticket;request.image=image.map_image;
  c3x_renderer_gpu_result_v1 result={};
  assert(client.images(request,result,nullptr,0)==C3X_RENDERER_RESULT_OK);client.stats();
 };
 assert(client.camera_begin(request,camera)==C3X_RENDERER_RESULT_PENDING);adopt();
 for(int result:{C3X_RENDERER_RESULT_OK,C3X_RENDERER_RESULT_ERROR,C3X_RENDERER_RESULT_SUPERSEDED}){
  auto previous=camera;auto image=view.image;int begins=state.begins,adoptions=state.adoptions;
  // Ordinary clock-only repetition must keep the current animated camera.
  ++frame.presentation_time_ticks;++frame.visible_animation_count;
  assert(client.camera_begin(request,camera)==C3X_RENDERER_RESULT_PENDING&&camera==previous);
  assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_OK);client.stats();
  assert(state.begins==begins&&state.adoptions==adoptions&&state.animating);
  state.preparation_result=result;
  assert(client.prepare_world_loading(request.identity)==result&&!state.animating);
  draw(image); // Completed native pixels and aliases survive world preparation.
  assert(client.camera_poll(previous,view)==C3X_RENDERER_RESULT_SUPERSEDED);
  assert(client.camera_begin(request,camera)==C3X_RENDERER_RESULT_PENDING&&camera!=previous);
  draw(image); // Also survive a pending replacement, until ordered adoption.
  adopt();assert(state.begins==begins+1&&state.adoptions==adoptions+1&&state.animating);
 }
 assert(state.preparations==3&&state.draws==6&&client.alive());
 // The synchronous transport continues to return its actual preparation status.
 State direct;c3x_remote_scene::AsyncSceneClient<Fake> sync(false,{},direct);
 direct.preparation_result=C3X_RENDERER_RESULT_ERROR;
 assert(sync.prepare_world_loading(request.identity)==C3X_RENDERER_RESULT_ERROR&&direct.preparations==1);
}
''')

    def test_repeated_visual_permission_posts_once_until_reset(self):
        # Civ III sends the same ambient permission on every present. Each post
        # cost two helper round trips on the saturated busy-map transport.
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <vector>
struct State {std::vector<unsigned> sets;int resets=0;};
struct Fake {
 State& state;explicit Fake(State& value):state(value){}
 bool alive()const{return true;}
 void publication_pressure(std::size_t){}
 int stats(){return 0;}
 int reset(){++state.resets;return C3X_RENDERER_RESULT_OK;}
 int visual_policy(unsigned value){if(value<2)state.sets.push_back(value);return 1;}
};
int main(){
 State state;c3x_remote_scene::AsyncSceneClient<Fake> client(true,[](char const*){assert(false);},state);
 for(int n=0;n<5;++n)assert(client.visual_policy(1)==C3X_RENDERER_RESULT_OK);
 client.stats();assert((state.sets==std::vector<unsigned>{1})&&client.visual_policy(2)==1);
 client.visual_policy(0);client.visual_policy(0);client.visual_policy(1);client.stats();
 assert((state.sets==std::vector<unsigned>{1,0,1}));
 // A reset clears the helper's state, so the same permission is sent again.
 assert(client.reset()==C3X_RENDERER_RESULT_OK&&state.resets==1);
 client.visual_policy(1);client.visual_policy(1);client.stats();
 assert((state.sets==std::vector<unsigned>{1,0,1,1}));
}
''')

    def test_consecutive_tactical_strokes_join_one_ordered_record(self):
        # Every native line was its own helper round trip (~150 a second on the
        # busy map). Lines into the same canvas that queue back to back now
        # share one record; anything else between them keeps its place.
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <future>
#include <vector>
struct Draw {std::vector<float> xs;long long destination;bool animated;};
struct State {std::promise<void> entered,release;std::shared_future<void> held=release.get_future().share();
 bool stalled=false;std::vector<Draw> draws;std::vector<int> units;};
struct Fake {
 State& state;explicit Fake(State& value):state(value){}
 bool alive()const{return true;}
 void publication_pressure(std::size_t){}
 int stats(){return 0;}
 int visual_policy(unsigned value){if(value<2&&!state.stalled){state.stalled=true;state.entered.set_value();state.held.wait();}return 1;}
 int set_units(int value){state.units.push_back(value);state.draws.push_back({{},-1,false});return C3X_RENDERER_RESULT_OK;}
 int tactical(c3x_renderer::tactical::Input const& capture,c3x_renderer_gpu_unit_v1 const& target){
  Draw draw{{},target.destination,capture.animated};for(auto const& p:capture.primitives)draw.xs.push_back(p.shape[0]);
  state.draws.push_back(draw);return C3X_RENDERER_RESULT_OK;}
};
int main(){
 State state;c3x_remote_scene::AsyncSceneClient<Fake> client(true,[](char const*){assert(false);},state);
 client.visual_policy(1);state.entered.get_future().get(); // the consumer is busy
 auto line=[&](long long destination,float x,bool animated=false){
  c3x_renderer::tactical::Input capture;capture.line(x,0,x+4,0);capture.animated=animated;
  c3x_renderer_gpu_unit_v1 target={sizeof(target)};target.destination=destination;target.clip[2]=640;target.clip[3]=480;
  assert(client.tactical(capture,target)==C3X_RENDERER_RESULT_OK);};
 line(1,0);line(1,1);line(1,2);line(2,10);line(1,3);
 assert(client.set_units(1)==C3X_RENDERER_RESULT_OK);
 line(1,4);line(1,5,true);line(1,6);
 state.release.set_value();client.stats();
 auto& d=state.draws;assert(d.size()==7);
 assert((d[0].xs==std::vector<float>{0.f,1.f,2.f})&&d[0].destination==1&&!d[0].animated);
 assert((d[1].xs==std::vector<float>{10.f})&&d[1].destination==2);
 assert((d[2].xs==std::vector<float>{3.f})&&d[2].destination==1);
 assert(d[3].destination==-1&&(state.units==std::vector<int>{1}));
 assert((d[4].xs==std::vector<float>{4.f})&&!d[4].animated);
 assert((d[5].xs==std::vector<float>{5.f})&&d[5].animated);
 assert((d[6].xs==std::vector<float>{6.f})&&!d[6].animated);
}
''')

    def test_large_tactical_captures_keep_one_work_unit_each(self):
        # Counting each primitive as queue work exhausted the 65536-unit budget
        # behind a busy-map backlog (route and grid captures carry thousands).
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <future>
struct State {std::promise<void> entered,release;std::shared_future<void> held=release.get_future().share();
 bool stalled=false;std::size_t primitives=0,records=0;float last=-1;bool ordered=true;};
struct Fake {
 State& state;explicit Fake(State& value):state(value){}
 bool alive()const{return true;}
 void publication_pressure(std::size_t){}
 int stats(){return 0;}
 int visual_policy(unsigned value){if(value<2&&!state.stalled){state.stalled=true;state.entered.set_value();state.held.wait();}return 1;}
 int tactical(c3x_renderer::tactical::Input const& capture,c3x_renderer_gpu_unit_v1 const&){
  ++state.records;for(auto const& p:capture.primitives){state.ordered&=p.shape[0]>state.last;state.last=p.shape[0];++state.primitives;}
  return C3X_RENDERER_RESULT_OK;}
};
int main(){
 State state;bool failed=false;
 c3x_remote_scene::AsyncSceneClient<Fake> client(true,[&](char const*){failed=true;},state);
 client.visual_policy(1);state.entered.get_future().get();
 c3x_renderer_gpu_unit_v1 target={sizeof(target)};target.destination=1;target.clip[2]=640;target.clip[3]=480;
 float x=0;
 for(int n=0;n<30;++n){c3x_renderer::tactical::Input capture;
  for(int k=0;k<3000;++k){capture.line(x,0,x+1,0);++x;}
  assert(client.tactical(capture,target)==C3X_RENDERER_RESULT_OK);}
 state.release.set_value();client.stats();
 assert(!failed&&client.alive());
 assert(state.primitives==90000&&state.ordered&&state.records>=8&&state.records<=30);
}
''')

    def test_busy_native_burst_has_independent_work_and_packet_bounds(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
int main(){
 std::promise<void> entered,release;auto held=release.get_future();unsigned executed=0;
 c3x_async::Publication queue;
 assert(queue.post(1,[&]{entered.set_value();held.wait();}));entered.get_future().get();
 // Three measured startup-sized bursts while the consumer is preparing: each
 // is about 2200 packets, 7400 operations and 6 MiB. Operations != packets.
 for(unsigned burst=0;burst<3;++burst)for(unsigned packet=0;packet<2196;++packet)
  assert(queue.post(2734,[&]{++executed;},0,"native-busy",packet<803?4:3));
 auto status=queue.status();assert(status.records==6589 && status.units==22174);
 assert(status.bytes<128u*1024u*1024u && status.rejected==0 && queue.healthy());
 release.set_value();queue.stop();status=queue.status();
 assert(executed==6588 && status.accepted==status.executed && !status.records && !status.units && !status.bytes);
 // Retain the independent packet ceiling even when semantic work is tiny.
 std::promise<void> entered2,release2;auto held2=release2.get_future();c3x_async::Publication bounded;
 assert(bounded.post(1,[&]{entered2.set_value();held2.wait();}));entered2.get_future().get();
 for(unsigned n=1;n<8192;++n)assert(bounded.post(1,[]{}));
 assert(!bounded.post(1,[]{}));assert(bounded.status().peak_records==8192);
 release2.set_value();bounded.stop();assert(bounded.status().rejected==1);
}
''')

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
struct State {bool custom_renderer_unit_bootstrap=false,custom_renderer_unit_bootstrap_failed=false;
 unsigned custom_renderer_unit_bootstrap_copies=0;} state,*is=&state;
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
 state.custom_renderer_unit_bootstrap=true;reply=0;
 assert(!forward()&&state.custom_renderer_unit_bootstrap_failed&&!state.custom_renderer_unit_bootstrap_copies);
 state.custom_renderer_unit_bootstrap_failed=false;reply=1;
 assert(forward()&&!state.custom_renderer_unit_bootstrap_failed&&state.custom_renderer_unit_bootstrap_copies==1);
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
 assert(queue.accepted()==queue.completed()+queue.status().superseded);
}
''')

    def test_camera_request_runs_ahead_of_queued_ui_but_not_state(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
int main(){
 std::promise<void> entered,release;auto held=release.get_future();std::vector<std::string> seen;
 c3x_async::Publication queue({},1<<20,64);
 auto ui=[](char const* label){return label&&(!std::strcmp(label,"images")||!std::strcmp(label,"present"));};
 auto add=[&](char const* name,char const* label,unsigned key=0,bool (*passes)(char const*)=nullptr){
  assert(queue.post(32,[&seen,name]{seen.push_back(name);},key,label,1,passes));};
 assert(queue.post(32,[&]{entered.set_value();held.wait();}));entered.get_future().get();
 add("fact","state");add("ui1","images");add("present1","present");
 add("camera1","camera-begin",1,ui);  // ahead of queued UI, behind the fact
 add("ui2","images");add("fact2","state");add("ui3","images");
 add("camera2","camera-begin",1,ui);  // replaces camera1; stops at fact2
 release.set_value();queue.stop();assert(queue.healthy());
 assert((seen==std::vector<std::string>{"fact","ui1","present1","ui2","fact2","camera2","ui3"}));
 assert(queue.status().superseded==1);
}
''')

    def test_camera_request_runs_during_the_in_flight_image_wait(self):
        # A busy step's camera request waited 121 ms (p50) behind the image
        # batch in flight (performance review 15). The image wait now sends a
        # posted camera request, but only one allowed to pass every entry
        # queued before it, and it is told when such a request is posted.
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <atomic>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
int main(){
 std::vector<std::string> seen;std::vector<std::string> notified;
 c3x_async::Publication queue({},1<<20,64);
 auto ui=[](char const* label){return label&&(!std::strcmp(label,"images")||!std::strcmp(label,"present"));};
 std::promise<void> entered,posted,blocked;auto camera_posted=posted.get_future(),fact_posted=blocked.get_future();
 queue.on_passing_post([&](char const* label){notified.push_back(label);});
 std::atomic<int> first{-1},second{-1};
 assert(queue.post(32,[&]{seen.push_back("images-start");entered.set_value();camera_posted.wait();
   first=queue.run_passing("images","camera-begin");    // camera1 runs inside this wait
   fact_posted.wait();
   second=queue.run_passing("images","camera-begin");   // camera2 is behind a fact: no
   seen.push_back("images-end");},0,"images"));
 entered.get_future().get();
 assert(queue.post(32,[&]{seen.push_back("ui");},0,"images"));
 assert(queue.post(32,[&]{seen.push_back("camera1");},1,"camera-begin",1,ui));posted.set_value();
 while(first<0)std::this_thread::yield();
 assert(queue.post(32,[&]{seen.push_back("fact");},0,"state"));
 assert(queue.post(32,[&]{seen.push_back("camera2");},1,"camera-begin",1,ui));blocked.set_value();
 queue.stop();assert(queue.healthy());
 assert(first==1&&second==0);
 assert((seen==std::vector<std::string>{"images-start","camera1","images-end","ui","fact","camera2"}));
 assert((notified==std::vector<std::string>{"camera-begin","camera-begin"}));
 assert(queue.status().executed==5&&queue.status().records==0);
}
''')

    def test_helper_accepts_only_a_camera_begin_during_an_image_wait(self):
        # The first camera-lane build sent a camera request during an image
        # wait; the helper's reliable-prefix rule faulted the whole UI session
        # (October 8, b50). The helper must accept a camera begin there, and the
        # bridge may send only that during the wait.
        from pathlib import Path
        root=Path(__file__).resolve().parents[2]
        helper=(root/'Renderer/native/helper_trial/scene_workload.cpp').read_text()
        rule=helper[helper.index('require(!image_batches->status().bytes||'):helper.index('"reliable prefix requires image execution receipt");')]
        self.assertIn('(wire.kind==unsigned(Kind::image_commands)&&wire.subtype==2)',rule)
        self.assertIn('(wire.live&&wire.kind==unsigned(Kind::camera)&&wire.subtype==1)',rule)
        self.assertEqual(rule.count('wire.kind=='),2)
        client=(root/'Renderer/sandbox/async_scene_client.h').read_text()
        self.assertIn('queue.run_passing("images","camera-begin")',client)

    def test_scene_facts_and_reveal_pass_canvas_backlog_without_losing_reliable_work(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <cstring>
#include <vector>
int main(){
 std::promise<void> entered,release;auto held=release.get_future();std::vector<int> seen;
 c3x_async::Publication queue({},1<<20,128);
 assert(queue.post(1,[&]{entered.set_value();held.wait();}));entered.get_future().get();
 auto ui=[](char const* label){return label&&!std::strcmp(label,"images");};
 auto image=[&](int id){assert(queue.post(10,[&,id]{seen.push_back(id);},0,"images"));};
 auto fact=[&](int id,bool independent){auto group=std::make_shared<std::vector<int>>(1,id);
  assert(queue.post_group(10,1,3,group,[&](auto& values){seen.insert(seen.end(),values.begin(),values.end());},
   [](auto& a,auto& b){a.insert(a.end(),b.begin(),b.end());},1000,100,100,"state",independent?ui:nullptr));};
 image(100);fact(1,true);image(101);fact(2,true);image(102);
 assert(queue.post(10,[&]{seen.push_back(3);},0,"world-delta",1,ui));
 assert(queue.post(10,[&]{seen.push_back(4);},1,"camera-begin",1,ui));
 // A canvas-bound fact retains the create/use dependency. Later scene-only
 // facts may pass later images, but cannot jump this or the camera barrier.
 image(103);fact(5,false);image(104);fact(6,true);
 auto pending=queue.status();assert(pending.records==12&&pending.bytes==111);
 release.set_value();queue.stop();
 assert((seen==std::vector<int>{1,2,3,4,100,101,102,103,5,6,104}));
 auto done=queue.status();assert(done.accepted==done.executed&&done.bytes==0&&done.records==0&&done.units==0);
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
 void supersede_pending_camera(){}
 void publication_pressure(std::size_t){}
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
 int images_batch(std::vector<c3x_remote_scene::ImageBatch::Operation>& operations,std::vector<c3x_remote_scene::ImageBatch::Reply>& replies){
  replies=c3x_remote_scene::ImageBatch::execute(operations,[&](auto const& request,auto& result){return images(request,result,nullptr,0);});return C3X_RENDERER_RESULT_OK;
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
 tile.tile_x=3;assert(client.camera_begin(request,camera)==C3X_RENDERER_RESULT_PENDING);client.stats();
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

    def test_adoption_observer_binds_local_remote_and_source_tickets_after_cancellation(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
struct Receipt {long long camera,image,remote,source,session;int anchor;};
struct State {
 std::mutex mutex;std::condition_variable wake;bool held=false,entered=false,refuse=false;
 long long cancelled=0;unsigned attempts=0;std::atomic<unsigned> observed{0},errors{0};
 std::vector<Receipt> receipts;
 void barrier(){std::unique_lock<std::mutex> lock(mutex);entered=true;wake.notify_all();wake.wait(lock,[&]{return !held;});}
 void wait(){std::unique_lock<std::mutex> lock(mutex);assert(wake.wait_for(lock,2s,[&]{return entered;}));}
 void release(){std::lock_guard<std::mutex> lock(mutex);held=false;wake.notify_all();}
};
struct Fake {
 State& state;long long next=40,source=700;c3x_renderer_frame_v1 frame={};c3x_renderer_tile_v1 tile={};
 explicit Fake(State& value):state(value){}
 bool alive()const{return true;}
 void supersede_pending_camera(){}
 void publication_pressure(std::size_t){}
 int stats(){return 0;}
 int camera_begin(c3x_renderer_camera_request_v1 const& request,long long& ticket){
  frame=*request.frame;tile=frame.tiles[0];frame.tiles=&tile;ticket=++next;return C3X_RENDERER_RESULT_PENDING;
 }
 int camera_ready(long long ticket,c3x_renderer_gpu_camera_view_v1& value){
  value={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(value)};
  value.camera.ticket=ticket;value.camera.frame=frame;
  value.camera.output={C3X_RENDERER_API_VERSION,sizeof(value.camera.output)};
  value.camera.output.width=640;value.camera.output.height=480;
  value.image={sizeof(value.image)};value.image.width=640;value.image.height=480;
  return C3X_RENDERER_RESULT_OK;
 }
 int camera_poll(long long ticket,c3x_renderer_gpu_camera_view_v1& value){
  ++state.attempts;state.barrier();
  if(state.refuse)return C3X_RENDERER_RESULT_SUPERSEDED;
  camera_ready(ticket,value);value.image.ticket=++source;
  value.image.map_image=1000+source;value.image.session=9001;return C3X_RENDERER_RESULT_OK;
 }
 int camera_cancel(long long ticket){state.cancelled=ticket;return C3X_RENDERER_RESULT_OK;}
};
int main(){
 c3x_renderer_tile_v1 tile={};tile.anchor_x=17;
 c3x_renderer_frame_v1 frame={};frame.tiles=&tile;frame.tile_count=1;
 c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,{}};
 c3x_renderer_gpu_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
 State state;
 {
  c3x_remote_scene::AsyncSceneClient<Fake> client(true,[&](char const*){++state.errors;},state);
  client.observe_camera_adoption([&](long long local_camera,long long local_image,auto const& actual){
   assert(state.attempts==1);
   state.receipts.push_back({local_camera,local_image,actual.camera.ticket,
    actual.image.ticket,actual.image.session,actual.camera.frame.tiles[0].anchor_x});
   ++state.observed;
  });
  long long camera=0;assert(client.camera_begin(request,camera)==C3X_RENDERER_RESULT_PENDING&&camera==1);
  client.stats();assert(state.observed==0);
  assert(client.camera_cancel(camera)==C3X_RENDERER_RESULT_OK);client.stats();
  assert(state.cancelled==41&&state.observed==0&&state.attempts==0);
  assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_SUPERSEDED);
  tile.anchor_x=33;
  assert(client.camera_begin(request,camera)==C3X_RENDERER_RESULT_PENDING&&camera==2);client.stats();
  assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_PENDING);client.stats();
  assert(state.observed==0&&state.attempts==0); // Readiness is not adoption.
  state.held=true;
  assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_OK);
  state.wait();assert(state.observed==0); // Caller admission only queued the helper adoption.
  assert(view.camera.ticket==2&&view.image.ticket==2&&view.image.session!=9001);
  state.release();client.stats();
  assert(state.observed==1&&state.receipts.size()==1&&state.errors==0);
  auto receipt=state.receipts[0];
  assert(receipt.camera==2&&receipt.image==2&&receipt.remote==42&&receipt.source==701);
  assert(receipt.camera!=receipt.remote&&receipt.remote!=receipt.source&&receipt.camera!=receipt.source);
  assert(receipt.session==9001&&receipt.anchor==33);
  assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_OK);client.stats();
  assert(state.observed==1&&state.attempts==1); // Repeated polls never execute another adoption.
 }
 State refused;refused.refuse=true;
 {
  c3x_remote_scene::AsyncSceneClient<Fake> client(true,[&](char const*){++refused.errors;},refused);
  client.observe_camera_adoption([&](long long,long long,auto const&){++refused.observed;});
  long long camera=0;assert(client.camera_begin(request,camera)==C3X_RENDERER_RESULT_PENDING);client.stats();
  assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_PENDING);client.stats();
  assert(client.camera_poll(camera,view)==C3X_RENDERER_RESULT_OK);
  auto deadline=std::chrono::steady_clock::now()+2s;
  while(!refused.errors&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
  assert(refused.errors==1&&refused.observed==0&&!client.alive());
 }
 assert(refused.attempts==1&&refused.observed==0);
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
