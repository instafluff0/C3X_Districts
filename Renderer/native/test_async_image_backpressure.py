"""Reliable image admission waits for bounded owned-payload capacity."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class AsyncImageBackpressureTests(unittest.TestCase):
    def test_native_byte_pressure_waits_before_copy_and_prior_payload_release(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <vector>
using namespace std::chrono_literals;
struct Payload {std::vector<int> values;};
int main(){
 std::promise<void> entered,release;auto held=release.get_future();
 std::vector<int> seen;std::atomic<unsigned> factories{0};
 c3x_async::Publication queue;
 auto owner=std::make_shared<Payload>();std::weak_ptr<Payload> prior=owner;
 assert(queue.post(124527040,[&,owner=std::move(owner)]{entered.set_value();held.wait();seen.push_back(0);}));
 entered.get_future().get();
 auto pending=std::async(std::launch::async,[&]{
  return queue.post_group_wait<Payload>(11289688,45,2,[&]{
   assert(prior.expired());++factories;auto value=std::make_shared<Payload>();value->values={2};return value;
  },[&](Payload& value){seen.insert(seen.end(),value.values.begin(),value.values.end());},
   [](Payload& target,Payload& value){target.values.insert(target.values.end(),value.values.begin(),value.values.end());},
   1024*1024,4096,64,"images");
 });
 assert(pending.wait_for(100ms)==std::future_status::timeout&&factories==0);
 auto status=queue.status();assert(status.bytes==124527040&&status.accepted==1&&status.rejected==0);
 // An ordinary fitting publication remains nonblocking while images wait.
 assert(queue.post(0,[&]{seen.push_back(1);}));
 release.set_value();assert(pending.wait_for(2s)==std::future_status::ready&&pending.get());queue.stop();
 status=queue.status();assert(seen==std::vector<int>({0,1,2})&&factories==1);
 assert(queue.healthy()&&status.accepted==3&&status.executed==3&&status.rejected==0);
 assert(!status.bytes&&!status.units&&!status.records&&status.peak_bytes<=128u*1024u*1024u);
}
''')

    def test_independent_bounds_and_stop_fault_oversized_wakes(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
using namespace std::chrono_literals;
struct Payload {};
bool image(c3x_async::Publication& queue,std::size_t bytes,std::size_t work,std::atomic<unsigned>& made){
 return queue.post_group_wait<Payload>(bytes,work,2,[&]{++made;return std::make_shared<Payload>();},
  [](Payload&){},[](Payload&,Payload&){},64,64,64,"images");
}
int main(){
 // Each budget independently makes a reliable image wait without faulting.
 for(unsigned kind=0;kind<3;++kind){
  std::promise<void> entered,release;auto held=release.get_future();std::atomic<unsigned> made{0};
  c3x_async::Publication queue({},16,kind==1?1:8,3);
  assert(queue.post(kind==0?16:1,[&]{entered.set_value();held.wait();},0,nullptr,kind==2?3:1));
  entered.get_future().get();
  auto pending=std::async(std::launch::async,[&]{return image(queue,1,1,made);});
  assert(pending.wait_for(50ms)==std::future_status::timeout&&made==0&&queue.healthy());
  release.set_value();assert(pending.wait_for(2s)==std::future_status::ready&&pending.get());queue.stop();
  auto s=queue.status();assert(s.accepted==2&&s.executed==2&&!s.rejected);
  assert(s.peak_bytes<=16&&s.peak_records<=(kind==1?1:8)&&s.peak_units<=3);
 }
 // A wait cannot hide an ordinary nonblocking overflow, explicit fault or stop.
 for(unsigned kind=0;kind<3;++kind){
  std::promise<void> entered,release;auto held=release.get_future();std::atomic<unsigned> made{0};
  c3x_async::Publication queue({},16,8,3);
  assert(queue.post(16,[&]{entered.set_value();held.wait();}));entered.get_future().get();
  auto pending=std::async(std::launch::async,[&]{return image(queue,1,1,made);});
  assert(pending.wait_for(50ms)==std::future_status::timeout&&made==0);
  std::future<void> stopped;
  if(kind==0){auto ordinary=std::async(std::launch::async,[&]{return queue.post(1,[]{});});
   assert(ordinary.wait_for(2s)==std::future_status::ready&&!ordinary.get());}
  if(kind==1)queue.fail("helper lost");
  if(kind==2)stopped=std::async(std::launch::async,[&]{queue.stop();});
  assert(pending.wait_for(2s)==std::future_status::ready&&!pending.get()&&made==0);
  release.set_value();if(stopped.valid())stopped.get();else queue.stop();
  assert(!queue.status().bytes&&!queue.status().records);
 }
 // Impossible requests fail immediately and never construct an owned packet.
 for(unsigned kind=0;kind<3;++kind){
  std::atomic<unsigned> made{0};c3x_async::Publication queue({},16,kind==2?0:8,3);
  auto result=std::async(std::launch::async,[&]{return image(queue,kind==0?17:1,kind==1?4:1,made);});
  assert(result.wait_for(2s)==std::future_status::ready&&!result.get()&&made==0);
  queue.stop();assert(queue.status().accepted==0&&queue.status().rejected==1);
 }
}
''')

    def test_copy_and_append_exceptions_leave_no_reserved_budget(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <vector>
struct Payload {std::vector<int> values;};
int main(){
 std::promise<void> entered,release;auto held=release.get_future();std::vector<int> seen;
 c3x_async::Publication queue({},64,8,8);
 assert(queue.post(8,[&]{entered.set_value();held.wait();}));entered.get_future().get();
 auto execute=[&](Payload& value){seen.insert(seen.end(),value.values.begin(),value.values.end());};
 auto append=[](Payload& target,Payload& value){target.values.push_back(value.values.front());};
 try{queue.post_group_wait<Payload>(8,1,2,[]()->std::shared_ptr<Payload>{throw std::bad_alloc();},execute,append,64,8,8,"images");assert(false);}
 catch(std::bad_alloc const&){}
 assert(queue.status().bytes==8&&queue.status().accepted==1&&queue.healthy());
 auto make=[](int value){auto packet=std::make_shared<Payload>();packet->values={value};return packet;};
 assert(queue.post_group_wait<Payload>(8,1,2,[&]{return make(11);},execute,append,64,8,8,"images"));
 std::weak_ptr<Payload> failed;
 try{queue.post_group_wait<Payload>(8,1,2,[&]{auto value=make(99);failed=value;return value;},execute,
  [](Payload&,Payload&){throw std::bad_alloc();},64,8,8,"images");assert(false);}
 catch(std::bad_alloc const&){}
 assert(failed.expired()&&queue.status().bytes==16&&queue.status().records==2&&queue.status().accepted==2);
 assert(queue.post_group_wait<Payload>(8,1,2,[&]{return make(12);},execute,append,64,8,8,"images"));
 release.set_value();queue.stop();auto s=queue.status();
 assert(seen==std::vector<int>({11,12})&&s.accepted==3&&s.executed==3&&s.rejected==0&&queue.healthy());
 assert(!s.bytes&&!s.units&&!s.records);
}
''')

    def test_real_async_client_full_window_uploads_keep_order_and_owned_pixels(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
using namespace std::chrono_literals;
struct State {
 std::mutex mutex;std::condition_variable wake;bool held=false,entered=false;
 std::vector<unsigned> revisions;std::vector<int> actions;
 void barrier(){std::unique_lock<std::mutex> lock(mutex);if(!held)return;entered=true;wake.notify_all();wake.wait(lock,[&]{return !held;});}
 void wait(){std::unique_lock<std::mutex> lock(mutex);assert(wake.wait_for(lock,2s,[&]{return entered;}));}
};
struct Fake {
 State& state;explicit Fake(State& value):state(value){}
 bool alive()const{return true;}void publication_pressure(std::size_t){}void supersede_pending_camera(){}
 unsigned frames()const{return 0;}int stats(){return 0;}
 int camera_begin(c3x_renderer_camera_request_v1 const&,long long& ticket){ticket=10;return C3X_RENDERER_RESULT_PENDING;}
 int camera_ready(long long,c3x_renderer_gpu_camera_view_v1& view){
  view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};view.camera.ticket=10;
  view.camera.output={C3X_RENDERER_API_VERSION,sizeof(view.camera.output)};
  view.camera.output.width=2240;view.camera.output.height=1260;
  view.image={sizeof(view.image)};view.image.ticket=20;view.image.map_image=30;
  view.image.width=2240;view.image.height=1260;return C3X_RENDERER_RESULT_OK;
 }
 int camera_poll(long long ticket,c3x_renderer_gpu_camera_view_v1& view){return camera_ready(ticket,view);}
 int images(c3x_renderer_gpu_images_v1 const&,c3x_renderer_gpu_result_v1&,unsigned*,unsigned){assert(false);return 0;}
 int images_batch(std::vector<c3x_remote_scene::ImageBatch::Operation>& operations,std::vector<c3x_remote_scene::ImageBatch::Reply>& replies){
  for(auto& operation:operations){auto& v=operation.image.value;
   state.actions.push_back(v.action);if(v.action==C3X_GPU_UPLOAD){state.barrier();
    assert(v.ticket==20&&v.image==40&&v.pixel_count==2240*1260);
    assert(v.pixels[0]==unsigned(v.revision)&&v.pixels[v.pixel_count-1]==unsigned(v.revision));
    state.revisions.push_back(unsigned(v.revision));}
   c3x_remote_scene::ImageBatch::Reply reply;reply.code=C3X_RENDERER_RESULT_OK;reply.value.image=40;replies.push_back(reply);
  }return C3X_RENDERER_RESULT_OK;
 }
};
int main(){
 State state;c3x_remote_scene::AsyncSceneClient<Fake> client(true,[](char const*){assert(false);},state);
 c3x_renderer_frame_v1 frame={};c3x_renderer_camera_request_v1 camera={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(camera),&frame,{}};
 long long ticket=0;assert(client.camera_begin(camera,ticket)==C3X_RENDERER_RESULT_PENDING);client.stats();
 c3x_renderer_gpu_camera_view_v1 view={};assert(client.camera_poll(ticket,view)==C3X_RENDERER_RESULT_PENDING);client.stats();
 assert(client.camera_poll(ticket,view)==C3X_RENDERER_RESULT_OK);client.stats();
 c3x_renderer_gpu_images_v1 request={};request.struct_size=sizeof(request);request.ticket=ticket;
 request.action=C3X_GPU_CREATE;request.width=2240;request.height=1260;request.format=C3X_GPU_BGRA32;
 c3x_renderer_gpu_result_v1 result={};assert(client.images(request,result,nullptr,0)==C3X_RENDERER_RESULT_OK);
 auto image=result.image;client.stats();
 {std::lock_guard<std::mutex> lock(state.mutex);state.held=true;}
 std::vector<unsigned> pixels(2240*1260);
 request.action=C3X_GPU_UPLOAD;request.image=image;request.pixels=pixels.data();request.pixel_count=unsigned(pixels.size());
 for(unsigned revision=1;revision<=11;++revision){std::fill(pixels.begin(),pixels.end(),revision);request.revision=revision;
  assert(client.images(request,result,nullptr,0)==C3X_RENDERER_RESULT_OK);if(revision==1)state.wait();}
 std::fill(pixels.begin(),pixels.end(),12);request.revision=12;
 auto pending=std::async(std::launch::async,[&]{return client.images(request,result,nullptr,0);});
 assert(pending.wait_for(100ms)==std::future_status::timeout);
 auto blocked=client.publication_status();assert(blocked.records==11&&blocked.rejected==0&&blocked.bytes<128u*1024u*1024u);
 {std::lock_guard<std::mutex> lock(state.mutex);state.held=false;state.wake.notify_all();}
 assert(pending.wait_for(2s)==std::future_status::ready&&pending.get()==C3X_RENDERER_RESULT_OK);
 std::fill(pixels.begin(),pixels.end(),999);client.stats();
 assert(state.revisions==std::vector<unsigned>({1,2,3,4,5,6,7,8,9,10,11,12}));
 request.action=C3X_GPU_DESTROY;request.pixels=nullptr;request.pixel_count=0;
 assert(client.images(request,result,nullptr,0)==C3X_RENDERER_RESULT_OK);client.stats();
 assert(state.actions.front()==C3X_GPU_CREATE&&state.actions.back()==C3X_GPU_DESTROY);
 auto done=client.publication_status();auto deadline=std::chrono::steady_clock::now()+2s;
 while(done.records&&std::chrono::steady_clock::now()<deadline){std::this_thread::yield();done=client.publication_status();}
 assert(client.alive()&&!done.records&&done.accepted==done.executed&&done.rejected==0);
 assert(done.peak_bytes<=128u*1024u*1024u&&done.peak_records<=8192&&done.peak_units<=65536);
}
''')

    def test_native_refresh_releases_word_lease_before_image_capacity_wait(self):
        source = (ROOT / 'Renderer/native/native_image_adapter.h').read_text()
        start = source.index('    bool refresh(Image& image){')
        refresh = source[start:source.index('    // A native-format GPU mirror', start)]
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace std::chrono_literals;
struct Native {std::uint16_t words[4]={1,2,3,4};int stride=2;};
std::atomic<unsigned> leases{0},reads{0};
void GdiFlush(){}void OutputDebugStringA(char const*){}
int field(void* p,int offset){assert(offset==0x40);return static_cast<Native*>(p)->stride;}
namespace c3x_native_access {
 std::uint16_t* words(void* p,void*){++leases;++reads;return static_cast<Native*>(p)->words;}
 void release_words(void*,void*){--leases;}
}
struct Image {void* native;unsigned gpu=0,width=2,height=2;bool cpu_uploaded=false,owned=false;
 std::vector<std::uint16_t> cpu=std::vector<std::uint16_t>(4);std::uint64_t revision=0;};
struct Gpu {
 c3x_async::Publication& queue;std::vector<unsigned>& received;
 bool upload(unsigned,std::uint64_t,unsigned const* pixels,std::size_t count){
  assert(leases==0);
  return queue.post_group_wait<std::vector<unsigned>>(count*4,1,2,[&]{
   assert(leases==0);return std::make_shared<std::vector<unsigned>>(pixels,pixels+count);
  },[this](auto& value){assert(leases==0);received=value;},[](auto&,auto&){assert(false);},0,0,0,"images");
 }
};
struct Harness {
 Gpu gpu;void* get_bits=nullptr;void* release_bits=nullptr;
 unsigned large_uploads=0,source_evictions=0;std::size_t cpu_bytes=8;
 struct {unsigned source_checks=0,source_reuses=0;std::size_t source_expanded_bytes=0;} counters;
''' + refresh + r'''
};
int main(){
 std::promise<void> entered,release;auto held=release.get_future();std::vector<unsigned> received;
 c3x_async::Publication queue({},32,8,8);
 assert(queue.post(32,[&]{entered.set_value();held.wait();}));entered.get_future().get();
 Native native;Image image{&native};Harness h{{queue,received}};
 auto pending=std::async(std::launch::async,[&]{return h.refresh(image);});
 assert(pending.wait_for(100ms)==std::future_status::timeout&&leases==0&&reads==1);
 release.set_value();assert(pending.wait_for(2s)==std::future_status::ready&&pending.get());queue.stop();
 assert(received==std::vector<unsigned>({1,2,3,4})&&image.revision==1&&h.cpu_bytes==8&&leases==0);
}
''')


if __name__ == '__main__':
    unittest.main()
