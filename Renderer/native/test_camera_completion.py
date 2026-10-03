"""Immutable camera inspection bypasses queued RPCs without adopting early."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class CameraCompletionTests(unittest.TestCase):
    def test_bounded_codec_owns_arrays_and_refuses_unproved_descriptors(self):
        run_cpp(r'''
#include "Renderer/native/remote_scene_output.h"
#include <cassert>
#include <memory>
using namespace c3x_remote_scene;
int main(){
 auto slot=std::make_unique<CameraCompletionSlot>();
 c3x_renderer_tile_v1 tiles[2]={};tiles[0].anchor_x=37;tiles[1].visibility_mask=9;
 unsigned replacements[2]={3,7},fallback=1;
 c3x_renderer_gpu_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
 view.camera.version=C3X_RENDERER_CAMERA_VIEW_VERSION;view.camera.struct_size=sizeof(view.camera);
 view.camera.ticket=91;view.camera.identity.map_epoch=8;view.camera.identity.viewer_epoch=10;
 view.camera.frame.api_version=C3X_RENDERER_API_VERSION;view.camera.frame.struct_size=sizeof(view.camera.frame);
 view.camera.frame.target_width=640;view.camera.frame.target_height=480;
 view.camera.frame.tiles=tiles;view.camera.frame.tile_count=2;view.camera.frame.world_topology_revision=99;
 view.camera.output={C3X_RENDERER_API_VERSION,sizeof(view.camera.output)};
 view.camera.output.width=640;view.camera.output.height=480;view.camera.output.stride_bytes=2560;
 view.camera.output.replacement_tile_count=2;view.camera.output.replacement_tile_flags=replacements;
 view.camera.output.fallback_tile_count=1;view.camera.output.fallback_tile_indices=&fallback;
 view.image={sizeof(view.image)};view.image.width=640;view.image.height=480;
 view.image.device_generation=5;view.image.content_revision=12;view.image.presentation_time_ticks=300;
 view.pixel_phase_x=37;view.pixel_phase_y=-3;
 publish_camera_completion(*slot,91,C3X_RENDERER_RESULT_OK,&view);
 assert(slot->version==camera_completion_version&&slot->size<camera_completion_capacity);
 tiles[0].anchor_x=999;replacements[1]=0;view.camera.identity.map_epoch=100;
 c3x_inputs::Bytes bytes;int code=0;
 assert(snapshot_camera_completion(*slot,90,bytes,code)&&code==C3X_RENDERER_RESULT_SUPERSEDED&&bytes.empty());
 assert(snapshot_camera_completion(*slot,92,bytes,code)&&code==C3X_RENDERER_RESULT_PENDING&&bytes.empty());
 assert(snapshot_camera_completion(*slot,91,bytes,code)&&code==C3X_RENDERER_RESULT_OK);
 CameraOutput copied;c3x_inputs::Reader in{bytes};decode_camera(in,copied);
 assert(copied.value.camera.ticket==91&&copied.value.camera.identity.map_epoch==8);
 assert(copied.value.camera.frame.tiles[0].anchor_x==37&&copied.output.replacements[1]==7);
 assert(copied.value.camera.frame.world_topology_revision==99&&copied.value.pixel_phase_y==-3);
 assert(copied.output.pixels.empty()&&copied.output.gpu.device_generation==5);
 // Supersession and failed work carry no arrays or image retirement authority.
 publish_camera_completion(*slot,92,C3X_RENDERER_RESULT_DEVICE_ERROR,nullptr);
 bytes.clear();assert(snapshot_camera_completion(*slot,92,bytes,code)&&code==C3X_RENDERER_RESULT_DEVICE_ERROR&&bytes.empty());
 assert(copied.value.camera.frame.tiles[0].anchor_x==37); // Caller lease survives overwrite.
 publish_camera_completion(*slot,93,C3X_RENDERER_RESULT_OK,&view);
 assert(!snapshot_camera_completion(*slot,93,bytes,code)); // Wrong exact ticket.
 view.camera.ticket=93;unsigned pixel=1;view.camera.output.bgra_pixels=&pixel;
 publish_camera_completion(*slot,93,C3X_RENDERER_RESULT_OK,&view);
 assert(!snapshot_camera_completion(*slot,93,bytes,code)); // CPU pixels are never a GPU completion.
 view.camera.output.bgra_pixels=nullptr;view.camera.frame.tile_count=8193;
 publish_camera_completion(*slot,93,C3X_RENDERER_RESULT_OK,&view);
 assert(!snapshot_camera_completion(*slot,93,bytes,code)); // Encode refusal uses RPC fallback.
 slot->version=camera_completion_version;slot->size=camera_completion_capacity+1;
 assert(!snapshot_camera_completion(*slot,93,bytes,code));
 slot->size=0;slot->code=99;assert(!snapshot_camera_completion(*slot,93,bytes,code));
}
''')

    def test_native_reader_is_nonblocking_and_abandoned_slot_falls_back(self):
        source=(ROOT/'Renderer/native/helper_trial/scene_client.h').read_text()
        methods='    void prepare_camera_receipt(){'+source.split('    void prepare_camera_receipt(){',1)[1].split('    void wait_image_receipt()',1)[0]
        run_cpp(r'''
#include "Renderer/native/remote_scene_output.h"
#include <cassert>
#include <memory>
using LONG=std::int32_t;using HANDLE=int;using DWORD=unsigned;
constexpr int WAIT_OBJECT_0=0,WAIT_TIMEOUT=1,WAIT_ABANDONED=2,WAIT_FAILED=3;
int wait_result=WAIT_TIMEOUT,releases=0,waits=0;
DWORD GetCurrentThreadId(){return 17;}
LONG InterlockedExchange(volatile LONG* p,LONG n){auto old=*p;*p=n;return old;}
LONG InterlockedCompareExchange(volatile LONG* p,LONG n,LONG old){auto was=*p;if(was==old)*p=n;return was;}
int WaitForSingleObject(HANDLE,unsigned ms){assert(!ms);++waits;return wait_result;}
void ReleaseMutex(HANDLE){++releases;}
struct Wire{LONG camera_receiver_thread=0,camera_completion_available=1;c3x_remote_scene::CameraCompletionSlot camera_completion;};
struct Reader{
 std::unique_ptr<Wire> owned=std::make_unique<Wire>();Wire* wire=owned.get();HANDLE camera_mutex=1;bool live=true;
 bool alive(){return live;}
''' + methods + r'''
};
int main(){Reader reader;reader.prepare_camera_receipt();assert(reader.wire->camera_receiver_thread==17);
 c3x_inputs::Bytes bytes;int code=0;
 for(int n=0;n<1000;++n)assert(reader.camera_completion(7,bytes,code)&&code==C3X_RENDERER_RESULT_PENDING);
 assert(!releases&&waits==1000&&bytes.empty());
 wait_result=WAIT_ABANDONED;assert(!reader.camera_completion(7,bytes,code)&&releases==1);
 wait_result=WAIT_FAILED;assert(!reader.camera_completion(7,bytes,code)&&releases==1);
 wait_result=WAIT_OBJECT_0;auto& slot=reader.wire->camera_completion;
 c3x_remote_scene::publish_camera_completion(slot,7,C3X_RENDERER_RESULT_ERROR,nullptr);
 assert(reader.camera_completion(7,bytes,code)&&code==C3X_RENDERER_RESULT_ERROR&&releases==2);
 reader.wire->camera_completion_available=0;auto before=waits;
 assert(!reader.camera_completion(7,bytes,code)&&waits==before);
 reader.wire->camera_completion_available=1;reader.live=false;
 assert(!reader.camera_completion(7,bytes,code)&&waits==before);
}
''')

    def test_actual_helper_publisher_refuses_contention_and_rebinds_after_retirement(self):
        source=(ROOT/'Renderer/native/helper_trial/scene_workload.cpp').read_text()
        methods='    void publish_camera('+source.split('    void publish_camera(',1)[1].split('    void start_control(',1)[0]
        retirement='if(retirement&&completion_registered){'+source.split('if(retirement&&completion_registered){',1)[1].split('\n            }',1)[0]+'\n}'
        run_cpp(r'''
#include "Renderer/native/remote_scene_output.h"
#include <cassert>
#include <memory>
using LONG=std::int32_t;using HANDLE=int;using DWORD=unsigned;using UINT=unsigned;
using WPARAM=std::uintptr_t;using LPARAM=std::intptr_t;
constexpr int WAIT_OBJECT_0=0,WAIT_TIMEOUT=1,WAIT_ABANDONED=2,WAIT_FAILED=3;
int wait_result=WAIT_OBJECT_0,releases=0,wakes=0,registrations=0,retirements=0;
c3x_renderer_camera_completion_fn callback=nullptr;void* callback_owner=nullptr;
LONG InterlockedExchange(volatile LONG* p,LONG n){auto old=*p;*p=n;return old;}
LONG InterlockedCompareExchange(volatile LONG* p,LONG n,LONG old){auto was=*p;if(was==old)*p=n;return was;}
int WaitForSingleObject(HANDLE,unsigned ms){assert(!ms);return wait_result;}
void ReleaseMutex(HANDLE){++releases;}
bool PostThreadMessageA(DWORD thread,UINT message,WPARAM,LPARAM){assert(thread==17&&message==12);++wakes;return true;}
int observe(c3x_renderer_camera_completion_fn fn,void* owner){callback=fn;callback_owner=owner;if(fn)++registrations;else ++retirements;return C3X_RENDERER_RESULT_OK;}
using c3x_inputs::require;
struct Wire{LONG camera_receiver_thread=17,camera_completion_available=0;c3x_remote_scene::CameraCompletionSlot camera_completion;};
struct Core{
 std::unique_ptr<Wire> owned=std::make_unique<Wire>();Wire* completion_wire=owned.get();HANDLE completion_mutex=1;
 UINT completion_message=12;bool completion_registered=false;c3x_renderer_observe_camera_completion_fn observe_camera_completion=observe;
''' + methods + r'''
 void retire(){bool retirement=true;Wire& wire=*completion_wire;''' + retirement + r'''}
};
int main(){Core helper;helper.register_camera_completion();
 assert(helper.completion_registered&&registrations==1&&helper.completion_wire->camera_completion_available==1);
 assert(callback_owner==&helper);helper.register_camera_completion();assert(registrations==1);
 auto previous=helper.completion_wire->camera_completion.ticket;wait_result=WAIT_TIMEOUT;
 callback(callback_owner,91,C3X_RENDERER_RESULT_ERROR,nullptr);
 assert(!helper.completion_wire->camera_completion_available&&helper.completion_wire->camera_completion.ticket==previous);
 assert(wakes==2&&releases==1); // Contention cannot hold the renderer state gate.
 wait_result=WAIT_OBJECT_0;callback(callback_owner,92,C3X_RENDERER_RESULT_SUPERSEDED,nullptr);
 assert(helper.completion_wire->camera_completion_available&&helper.completion_wire->camera_completion.ticket==92);
 assert(helper.completion_wire->camera_completion.code==C3X_RENDERER_RESULT_SUPERSEDED&&wakes==3);
 helper.retire();assert(!helper.completion_registered&&!callback&&retirements==1);
 assert(!helper.completion_wire->camera_completion_available);
 helper.register_camera_completion();assert(helper.completion_registered&&registrations==2&&callback_owner==&helper);
 wait_result=WAIT_FAILED;callback(callback_owner,93,C3X_RENDERER_RESULT_ERROR,nullptr);
 assert(!helper.completion_wire->camera_completion_available&&wakes==5);
}
''')

    def test_actual_worker_terminal_notifications_and_observer_retirement(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        setter='    int observe_camera_completion('+source.split('    int observe_camera_completion(',1)[1].split('    int poll_gpu_camera_view(',1)[0]
        notify='    int camera_ready_view_locked('+source.split('    int camera_ready_view_locked(',1)[1].split('    int begin_gpu_camera_locked(',1)[0]
        cancel='    int camera_cancel('+source.split('    int camera_cancel(',1)[1].split('    int blit(',1)[0]
        drain='    void drain_camera_locked('+source.split('    void drain_camera_locked(',1)[1].split('    // A foreground draw/transfer',1)[0]
        run_cpp(r'''
#include "Renderer/native/camera_completion.h"
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <vector>
using DWORD=unsigned;using UINT=unsigned;using WPARAM=std::uintptr_t;using LPARAM=std::intptr_t;
unsigned wakes=0;bool PostThreadMessageA(DWORD,UINT,WPARAM,LPARAM){++wakes;return true;}
struct Published{
 struct {std::shared_ptr<int> texture;}resident;
 c3x_renderer_frame_v1 frame={};c3x_renderer_output_v1 output={};c3x_renderer_camera_identity_v1 identity={};
 int phase_x=37,phase_y=-3;
 void clear(){resident.texture.reset();}
};
struct Worker{
 std::mutex call_mutex,state_mutex;std::condition_variable completed,wake;
 c3x_renderer_camera_completion_fn camera_completion_observer=nullptr;void* camera_completion_context=nullptr;
 long long camera_ticket=91;int camera_result=C3X_RENDERER_RESULT_PENDING;
 bool camera_gpu=true,camera_active=false,camera_pending=false,camera_paused=false,camera_ready_prepared=true,unit_pixels_active=false;
 DWORD camera_notify_thread=17;UINT camera_notify_message=12;
 std::atomic<bool> foreground_pending{false},camera_cancelled{false};std::vector<int> camera_pending_tiles,camera_pending_topology;
 Published camera_ready;
 void pause_ahead_locked(std::unique_lock<std::mutex>&,bool){}
''' + setter + notify + cancel + drain + r'''
 void finish(){std::lock_guard<std::mutex> lock(state_mutex);notify_camera_completion_locked(camera_ticket);}
 void drain(){std::unique_lock<std::mutex> lock(state_mutex);drain_camera_locked(lock);}
};
struct Receipt{
 std::mutex mutex;std::condition_variable wake;bool held=false,entered=false;
 unsigned calls=0;long long ticket=0;int code=0;bool descriptor=false;
};
void observed(void* owner,long long ticket,int code,c3x_renderer_gpu_camera_view_v1 const* view){
 auto& r=*static_cast<Receipt*>(owner);std::unique_lock<std::mutex> lock(r.mutex);
 ++r.calls;r.ticket=ticket;r.code=code;r.descriptor=view!=nullptr;
 if(view){assert(view->camera.ticket==ticket&&view->pixel_phase_x==37&&view->image.prepared==1);}
 if(r.held){r.entered=true;r.wake.notify_all();r.wake.wait(lock,[&]{return !r.held;});}
}
int main(){using namespace std::chrono_literals;Worker worker;Receipt r;
 assert(worker.observe_camera_completion(observed,&r)==C3X_RENDERER_RESULT_OK);
 worker.finish();assert(!r.calls&&!wakes);worker.camera_result=C3X_RENDERER_RESULT_OK;
 worker.camera_ready.resident.texture=std::make_shared<int>(1);worker.finish();
 assert(r.calls==1&&r.ticket==91&&r.descriptor&&r.code==C3X_RENDERER_RESULT_OK&&wakes==1);
 assert(worker.camera_ready.resident.texture); // Inspection does not retire/import the map.
 r.held=true;auto completion=std::async(std::launch::async,[&]{worker.finish();});
 {std::unique_lock<std::mutex> lock(r.mutex);assert(r.wake.wait_for(lock,2s,[&]{return r.entered;}));}
 auto retiring=std::async(std::launch::async,[&]{return worker.observe_camera_completion(nullptr,nullptr);});
 assert(retiring.wait_for(2ms)==std::future_status::timeout);
 {std::lock_guard<std::mutex> lock(r.mutex);r.held=false;r.wake.notify_all();}
 completion.get();assert(retiring.get()==C3X_RENDERER_RESULT_OK);auto calls=r.calls;
 worker.finish();assert(r.calls==calls); // No callback retains a dead helper context.
 worker.observe_camera_completion(observed,&r);worker.camera_result=C3X_RENDERER_RESULT_PENDING;worker.camera_pending=true;
 assert(worker.camera_cancel(91)==C3X_RENDERER_RESULT_OK);
 assert(r.code==C3X_RENDERER_RESULT_SUPERSEDED&&!r.descriptor&&!worker.camera_pending&&!worker.camera_ready.resident.texture);
 calls=r.calls;assert(worker.camera_cancel(90)==C3X_RENDERER_RESULT_SUPERSEDED&&r.calls==calls);
 worker.camera_result=C3X_RENDERER_RESULT_PENDING;worker.camera_pending=true;worker.drain();
 assert(r.calls==calls+1&&r.code==C3X_RENDERER_RESULT_SUPERSEDED&&!r.descriptor);
 worker.camera_result=C3X_RENDERER_RESULT_DEVICE_ERROR;worker.finish();
 assert(r.code==C3X_RENDERER_RESULT_DEVICE_ERROR&&!r.descriptor);
}
''')

    def test_completed_camera_bypasses_blocked_transport_but_adopts_after_old_prefix(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
using namespace c3x_remote_scene;
struct State{
 std::mutex mutex;std::condition_variable wake;bool held=false,entered=false,mailbox=true,corrupt=false;
 std::unique_ptr<CameraCompletionSlot> slot=std::make_unique<CameraCompletionSlot>();
 c3x_inputs::Frame frame;c3x_renderer_camera_identity_v1 identity={};long long remote=0,current=0;
 unsigned ready_rpcs=0,adoptions=0,receivers=0;std::vector<int> order;
 void hold(){std::lock_guard<std::mutex> lock(mutex);held=true;entered=false;}
 void barrier(){std::unique_lock<std::mutex> lock(mutex);if(!held)return;entered=true;wake.notify_all();wake.wait(lock,[&]{return !held;});}
 void wait(){std::unique_lock<std::mutex> lock(mutex);assert(wake.wait_for(lock,2s,[&]{return entered;}));}
 void release(){std::lock_guard<std::mutex> lock(mutex);held=false;wake.notify_all();}
 c3x_renderer_gpu_camera_view_v1 view(){
  c3x_renderer_gpu_camera_view_v1 v={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(v)};
  v.camera.version=C3X_RENDERER_CAMERA_VIEW_VERSION;v.camera.struct_size=sizeof(v.camera);
  v.camera.ticket=remote;v.camera.frame=frame.value;v.camera.identity=identity;
  v.camera.frame.world_topology=nullptr;v.camera.frame.world_topology_count=0;
  v.camera.output={C3X_RENDERER_API_VERSION,sizeof(v.camera.output)};
  v.camera.output.width=640;v.camera.output.height=480;v.camera.output.stride_bytes=2560;
  v.image={sizeof(v.image)};v.image.width=640;v.image.height=480;
  if(corrupt)++v.camera.identity.viewer_epoch;
  return v;
 }
 void complete(int code=C3X_RENDERER_RESULT_OK){std::lock_guard<std::mutex> lock(mutex);
  auto v=view();publish_camera_completion(*slot,remote,code,code==C3X_RENDERER_RESULT_OK?&v:nullptr);}
};
struct Fake{
 State& s;explicit Fake(State& value):s(value){}
 bool alive()const{return true;}void publication_pressure(std::size_t){}void supersede_pending_camera(){}
 void prepare_camera_receipt(){++s.receivers;}
 int stats(){return 0;}
 int camera_begin(c3x_renderer_camera_request_v1 const& request,long long& ticket){
  s.frame.value=*request.frame;s.frame.tiles.assign(request.frame->tiles,request.frame->tiles+request.frame->tile_count);
  if(request.frame->world_topology_count)s.frame.topology.assign(request.frame->world_topology,request.frame->world_topology+request.frame->world_topology_count);
  s.frame.bind();s.identity=request.identity;ticket=++s.remote;
  publish_camera_completion(*s.slot,0,C3X_RENDERER_RESULT_PENDING,nullptr);return C3X_RENDERER_RESULT_PENDING;
 }
 bool camera_completion(long long ticket,CameraOutput& value,int& code){
  std::lock_guard<std::mutex> lock(s.mutex);if(!s.mailbox)return false;
  c3x_inputs::Bytes bytes;if(!snapshot_camera_completion(*s.slot,ticket,bytes,code))return false;
  if(code==C3X_RENDERER_RESULT_OK){c3x_inputs::Reader in{bytes};decode_camera(in,value);}return true;
 }
 int camera_ready(long long ticket,c3x_renderer_gpu_camera_view_v1& value){++s.ready_rpcs;assert(ticket==s.remote);
  value=s.view();return C3X_RENDERER_RESULT_OK;}
 int camera_poll(long long ticket,c3x_renderer_gpu_camera_view_v1& value){assert(ticket==s.remote);++s.adoptions;
  s.current=ticket;s.order.push_back(2);value=s.view();value.image.ticket=100+ticket;value.image.map_image=1000+ticket;
  return C3X_RENDERER_RESULT_OK;}
 int camera_cancel(long long){return C3X_RENDERER_RESULT_OK;}
 int images(c3x_renderer_gpu_images_v1 const&,c3x_renderer_gpu_result_v1&,unsigned*,unsigned){assert(false);return 0;}
 int images_batch(std::vector<ImageBatch::Operation>& operations,std::vector<ImageBatch::Reply>& replies){
  for(auto& operation:operations){auto& v=operation.image.value;
   assert(v.ticket==100+s.current);
   if(v.action==C3X_GPU_SUBMIT){s.barrier();assert(operation.image.commands[0].source==1000+s.current);s.order.push_back(s.current==1?1:3);}
   ImageBatch::Reply reply;reply.code=C3X_RENDERER_RESULT_OK;reply.value.image=2000;replies.push_back(reply);
  }return C3X_RENDERER_RESULT_OK;
 }
};
int main(){
 State s;AsyncSceneClient<Fake> client(true,[](char const*){assert(false);},s);
 c3x_renderer_tile_v1 tile={};tile.anchor_x=17;unsigned topology=9;
 c3x_renderer_frame_v1 frame={};frame.tiles=&tile;frame.tile_count=1;frame.target_width=640;frame.target_height=480;
 frame.world_topology=&topology;frame.world_topology_count=1;frame.world_topology_revision=10;
 c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,{}};
 long long ticket=0;assert(client.camera_begin(request,ticket)==C3X_RENDERER_RESULT_PENDING);client.stats();
 auto admitted=client.publication_status().accepted;c3x_renderer_gpu_camera_view_v1 shown={};shown.image.ticket=77;
 for(int n=0;n<1000;++n)assert(client.camera_poll(ticket,shown)==C3X_RENDERER_RESULT_PENDING);
 assert(client.publication_status().accepted==admitted&&!s.ready_rpcs&&!s.adoptions&&shown.image.ticket==77);
 s.complete();assert(client.camera_poll(ticket,shown)==C3X_RENDERER_RESULT_OK);client.stats();
 assert(s.adoptions==1&&!s.ready_rpcs&&shown.camera.frame.tiles[0].anchor_x==17&&s.receivers==1);
 auto old=shown.image;
 c3x_renderer_gpu_images_v1 image={};image.struct_size=sizeof(image);image.action=C3X_GPU_CREATE;image.ticket=old.ticket;
 c3x_renderer_gpu_result_v1 made={sizeof(made)};assert(client.images(image,made,nullptr,0)==C3X_RENDERER_RESULT_OK);client.stats();
 auto local_image=made.image;
 tile.anchor_x=33;assert(client.camera_begin(request,ticket)==C3X_RENDERER_RESULT_PENDING);client.stats();
 s.order.clear();s.hold();c3x_renderer_gpu_command_v1 draw={};draw.source=old.map_image;draw.destination=local_image;
 image.action=C3X_GPU_SUBMIT;image.commands=&draw;image.command_count=1;
 assert(client.images(image,made,nullptr,0)==C3X_RENDERER_RESULT_OK);s.wait();s.complete();
 auto before=std::chrono::steady_clock::now();assert(client.camera_poll(ticket,shown)==C3X_RENDERER_RESULT_OK);
 assert(std::chrono::steady_clock::now()-before<100ms&&s.adoptions==1&&!s.ready_rpcs);
 image.ticket=shown.image.ticket;draw.source=shown.image.map_image;
 assert(client.images(image,made,nullptr,0)==C3X_RENDERER_RESULT_OK);s.release();client.stats();
 assert(s.order==std::vector<int>({1,2,3})&&s.adoptions==2&&shown.camera.frame.tiles[0].anchor_x==33);
 assert(client.camera_cancel(ticket)==C3X_RENDERER_RESULT_OK);client.stats();
 assert(client.camera_poll(ticket,shown)==C3X_RENDERER_RESULT_SUPERSEDED);
 // A corrupt source cannot publish aliases or redirect its destination.
 tile.anchor_x=49;assert(client.camera_begin(request,ticket)==C3X_RENDERER_RESULT_PENDING);client.stats();s.corrupt=true;s.complete();
 admitted=client.publication_status().accepted;
 assert(client.camera_poll(ticket,shown)==C3X_RENDERER_RESULT_SUPERSEDED&&client.publication_status().accepted==admitted);
 assert(s.adoptions==2);client.camera_cancel(ticket);client.stats();s.corrupt=false;
 // Errors are surfaced without touching the last adopted frame.
 tile.anchor_x=65;assert(client.camera_begin(request,ticket)==C3X_RENDERER_RESULT_PENDING);client.stats();s.complete(C3X_RENDERER_RESULT_DEVICE_ERROR);
 auto previous=shown.image.ticket;
 assert(client.camera_poll(ticket,shown)==C3X_RENDERER_RESULT_DEVICE_ERROR&&shown.image.ticket==previous);
 client.camera_cancel(ticket);client.stats();
 // Unsupported/refused mailbox retains the original exact readiness path.
 s.mailbox=false;tile.anchor_x=81;assert(client.camera_begin(request,ticket)==C3X_RENDERER_RESULT_PENDING);client.stats();
 assert(client.camera_poll(ticket,shown)==C3X_RENDERER_RESULT_PENDING);client.stats();
 assert(client.camera_poll(ticket,shown)==C3X_RENDERER_RESULT_OK);client.stats();
 assert(s.ready_rpcs==1&&s.adoptions==3&&shown.camera.frame.tiles[0].anchor_x==81);
}
''')


if __name__=='__main__':
    unittest.main()
