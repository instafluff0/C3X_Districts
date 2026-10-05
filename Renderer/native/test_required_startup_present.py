"""Execute the startup delivery gates without a window, readback, or VM."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_preparation_cancellation import block_at


class RequiredStartupPresentTests(unittest.TestCase):
    def test_startup_join_preserves_nearly_full_reliable_prefix_and_existing_caps(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
int main(){std::promise<void> entered,release;auto held=release.get_future();unsigned next=0;
 c3x_async::Publication queue;
 assert(queue.post(1,[&]{entered.set_value();held.wait();}));entered.get_future().get();
 for(unsigned n=1;n<8191;++n)assert(queue.post(1,[&,n]{assert(++next==n);},0,"reliable"));
 auto receipt=std::async(std::launch::async,[&]{return queue.setup([&]{assert(next==8190);return 41;});});
 auto deadline=std::chrono::steady_clock::now()+2s;
 while(queue.status().records<8192){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
 assert(receipt.wait_for(20ms)==std::future_status::timeout);release.set_value();assert(receipt.get()==41);
 queue.stop();auto s=queue.status();assert(queue.healthy()&&s.peak_records==8192);
 assert(s.accepted==8192&&s.executed==8192&&!s.superseded&&!s.abandoned&&!s.rejected&&!s.records&&!s.units&&!s.bytes);
}
''')

    def test_worker_joins_exact_revision_and_refuses_failed_or_superseded_delivery(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        start = source.index("    int trial_required_present_shared(")
        method = source[start:source.index("    int trial_visual_shared(", start)]
        start = source.index("struct PublishedMapFrame {")
        publication = source[start:source.index("\n// Cheap, deliberately provisional", start)]
        start = source.index("{publication.clear();completed_phase_x=completed_phase_y=0;}")
        retired_cpu = block_at(source, start)
        run_cpp(r'''
#include "Renderer/native/gpu_frame_api.h"
#include "Renderer/native/render_core/scene_publication.h"
#include <cassert>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
using DWORD=unsigned;
struct LARGE_INTEGER {long long QuadPart=0;};
bool clock_ok=true;unsigned pauses=0;
bool QueryPerformanceCounter(LARGE_INTEGER* p){p->QuadPart=17;return clock_ok;}
bool QueryPerformanceFrequency(LARGE_INTEGER* p){p->QuadPart=1000;return clock_ok;}
void Sleep(unsigned n){assert(n==1);++pauses;}
''' + publication + r'''
struct Session {long long ticket=41;std::uint64_t revision=7;bool ready=true;
 long long current_ticket(){return ticket;}std::uint64_t committed_revision(){return revision;}bool visual_ready(){return ready;}};
struct Owner {
 std::mutex call_mutex,state_mutex;
 void drain_facts_locked(){}
 struct {std::unique_ptr<Session> gpu_composition=std::make_unique<Session>();}renderer_state;
 PublishedMapFrame publication,gpu_publication;
 c3x_renderer::render_core::ScenePublication scene_changes;
 c3x_renderer_frame_v1 source_frame{};
 Owner(){publication.identity=gpu_publication.identity={1,2,3,4};
  assert(scene_changes.capture(source_frame,gpu_publication.identity));}
 int completed_phase_x=17,completed_phase_y=19;
 void complete_gpu_import()''' + retired_cpu + r'''
 void change_scope(bool viewer){auto identity=scene_changes.state()->identity;
  if(viewer)++identity.viewer_epoch;else ++identity.scene_epoch;
  assert(scene_changes.capture(source_frame,identity));}
 long long gpu_camera_front_ticket=9;std::atomic<long long> camera_obsolete_through{0};
 bool trial_surface_swap=true;std::uint64_t trial_presented_front_revision=0;
 c3x_renderer_gpu_present_v1 gpu_present{};unsigned trial_consumer_pid=0,trial_width=0,trial_height=0;
 std::uint64_t trial_handle=0;long long visual_ticks=0,visual_frequency=0,visual_last=0;
 unsigned commits=0,attempts=0;int mode=0;
 enum class Command{trial_present_shared,trial_required_visual_shared};
 void advance_visual_clock(){}
 int submit_locked(std::unique_lock<std::mutex>& lock,Command command){
  assert(lock.owns_lock());assert(!call_mutex.try_lock());
  auto& s=*renderer_state.gpu_composition;
  if(command==Command::trial_present_shared){++commits;
   assert(gpu_present.ticket==41&&gpu_present.image==73&&!trial_consumer_pid);
   if(mode==1)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
   if(mode==2){change_scope(true);return C3X_RENDERER_RESULT_OK;}
   if(mode==10)s.ready=false;
   if(mode==11){s.revision=0;return C3X_RENDERER_RESULT_OK;}
   ++s.revision;return C3X_RENDERER_RESULT_OK;
  }
  ++attempts;assert(visual_ticks==17&&visual_frequency==1000);
  if(mode==3)return C3X_RENDERER_RESULT_DEVICE_ERROR;
  if(mode==4)return C3X_RENDERER_RESULT_PENDING;
  if(mode==5){camera_obsolete_through=9;return C3X_RENDERER_RESULT_BUSY;}
  if(mode==6){++gpu_publication.identity.map_epoch;return C3X_RENDERER_RESULT_BUSY;}
  if(mode==7){++s.revision;return C3X_RENDERER_RESULT_BUSY;}
  if(mode==8){++s.ticket;return C3X_RENDERER_RESULT_BUSY;}
  if(mode==9){trial_presented_front_revision=s.revision;change_scope(false);return C3X_RENDERER_RESULT_OK;}
  if(mode==12){scene_changes.reset();return C3X_RENDERER_RESULT_BUSY;}
  if(mode==13){auto identity=scene_changes.state()->identity;scene_changes.reset();
   assert(scene_changes.capture(source_frame,identity));return C3X_RENDERER_RESULT_BUSY;}
  if(attempts==1)return C3X_RENDERER_RESULT_BUSY; // OS permit remains denied.
  if(attempts==2)return C3X_RENDERER_RESULT_OK; // Non-S_OK Present has no receipt.
  trial_presented_front_revision=s.revision;trial_width=640;trial_height=480;
  return C3X_RENDERER_RESULT_OK;
 }
''' + method + r'''
};
int main(){c3x_renderer_gpu_present_v1 request{sizeof(request)};request.ticket=41;request.image=73;
 request.width=640;request.height=480;std::uint64_t handle=99;unsigned w=99,h=99;
 Owner ok;ok.complete_gpu_import();
 assert(!ok.publication.identity.map_epoch && ok.gpu_publication.identity.map_epoch==1);
 assert(!ok.completed_phase_x && !ok.completed_phase_y);
 assert(ok.trial_required_present_shared(request,0,handle,w,h)==C3X_RENDERER_RESULT_OK);
 assert(ok.commits==1&&ok.attempts==3&&pauses==2&&handle==0&&w==640&&h==480);
 assert(ok.trial_presented_front_revision==ok.renderer_state.gpu_composition->revision);
 for(int mode=1;mode<=13;++mode){Owner s;s.mode=mode;
  int expected=mode==1?C3X_RENDERER_RESULT_BAD_ARGUMENT:mode==3?C3X_RENDERER_RESULT_DEVICE_ERROR:
   (mode==4||mode==10||mode==11)?C3X_RENDERER_RESULT_ERROR:C3X_RENDERER_RESULT_SUPERSEDED;
  assert(s.trial_required_present_shared(request,0,handle,w,h)==expected&&handle==0&&w==0&&h==0);
 }
 Owner obsolete;obsolete.camera_obsolete_through=9;
 assert(obsolete.trial_required_present_shared(request,0,handle,w,h)==C3X_RENDERER_RESULT_SUPERSEDED&&!obsolete.commits);
 Owner wrong_scope;wrong_scope.change_scope(true);
 assert(wrong_scope.trial_required_present_shared(request,0,handle,w,h)==C3X_RENDERER_RESULT_SUPERSEDED&&!wrong_scope.commits);
 Owner wrong_scene;wrong_scene.change_scope(false);
 assert(wrong_scene.trial_required_present_shared(request,0,handle,w,h)==C3X_RENDERER_RESULT_SUPERSEDED&&!wrong_scene.commits);
 Owner no_authority;no_authority.scene_changes.reset();
 assert(no_authority.trial_required_present_shared(request,0,handle,w,h)==C3X_RENDERER_RESULT_SUPERSEDED&&!no_authority.commits);
 Owner missing;missing.trial_surface_swap=false;
 assert(missing.trial_required_present_shared(request,0,handle,w,h)==C3X_RENDERER_RESULT_SUPERSEDED&&!missing.commits);
 Owner wrong_ticket;request.ticket=42;
 assert(wrong_ticket.trial_required_present_shared(request,0,handle,w,h)==C3X_RENDERER_RESULT_SUPERSEDED&&!wrong_ticket.commits);
 request.ticket=41;Owner clock;clock_ok=false;
 assert(clock.trial_required_present_shared(request,0,handle,w,h)==C3X_RENDERER_RESULT_ERROR&&!clock.attempts);
}
''')

    def test_final_native_transfer_is_startup_only_optional_and_scope_safe(self):
        source = (ROOT / "injected_code.c").read_text()
        # Injection compiles linearly: this early hook's void logger must be
        # defined before its call, rather than relying on implicit C int.
        logger = source.index("\nvoid\nlog_custom_renderer_event (")
        graph_present = source.index("\nint __fastcall\npatch_JGL_Graphsy_present (")
        self.assertLess(logger, graph_present)
        start = source.index("\tif (! is->current_config.enable_custom_rendering) is->custom_renderer_first_front_pending = false;")
        block = source[start:source.index("\n\t// JGL 0x3baa0", start)]
        arm_start = source.index("\tif (first_map_delivery && is->custom_renderer_composited")
        # This arm is a single unbraced if; extract its real two-line statement.
        arm = source[arm_start:source.index("\n}", arm_start)]
        run_cpp(r'''
#include <cassert>
#include <cstring>
using HMODULE=void*;struct RECT{};
enum {IS_OK=1,IS_INIT_FAILED=2,IS_LOADING=3,C3X_NATIVE_IMAGE_PRESENT=4,
 C3X_RENDERER_RESULT_OK=0,C3X_RENDERER_RESULT_ERROR=-1};
struct Main_Screen_Form {bool is_now_loading_game=false;}screen,other_screen;
Main_Screen_Form* p_main_screen_form=&screen;
struct {struct {void* Tiles=reinterpret_cast<void*>(8);}Map;}bic;auto p_bic_data=&bic;
struct State {struct {bool enable_custom_rendering=true;}current_config;
 bool custom_renderer_first_front_pending=true,custom_renderer_draw_in_progress=false,
  custom_renderer_frame_active=false,custom_renderer_capture_only=false,custom_renderer_composited=true;
 int custom_renderer_init_state=IS_OK,custom_renderer_viewer_civ_id=7;
 HMODULE custom_renderer_module=reinterpret_cast<void*>(3),custom_renderer_native_module=reinterpret_cast<void*>(3);
 long long custom_renderer_map_epoch=1,custom_renderer_viewer_epoch=2,custom_renderer_display_viewer_epoch=2;
 unsigned custom_renderer_presented_frames=0;
}state,*is=&state;
bool supported=true;unsigned color_seen=99,calls=0;int reply=1,mode=0,logs=0;
void* proc(HMODULE module,char const* name){assert(module==is->custom_renderer_module);
 assert(std::strcmp(name,"c3x_renderer_native_required_present_supported")==0);return supported?reinterpret_cast<void*>(1):nullptr;}
auto p_GetProcAddress=proc;
void log_custom_renderer_event(char const*,int){++logs;}
int translate_custom_renderer_native(int op,void*,void*,RECT*,void*,unsigned color){
 assert(op==C3X_NATIVE_IMAGE_PRESENT);++calls;color_seen=color;
 if(mode==1){is->custom_renderer_module=reinterpret_cast<void*>(4);is->custom_renderer_init_state=IS_LOADING;}
 if(mode==2)++is->custom_renderer_map_epoch;
 if(mode==3)++is->custom_renderer_viewer_epoch;
 if(mode==4)++is->custom_renderer_viewer_civ_id;
 if(mode==5)is->current_config.enable_custom_rendering=false;
 if(mode==6)p_main_screen_form=&other_screen;
 if(mode==7)p_bic_data->Map.Tiles=reinterpret_cast<void*>(9);
 if(mode==8)p_bic_data=nullptr;
 return reply;
}
int transfer(){void* image=nullptr;void* graph=nullptr;RECT* rect=nullptr;
''' + block + r'''
 return 917;
}
void arm(bool first_map_delivery){
''' + arm + r'''
}
void reset(){state={};p_main_screen_form=&screen;p_bic_data=&bic;screen={};bic.Map.Tiles=reinterpret_cast<void*>(8);
 supported=true;reply=1;mode=0;logs=0;calls=0;color_seen=99;}
int main(){reset();assert(transfer()==0&&color_seen==1&&!state.custom_renderer_first_front_pending&&logs==1);
 reset();assert(transfer()==0&&color_seen==1);assert(transfer()==0&&color_seen==0&&calls==2);
 reset();supported=false;assert(transfer()==0&&color_seen==0&&state.custom_renderer_first_front_pending&&!logs); // old bridge
 reset();state.current_config.enable_custom_rendering=false;reply=0;
 assert(transfer()==917&&color_seen==0&&!state.custom_renderer_first_front_pending&&!logs); // original config-off return
 for(unsigned guard=0;guard<9;++guard){reset();
  if(guard==0)state.custom_renderer_draw_in_progress=true;
  if(guard==1)state.custom_renderer_frame_active=true;
  if(guard==2)state.custom_renderer_capture_only=true;
  if(guard==3)screen.is_now_loading_game=true;
  if(guard==4)state.custom_renderer_display_viewer_epoch=0;
  if(guard==5)state.custom_renderer_native_module=reinterpret_cast<void*>(4);
  if(guard==6)state.custom_renderer_first_front_pending=false;
  if(guard==7)p_bic_data=nullptr;
  if(guard==8)bic.Map.Tiles=nullptr;
  assert(transfer()==0&&color_seen==0&&!logs);
 }
 for(int n=1;n<=8;++n){reset();mode=n;assert(transfer()==0&&color_seen==1&&!logs);
  assert(state.custom_renderer_first_front_pending&&state.custom_renderer_init_state==(n==1?IS_LOADING:IS_OK));}
 reset();reply=-1;assert(transfer()==0&&color_seen==1&&state.custom_renderer_first_front_pending&&state.custom_renderer_init_state==IS_INIT_FAILED);
 reset();state.custom_renderer_first_front_pending=false;arm(true);assert(state.custom_renderer_first_front_pending);
 reset();state.custom_renderer_first_front_pending=false;arm(false);assert(!state.custom_renderer_first_front_pending); // interturn/map redraw
 reset();state.custom_renderer_first_front_pending=false;state.custom_renderer_composited=false;arm(true);assert(!state.custom_renderer_first_front_pending);
 reset();state.custom_renderer_first_front_pending=false;state.custom_renderer_display_viewer_epoch=0;arm(true);assert(!state.custom_renderer_first_front_pending);
}
''')

    def test_required_async_transfer_joins_reliable_prefix_and_copies_identities(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <chrono>
#include <cstring>
using namespace std::chrono_literals;
struct Shared {std::uint64_t handle=0;unsigned width=0,height=0;};
struct State {std::promise<void> entered,release;std::shared_future<void> released=release.get_future().share();
 std::atomic<unsigned> units{0},required{0},ordinary{0},adopted{0};};
struct Fake {
 State& state;explicit Fake(State& s):state(s){}
 void publication_pressure(std::size_t){}void supersede_pending_camera(){}
 int camera_begin(c3x_renderer_camera_request_v1 const&,long long& t){t=31;return C3X_RENDERER_RESULT_PENDING;}
 int camera_ready(long long t,c3x_renderer_gpu_camera_view_v1& v){assert(t==31);
  v={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(v)};v.camera.ticket=31;
  v.image={sizeof(v.image)};v.image.ticket=41;v.image.map_image=73;v.image.session=2;
  v.camera.output={C3X_RENDERER_API_VERSION,sizeof(v.camera.output)};return C3X_RENDERER_RESULT_OK;}
 int camera_poll(long long t,c3x_renderer_gpu_camera_view_v1& v){++state.adopted;return camera_ready(t,v);}
 int unit_state(c3x_renderer_unit_state_v1 const&){state.entered.set_value();state.released.wait();++state.units;return C3X_RENDERER_RESULT_OK;}
 int visual_policy(unsigned){return 1;}
 int present(c3x_renderer_gpu_present_v1 const& v,Shared&){assert(v.ticket==41&&v.image==73);++state.ordinary;return C3X_RENDERER_RESULT_OK;}
 int present_required(c3x_renderer_gpu_present_v1 const& v,Shared& frame){
  assert(v.ticket==41&&v.image==73&&v.width==640&&v.height==480&&state.adopted==1&&state.units==1);
  ++state.required;frame.width=640;frame.height=480;return C3X_RENDERER_RESULT_OK;}
};
int main(){State state;c3x_remote_scene::AsyncSceneClient<Fake> client(true,[](char const*){assert(false);},state);
 c3x_renderer_frame_v1 frame{};c3x_renderer_camera_request_v1 request{C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,{}};
 long long ticket=0;assert(client.camera_begin(request,ticket)==C3X_RENDERER_RESULT_PENDING);
 c3x_renderer_gpu_camera_view_v1 view{C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
 auto deadline=std::chrono::steady_clock::now()+2s;
 while(client.camera_poll(ticket,view)!=C3X_RENDERER_RESULT_OK){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::sleep_for(1ms);}
 c3x_renderer_unit_state_v1 unit{sizeof(unit)};assert(client.unit_state(unit)==C3X_RENDERER_RESULT_OK);state.entered.get_future().get();
 c3x_renderer_gpu_present_v1 value{sizeof(value)};value.ticket=view.image.ticket;value.image=view.image.map_image;value.width=640;value.height=480;
 Shared shared;auto done=std::async(std::launch::async,[&]{return client.present_required(value,shared);});
 deadline=std::chrono::steady_clock::now()+2s;
 while(client.publication_status().records<2){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
 value.ticket=value.image=999;value.width=value.height=1;
 assert(done.wait_for(20ms)==std::future_status::timeout&&state.required==0);
 state.release.set_value();assert(done.get()==C3X_RENDERER_RESULT_OK&&state.required==1&&shared.width==640&&shared.height==480);
 value.ticket=view.image.ticket;value.image=view.image.map_image;
 assert(client.present(value,shared)==C3X_RENDERER_RESULT_OK);
 deadline=std::chrono::steady_clock::now()+2s;
 while(!state.ordinary){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
 assert(client.publication_status().rejected==0);
}
''')

    def test_helper_dedicated_wire_subtype_requires_actual_delivery_capability(self):
        source = (ROOT / "Renderer/native/helper_trial/scene_workload.cpp").read_text()
        start = source.index("if(wire.kind==unsigned(Kind::presentation)&&(wire.subtype==0||wire.subtype==3))")
        block = block_at(source, start)
        run_cpp(r'''
#include "Renderer/native/gpu_frame_api.h"
#include <cassert>
#include <map>
#include <stdexcept>
using LONG=long;
LONG InterlockedIncrement(volatile LONG* p){return ++*p;}
void require(bool value,char const* error){if(!value)throw std::runtime_error(error);}
enum class Kind{presentation=7};
struct Reader {long long values[10]{0,41,73,640,480,0,0,640,480,1};unsigned at=0;
 template<class T>void operator()(T& value){value=T(values[at++]);}
 unsigned u32(){return unsigned(values[at++]);}void done(){assert(at==10);}};
using Present=int(*)(c3x_renderer_gpu_present_v1 const*,unsigned,std::uint64_t*,unsigned*,unsigned*);
unsigned required_calls=0,ordinary_calls=0;int reply=C3X_RENDERER_RESULT_OK;
int present_common(c3x_renderer_gpu_present_v1 const* v,unsigned pid,std::uint64_t* handle,unsigned* w,unsigned* h){
 assert(!pid&&!v->action&&v->ticket==41&&v->image==73&&v->width==640&&v->height==480);
 *handle=0;*w=640;*h=480;return reply;}
int ordinary(c3x_renderer_gpu_present_v1 const* v,unsigned pid,std::uint64_t* handle,unsigned* w,unsigned* h){
 ++ordinary_calls;return present_common(v,pid,handle,w,h);}
int required(c3x_renderer_gpu_present_v1 const* v,unsigned pid,std::uint64_t* handle,unsigned* w,unsigned* h){
 ++required_calls;return present_common(v,pid,handle,w,h);}
struct Core {
 struct Wire {unsigned kind=unsigned(Kind::presentation),subtype=3,live=1,expected_code=C3X_RENDERER_RESULT_OK,
  code=0,executed=1,consumer_pid=0,width=0,height=0;std::uint64_t shared_handle=0;LONG visual_frames=0;}wire;
 Wire* telemetry=&wire;Reader in;Present present_shared=ordinary,required_present_shared=required;
 bool direct_surface_bound=true,direct_display_ready=false;unsigned starts=0;
 std::map<long long,long long> ticket_ids,image_ids;
 void stop_direct_cadence(){direct_display_ready=false;}
 void start_direct_cadence(){assert(direct_display_ready);++starts;}
 void offer(){
''' + block + r'''
 }
};
int main(){Core gate;gate.offer();assert(required_calls==1&&!ordinary_calls&&gate.wire.visual_frames==0&&gate.starts==1);
 Core normal;normal.wire.subtype=0;normal.offer();assert(ordinary_calls==1&&normal.wire.visual_frames==0&&normal.starts==1);
 Core missing;missing.required_present_shared=nullptr;bool refused=false;
 try{missing.offer();}catch(std::runtime_error const&){refused=true;}
 assert(refused&&!missing.starts&&required_calls==1); // mixed new-helper/old-core trio cannot claim readiness
 Core unbound;unbound.direct_surface_bound=false;refused=false;
 try{unbound.offer();}catch(std::runtime_error const&){refused=true;}
 assert(refused&&!unbound.starts&&required_calls==1);
 Core failed;reply=C3X_RENDERER_RESULT_SUPERSEDED;failed.offer();
 assert(failed.wire.code==unsigned(reply)&&!failed.starts&&!failed.wire.visual_frames);
}
''')


if __name__ == "__main__":
    unittest.main()
