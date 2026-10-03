"""Execute the production timer/advancement split without launching Civ III."""
import unittest
import csv
import os
from pathlib import Path
from Renderer.native.native_cpp_test import run_cpp
from Renderer.tools.audit_native_visual_cadence import audit
ROOT=Path(__file__).resolve().parents[2]


class NativeVisualCadenceTests(unittest.TestCase):
    def test_presentation_permit_is_nonblocking_and_survives_noop(self):
        run_cpp(r'''
#include <windows.h>
#include <cassert>
#include "Renderer/native/presentation_permit.h"
int main(){
 c3x_renderer::PresentationPermit permit;
 assert(!permit.ready());
 auto signal=CreateEventW(nullptr,FALSE,FALSE,nullptr);assert(signal);
 permit.reset(signal);
 auto begin=GetTickCount64();
 for(unsigned n=0;n<1000;++n)assert(!permit.ready());
 assert(GetTickCount64()-begin<1000);
 SetEvent(signal);assert(permit.ready());
 for(unsigned n=0;n<1000;++n)assert(permit.ready()); // unchanged static front
 permit.presented();assert(!permit.ready());
 SetEvent(signal);assert(permit.ready());permit.reset();assert(!permit.ready());
 auto replacement=CreateEventW(nullptr,FALSE,TRUE,nullptr);assert(replacement);
 permit.reset(replacement);assert(permit.ready());permit.presented();assert(!permit.ready());
}
''')

    def test_authorized_gog_symbols_enable_the_native_branch(self):
        expected={
            'p_main_animation_timer':('define',0x009F6500),
            'Timer_reset_and_activate':('define',0x006205D0),
            'Units_Image_Data_advance_animations':('inlead',0x00405FC0),
            'p_native_timer_inhibited':('define',0x0072C2C4),
            'p_native_game_ending':('define',0x00CC37BC),
            'Advisor_GUI_open':('inlead',0x0049D070),
        }
        found=set()
        for row in csv.reader((ROOT/'civ_prog_objects.csv').read_text().replace('\t',' ').splitlines(),skipinitialspace=True):
            row=[item.strip() for item in row]
            if len(row)!=6 or row[4] not in expected:continue
            self.assertNotIn(row[4],found)
            found.add(row[4])
            self.assertEqual((row[0],int(row[1],0)),expected[row[4]])
            other_builds=[0x4A3AF0,0x49D100] if row[4]=='Advisor_GUI_open' else [0,0]
            self.assertEqual([int(row[2],0),int(row[3],0)],other_builds)
        self.assertEqual(found,set(expected))

    def test_gog_bytes_and_four_argument_abi(self):
        original=ROOT/'Renderer/native/build/unit-audit-original.exe'
        if not original.exists():self.skipTest('local GOG executable required for byte audit')
        self.assertEqual(audit(original)['status'],'pass')

    def test_native_timer_does_not_drive_resident_visuals(self):
        source=(ROOT/'injected_code.c').read_text()
        body=source[source.index('void __stdcall\npatch_on_timer_0x9F6500'):source.index('void __fastcall\npatch_Units_Image_Data_load_animated_effect')]
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#define __stdcall
const int C3X_NATIVE_VISUAL_POLICY=116;
const int C3X_RENDERER_RESULT_DEVICE_ERROR=5;
char animator_bytes[64]={};struct {struct {char* field_18E4=animator_bytes;}animator;} screen;auto p_main_screen_form=&screen;
const unsigned C3X_RENDERER_DIRTY_SCENE=1;
int native_calls=0,legacy_redraws=0,effects=0;bool resident=true,reenter=false;
int policy(int,void*,void*,void const*,void const*,unsigned color){assert(color==2);return resident;}
struct State{bool custom_renderer_redraw_pending=false;unsigned custom_renderer_dirty_flags=0;bool custom_renderer_timer_running=false;struct{bool enable_custom_animations=true,enable_custom_rendering=true;}current_config;
 int (*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned)=policy;
 int (*custom_renderer_backend_healthy)()=nullptr;};
State state;State* is=&state;unsigned debug=0;unsigned* p_debug_mode_bits=&debug;
void log_custom_renderer_event(char const*,int){}
void patch_on_timer_0x9F6500();
void on_timer_0x9F6500(){++native_calls;if(reenter)patch_on_timer_0x9F6500();}
void custom_renderer_scheduler_tick(){++legacy_redraws;}
void clear_active_custom_tile_animation_effects(){++effects;}
void tile_animation_scheduler_tick(){++effects;}
'''+body+r'''
int main(){
 for(int i=0;i<100;++i)patch_on_timer_0x9F6500();
 assert(native_calls==100&&legacy_redraws==0&&effects==0);
 reenter=true;patch_on_timer_0x9F6500();assert(native_calls==101&&!state.custom_renderer_timer_running);
 state.custom_renderer_redraw_pending=true;state.custom_renderer_dirty_flags=C3X_RENDERER_DIRTY_SCENE;
 patch_on_timer_0x9F6500();assert(animator_bytes[10]==1);--native_calls;
 state.custom_renderer_redraw_pending=false;animator_bytes[10]=0;
 resident=false;patch_on_timer_0x9F6500();assert(legacy_redraws==1&&native_calls==102);
 state.current_config.enable_custom_rendering=false;patch_on_timer_0x9F6500();assert(effects==1&&native_calls==103);
}
''')
        self.assertNotIn('Animator_update (',body)
        self.assertNotIn('Timer_reset_and_activate (',body)

    def test_independent_cadence_stops_and_has_no_catchup_queue(self):
        run_cpp(r'''
#include <cassert>
#include <atomic>
#include "Renderer/native/visual_cadence.h"
int main(){
 using namespace std::chrono;
 c3x_renderer::VisualCadence cadence;std::atomic<unsigned> calls{0};
 steady_clock::time_point last_end;bool first=true;
 cadence.enable([&]{
  auto begin=steady_clock::now();
  if(!first)assert(begin-last_end>=milliseconds(9));
  first=false;++calls;std::this_thread::sleep_for(milliseconds(60));last_end=steady_clock::now();
 });
 std::this_thread::sleep_for(milliseconds(260));cadence.stop();
 unsigned finished=calls;assert(finished>=2&&finished<=4);
 std::this_thread::sleep_for(milliseconds(80));assert(calls==finished);
 cadence.enable([&]{++calls;});std::this_thread::sleep_for(milliseconds(100));
 cadence.disable();cadence.stop();assert(calls>finished);
 // Publishing UI repeatedly must not interrupt the cadence's minimum pause.
 c3x_renderer::VisualCadence stable(milliseconds(50),milliseconds(40));
 calls=0;steady_clock::time_point prior;
 auto sample=[&]{auto now=steady_clock::now();if(calls)assert(now-prior>=milliseconds(35));prior=now;++calls;};
 stable.enable(sample);
 for(unsigned n=0;n<100;++n){stable.enable(sample);std::this_thread::sleep_for(milliseconds(2));}
 stable.stop();assert(calls>=2&&calls<=7);
}
''')

    def test_delivery_backpressure_failure_and_native_ownership(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        body='    int visual_frame('+source.split('    int visual_frame(',1)[1].split('    int present_gpu(',1)[0]
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <memory>
#include "Renderer/native/gpu_frame_api.h"
namespace c3x_inputs {
enum class Kind {visual};
struct Writer {void u32(unsigned){}};
struct Call {bool completed=false;template<class F>Call(Kind,unsigned,F f){Writer out;f(out);}int result(int code){completed=true;return code;}};
}
struct LARGE_INTEGER{long long QuadPart=0;};void QueryPerformanceCounter(LARGE_INTEGER* q){++q->QuadPart;}
unsigned GetEnvironmentVariableA(char const*,char*,unsigned){return 0;}
using HWND=void*;constexpr int GA_ROOT=2;
bool visible=true,minimized=false;bool IsWindowVisible(HWND){return visible;}bool IsIconic(HWND){return minimized;}
struct Session{bool active=true;bool visual_active(){return active;}void stop_visuals(){active=false;}
 unsigned visual_bytes(){return 0;}std::size_t visual_nodes(){return 0;}std::size_t visual_sources(){return 0;}
 unsigned visual_sample_allocations(){return 0;}unsigned visual_sample_imports(){return 0;}};
struct State{
 std::mutex call_mutex,state_mutex;bool running=true,visual_delivery=true,visual_allowed=true,visual_present_pending=false;
 bool camera_active=false,camera_pending=false,camera_gpu=false;int camera_result=0,camera_ticket=0,gpu_camera_front_ticket=0;
 long long visual_last=0;unsigned long long visual_frames=0,visual_map_samples=0;
 struct{c3x_renderer_frame_v1 frame={};}gpu_publication;
 long long visual_ticks=0,visual_frequency=1;
 struct{void* window=nullptr;}gpu_present;
 struct Presenter{bool caller=true;int result=1;unsigned calls=0;bool caller_thread(){return caller;}bool view(){return true;}
  int present(bool){++calls;return result;}}gpu_presenter;
 struct{std::unique_ptr<Session> gpu_composition=std::make_unique<Session>();
  struct{double milliseconds(long long){return 0;}void write(char const*,char const*,bool){}}trace;}renderer_state;
 enum class Command{visual_frame};int draw_result=1;unsigned draws=0;
 int submit_locked(std::unique_lock<std::mutex>&,Command){++draws;return draw_result;}
 void snapshot_fresh_units(c3x_renderer_frame_v1 const&,long long,long long){}
 void advance_visual_clock(){}void stop_visual_delivery(){visual_delivery=false;visual_present_pending=false;}
 struct ForegroundCameraPause{ForegroundCameraPause(State&,std::unique_lock<std::mutex>&,char const*){}};
'''+body+r'''
};
int main(){
 State hidden;visible=false;assert(hidden.visual_frame(true)==C3X_RENDERER_RESULT_PENDING&&!hidden.draws);
 visible=true;minimized=true;assert(hidden.visual_frame(true)==C3X_RENDERER_RESULT_PENDING&&!hidden.draws);
 minimized=false;assert(hidden.visual_frame(true)==1&&hidden.draws==1); // no foreground/focus predicate
 State s;s.gpu_presenter.caller=false;
 assert(s.visual_frame()==C3X_RENDERER_RESULT_PENDING&&!s.draws);
 s.camera_pending=true;assert(s.visual_frame(true)==C3X_RENDERER_RESULT_PENDING&&!s.draws);
 s.camera_pending=false;s.gpu_presenter.result=C3X_RENDERER_RESULT_PENDING;
 assert(s.visual_frame(true)==C3X_RENDERER_RESULT_PENDING&&s.visual_present_pending&&!s.visual_frames);
 // A render revision need not change for an outstanding Present to retry.
 s.draw_result=C3X_RENDERER_RESULT_PENDING;s.gpu_presenter.result=1;
 assert(s.visual_frame(true)==1&&!s.visual_present_pending&&s.visual_frames==1&&s.gpu_presenter.calls==2);
 s.draw_result=1;s.gpu_presenter.result=C3X_RENDERER_RESULT_ERROR;
 assert(s.visual_frame(true)==C3X_RENDERER_RESULT_ERROR&&!s.visual_delivery&&!s.renderer_state.gpu_composition->active);
 unsigned stopped=s.draws;assert(s.visual_frame(true)==C3X_RENDERER_RESULT_PENDING&&s.draws==stopped);
 State failed;failed.draw_result=C3X_RENDERER_RESULT_ERROR;
 assert(failed.visual_frame(true)==C3X_RENDERER_RESULT_ERROR&&!failed.gpu_presenter.calls&&!failed.visual_delivery);
}
''')

    @unittest.skipUnless(os.name == 'posix', 'host-only presentation stubs')
    def test_actual_direct_surface_readiness_controls_retry_and_grant_consumption(self):
        source = (ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        branch = source.split('}else if(command==Command::trial_visual_shared||command==Command::trial_required_visual_shared){', 1)[1]
        block = 'bool presentation_ready=' + branch.split('bool presentation_ready=', 1)[1].split(
            'if(result==C3X_RENDERER_RESULT_OK){trial_width=', 1)[0]
        run_cpp(r'''
#include <cassert>
#include <atomic>
#include <cstdio>
#include <memory>
#include <utility>
#include "Renderer/native/c3x_renderer_api.h"
using HRESULT=int;constexpr HRESULT S_OK=0;bool FAILED(HRESULT code){return code<0;}
struct LARGE_INTEGER{long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER* value){++value->QuadPart;}
template<std::size_t N,class... Args>int sprintf_s(char (&buffer)[N],char const* format,Args... args){
 return std::snprintf(buffer,N,format,args...);
}
struct Session{
 int draw=1;unsigned samples=0,published=0;
 int visual_frame(long long ticks,long long frequency,void*,void*,void*){
  assert(ticks==17&&frequency==1000);++samples;return draw;
 }
 void did_present(){++published;}
 std::uint64_t committed_revision()const{return 9;}
 unsigned presented_zoom(){return 81920;}
 std::pair<unsigned,unsigned> visual_publication(){return {7,8};}
 struct Work{unsigned operations=3,assemblies=1,copies=2,copied_pixels=64,assembly_pixels=32;
  unsigned selected_borrows=1,selected_owned=0,direct_native_images=1,avoided_copy_pixels=128;};
 Work visual_work(){return {};}
};
struct Swap{HRESULT result=S_OK;unsigned presents=0;
 HRESULT Present(unsigned sync,unsigned flags){assert(!sync&&!flags);++presents;return result;}
};
struct Surface{void* Get(){return nullptr;}};
struct Owner{
 enum class Command{trial_visual_shared,trial_required_visual_shared};
 Command command=Command::trial_visual_shared;
 struct Permit{bool admitted=false;unsigned polls=0,consumed=0;
  bool ready(){++polls;return admitted;}void presented(){assert(admitted);admitted=false;++consumed;}
 }trial_surface_permit;
 struct Trace{LARGE_INTEGER frequency{1000};unsigned rows=0;
  double milliseconds(long long ticks){return double(ticks);}
  void write(char const*,char const*,bool){++rows;}
 };
 struct{std::unique_ptr<Session> gpu_composition=std::make_unique<Session>();Trace trace;}renderer_state;
 Surface trial_surface_view,trial_surface_back,trial_surface_buffer;
 std::unique_ptr<Swap> trial_surface_swap=std::make_unique<Swap>();
 std::atomic<unsigned> presented_zoom_q16{65536};unsigned route_present_index=0;
 std::atomic<bool> trial_front_pending{true};std::uint64_t trial_presented_front_revision=0;
 std::atomic<std::uint64_t> trial_visual_permit_denials{0};
 long long visual_ticks=17,visual_frequency=1000;
 int offer(){
  bool phase_probe=true,route_witness=true;
  LARGE_INTEGER started{},prepared{},sampled{},finished{};
  int result=C3X_RENDERER_RESULT_ERROR;
''' + block + r'''
  return result;
 }
};
int main(){
 Owner denied;assert(denied.offer()==C3X_RENDERER_RESULT_BUSY);
 assert(denied.trial_surface_permit.polls==1&&!denied.renderer_state.gpu_composition->samples);
 assert(!denied.trial_surface_swap->presents&&!denied.trial_surface_permit.consumed);
 assert(denied.trial_front_pending&&denied.trial_presented_front_revision==0);
 assert(denied.trial_visual_permit_denials==1);
 assert(denied.presented_zoom_q16==65536&&!denied.route_present_index&&!denied.renderer_state.trace.rows);
 Owner noop;noop.trial_surface_permit.admitted=true;noop.renderer_state.gpu_composition->draw=0;
 for(unsigned i=0;i<3;++i)assert(noop.offer()==C3X_RENDERER_RESULT_PENDING);
 assert(noop.trial_surface_permit.admitted&&!noop.trial_surface_permit.consumed);
 assert(noop.renderer_state.gpu_composition->samples==3&&!noop.renderer_state.gpu_composition->published);
 assert(!noop.trial_surface_swap->presents&&noop.presented_zoom_q16==65536);
 assert(noop.trial_front_pending&&noop.trial_presented_front_revision==0);
 assert(noop.trial_visual_permit_denials==0);
 Owner drawn;drawn.trial_surface_permit.admitted=true;
 assert(drawn.offer()==C3X_RENDERER_RESULT_OK);
 assert(drawn.renderer_state.gpu_composition->samples==1&&drawn.renderer_state.gpu_composition->published==1);
 assert(drawn.trial_surface_swap->presents==1&&drawn.trial_surface_permit.consumed==1&&!drawn.trial_surface_permit.admitted);
 assert(drawn.presented_zoom_q16==81920&&drawn.route_present_index==1&&drawn.renderer_state.trace.rows==2);
 assert(!drawn.trial_front_pending&&drawn.trial_presented_front_revision==9);
 assert(drawn.offer()==C3X_RENDERER_RESULT_BUSY&&drawn.renderer_state.gpu_composition->samples==1);
 assert(drawn.trial_visual_permit_denials==1);
 Owner failed;failed.trial_surface_permit.admitted=true;failed.trial_surface_swap->result=-1;
 assert(failed.offer()==C3X_RENDERER_RESULT_DEVICE_ERROR);
 assert(failed.renderer_state.gpu_composition->samples==1&&failed.trial_surface_swap->presents==1);
 assert(failed.trial_surface_permit.admitted&&!failed.trial_surface_permit.consumed);
 assert(!failed.renderer_state.gpu_composition->published&&failed.presented_zoom_q16==65536&&!failed.route_present_index);
 assert(failed.trial_front_pending&&failed.trial_presented_front_revision==0);
 // Existing positive DXGI statuses are not successful publication evidence.
 Owner status;status.trial_surface_permit.admitted=true;status.trial_surface_swap->result=1;
 assert(status.offer()==C3X_RENDERER_RESULT_OK&&status.trial_surface_permit.admitted);
 assert(!status.renderer_state.gpu_composition->published&&!status.route_present_index&&status.presented_zoom_q16==65536);
 assert(status.trial_front_pending&&status.trial_presented_front_revision==0);
 // Required startup reuses the prepared front after a non-S_OK receipt. It
 // actually presents despite unchanged source work; ordinary static no-op does not.
 status.command=Owner::Command::trial_required_visual_shared;status.renderer_state.gpu_composition->draw=2;
 status.trial_surface_swap->result=S_OK;assert(status.offer()==C3X_RENDERER_RESULT_OK);
 assert(status.trial_surface_swap->presents==2&&status.renderer_state.gpu_composition->published==1);
 assert(!status.trial_front_pending&&status.trial_presented_front_revision==9&&status.trial_surface_permit.consumed==1);
}
''')

    def test_busy_offer_retries_without_reducing_static_frame_period(self):
        run_cpp(r'''
#include <cassert>
#include <atomic>
#include "Renderer/native/visual_cadence.h"
int main(){using namespace std::chrono;
 c3x_renderer::VisualCadence cadence(milliseconds(200),milliseconds(5));
 std::mutex mutex;std::condition_variable wake;std::atomic<unsigned> offers{0};bool complete=false;
 cadence.enable_retrying([&]{auto n=++offers;if(n<4)return true;
  {std::lock_guard<std::mutex> lock(mutex);complete=true;}wake.notify_one();return false;});
 {std::unique_lock<std::mutex> lock(mutex);assert(wake.wait_for(lock,milliseconds(250),[&]{return complete;}));}
 unsigned first=offers;assert(first==4);std::this_thread::sleep_for(milliseconds(70));
 assert(offers==first); // Completed or static work does not poll at the busy cadence.
 cadence.stop();unsigned stopped=offers;std::this_thread::sleep_for(milliseconds(30));assert(offers==stopped);
}
''')

    def test_direct_visual_busy_is_distinct_from_unchanged(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        body='    int trial_visual_shared('+source.split('    int trial_visual_shared(',1)[1].split('    int trial_priority_front_pending(',1)[0]
        run_cpp(r'''
#include <cassert>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>
#include "Renderer/native/c3x_renderer_api.h"
using DWORD=unsigned;
struct State{std::mutex call_mutex,state_mutex;DWORD trial_consumer_pid=0;
 std::atomic<std::uint64_t> trial_visual_call_busy{0},trial_visual_state_busy{0};
 std::uint64_t trial_handle=0;unsigned trial_width=0,trial_height=0,submits=0;
 long long visual_ticks=0,visual_frequency=0;int result=C3X_RENDERER_RESULT_PENDING;
 enum class Command{trial_visual_shared};
 int submit_locked(std::unique_lock<std::mutex>&,Command){++submits;return result;}
'''+body+r'''
};
int main(){State state;std::uint64_t handle=0;unsigned w=0,h=0;
 for(auto* gate:{&state.call_mutex,&state.state_mutex}){
  std::atomic<bool> held{false},release{false};std::thread owner([&]{std::lock_guard<std::mutex> lock(*gate);
   held=true;while(!release)std::this_thread::yield();});
  while(!held)std::this_thread::yield();
  assert(state.trial_visual_shared(123,1000,0,handle,w,h)==C3X_RENDERER_RESULT_BUSY);
  assert(state.submits==0&&state.visual_ticks==0);release=true;owner.join();
 }
 assert(state.trial_visual_call_busy==1&&state.trial_visual_state_busy==1);
 assert(state.trial_visual_shared(456,1000,0,handle,w,h)==C3X_RENDERER_RESULT_PENDING);
 assert(state.submits==1&&state.visual_ticks==456);
 state.result=C3X_RENDERER_RESULT_OK;assert(state.trial_visual_shared(789,1000,0,handle,w,h)==C3X_RENDERER_RESULT_OK);
 assert(state.submits==2&&state.visual_ticks==789);
}
''')
