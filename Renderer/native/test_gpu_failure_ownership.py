"""Exercise production failure branches without allowing stale GDI or lost native writes."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class GpuFailureOwnershipTests(unittest.TestCase):
    def test_rejected_present_keeps_front_until_successful_native_handoff(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        body = '    int present_gpu(' + source.split('    int present_gpu(', 1)[1].split('    // Native final transfer', 1)[0]
        run_cpp(r'''
#include <cassert>
#include <mutex>
#include <memory>
#include <stdexcept>
#include "Renderer/native/gpu_frame_api.h"
using HWND=void*;
void OutputDebugStringA(char const*){}
namespace c3x_inputs {
enum class Kind{presentation};
struct Writer{template<class T>void operator()(T){}void u32(unsigned){}};
struct Call{template<class F>Call(Kind,unsigned,F f){Writer w;f(w);}int result(int code){return code;}};
struct Runtime{void gameplay(){}};Runtime& runtime(){static Runtime r;return r;}
struct Realtime{void offer(int){}};Realtime& realtime_replay(){static Realtime r;return r;}
}
struct State {
 std::mutex call_mutex,state_mutex;bool gpu_presentation=true,visual_present_pending=false,visual_delivery=false;
 c3x_renderer_gpu_present_v1 gpu_present={};int result=1;bool explode=false;
 struct Session{int ticket=7;bool active=true;int current_ticket(){return ticket;}void stop_visuals(){active=false;}bool visual_ready(){return active;}};
 struct{std::unique_ptr<Session> gpu_composition=std::make_unique<Session>();void* device=nullptr;}renderer_state;
 struct Presenter{unsigned resets=0,releases=0;bool caller_thread(){return true;}void reset(){++resets;}
  void release_native(){++releases;}bool prepare(HWND,void*,int,int,bool){return true;}int present(){return 1;}}gpu_presenter;
 struct Cadence{template<class F>void enable(F){}}visual_cadence;
 enum class Command{gpu_present};
 void start_locked(){}void drain_camera_locked(std::unique_lock<std::mutex>&,bool){}
 void stop_visual_delivery(){visual_delivery=false;}void advance_visual_clock(){}
 int visual_frame(bool){return 1;}
 int submit_locked(std::unique_lock<std::mutex>&,Command){if(explode)throw std::bad_alloc();return result;}
''' + body + r'''
};
int main(){
 for(bool exception:{false,true}){
  State s;c3x_renderer_gpu_present_v1 r={sizeof(r)};r.ticket=7;r.width=320;r.height=240;r.area[2]=320;r.area[3]=240;
  auto* session=s.renderer_state.gpu_composition.get();s.explode=exception;s.result=C3X_RENDERER_RESULT_ERROR;
  assert(s.present_gpu(r)!=C3X_RENDERER_RESULT_OK);
  assert(s.gpu_presenter.resets==0&&s.gpu_presenter.releases==0&&s.renderer_state.gpu_composition.get()==session);
  s.explode=false;r.action=2;
  assert(s.present_gpu(r)!=C3X_RENDERER_RESULT_OK&&s.gpu_presenter.releases==0);
  s.result=C3X_RENDERER_RESULT_OK;
  assert(s.present_gpu(r)==C3X_RENDERER_RESULT_OK&&s.gpu_presenter.releases==1);
 }
}
''')

    def test_worker_exception_preserves_gpu_owned_native_canvases(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        start = source.index('                if(command==Command::unit)renderer_state.unit_bodies.reset_gpu();')
        body = source[start:source.index('                output = ', start)]
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include <stdexcept>
struct State {
 int gpu_composition=0;unsigned resets=0;
 struct {unsigned resets=0;void reset_gpu(){++resets;}}unit_bodies;
 struct{void write(char const*,char const*,bool){}}trace;
 void reset(){++resets;gpu_composition=0;}
};
enum class Command{unit,native_screen,visual_frame,gpu_present,gpu_images,gpu_unit,tactical,render,gpu_render};
int main(){
 for(auto command:{Command::gpu_present,Command::gpu_images,Command::gpu_unit,Command::tactical,Command::render,Command::gpu_render}){
  State renderer_state;renderer_state.gpu_composition=1;
  try{throw std::bad_alloc();}catch(...){
''' + body + r'''
  }
  assert(renderer_state.gpu_composition==1 && renderer_state.resets==0);
 }
}
''')
