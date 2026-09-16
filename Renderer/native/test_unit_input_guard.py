"""Run the production compatibility scheduler against native press ownership."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


def scheduler_program(source):
    source = source[source.index('extern "C" __declspec(dllexport) int c3x_renderer_schedule_idle('):]
    source = source[:source.index('\n}\n') + 3]
    return r'''
#define __declspec(x)
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
constexpr int VK_LBUTTON=1,VK_RBUTTON=2,VK_MBUTTON=4;
int held=0,queries=0,logs=0;
short GetKeyState(int key){++queries;return key==held?short(0x8000):0;}
void OutputDebugStringA(char const*){++logs;}
int c3x_renderer_schedule(c3x_renderer_schedule_v1 const* in,c3x_renderer_schedule_result_v1* out){
 if(!in || !out)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
 *out={};out->request_redraw=1;out->dirty_flags=123;out->phase_millionths=456;out->skipped_frame_count=2;
 return C3X_RENDERER_RESULT_OK;
}
''' + source + r'''
int main(){
 c3x_renderer_schedule_v1 in={};c3x_renderer_schedule_result_v1 out={};
 assert(c3x_renderer_schedule_idle(nullptr,&out)==C3X_RENDERER_RESULT_BAD_ARGUMENT && !queries);
 in.frequency=1000;
 // Native m01_Show_Enabled clears pressed-form ownership when command buttons
 // are rebuilt by a full map redraw. This is independent of gesture duration.
 for(bool selected:{true,false})for(int button:{VK_LBUTTON,VK_RBUTTON,VK_MBUTTON}){
  bool native_press_owned=true;
  in.state_flags=C3X_RENDERER_SCHEDULER_MAP_VISIBLE|C3X_RENDERER_SCHEDULER_FOCUSED;
  if(selected)in.state_flags|=C3X_RENDERER_SCHEDULER_PATHFINDER_HOLD; // old pre-drag latch
  held=button;
  for(int elapsed:{0,126,199,499,500,2000}){
   in.now_ticks=10000+elapsed;
   assert(c3x_renderer_schedule_idle(&in,&out)==C3X_RENDERER_RESULT_OK);
   if(selected&&out.request_redraw)native_press_owned=false;
   assert(native_press_owned); // old selected-unit bypass fails at elapsed=0
   assert(!out.request_redraw&&!out.dirty_flags&&!out.skipped_frame_count&&out.rebase_clock);
   assert(out.phase_millionths==456); // observations never consume native input
  }
  // Native release/click or native hold/action dispatch owns the gesture.
  // A stale native pre-drag bit must not inhibit rendering after OS release.
  held=0;in.now_ticks+=1;c3x_renderer_schedule_idle(&in,&out);
  assert(native_press_owned&&out.request_redraw&&out.dirty_flags==123&&!out.rebase_clock);
 }
 assert(logs==42&&queries==126);
}
'''


class UnitInputGuardTests(unittest.TestCase):
    def test_native_press_has_no_selected_unit_or_elapsed_time_exception(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        run_cpp(scheduler_program(source))
        wrapper = source[source.index('extern "C" __declspec(dllexport) int c3x_renderer_schedule_idle('):]
        wrapper = wrapper[:wrapper.index('\n}\n') + 3]
        for forbidden in ('GetQueueStatus', 'GetAsyncKeyState', 'PeekMessage', 'static ', 'held_ticks'):
            self.assertNotIn(forbidden, wrapper)
        injected = (ROOT / 'injected_code.c').read_text()
        scheduler = injected[injected.index('void\ncustom_renderer_scheduler_tick'):]
        scheduler = scheduler[:scheduler.index('\n}\n') + 3]
        self.assertNotIn('C3X_RENDERER_SCHEDULER_PATHFINDER_HOLD', scheduler)
