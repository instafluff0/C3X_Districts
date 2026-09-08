"""Exercise the actual DLL ambient scheduler wrapper without sending input."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

class UnitInputGuardTests(unittest.TestCase):
    def test_click_window_release_and_sustained_hold_do_not_consume_input(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++ compiler unavailable')
        root=Path(__file__).parent
        source=(root/'c3x_renderer.cpp').read_text()
        source=source[source.index('extern "C" __declspec(dllexport) int c3x_renderer_schedule_idle('):]
        self.assertNotIn('GetQueueStatus',source)
        self.assertNotIn('GetAsyncKeyState',source)
        self.assertNotIn('PeekMessage',source)
        code=r'''
#define __declspec(x)
#include "c3x_renderer_api.h"
#include <cassert>
constexpr int VK_LBUTTON=1,VK_RBUTTON=2,VK_MBUTTON=4;
int held=0,queries=0,logs=0,callback_logs=0,click_logs=0,release_logs=0,request_logs=0,pathfinder_logs=0;
short GetKeyState(int key){++queries;return key==held?short(0x8000):0;}
void OutputDebugStringA(char const* line){
 ++logs;
 if(std::strstr(line,"stage=scheduler-callback"))++callback_logs;
 if(std::strstr(line,"reason=click-decision-guard"))++click_logs;
 if(std::strstr(line,"reason=release-guard"))++release_logs;
 if(std::strstr(line,"reason=redraw-request"))++request_logs;
 if(std::strstr(line,"pathfinder=1"))++pathfinder_logs;
}
int c3x_renderer_schedule(c3x_renderer_schedule_v1 const* in,c3x_renderer_schedule_result_v1* out){
 if(!in || !out)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
 *out={};out->request_redraw=1;out->dirty_flags=123;out->phase_millionths=456;out->skipped_frame_count=2;
 return C3X_RENDERER_RESULT_OK;
}
'''+source+r'''
int main(){
 c3x_renderer_schedule_v1 in={};c3x_renderer_schedule_result_v1 out={};
 assert(c3x_renderer_schedule_idle(nullptr,&out)==C3X_RENDERER_RESULT_BAD_ARGUMENT && !queries);
 in.frequency=1000;
 in.state_flags=C3X_RENDERER_SCHEDULER_MAP_VISIBLE|C3X_RENDERER_SCHEDULER_FOCUSED;
 in.now_ticks=1000;held=VK_LBUTTON;
 in.state_flags|=C3X_RENDERER_SCHEDULER_PATHFINDER_HOLD;
 c3x_renderer_schedule_idle(&in,&out);
 assert(out.request_redraw && out.dirty_flags==123 && !out.rebase_clock); // map hold animates immediately
 in.now_ticks+=66;held=0;in.state_flags&=~C3X_RENDERER_SCHEDULER_PATHFINDER_HOLD;
 c3x_renderer_schedule_idle(&in,&out);assert(!out.request_redraw); // release still settles
 in.now_ticks+=66;c3x_renderer_schedule_idle(&in,&out);assert(out.request_redraw);
 for(int button:{VK_LBUTTON,VK_RBUTTON,VK_MBUTTON}){
   in.now_ticks+=1000;held=button;c3x_renderer_schedule_idle(&in,&out);
   assert(!out.request_redraw && !out.dirty_flags && !out.skipped_frame_count && out.rebase_clock && out.phase_millionths==456);
   in.now_ticks+=200;c3x_renderer_schedule_idle(&in,&out);assert(!out.request_redraw); // native click decision
   in.now_ticks+=51;c3x_renderer_schedule_idle(&in,&out);assert(out.request_redraw); // sustained pathfinder hold
   in.now_ticks+=66;held=0;c3x_renderer_schedule_idle(&in,&out);assert(!out.request_redraw); // release settles
   in.now_ticks+=66;c3x_renderer_schedule_idle(&in,&out);assert(out.request_redraw && out.phase_millionths==456);
 }
 assert(logs==18 && callback_logs==18); // one complete scheduler decision record per callback
 assert(click_logs==6 && release_logs==4 && request_logs==8 && pathfinder_logs==1);
}
'''
        code='#include <cstdio>\n#include <cstring>\n#include <initializer_list>\n'+code
        with tempfile.TemporaryDirectory() as folder:
            cpp=Path(folder)/'input.cpp';exe=Path(folder)/'input';cpp.write_text(code)
            subprocess.run([compiler,'-std=c++17','-I',str(root),str(cpp),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True)
