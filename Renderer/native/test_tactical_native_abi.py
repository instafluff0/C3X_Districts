"""Exercise the real injected route-cursor wrapper with Civ III's x86 ABI."""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp


class TacticalNativeAbiTests(unittest.TestCase):
    def test_route_cursor_callee_cleanup_on_custom_and_native_paths(self):
        source = (Path(__file__).resolve().parents[2] / "injected_code.c").read_text()
        name = source.index("patch_Main_Screen_Form_draw_route_cursor (")
        start = source.rfind("\nvoid ", 0, name) + 1
        end = source.index("\n}\n", name) + 3
        wrapper = source[start:end]
        run_cpp(r'''
#include <windows.h>
#include <cassert>
#include <cstdio>
enum {C3X_NATIVE_TACTICAL_CAPABLE=1,C3X_NATIVE_TACTICAL_TARGET=2};
int native_calls=0,custom_calls=0,capable=1,last_x=0,last_y=0;
void __stdcall native_cursor(int x,int y){++native_calls;last_x=x;last_y=y;}
// Keep the current patch-table type to exercise the forwarding ABI repair too.
#define Main_Screen_Form_draw_route_cursor ((void (__cdecl *)(int,int))native_cursor)
struct Screen {struct {struct {struct {struct {void* Image;} JGL;} Canvas;} Data;} Units_Control;} screen;
auto p_main_screen_form=&screen;
int image(int op,void* target,void*,void const* data,void const*,unsigned){
 if(op==C3X_NATIVE_TACTICAL_CAPABLE)return capable;
 assert(op==C3X_NATIVE_TACTICAL_TARGET && target==screen.Units_Control.Data.Canvas.JGL.Image);
 auto point=static_cast<int const*>(data);++custom_calls;last_x=point[0];last_y=point[1];return 1;
}
struct State {struct {bool enable_custom_rendering;} current_config;
 int (*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned);
} state;
auto is=&state;
''' + wrapper + r'''
// Reproduce the native caller: push two coordinates, call, no caller cleanup.
// Restore ESP before asserting so the old cdecl wrapper gives a precise failure.
void check_stack(int x,int y){
 unsigned before=0,after=0;
 void* entry=reinterpret_cast<void*>(patch_Main_Screen_Form_draw_route_cursor);
 __asm {
  mov before,esp
  push y
  push x
  mov eax,entry
  call eax
  mov after,esp
  mov esp,before
 }
 if(before!=after){std::fprintf(stderr,"FAIL route cursor stack delta=%d\n",int(after-before));ExitProcess(1);}
 assert(last_x==x && last_y==y);
}
int main(){
 for(int mode=0;mode<4;++mode){
  state.current_config.enable_custom_rendering=mode!=0;
  state.custom_renderer_native_image=mode==1?nullptr:image;
  capable=mode!=2;
  int old_native=native_calls,old_custom=custom_calls;
  for(int n=0;n<100;++n)check_stack(1110+n,507-n);
  assert(native_calls-old_native==(mode==3?0:100));
  assert(custom_calls-old_custom==(mode==3?100:0));
 }
 std::puts("PASS native route cursor: callee pops two arguments; repeated custom, config-off, unavailable and fallback calls preserve stack and coordinates");
}
''')


if __name__ == "__main__":
    unittest.main()
