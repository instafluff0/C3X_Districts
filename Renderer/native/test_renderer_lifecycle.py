"""Execute the loading barrier and config-off menu boundary with controlled readiness."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
ROOT = Path(__file__).resolve().parents[2]

class RendererLifecycleTests(unittest.TestCase):
    def test_first_map_wait_has_one_loading_scope_and_a_deadline(self):
        source=(ROOT/'injected_code.c').read_text()
        block=source.split('// Cold start is part of loading.',1)[1].split('\n\tbool gpu_map',1)[0]
        block=block[block.index('if ('):].replace('(void *)(*p_GetProcAddress)', '(Sleeper)(*p_GetProcAddress)')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#define WINAPI
using DWORD=unsigned;using Sleeper=void(*)(DWORD);
struct LARGE_INTEGER {long long QuadPart;};
enum {C3X_RENDERER_RESULT_ERROR=0,C3X_RENDERER_RESULT_OK=1,C3X_RENDERER_RESULT_PENDING=4,C3X_NATIVE_MAP_PREPARE=1,C3X_NATIVE_MAP_CANCEL=2};
struct {struct {int field_574[4]={};}GUI;} form,*p_main_screen_form=&form;
long long ticks=0;int sleeps=0,polls=0,cancels=0,remaining=0,reply=1;
void sleep_ms(DWORD ms){assert(ms==5);ticks+=ms;++sleeps;}
void QueryPerformanceCounter(LARGE_INTEGER* t){t->QuadPart=ticks;}
Sleeper get_proc(void*,char const*){return sleep_ms;}auto p_GetProcAddress=get_proc;
void log_custom_renderer_event(char const*,int){}
int native_map(int action,void*,void*,void*){if(action==C3X_NATIVE_MAP_CANCEL){++cancels;return 1;}++polls;return --remaining>0?4:reply;}
struct {unsigned custom_renderer_presented_frames=0;LARGE_INTEGER custom_renderer_qpc_frequency{1000};void* kernel32=nullptr;decltype(&native_map)custom_renderer_native_map=native_map;} state,*is=&state;
int execute(int resident_result){void* image=nullptr;int request=0,displayed=0;
'''+block+r'''
 return resident_result;
}
int main(){
 form.GUI.field_574[3]=1;remaining=8;
 assert(execute(4)==1&&sleeps==8&&polls==8&&!cancels);
 state.custom_renderer_presented_frames=1;auto before=sleeps;
 assert(execute(4)==4&&sleeps==before); // warm map never blocks
 state.custom_renderer_presented_frames=0;form.GUI.field_574[3]=0;
 assert(execute(4)==4&&sleeps==before); // never wait behind a black screen
 form.GUI.field_574[3]=1;remaining=2;reply=3;
 assert(execute(4)==3&&sleeps==before+2); // failure exits promptly
 remaining=100000;reply=1;ticks=0;
 assert(execute(4)==0&&ticks==60000&&cancels==1); // bounded cold-start failure
}
''')

    def test_menu_unloads_only_custom_scene_before_native_background(self):
        source=(ROOT/'injected_code.c').read_text()
        start=source.index('int __fastcall\npatch_Sprite_draw_main_menu_background')
        wrapper=source[start:source.index('\n}\n',start)+3].replace('this','self')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#define __fastcall
constexpr int __=0;
struct Sprite{};struct PCX_Image{};struct PCX_Color_Table{};
struct {struct {bool enable_custom_rendering=false;}current_config;void*custom_renderer_module=nullptr;}state,*is=&state;
int calls=0,unloads=0;Sprite sprite;PCX_Image canvas;PCX_Color_Table palette;
void unload_custom_renderer(){++unloads;state.custom_renderer_module=nullptr;}
int Sprite_draw(Sprite* s,int,PCX_Image* c,int x,int y,PCX_Color_Table* p){
 assert(s==&sprite&&c==&canvas&&x==4&&y==9&&p==&palette);++calls;
 if(state.current_config.enable_custom_rendering)assert(!state.custom_renderer_module);
 return 73;
}
'''+wrapper+r'''
int main(){
 state.custom_renderer_module=&sprite;
 assert(patch_Sprite_draw_main_menu_background(&sprite,0,&canvas,4,9,&palette)==73&&unloads==0&&calls==1);
 state.current_config.enable_custom_rendering=true;
 assert(patch_Sprite_draw_main_menu_background(&sprite,0,&canvas,4,9,&palette)==73&&unloads==1&&calls==2);
 assert(patch_Sprite_draw_main_menu_background(&sprite,0,&canvas,4,9,&palette)==73&&unloads==1&&calls==3);
}
''')
