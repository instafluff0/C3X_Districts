"""Opt-in diagnostic input delegates when disabled and stays within its test scope."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ScriptedGameInputTests(unittest.TestCase):
    def test_guard_load_and_bounded_camera_commands(self):
        source = (Path(__file__).resolve().parents[2] / 'injected_code.c').read_text()
        block = source.split('patch_Main_Screen_Form_m82_handle_key_event (', 1)[1]
        block = block.split('{', 1)[1].split('char s[200]', 1)[0]
        block = block.replace('return;', 'return 1;')
        block = block.replace('(void *)(*p_GetProcAddress)', '(Env)(*p_GetProcAddress)')
        ready = source.split('patch_show_intro_after_load_popup (', 1)[1].split('{', 1)[1].split('if (! is->suppress_intro_after_load_popup)', 1)[0]
        ready = ready.replace('(void *)(*p_GetProcAddress)', '(Env)(*p_GetProcAddress)')
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <cstring>
#define WINAPI
#define Main_Screen_Form_move_camera enabled
#define MAX_PATH 260
#define VK_RETURN 13
using DWORD=unsigned;using LPCSTR=char const*;using LPSTR=char*;
using Env=DWORD(*)(LPCSTR,LPSTR,DWORD);
struct Unit {struct {int X=42,Y=43;}Body;} unit;
int messages=0;
void show_map_specific_text(int x,int y,char const* text,bool pause){assert(x==42&&y==43&&text&&!pause);++messages;}
struct Main_Screen_Form {int field_2E194=0,camera_x=10,camera_y=20;bool is_now_loading_game=false;Unit* Current_Unit=nullptr;
 struct {bool is_enabled=false;} GUI;} form;
auto p_main_screen_form=&form;
struct Bic {struct {void* Tiles=nullptr;} Map;} bic;auto p_bic_data=&bic;
struct State {struct {bool enable_custom_rendering=false;} current_config;
 char custom_renderer_test_save[MAX_PATH]={};unsigned custom_renderer_test_step=0;
 char const* load_file_path_override=nullptr;int suppress_intro_after_load_popup=0;
 bool custom_renderer_display_valid=false;void* kernel32=nullptr;} state;auto is=&state;
int env_calls=0,original_calls=0,moves=0;constexpr int __=0;
int patch_show_popup(void*,int,int,int){return 81;}DWORD env_length=9;
DWORD environment(LPCSTR key,LPSTR buffer,DWORD size){++env_calls;if(std::strcmp(key,"C3X_RENDERER_GAME_TEST_MODE")==0)return 0;
 assert(std::strcmp(key,"C3X_RENDERER_GAME_TEST_SAVE")==0&&size==MAX_PATH);
 std::strcpy(buffer,"input.SAV");return env_length;}
Env proc(void*,char const*){return environment;}auto p_GetProcAddress=proc;
void debug(char const*){}auto p_OutputDebugStringA=debug;
int Main_Screen_Form_handle_key_down(Main_Screen_Form* p,int edx,int character,int key){
 ++original_calls;assert(p==&form&&edx==19&&character==0&&key==VK_RETURN&&p->field_2E194==3);return 91;}
void patch_Main_Screen_Form_move_camera(Main_Screen_Form* p,int edx,int x,int y,int reason,bool bounds){
 assert(p==&form&&edx==19&&reason==1&&!bounds);++moves;p->camera_x=x;p->camera_y=y;}
int command(Main_Screen_Form* self,int edx,int virtual_key_code,int is_down=1){
''' + block.replace('this', 'self') + r'''
 return 73; // Continue the existing shared gameplay key hook unchanged.
}
int checkpoint(void* self=nullptr,int param_1=0,int param_2=0){
''' + ready.replace('this', 'self') + r'''
 return 17; // Continue the ordinary native load confirmation.
}
int main(){
 assert(checkpoint()==17&&env_calls==0);
 state.current_config.enable_custom_rendering=true;
 env_length=0;assert(checkpoint()==17&&state.custom_renderer_test_step==0);
 env_length=MAX_PATH;assert(checkpoint()==17&&state.custom_renderer_test_step==0);
 env_length=9;assert(checkpoint()==0&&state.custom_renderer_test_step==1);
 state.custom_renderer_test_step=0;state.current_config.enable_custom_rendering=false;env_calls=0;
 assert(command(&form,19,0x87)==73&&env_calls==0);
 assert(command(&form,19,0x86)==73&&env_calls==0&&messages==0);
 state.current_config.enable_custom_rendering=true;
 assert(command(&form,19,0x85)==73&&env_calls==0);
 assert(command(&form,19,0x87,0)==73&&env_calls==0);
 env_length=0;assert(command(&form,19,0x87)==73&&state.custom_renderer_test_step==0);
 env_length=MAX_PATH;assert(command(&form,19,0x87)==73&&state.custom_renderer_test_step==0);
 env_length=9;assert(command(&form,19,0x87)==1&&original_calls==0);
 state.custom_renderer_test_step=1; // The verified native post-load checkpoint.
 assert(command(&form,19,0x87)==1&&moves==0);
 form.GUI.is_enabled=true;bic.Map.Tiles=&bic;state.custom_renderer_display_valid=true;
 form.is_now_loading_game=true;assert(command(&form,19,0x87)==1&&moves==0);
 form.is_now_loading_game=false;
 for(int n=0;n<40;++n)assert(command(&form,19,0x87)==1);
 assert(moves==32&&state.custom_renderer_test_step==33&&form.camera_x==10&&form.camera_y==20);
 assert(command(&form,19,0x86)==1&&messages==0);
 form.Current_Unit=&unit;assert(command(&form,19,0x86)==1&&messages==2&&moves==32);
 form.is_now_loading_game=true;assert(command(&form,19,0x86)==1&&messages==2);
 form.is_now_loading_game=false;env_length=0;assert(command(&form,19,0x86)==73&&messages==2);
}
''')


if __name__ == '__main__':
    unittest.main()
