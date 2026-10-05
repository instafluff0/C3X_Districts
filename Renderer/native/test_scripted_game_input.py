"""Opt-in diagnostic input delegates when disabled and stays within its test scope."""
from pathlib import Path
import re
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ScriptedGameInputTests(unittest.TestCase):
    def test_diagnostic_presentation_counter_abi(self):
        script = (Path(__file__).resolve().parents[1] / 'tools/scripted_game_test.ps1').read_text()
        version, offset = re.search(r'WireVersion = (\d+), FrameOffset = (\d+)', script).groups()
        run_cpp('''
#include <cstddef>
#include "Renderer/native/helper_trial/scene_wire.h"
static_assert(c3x_helper_trial::wire_version == ''' + version + ''', "Diagnostic wire version");
static_assert(offsetof(c3x_helper_trial::Wire, visual_frames) == ''' + offset + ''', "Diagnostic counter offset");
static_assert(offsetof(c3x_helper_trial::Wire, presented_zoom_q16) == 232, "Presented zoom counter offset");
int main() {}
''')

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
#include <cstdint>
#define WINAPI
#define Main_Screen_Form_move_camera enabled
#define MAX_PATH 260
#define VK_RETURN 13
using DWORD=unsigned;using LPCSTR=char const*;using LPSTR=char*;
using Env=DWORD(*)(LPCSTR,LPSTR,DWORD);
struct Unit {struct {int X=42,Y=43;}Body;} unit;
int messages=0,combats=0,routes=0,script_messages=0,script_parameters=0,fixture_refusals=0;
bool combat_mode=false,route_mode=false,hud_mode=false,interaction_mode=false;
char const* fixture_message="NEWSCILEADER";DWORD fixture_message_length=12;
void show_map_specific_text(int x,int y,char const* text,bool pause){assert(x==42&&y==43&&text&&!pause);++messages;}
struct Main_Screen_Form {int field_2E194=0,camera_x=10,camera_y=20,Player_CivID=1;bool is_now_loading_game=false;Unit* Current_Unit=nullptr;
 struct {bool is_enabled=false;} GUI;} form;
auto p_main_screen_form=&form;
void run_custom_renderer_combat_test(Main_Screen_Form* p,int key){assert(p==&form&&(key==0x84||key==0x85));++combats;}
void run_custom_renderer_test_route(Main_Screen_Form* p,int edx){assert(p==&form&&edx==19);++routes;}
struct Race {char CountryName[40]="Japan";int ScientificLeadersCount=1;std::intptr_t ScientificLeaders=0;} races[2];
struct Leader {int RaceID=0;} leaders[32];
struct Bic {struct {void* Tiles=nullptr;} Map;Race* Races=races;int RacesCount=2;} bic;auto p_bic_data=&bic;
struct State {struct {bool enable_custom_rendering=false;} current_config;
 char custom_renderer_test_save[MAX_PATH]={};unsigned custom_renderer_test_step=0;
 unsigned custom_renderer_test_route_step=0;bool custom_renderer_test_route_adopted=false,custom_renderer_test_route_resolving=false;
 char const* load_file_path_override=nullptr;int suppress_intro_after_load_popup=0;
 bool custom_renderer_display_valid=false;void* kernel32=nullptr;} state;auto is=&state;
int env_calls=0,original_calls=0,moves=0;constexpr int __=0;
int patch_show_popup(void*,int,int,int){return 81;}DWORD env_length=9;
DWORD environment(LPCSTR key,LPSTR buffer,DWORD size){++env_calls;if(std::strcmp(key,"C3X_RENDERER_GAME_TEST_MODE")==0){if(interaction_mode){std::strcpy(buffer,"interaction");return 11;}if(combat_mode){std::strcpy(buffer,"combat");return 6;}if(route_mode){std::strcpy(buffer,"matched-route");return 13;}if(hud_mode){std::strcpy(buffer,"hud");return 3;}return 0;}
 if(std::strcmp(key,"C3X_RENDERER_GAME_TEST_MESSAGE")==0){assert(size==32);std::strcpy(buffer,fixture_message);return fixture_message_length;}
 assert(std::strcmp(key,"C3X_RENDERER_GAME_TEST_SAVE")==0&&size==MAX_PATH);
 std::strcpy(buffer,"input.SAV");return env_length;}
Env proc(void*,char const*){return environment;}auto p_GetProcAddress=proc;
void debug(char const* line){if(std::strstr(line,"fixture-source-mismatch"))++fixture_refusals;}auto p_OutputDebugStringA=debug;
int set_popup_str_param(int index,char* name,int a,int b){assert(index==0&&std::strcmp(name,"Aida Yasuki")==0&&a==-1&&b==-1);++script_parameters;return 0;}
void patch_Main_Screen_Form_show_map_message(Main_Screen_Form* p,int edx,int x,int y,char* key,bool pause){
 assert(p==&form&&edx==__&&x==42&&y==43&&std::strcmp(key,"NEWSCILEADER")==0&&!pause);++script_messages;}
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
 combat_mode=true;assert(checkpoint()==0);combat_mode=false;
 interaction_mode=true;assert(checkpoint()==81);interaction_mode=false;
 state.custom_renderer_test_step=0;state.current_config.enable_custom_rendering=false;env_calls=0;
 assert(command(&form,19,0x87)==73&&env_calls==0);
 assert(command(&form,19,0x86)==73&&env_calls==0&&messages==0);
 assert(command(&form,19,0x84)==73&&env_calls==0&&combats==0);
 state.current_config.enable_custom_rendering=true;
 assert(command(&form,19,0x83)==73&&env_calls==0);
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
 combat_mode=true;assert(command(&form,19,0x84)==73&&combats==0);
 env_length=9;combat_mode=false;assert(command(&form,19,0x84)==73&&combats==0);
 combat_mode=true;assert(command(&form,19,0x84)==1&&combats==1);
 assert(command(&form,19,0x85)==1&&combats==2);
 assert(command(&form,19,0x85,0)==73&&combats==2);
 state.current_config.enable_custom_rendering=false;
 assert(command(&form,19,0x84)==73&&combats==2);
 route_mode=true;combat_mode=false;
 assert(command(&form,19,0x87)==73&&routes==0);
 state.current_config.enable_custom_rendering=true;env_length=0;
 assert(command(&form,19,0x87)==73&&routes==0);
 env_length=9;
 assert(command(&form,19,0x87)==1&&routes==1&&moves==32);
 assert(command(&form,19,0x87,0)==73&&routes==1);
 assert(command(&form,19,0x86)==1&&routes==1);
 state.custom_renderer_test_route_step=7;state.custom_renderer_test_route_adopted=true;
 assert(checkpoint()==0&&state.custom_renderer_test_route_step==0&&!state.custom_renderer_test_route_adopted);
 // The official native script fixture has independent, exact opt-in gates.
 route_mode=false;hud_mode=true;char actual_name[]="Aida Yasuki";
 races[0].ScientificLeaders=reinterpret_cast<std::intptr_t>(actual_name);
 int old_messages=messages,old_moves=moves;Leader old_leader=leaders[1];
 assert(command(&form,19,0x86)==1&&script_messages==1&&script_parameters==1);
 assert(messages==old_messages&&moves==old_moves&&leaders[1].RaceID==old_leader.RaceID);
 auto refuse_fixture=[&](){int before=fixture_refusals;
  assert(command(&form,19,0x86)==1&&fixture_refusals==before+1&&script_messages==1&&script_parameters==1&&messages==old_messages);};
 form.Player_CivID=0;refuse_fixture();form.Player_CivID=32;refuse_fixture();form.Player_CivID=1;
 leaders[1].RaceID=-1;refuse_fixture();leaders[1].RaceID=2;refuse_fixture();leaders[1].RaceID=0;
 bic.Races=nullptr;refuse_fixture();bic.Races=races;
 races[0].ScientificLeadersCount=0;refuse_fixture();races[0].ScientificLeadersCount=1;
 races[0].ScientificLeaders=0;refuse_fixture();races[0].ScientificLeaders=reinterpret_cast<std::intptr_t>(actual_name);
 std::strcpy(races[0].CountryName,"France");refuse_fixture();std::strcpy(races[0].CountryName,"Japan");
 char wrong_name[]="Kiyosi Ito";races[0].ScientificLeaders=reinterpret_cast<std::intptr_t>(wrong_name);refuse_fixture();
 races[0].ScientificLeaders=reinterpret_cast<std::intptr_t>(actual_name);
 // Mismatched or absent message selection preserves the ordinary two messages.
 fixture_message="NEWLEADER";fixture_message_length=9;
 assert(command(&form,19,0x86)==1&&messages==old_messages+2&&script_messages==1);
 fixture_message="NEWSCILEADER";fixture_message_length=0;
 assert(command(&form,19,0x86)==1&&messages==old_messages+4&&script_messages==1);
 fixture_message_length=12;hud_mode=false;
 assert(command(&form,19,0x86)==1&&messages==old_messages+6&&script_messages==1);hud_mode=true;
 // No fixture or environment access when disabled, released or off owner form.
 int calls_before=env_calls;state.current_config.enable_custom_rendering=false;
 assert(command(&form,19,0x86)==73&&env_calls==calls_before&&script_messages==1);
 state.current_config.enable_custom_rendering=true;
 assert(command(&form,19,0x86,0)==73&&env_calls==calls_before&&script_messages==1);
 Main_Screen_Form other;assert(command(&other,19,0x86)==73&&env_calls==calls_before&&script_messages==1);
 env_length=0;assert(command(&form,19,0x86)==73&&script_messages==1);
 env_length=MAX_PATH;assert(command(&form,19,0x86)==73&&script_messages==1);env_length=9;
 form.is_now_loading_game=true;assert(command(&form,19,0x86)==1&&script_messages==1);form.is_now_loading_game=false;
 form.GUI.is_enabled=false;assert(command(&form,19,0x86)==1&&script_messages==1);form.GUI.is_enabled=true;
 state.custom_renderer_display_valid=false;assert(command(&form,19,0x86)==1&&script_messages==1);state.custom_renderer_display_valid=true;
 form.Current_Unit=nullptr;assert(command(&form,19,0x86)==1&&script_messages==1);form.Current_Unit=&unit;
 assert(command(&form,19,0x86)==1&&script_messages==2&&script_parameters==2&&messages==old_messages+6);
}
''')


if __name__ == '__main__':
    unittest.main()
