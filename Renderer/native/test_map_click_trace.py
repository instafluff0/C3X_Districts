"""The temporary click trace observes existing hooks without driving input."""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class MapClickTraceTests(unittest.TestCase):
    def test_trace_preserves_native_dispatch_and_config_off(self):
        source = (ROOT / "injected_code.c").read_text()
        trace = source[source.index("// Temporary, event-bounded diagnosis"):
                       source.index("void __fastcall \npatch_Main_GUI_handle_click_in_status_panel")]
        hover = source[source.index("void __fastcall\npatch_Main_Screen_Form_process_mouse_hover"):
                       source.index("bool __fastcall\npatch_Unit_can_disembark_anything")]
        code = (trace + hover).replace("this", "screen")
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#define __fastcall
constexpr int __=0;
struct Unit{struct{int ID=7;}Body;};
struct City{struct{int ID=9;}Body;};
struct Timer{void* timer_id=nullptr;};
struct Main_Screen_Form{
 int mouse_x=12,mouse_y=34,Mode_Action=0,field_4ED0=0,field_4E80[6]={};
 int field_4DC0[25]={},camera_x=123,camera_y=-456;
 Unit* Current_Unit=nullptr;City* Selected_City=nullptr;
 Timer timer_1,timer_2;struct{unsigned Status1=0,Status2=1;}Base_Data;
};
struct{struct{struct{int Status2=0;}Data;}Base;}city_form,*p_city_form=&city_form;
struct State{
 struct{bool enable_custom_rendering=true;}current_config;
 int sb_activated_by_button=1,custom_renderer_zoom_tile_width=192,custom_renderer_zoom_native_tile_width=128;
 long long custom_renderer_zoom_translate_x_fp=-456,custom_renderer_zoom_translate_y_fp=789;
}state,*is=&state;
struct LARGE_INTEGER{long long QuadPart;};
int queries=0,clock_reads=0,native_clicks=0,native_hovers=0,hud_updates=0;
int pick_result=0;bool change_action=false;City target_city;
std::vector<std::string> logs;
void log_line(char* line){logs.emplace_back(line);}
auto output=log_line;auto p_OutputDebugStringA=&output;
void QueryPerformanceCounter(LARGE_INTEGER* out){out->QuadPart=123456789LL;++clock_reads;}
int patch_Main_Screen_Form_get_tile_coords_under_mouse(Main_Screen_Form*,int,int x,int y,int* tx,int* ty){
 assert(x==12&&y==34);++queries;*tx=6;*ty=8;return pick_result;
}
City* city_at(int x,int y){assert(x==6&&y==8);return &target_city;}
void Main_Screen_Form_handle_left_click_on_map_1(Main_Screen_Form* screen,int,int x,int y){
 assert(x==12&&y==34&&state.sb_activated_by_button==2);++native_clicks;
 screen->Selected_City=&target_city;
}
void Main_Screen_Form_process_mouse_hover(Main_Screen_Form* screen,int,int x,int y){
 assert(x==12&&y==34);++native_hovers;if(change_action)screen->Mode_Action=1;
}
void update_combat_odds_hud_for_hover(Main_Screen_Form*,int x,int y){assert(x==12&&y==34);++hud_updates;}
void combat_odds_hud_request_redraw_if_layout_stale(Main_Screen_Form*){++hud_updates;}
''' + code + r'''
int main(){
 Main_Screen_Form screen;Unit unit;
 for(bool selected:{false,true}){
  screen.Current_Unit=selected?&unit:nullptr;screen.Selected_City=nullptr;
  state.sb_activated_by_button=1;logs.clear();
  patch_Main_Screen_Form_handle_left_click_on_map_1(&screen,0,12,34);
  assert(logs.size()==2&&state.sb_activated_by_button==0);
  assert(logs[0].find("phase=click-enter")!=std::string::npos);
  assert(logs[0].find("selected_city=-1")!=std::string::npos);
  assert(logs[1].find("selected_city=9")!=std::string::npos);
  assert(logs[0].find(selected?"current_unit=7":"current_unit=-1")!=std::string::npos);
 }
 assert(native_clicks==2&&queries==4&&clock_reads==4);
 logs.clear();patch_Main_Screen_Form_process_mouse_hover(&screen,0,12,34);
 assert(logs.empty()&&queries==4); // ordinary hover produces no trace work
 change_action=true;patch_Main_Screen_Form_process_mouse_hover(&screen,0,12,34);
 assert(logs.size()==1&&logs[0].find("phase=hover-action")!=std::string::npos);
 assert(native_hovers==2&&hud_updates==4&&screen.Mode_Action==1);
 pick_result=1;trace_custom_renderer_map_click(&screen,"goto-command",12,34);
 assert(logs.back().find("pick=1 tile=6,8 city=-1")!=std::string::npos);
 int saved_queries=queries,saved_reads=clock_reads;logs.clear();
 state.current_config.enable_custom_rendering=false;state.sb_activated_by_button=1;
 patch_Main_Screen_Form_handle_left_click_on_map_1(&screen,0,12,34);
 assert(native_clicks==3&&logs.empty()&&queries==saved_queries&&clock_reads==saved_reads);
}
''')
