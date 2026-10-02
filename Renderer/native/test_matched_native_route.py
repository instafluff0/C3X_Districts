"""Execute extracted diagnostic route parsing, readiness, ordering and exact-camera acknowledgements."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


ROOT = Path(__file__).resolve().parents[2]


def production_functions():
    source = (ROOT / "injected_code.c").read_text()
    start = source.index("int\nparse_custom_renderer_test_route (")
    end = source.index("void __fastcall\npatch_Main_Screen_Form_m82_handle_key_event (", start)
    return source[start:end]


class MatchedNativeRoute(unittest.TestCase):
    def test_actual_parser_whole_list_and_bounds(self):
        functions = production_functions().split("void\nlog_custom_renderer_test_route_resolved",1)[0]
        run_cpp("#include <cstddef>\n" + functions + r'''
#include <stdio.h>
#include <string.h>
int main(void) {
    int x=777,y=777,width=777; unsigned count=777;
    #define CHECK(r,s,mw,mh,result) do { \
        int actual=parse_custom_renderer_test_route(r,s,mw,mh,&x,&y,&width,&count); \
        if(actual != result) { fprintf(stderr,"line %d actual %d expected %d\n",__LINE__,actual,result); return 1; } \
    } while(0)
    CHECK("5280,1866,128;5344,1866,160;5280,1866,384",2,130,130,1);
    if(x!=5344||y!=1866||width!=160||count!=3)return 2;
    CHECK("0,0,128",1,130,130,1);
    CHECK("8319,4159,384",1,130,130,1);
    CHECK("8320,0,128",1,130,130,-4);
    CHECK("0,4160,128",1,130,130,-4);
    CHECK("0,0,127",1,130,130,-5);
    CHECK("0,0,128;1,1,999",1,130,130,-5);
    CHECK("0,0,128;99999,1,128",1,130,130,-4);
    CHECK("0,0,128;",1,130,130,-2);
    CHECK("0,0,128;;1,1,128",1,130,130,-2);
    CHECK("0,0,128 extra",1,130,130,-2);
    CHECK("0,0",1,130,130,-2);
    CHECK("-1,0,128",1,130,130,-2);
    CHECK("+1,0,128",1,130,130,-2);
    CHECK(" 1,0,128",1,130,130,-2);
    CHECK("1.0,0,128",1,130,130,-2);
    CHECK("2147483648,0,128",1,130,130,-2);
    CHECK("999999999999999999999999,0,128",1,130,130,-2);
    CHECK("0,0,128",0,130,130,-6);
    CHECK("0,0,128",2,130,130,-6);
    CHECK("",1,130,130,-1);
    CHECK(NULL,1,130,130,-1);
    CHECK("0,0,128",1,0,130,-7);
    CHECK("0,0,128",1,130,-1,-7);
    CHECK("0,0,128",1,2147483647,130,-7);
    char bounded[2049]; memset(bounded,'1',2048);bounded[2048]=0;
    CHECK(bounded,1,130,130,-1);
    char many[512]="0,0,128";
    for(unsigned i=1;i<32;i++)strcat(many,";0,0,128");
    CHECK(many,32,130,130,1);
    strcat(many,";0,0,128");CHECK(many,1,130,130,-3);
    x=y=width=777;count=777;
    CHECK("0,0,128;0,0,129",1,130,130,-5);
    if(x!=777||y!=777||width!=777||count!=777)return 3;
    puts("31 executable parser cases passed");return 0;
}
''')

    def test_actual_runner_and_camera_acknowledgements(self):
        functions = production_functions().replace('(void *)(*p_GetProcAddress)', '(Env)(*p_GetProcAddress)')
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#define WINAPI
#define Main_Screen_Form_move_camera enabled
#define ARRAY_LEN(a) (sizeof(a)/sizeof(a[0]))
using DWORD=unsigned;using LPCSTR=char const*;using LPSTR=char*;
using Env=DWORD(*)(LPCSTR,LPSTR,DWORD);
struct LARGE_INTEGER {long long QuadPart=0;};
long long clock_ticks=1000;bool clock_ok=true;
int QueryPerformanceCounter(LARGE_INTEGER* value){value->QuadPart=++clock_ticks;return clock_ok;}
struct Main_Screen_Form {bool is_now_loading_game=false;struct {bool is_enabled=true;} GUI;int camera_x=42,camera_y=43;} form,other;
auto p_main_screen_form=&form;
struct Bic {struct {int Width=130,Height=130;void* Tiles=(void*)1;}Map;bool is_zoomed_out=false;}bic;auto p_bic_data=&bic;
struct State {struct {bool enable_custom_rendering=true;}current_config;
 char custom_renderer_test_save[260]="input.SAV";unsigned custom_renderer_test_step=1,custom_renderer_test_route_step=0;
 int custom_renderer_test_route_x=0,custom_renderer_test_route_y=0,custom_renderer_test_route_width=0;
 bool custom_renderer_test_route_adopted=false,custom_renderer_test_route_resolving=false,custom_renderer_display_valid=true;
 struct {int camera_x=42,camera_y=43;}custom_renderer_display_view;
 int custom_renderer_zoom_target_width=128;LARGE_INTEGER custom_renderer_qpc_frequency{1000};void* kernel32=nullptr;
}state;auto is=&state;
char route[2049]="640,1000,160;700,1000,128";DWORD environment_length=0;int env_calls=0,moves=0,zooms=0;
bool zoom_enabled=true,queue_accepts=true;int native_override_x=-1,native_override_y=-1;
std::vector<std::string> logs,events;
DWORD environment(LPCSTR key,LPSTR buffer,DWORD size){++env_calls;assert(std::strcmp(key,"C3X_RENDERER_GAME_TEST_ROUTE")==0&&size==2048);
 unsigned length=std::strlen(route);if(length<size)std::strcpy(buffer,route);return environment_length?environment_length:length;}
Env proc(void*,char const*){return environment;}auto p_GetProcAddress=proc;
void debug(char const* line){logs.emplace_back(line);if(std::strstr(line,"stage=scripted-route-accepted"))events.emplace_back("accept");}
auto p_OutputDebugStringA=debug;
bool custom_renderer_zoom_enabled(){return zoom_enabled;}
bool advance_custom_renderer_zoom(Main_Screen_Form* self,int delta,bool wrap){assert(self==&form&&!wrap);++zooms;events.emplace_back("zoom");
 int levels[]={128,160,192,224,256,320,384};int current=0;for(int i=0;i<7;i++)if(levels[i]==state.custom_renderer_zoom_target_width)current=i;
 if(queue_accepts)state.custom_renderer_zoom_target_width=levels[current+delta];return true;}
void log_custom_renderer_test_route_resolved(int,int);
void patch_Main_Screen_Form_move_camera(Main_Screen_Form* self,int edx,int x,int y,int reason,bool bounds){assert(self==&form&&edx==19&&reason==1&&!bounds);++moves;events.emplace_back("move");
 self->camera_x=native_override_x<0?x:native_override_x;self->camera_y=native_override_y<0?y:native_override_y;
 log_custom_renderer_test_route_resolved(self->camera_x,self->camera_y);}
''' + functions + r'''
int main(){
 state.current_config.enable_custom_rendering=false;run_custom_renderer_test_route(&form,19);assert(env_calls==0&&moves==0);
 state.current_config.enable_custom_rendering=true;run_custom_renderer_test_route(&other,19);assert(env_calls==0);
 state.custom_renderer_test_save[0]=0;run_custom_renderer_test_route(&form,19);assert(env_calls==0);std::strcpy(state.custom_renderer_test_save,"input.SAV");
 auto refuse=[&](int reason){auto step=state.custom_renderer_test_step;auto moved=moves,zoomed=zooms;run_custom_renderer_test_route(&form,19);
 assert(step==state.custom_renderer_test_step&&moved==moves&&zoomed==zooms);assert(logs.back().find("reason="+std::to_string(reason))!=std::string::npos);};
 std::strcpy(route,"640,1000,160;700,1000,129");refuse(-5);std::strcpy(route,"640,1000,160;700,1000,128");
 environment_length=2048;refuse(-1);environment_length=0;
 form.is_now_loading_game=true;refuse(-8);form.is_now_loading_game=false;
 form.GUI.is_enabled=false;refuse(-8);form.GUI.is_enabled=true;
 bic.Map.Tiles=nullptr;refuse(-8);bic.Map.Tiles=(void*)1;
 state.custom_renderer_display_valid=false;refuse(-8);state.custom_renderer_display_valid=true;
 zoom_enabled=false;refuse(-8);zoom_enabled=true;
 bic.is_zoomed_out=true;refuse(-8);bic.is_zoomed_out=false;
 state.custom_renderer_qpc_frequency.QuadPart=0;refuse(-9);state.custom_renderer_qpc_frequency.QuadPart=1000;
 clock_ok=false;refuse(-9);clock_ok=true;
 logs.clear();events.clear();run_custom_renderer_test_route(&form,19);
 assert(state.custom_renderer_test_step==2&&moves==1&&state.custom_renderer_zoom_target_width==160);
 assert(events.size()==3&&events[0]=="accept"&&events[1]=="zoom"&&events[2]=="move");
 assert(logs[0].find("step=1 count=2 qpc=")!=std::string::npos&&logs[0].find("requested=640,1000")!=std::string::npos);
 assert(logs.back().find("native=640,1000")!=std::string::npos);
 auto n=logs.size();log_custom_renderer_test_route_adopted(640,1000,false);assert(logs.size()==n);
 state.custom_renderer_display_view.camera_x=640;state.custom_renderer_display_view.camera_y=1000;
 state.custom_renderer_display_valid=false;log_custom_renderer_test_route_adopted(640,1000,false);assert(logs.size()==n);
 state.custom_renderer_display_valid=true;log_custom_renderer_test_route_adopted(640,999,false);assert(logs.size()==n);
 log_custom_renderer_test_route_adopted(640,1000,false);assert(logs.size()==n+1&&state.custom_renderer_test_route_adopted);
 assert(logs.back().find("displayed=640,1000 valid=1 camera_already_adopted=0")!=std::string::npos);
 log_custom_renderer_test_route_adopted(640,1000,false);assert(logs.size()==n+1);
 native_override_x=701;native_override_y=1001;run_custom_renderer_test_route(&form,19);
 assert(logs.back().find("native=701,1001")!=std::string::npos&&state.custom_renderer_test_step==3);
 refuse(-6);
 state.custom_renderer_test_step=1;state.custom_renderer_display_view.camera_x=640;state.custom_renderer_display_view.camera_y=1000;
 native_override_x=native_override_y=-1;run_custom_renderer_test_route(&form,19);
 assert(logs[logs.size()-2].find("camera_already_adopted=1")!=std::string::npos);
 state.custom_renderer_test_step=2;queue_accepts=false;auto moved=moves;run_custom_renderer_test_route(&form,19);
 assert(moves==moved&&state.custom_renderer_test_route_step==0&&logs.back().find("reason=-10")!=std::string::npos);
}
''')


if __name__ == "__main__":unittest.main()

if __name__ == "__main__":
    unittest.main()
