"""Continuous and user-time day/night clock: C3X advances it between turns,
Renderer64 adopts a new time only when its previous lighting change has
refined and the camera is still."""
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
import unittest

STUBS = r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define __ 0
enum {DNCM_OFF=0,DNCM_TIMER,DNCM_USER_TIME,DNCM_EVERY_TURN,DNCM_SPECIFIED,DNCM_CONTINUOUS};
enum {SCM_OFF=0,SCM_TIMER,SCM_USER_SEASON,SCM_EVERY_TURN,SCM_ON_DAY_NIGHT_HOUR,SCM_SPECIFIED};
struct LARGE_INTEGER {long long QuadPart;};
struct SYSTEMTIME {unsigned short wYear,wMonth,wDayOfWeek,wDay,wHour,wMinute,wSecond,wMilliseconds;};
typedef SYSTEMTIME* LPSYSTEMTIME;
long long qpc_now=0;SYSTEMTIME local_time={};
void QueryPerformanceCounter(LARGE_INTEGER* value){value->QuadPart=qpc_now;}
void QueryPerformanceFrequency(LARGE_INTEGER* value){value->QuadPart=1000;}
void GetLocalTime(SYSTEMTIME* value){*value=local_time;}
int clamp(int lo,int hi,int v){return v<lo?lo:v>hi?hi:v;}
int not_below(int lo,int v){return v<lo?lo:v;}
struct {
 struct {int day_night_cycle_mode=DNCM_CONTINUOUS,day_night_game_minutes_per_real_minute=20,pinned_hour_for_day_night_cycle=0,
  fixed_hours_per_turn_for_day_night_cycle=1,elapsed_minutes_per_day_night_hour_transition=3,
  seasonal_cycle_mode=SCM_OFF,transition_season_on_day_night_hour=0;bool enable_custom_rendering=true;} current_config;
 bool day_night_cycle_unstarted=false,custom_renderer_redraw_pending=false;
 int current_day_night_cycle=12,day_night_minute=0,current_seasonal_cycle=0;
 unsigned custom_renderer_dirty_flags=0;
 LARGE_INTEGER last_day_night_cycle_update_time={0};
} state;auto is=&state;
int player_bits=1;int* p_player_bits=&player_bits;
struct {bool is_now_loading_game=false;} main_form;auto p_main_screen_form=&main_form;
int season_calls=0,season_hits=0;
int calculate_current_seasonal_cycle(bool hit){++season_calls;if(hit)++season_hits;return hit?(state.current_seasonal_cycle+1)%4:state.current_seasonal_cycle;}
'''


def injected_function(name, next_marker):
    source = (ROOT / 'injected_code.c').read_text()
    start = source.index(name)
    return source[start:source.index(next_marker, start)]


class DayNightClockTests(unittest.TestCase):
    def test_c3x_clock_advances_between_turns(self):
        body = injected_function('void\nadvance_day_night_clock ()', 'void __stdcall\npatch_on_timer_0x9F6500')
        run_cpp(STUBS + body + r'''
int clock(){return state.current_day_night_cycle*60+state.day_night_minute;}
void at(long long t){qpc_now=t;state.custom_renderer_redraw_pending=false;state.custom_renderer_dirty_flags=0;advance_day_night_clock();}
int main(){
 bool captured=false;auto capture=[&]{bool c=state.custom_renderer_redraw_pending;if(c)assert(state.custom_renderer_dirty_flags==C3X_RENDERER_DIRTY_SCENE);return c;};
 // Continuous at 20x: one game minute per 3 real seconds (3000 ticks). The
 // first tick anchors the clock and sends it to Renderer64 with one capture;
 // later minutes keep the remainder and need no capture (Renderer64 advances
 // them from the frame's clock); a new hour captures again.
 at(5000);assert(clock()==720 && capture() && state.last_day_night_cycle_update_time.QuadPart==5000);
 at(7999);assert(clock()==720 && !capture());
 at(8000);assert(clock()==721 && !capture() && state.last_day_night_cycle_update_time.QuadPart==8000);
 at(12500);assert(clock()==722 && state.last_day_night_cycle_update_time.QuadPart==11000);
 at(14000);assert(clock()==723 && !capture());
 state.day_night_minute=59;at(17000);assert(clock()==780 && capture());
 // Seasons on-day-night-hour switch the moment the clock reaches the hour, once.
 state.current_config.seasonal_cycle_mode=SCM_ON_DAY_NIGHT_HOUR;state.current_config.transition_season_on_day_night_hour=0;
 state.current_day_night_cycle=23;state.day_night_minute=58;state.last_day_night_cycle_update_time.QuadPart=20000;
 at(23000);assert(clock()==1439 && state.current_seasonal_cycle==0 && !capture());
 at(26000);assert(clock()==0 && state.current_seasonal_cycle==1 && season_hits==1 && capture());
 at(29000);assert(clock()==1 && state.current_seasonal_cycle==1 && season_hits==1 && !capture());
 // A long gap that jumps over the transition hour still switches once.
 state.current_day_night_cycle=23;state.day_night_minute=50;state.last_day_night_cycle_update_time.QuadPart=30000;
 at(30000+30*3000);assert(clock()==20 && state.current_seasonal_cycle==2 && season_hits==2);
 // Days of absence are bounded to one step; the clock still moves.
 state.last_day_night_cycle_update_time.QuadPart=0;at(1000);at(1000+1000LL*86400*10);assert(clock()==19 && season_hits==3);
 // User-time follows the system clock's hour and minute and records when that
 // minute began (QPC 1000/s here); only an unanchored clock or a new hour captures.
 state.current_config.day_night_cycle_mode=DNCM_USER_TIME;state.current_config.transition_season_on_day_night_hour=12;
 state.current_day_night_cycle=9;state.day_night_minute=14;state.last_day_night_cycle_update_time.QuadPart=0;
 local_time.wHour=9;local_time.wMinute=14;local_time.wSecond=30;local_time.wMilliseconds=250;
 at(100000);assert(clock()==554 && capture() && state.last_day_night_cycle_update_time.QuadPart==100000-30250);
 local_time.wMinute=15;local_time.wSecond=0;local_time.wMilliseconds=40;
 at(129800);assert(clock()==555 && !capture() && state.last_day_night_cycle_update_time.QuadPart==129760);
 at(130000);assert(!capture() && state.last_day_night_cycle_update_time.QuadPart==129760);
 int hits=season_hits;
 local_time.wHour=14;local_time.wMinute=0;at(140000);assert(clock()==840 && season_hits==hits && capture()); // a clock jump resynchronizes
 state.current_day_night_cycle=11;state.day_night_minute=59;local_time.wHour=12;local_time.wMinute=0;
 at(150000);assert(clock()==720 && season_hits==hits+1 && capture());
 // Nothing advances while unstarted, loading, outside a game, or in other modes.
 local_time.wMinute=5;
 state.day_night_cycle_unstarted=true;at(0);assert(clock()==720);state.day_night_cycle_unstarted=false;
 main_form.is_now_loading_game=true;at(0);assert(clock()==720);main_form.is_now_loading_game=false;
 player_bits=0;at(0);assert(clock()==720);player_bits=1;
 int idle_modes[]={DNCM_OFF,DNCM_TIMER,DNCM_EVERY_TURN,DNCM_SPECIFIED};
 for(int mode:idle_modes){state.current_config.day_night_cycle_mode=mode;at(1LL<<40);assert(clock()==720 && !state.custom_renderer_redraw_pending);}
 (void)captured;
 std::puts("PASS C3X day/night clock: continuous minutes with remainder, season at the hour, user-time minutes, idle modes");
}
''')

    def test_turn_end_keeps_the_running_clock(self):
        body = injected_function('int\ncalculate_current_day_night_cycle_hour ()', 'void __fastcall\npatch_Map_Renderer_load_images')
        run_cpp(STUBS + body + r'''
int main(){
 // The turn end must not reset a continuous clock to noon or step it.
 state.current_day_night_cycle=17;state.day_night_minute=34;
 assert(calculate_current_day_night_cycle_hour()==17 && state.day_night_minute==34);
 state.day_night_cycle_unstarted=true;assert(calculate_current_day_night_cycle_hour()==12 && !state.day_night_cycle_unstarted);
 // User-time samples the hour and the minute together under the custom renderer.
 state.current_config.day_night_cycle_mode=DNCM_USER_TIME;local_time.wHour=13;local_time.wMinute=7;local_time.wSecond=2;local_time.wMilliseconds=5;
 qpc_now=50000;assert(calculate_current_day_night_cycle_hour()==13 && state.day_night_minute==7);
 assert(state.last_day_night_cycle_update_time.QuadPart==50000-2005); // when that minute began
 state.current_config.enable_custom_rendering=false;state.day_night_minute=0;local_time.wMinute=9;
 assert(calculate_current_day_night_cycle_hour()==13 && state.day_night_minute==0);
 std::puts("PASS turn end keeps the continuous clock; user-time samples hour and minute together");
}
''')

    def test_frames_within_an_hour_are_identical(self):
        # A retried capture must converge: at 1440x minutes pass every few
        # timer ticks, and a per-minute frame clock made Civ III redraw every
        # tick (2026-10-23). Frames anchor at the hour's start instead.
        source = (ROOT / 'injected_code.c').read_text()
        start = source.index('\tif (((is->current_config.day_night_cycle_mode == DNCM_CONTINUOUS) || (is->current_config.day_night_cycle_mode == DNCM_USER_TIME)) &&\n\t    ! is->day_night_cycle_unstarted) {')
        block = source[start:source.index('\n\t}\n', start) + 4]
        run_cpp(STUBS + r'''
c3x_renderer_frame_v1 build(){c3x_renderer_frame_v1 frame={};
''' + block + r'''
 return frame;}
int main(){
 // Continuous 20x (QPC 1000/s, 3000 ticks per game minute): 12:07 began at 50000, so 12:00 began at 29000.
 state.current_day_night_cycle=12;state.day_night_minute=7;state.last_day_night_cycle_update_time.QuadPart=50000;
 auto a=build();assert(a.minute==0 && a.clock_rate==20 && a.clock_qpc==29000);
 state.day_night_minute=9;state.last_day_night_cycle_update_time.QuadPart=56000;
 auto b=build();assert(!std::memcmp(&a,&b,sizeof(a)));
 // User-time: real minutes; an unanchored clock sends no anchor.
 state.current_config.day_night_cycle_mode=DNCM_USER_TIME;state.day_night_minute=7;state.last_day_night_cycle_update_time.QuadPart=500000;
 auto c=build();assert(c.clock_rate==1 && c.clock_qpc==500000-7*60000);
 state.last_day_night_cycle_update_time.QuadPart=0;assert(build().clock_qpc==0);
 // Other modes and an unstarted clock send none.
 state.current_config.day_night_cycle_mode=DNCM_EVERY_TURN;assert(build().clock_rate==0);
 state.current_config.day_night_cycle_mode=DNCM_CONTINUOUS;state.day_night_cycle_unstarted=true;assert(build().clock_rate==0);
 std::puts("PASS frames within a day/night hour are identical (anchored at the hour's start)");
}
''')

    def test_continuous_mode_needs_the_custom_renderer(self):
        source = (ROOT / 'injected_code.c').read_text()
        start = source.index('\t// The continuous day/night cycle is computed by the custom renderer only')
        body = source[start:source.index('\tif (cfg->warn_about_unrecognized_names', start)]
        run_cpp(STUBS + r'''
#include <cstdio>
struct PopupForm;struct PopupVtable {void (*set_text_key_and_flags)(PopupForm*,int,char const*,char const*,int,int,int,int);};
struct PopupForm {PopupVtable* vtable;};
int popups=0;
void set_text(PopupForm*,int,char const*,char const*,int,int,int,int){}
PopupVtable popup_vtable={set_text};PopupForm popup_form={&popup_vtable};
PopupForm* get_popup_form(){return &popup_form;}
void PopupForm_add_text(PopupForm*,int,char const*,bool){}
void patch_show_popup(PopupForm*,int,int,int){++popups;}
struct Config {int day_night_cycle_mode,seasonal_cycle_mode;bool enable_custom_rendering;};
struct {char const* mod_script_path="";} script;
void validate(Config* cfg){auto is=&script;
''' + body + r'''
}
int main(){
 Config on{DNCM_CONTINUOUS,SCM_ON_DAY_NIGHT_HOUR,true};validate(&on);
 assert(on.day_night_cycle_mode==DNCM_CONTINUOUS && on.seasonal_cycle_mode==SCM_ON_DAY_NIGHT_HOUR && popups==0);
 // Without Renderer64 continuous is off, and seasons tied to its hour follow.
 Config off{DNCM_CONTINUOUS,SCM_ON_DAY_NIGHT_HOUR,false};validate(&off);
 assert(off.day_night_cycle_mode==DNCM_OFF && off.seasonal_cycle_mode==SCM_OFF && popups==2);
 Config user{DNCM_USER_TIME,SCM_OFF,false};validate(&user);assert(user.day_night_cycle_mode==DNCM_USER_TIME && popups==2);
 std::puts("PASS continuous day/night requires the custom renderer");
}
''')

    def test_renderer_adopts_a_new_time_only_when_settled(self):
        run_cpp(r'''
#include "Renderer/native/environment_runtime.h"
#include <cassert>
#include <cstdio>
using c3x_renderer::adopt_clock_time;
int main(){
 // The first frame and a season change apply at once, even mid-refinement, mid-scroll or before a step is due.
 assert(adopt_clock_time(-1.f,0,17.5f,0,true,false,false));
 assert(adopt_clock_time(17.5f,0,0.f,1,true,false,false));
 // A new minute waits for the previous change to refine, a still camera and a due step.
 assert(!adopt_clock_time(17.5f,0,17.55f,0,true,true,true));
 assert(!adopt_clock_time(17.5f,0,17.55f,0,false,false,true));
 assert(!adopt_clock_time(17.5f,0,17.55f,0,false,true,false));
 assert(adopt_clock_time(17.5f,0,17.55f,0,false,true,true));
 assert(!adopt_clock_time(17.5f,0,17.5f,0,false,true,true));
 assert(c3x_renderer::clock_step_seconds==10.0);
 std::puts("PASS Renderer64 adopts a new day/night time only after refinement, with a still camera");
}
''')

    def test_renderer_advances_whole_minutes_from_the_frame_clock(self):
        run_cpp(r'''
#include "Renderer/native/environment_runtime.h"
#include <cassert>
#include <cstdio>
using c3x_renderer::clock_hour;
int main(){
 // Other modes: the exact hour and minute, no advance.
 assert(clock_hour(17,30,0,0,999999,1000)==17.5f);
 // 20x from QPC 1000 (1000 ticks/s): a game minute per 3 s, whole minutes only.
 assert(clock_hour(17,30,20,1000,1000+2999,1000)==17.5f);
 assert(clock_hour(17,30,20,1000,1000+3000,1000)==17.5f+1.f/60.f);
 assert(clock_hour(17,30,20,1000,1000+30000,1000)==17.5f+10.f/60.f);
 // At most an hour past the frame (C3X captures each new hour); wraps at midnight.
 assert(clock_hour(17,30,20,1000,1000+1000LL*3600,1000)==18.5f);
 assert(clock_hour(23,50,20,1000,1000+30000,1000)==0.f);
 // An unanchored or future anchor does not advance.
 assert(clock_hour(9,5,1,0,500000,1000)==9.f+5.f/60.f && clock_hour(9,5,1,9000,8000,1000)==9.f+5.f/60.f);
 std::puts("PASS Renderer64 advances whole game minutes from the frame clock");
}
''')

    def test_lighting_change_refines_in_slices(self):
        run_cpp(r'''
#include "Renderer/sandbox/static_raster_state.h"
#include <cassert>
#include <cstdio>
using c3x_renderer::render_core::lighting_refine_budget;
int main(){
 double region=2400.*1400.;
 // A lighting change spends at most a sixteenth of the region per frame.
 assert(lighting_refine_budget(32e6,region,true)==region/16);
 assert(lighting_refine_budget(1e5,region,true)==1e5);
 // Other refinement keeps its CPU-steered budget; no budget stays none.
 assert(lighting_refine_budget(32e6,region,false)==32e6);
 assert(lighting_refine_budget(-1,region,true)==-1 && lighting_refine_budget(0,region,true)==0);
 std::puts("PASS a day/night lighting change refines behind its preview in sixteen slices");
}
''')
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        self.assertIn('lighting_refine_budget(refinement_budget(lane),', fresh)

    def test_frame_and_bridge_carry_the_minute(self):
        api = (ROOT / 'Renderer/native/c3x_renderer_api.h').read_text()
        codec = (ROOT / 'Renderer/native/input_recording/codec.h').read_text()
        renderer = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        self.assertIn('c3x_renderer_i32 minute;\n    c3x_renderer_i32 clock_rate;\n    c3x_renderer_i64 clock_qpc;', api)
        self.assertIn('io(v.minute);io(v.clock_rate);io(v.clock_qpc);', codec)
        # Both frame builders send the clock; the C3X timer captures only for a new anchor, hour or season.
        injected = (ROOT / 'injected_code.c').read_text()
        self.assertEqual(injected.count('frame.clock_qpc = (is->last_day_night_cycle_update_time.QuadPart == 0) ? 0 :'), 2)
        # Every lighting consumer reads one adopted time: the bridge's per-frame
        # upload uses frame_hour, units use the fresh pipeline's visual_hour.
        self.assertNotIn('evaluate_environment(\n            static_cast<float>(frame.hour)', renderer)
        self.assertNotIn('frame_settings.hour = static_cast<float>(frame.hour);', renderer)
        self.assertIn('frame_hour(frame), frame_season(frame)', renderer)
        self.assertIn('/30.f:visual_hour;', fresh)
        self.assertIn('renderer.lighting_hour=hour;', fresh)
        self.assertIn('sandbox_fresh.lighting_pending()', (ROOT / 'Renderer/sandbox/resident_scene.cpp').read_text())


if __name__ == '__main__':
    unittest.main()
