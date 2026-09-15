"""Execute the production timer/advancement split without launching Civ III."""
import unittest
import csv
from pathlib import Path
from Renderer.native.native_cpp_test import run_cpp
from Renderer.tools.audit_native_visual_cadence import audit
ROOT=Path(__file__).resolve().parents[2]


class NativeVisualCadenceTests(unittest.TestCase):
    def test_authorized_gog_symbols_enable_the_native_branch(self):
        expected={
            'p_main_animation_timer':('define',0x009F6500),
            'Timer_reset_and_activate':('define',0x006205D0),
            'Units_Image_Data_advance_animations':('inlead',0x00405FC0),
            'p_native_timer_inhibited':('define',0x0072C2C4),
            'p_native_game_ending':('define',0x00CC37BC),
            'Advisor_GUI_open':('inlead',0x0049D070),
        }
        found=set()
        for row in csv.reader((ROOT/'civ_prog_objects.csv').read_text().replace('\t',' ').splitlines(),skipinitialspace=True):
            row=[item.strip() for item in row]
            if len(row)!=6 or row[4] not in expected:continue
            self.assertNotIn(row[4],found)
            found.add(row[4])
            self.assertEqual((row[0],int(row[1],0)),expected[row[4]])
            other_builds=[0x4A3AF0,0x49D100] if row[4]=='Advisor_GUI_open' else [0,0]
            self.assertEqual([int(row[2],0),int(row[3],0)],other_builds)
        self.assertEqual(found,set(expected))

    def test_gog_bytes_and_four_argument_abi(self):
        original=ROOT/'Renderer/native/build/unit-audit-original.exe'
        if not original.exists():self.skipTest('local GOG executable required for byte audit')
        self.assertEqual(audit(original)['status'],'pass')

    def test_production_cadence_preserves_advancement_guards_and_lifecycle(self):
        source=(ROOT/'injected_code.c').read_text()
        body=source[source.index('// The intermediate native visual call'):source.index('void __fastcall\npatch_Units_Image_Data_load_animated_effect')]
        advisor=source[source.index('void __fastcall\npatch_Advisor_GUI_open'):source.index('void __fastcall\npatch_Main_Screen_Form_open_quick_build_chooser')]
        body=advisor+body
        unload=source.split('unload_custom_renderer ()\n{',1)[1].split('\tis->custom_renderer_frame_active = false;',1)[0]
        unload=unload.split('\tis->custom_renderer_native_timer_due.QuadPart = 0;',1)[0]+'\tis->custom_renderer_native_timer_due.QuadPart = 0;'
        # Compile-only fixture for the real TCC/native layouts, using the same
        # authorized table definitions as the enabled injected build.
        definitions={}
        needed={'p_main_screen_form','p_player_bits','p_debug_mode_bits','Animator_update','on_timer_0x9F6500',
                'p_main_animation_timer','Timer_reset_and_activate','Units_Image_Data_advance_animations',
                'p_native_timer_inhibited','p_native_game_ending','Advisor_GUI_open'}
        for row in csv.reader((ROOT/'civ_prog_objects.csv').read_text().replace('\t',' ').splitlines(),skipinitialspace=True):
            row=[item.strip() for item in row]
            if len(row)==6 and row[4] in needed:definitions[row[4]]=(row[5],row[1])
        self.assertEqual(set(definitions),needed)
        typed='#include "stdio.h"\n#include "C3X.h"\n#define __ 0\n'
        typed+='\n'.join(f'#define {name} (({kind}){address})' for name,(kind,address) in definitions.items())
        typed+='''
extern struct injected_state * is;
extern void (WINAPI ** p_OutputDebugStringA) (char *);
#define GetFocus is->GetFocus
bool is_online_game ();
void custom_renderer_scheduler_tick ();
void set_custom_renderer_native_probe (JGL_Image *);
void tile_animation_scheduler_tick ();
void clear_active_custom_tile_animation_effects ();
'''+body+'\nvoid unload_cadence () {\n'+unload+'\n}\n'
        def run_contract(program):
            out=ROOT/'Renderer/native/build/native-visual-cadence/contract.cpp'
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(program)
            out.with_name('cadence_types.c').write_text(typed)
            run_cpp(program)
        run_contract(r'''
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#define __fastcall
#define __stdcall
#define __ 0
#define p_main_animation_timer timer_ptr
#define Timer_reset_and_activate reset_timer
#define Units_Image_Data_advance_animations advance
#define p_native_timer_inhibited inhibited_ptr
#define p_native_game_ending ending_ptr
void set_custom_renderer_native_probe(void*){}
struct Advisor_GUI {};enum AdvisorKind {AK_DOMESTIC,AK_TRADE};
void Advisor_GUI_open(Advisor_GUI*,int,AdvisorKind);
struct LARGE_INTEGER {long long QuadPart=0;};
struct Unit {struct {int ID=1;struct {struct {int current_anim_type=13;} summary;} Animation;} Body;};
struct Animator {int field_18E4[22]{};int field_1AE0=0,field_1AE4=0;Unit* Units[1024]{};int Units_Count=0,Units2_Count=0;};
struct Screen {Animator animator;bool is_now_loading_game=false;Unit* Current_Unit=nullptr;int Mode_Action=0;} screen;
auto p_main_screen_form=&screen;
struct Timer {void* callback_fn_2=nullptr;unsigned* timer_id=reinterpret_cast<unsigned*>(1);
 void* callback_fn=reinterpret_cast<void*>(2);void* callback_param=reinterpret_cast<void*>(1);int duration=66;};
Timer timer;auto timer_ptr=&timer;
int inhibited=0,ending=0,players=1;auto inhibited_ptr=&inhibited;auto ending_ptr=&ending;auto p_player_bits=&players;
struct State {
 struct {bool enable_custom_rendering=true,enable_custom_animations=false;} current_config;
 int custom_renderer_init_state=1,saved_tile_count=-1;bool custom_renderer_modal=false,paused_for_popup=false,custom_renderer_draw_in_progress=false;
 bool custom_renderer_fast_timer=false,custom_renderer_visual_only=false,custom_renderer_timer_running=false;
 bool custom_renderer_redraw_pending=false;unsigned custom_renderer_requested_frames=0,custom_renderer_presented_frames=0;
 unsigned custom_renderer_visible_animation_count=0;
 LARGE_INTEGER custom_renderer_qpc_frequency{1000000},custom_renderer_native_timer_due;
} state;auto is=&state;
constexpr int IS_OK=1,AT_FORTRESS=11,AT_ROAD=13,AT_PLANT=18;
long long now=1000000;bool focus=true,online=false,fail_fast_timer=false,reenter=false,qpc_ok=true;
unsigned native_calls=0,visual_draws=0,advance_calls=0,effect_cursor=0;double native_seconds=0;
void* focus_fn(){return focus?reinterpret_cast<void*>(1):nullptr;}auto GetFocus=focus_fn;
bool is_online_game(){return online;}
bool QueryPerformanceCounter(LARGE_INTEGER* out){out->QuadPart=now;return qpc_ok;}
void debug(char const*){}auto p_OutputDebugStringA=debug;
int debug_bits=0;auto p_debug_mode_bits=&debug_bits;
void clear_active_custom_tile_animation_effects(){}void tile_animation_scheduler_tick(){}
void custom_renderer_scheduler_tick(){
 if(state.custom_renderer_visible_animation_count){*(bool*)(screen.animator.field_18E4+10)=true;++state.custom_renderer_requested_frames;}
}
void reset_timer(Timer* t,int,void* callback,void* param,int duration,int resolution){
 assert(callback==reinterpret_cast<void*>(2) && param==reinterpret_cast<void*>(1));
 assert(resolution==(duration==33?5:66));t->duration=duration;
 t->timer_id=fail_fast_timer && duration==33?nullptr:reinterpret_cast<unsigned*>(1);
}
struct Units_Image_Data {} data;
void advance(Units_Image_Data*,int,float elapsed,Unit** units,int count,void* effects){
 assert(units==screen.animator.Units && count==screen.animator.Units_Count && effects==&data);
 ++advance_calls;++effect_cursor;native_seconds+=elapsed;
}
void patch_on_timer_0x9F6500();
void patch_Units_Image_Data_advance_animations(Units_Image_Data*,int,float,Unit**,int,void*);
void Animator_update(Animator* a){
 // Audited order: native camera/canvas setup, advancement, draw/composite,
 // then commit native elapsed-time accumulator. Extra refresh must preserve it.
 long long prior=(static_cast<long long>(a->field_1AE4)<<32)|static_cast<unsigned>(a->field_1AE0);
 float elapsed=float(now-prior)/1000000;
 patch_Units_Image_Data_advance_animations(&data,0,elapsed,a->Units,a->Units_Count,&data);
 if(state.custom_renderer_visual_only){++visual_draws;assert(*(bool*)(a->field_18E4+0xd));}
 if(reenter){reenter=false;unsigned calls=native_calls;patch_on_timer_0x9F6500();assert(calls==native_calls);}
 a->field_1AE0=int(now);a->field_1AE4=int(now>>32);
}
void on_timer_0x9F6500(){
 ++native_calls;if(!inhibited && !ending && !online)Animator_update(&screen.animator);
}
''' + body.replace('this','self') + '\nvoid unload_cadence(){\n' + unload + '\n}\n' + r'''
void Advisor_GUI_open(Advisor_GUI* self,int edx,AdvisorKind kind){
 assert(self && edx==17 && state.custom_renderer_modal);
 assert(!custom_renderer_has_visual_work());
 now+=66000;unsigned before=visual_draws;patch_on_timer_0x9F6500();
 assert(timer.duration==66 && visual_draws==before);
 if(kind==AK_DOMESTIC)patch_Advisor_GUI_open(self,edx,AK_TRADE);
 assert(state.custom_renderer_modal);
}
int main(){
 Unit worker;screen.animator.Units[0]=&worker;screen.animator.Units_Count=1;
 screen.animator.field_1AE0=int(now);
 for(int i=0;i<61;++i){now=1000000+i*33000;patch_on_timer_0x9F6500();}
 assert(timer.duration==33 && native_calls==31 && visual_draws==30 && advance_calls==31 && effect_cursor==31);
 assert(native_seconds>1.979 && native_seconds<1.981);
 assert(!state.custom_renderer_visual_only && !state.custom_renderer_timer_running);
 assert(!*(bool*)(screen.animator.field_18E4+0xd));
 // Advisor construction, page switches and nested dialogs suspend extra work.
 Advisor_GUI advisor;
 patch_Advisor_GUI_open(&advisor,17,AK_DOMESTIC);assert(!state.custom_renderer_modal);
 state.custom_renderer_modal=true;patch_Advisor_GUI_open(&advisor,17,AK_TRADE);assert(state.custom_renderer_modal);
 state.custom_renderer_modal=false;
 now+=66000;patch_on_timer_0x9F6500();assert(timer.duration==33);
 // Ordinary native updates between timer callbacks remain authoritative.
 now+=10000;Animator_update(&screen.animator);unsigned before=advance_calls;
 now+=23000;reenter=true;patch_on_timer_0x9F6500();assert(advance_calls==before && !reenter);
 now+=33000;patch_on_timer_0x9F6500();assert(advance_calls==before+1);
 // Each native suspension condition restores cadence and suppresses extra work.
 auto tick=[&](){now+=66000;patch_on_timer_0x9F6500();};
 auto stopped=[&](){tick();assert(timer.duration==66 && !state.custom_renderer_fast_timer);};
 auto resumed=[&](){tick();assert(timer.duration==33 && state.custom_renderer_fast_timer);};
 inhibited=1;before=advance_calls;stopped();assert(advance_calls==before);inhibited=0;resumed();
 ending=1;before=advance_calls;stopped();assert(advance_calls==before);ending=0;resumed();
 online=true;stopped();online=false;resumed();
 qpc_ok=false;stopped();qpc_ok=true;resumed();
 timer.callback_fn_2=reinterpret_cast<void*>(1);stopped();timer.callback_fn_2=nullptr;resumed();
 state.custom_renderer_modal=true;stopped();state.custom_renderer_modal=false;resumed();
 state.paused_for_popup=true;stopped();state.paused_for_popup=false;resumed();
 focus=false;stopped();focus=true;resumed();
 screen.is_now_loading_game=true;stopped();screen.is_now_loading_game=false;resumed();
 players=0;stopped();players=1;resumed();
 screen.Mode_Action=1;stopped();screen.Mode_Action=0x7f00;resumed();screen.Mode_Action=0;
 screen.animator.Units2_Count=1;stopped();screen.animator.Units2_Count=0;resumed();
 state.custom_renderer_draw_in_progress=true;stopped();state.custom_renderer_draw_in_progress=false;resumed();
 *(bool*)(screen.animator.field_18E4+0xb)=true;stopped();*(bool*)(screen.animator.field_18E4+0xb)=false;resumed();
 worker.Body.Animation.summary.current_anim_type=1;stopped(); // Unselected idle: no extra timer work.
 screen.Current_Unit=&worker;resumed();screen.Current_Unit=nullptr;stopped();
 state.custom_renderer_visible_animation_count=1;stopped();state.custom_renderer_visible_animation_count=0;stopped(); // Map-only cadence remains native.
 worker.Body.Animation.summary.current_anim_type=13;fail_fast_timer=true;stopped();fail_fast_timer=false;resumed();
 state.current_config.enable_custom_rendering=false;stopped();state.current_config.enable_custom_rendering=true;resumed();
 // A stopped native timer cannot be resurrected by a late queued callback.
 timer.timer_id=nullptr;tick();assert(!state.custom_renderer_fast_timer && !timer.timer_id);
 // A blocked UI never causes a burst of catch-up native callbacks.
 timer.timer_id=reinterpret_cast<unsigned*>(1);timer.duration=66;now+=5000000;
 before=native_calls;tick();assert(native_calls==before+1);
 unload_cadence();assert(timer.duration==66 && !state.custom_renderer_fast_timer && !state.custom_renderer_native_timer_due.QuadPart);
 std::printf("native cadence: 61 visual opportunities / 31 native advances; lifecycle guards pass\n");
}
''')

if __name__=='__main__':unittest.main()
