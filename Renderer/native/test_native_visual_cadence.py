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

    def test_native_timer_does_not_drive_resident_visuals(self):
        source=(ROOT/'injected_code.c').read_text()
        body=source[source.index('void __stdcall\npatch_on_timer_0x9F6500'):source.index('void __fastcall\npatch_Units_Image_Data_load_animated_effect')]
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#define __stdcall
const int C3X_NATIVE_VISUAL_POLICY=116;
char animator_bytes[64]={};struct {struct {char* field_18E4=animator_bytes;}animator;} screen;auto p_main_screen_form=&screen;
const unsigned C3X_RENDERER_DIRTY_SCENE=1;
int native_calls=0,legacy_redraws=0,effects=0;bool resident=true,reenter=false;
int policy(int,void*,void*,void const*,void const*,unsigned color){assert(color==2);return resident;}
struct State{bool custom_renderer_redraw_pending=false;unsigned custom_renderer_dirty_flags=0;bool custom_renderer_timer_running=false;struct{bool enable_custom_animations=true,enable_custom_rendering=true;}current_config;
 int (*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned)=policy;};
State state;State* is=&state;unsigned debug=0;unsigned* p_debug_mode_bits=&debug;
void patch_on_timer_0x9F6500();
void on_timer_0x9F6500(){++native_calls;if(reenter)patch_on_timer_0x9F6500();}
void custom_renderer_scheduler_tick(){++legacy_redraws;}
void clear_active_custom_tile_animation_effects(){++effects;}
void tile_animation_scheduler_tick(){++effects;}
'''+body+r'''
int main(){
 for(int i=0;i<100;++i)patch_on_timer_0x9F6500();
 assert(native_calls==100&&legacy_redraws==0&&effects==0);
 reenter=true;patch_on_timer_0x9F6500();assert(native_calls==101&&!state.custom_renderer_timer_running);
 state.custom_renderer_redraw_pending=true;state.custom_renderer_dirty_flags=C3X_RENDERER_DIRTY_SCENE;
 patch_on_timer_0x9F6500();assert(animator_bytes[10]==1);--native_calls;
 state.custom_renderer_redraw_pending=false;animator_bytes[10]=0;
 resident=false;patch_on_timer_0x9F6500();assert(legacy_redraws==1&&native_calls==102);
 state.current_config.enable_custom_rendering=false;patch_on_timer_0x9F6500();assert(effects==1&&native_calls==103);
}
''')
        self.assertNotIn('Animator_update (',body)
        self.assertNotIn('Timer_reset_and_activate (',body)

    def test_independent_cadence_stops_and_has_no_catchup_queue(self):
        run_cpp(r'''
#include <cassert>
#include <atomic>
#include "Renderer/native/visual_cadence.h"
int main(){
 using namespace std::chrono;
 c3x_renderer::VisualCadence cadence;std::atomic<unsigned> calls{0};
 steady_clock::time_point last_end;bool first=true;
 cadence.enable([&]{
  auto begin=steady_clock::now();
  if(!first)assert(begin-last_end>=milliseconds(9));
  first=false;++calls;std::this_thread::sleep_for(milliseconds(60));last_end=steady_clock::now();
 });
 std::this_thread::sleep_for(milliseconds(260));cadence.stop();
 unsigned finished=calls;assert(finished>=2&&finished<=4);
 std::this_thread::sleep_for(milliseconds(80));assert(calls==finished);
 cadence.enable([&]{++calls;});std::this_thread::sleep_for(milliseconds(100));
 cadence.disable();cadence.stop();assert(calls>finished);
}
''')
