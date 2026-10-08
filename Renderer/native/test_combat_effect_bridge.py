"""Execute the injected combat-effect bridge against a mock Civ III.

The bridge reports two presentation facts through the existing unit-state
entry (`c3x_renderer_unit_state`), with no new patch-table symbols:

- `patch_Units_Image_Data_load_animated_effect`: after Civ III loads a bombard
  hit/miss effect (AE_Hit..AE_WaterMiss), its effect record holds the target
  tile; the source is the bombarding unit. Only when the renderer accepts is
  the low byte of FLC_Animation.Last cleared (pixels hidden; ticks, sound and
  the other bytes stay native).
- `watch_custom_renderer_effect_anims`: an Animator effect-list FLC that Civ III
  first kept hidden and then revealed is a bombing run's bomb release; one
  visible from the start (SAM shoot-down, SDI interception) is a standalone
  effect. Each is reported once with its tile and direction and hidden when
  the renderer draws it. Nothing native stays visible because a source unit is
  unknown: the renderer uses its default munition.
"""
import subprocess
import unittest
from pathlib import Path

from Renderer.native.test_effect_sampler import run_host

ROOT = Path(__file__).resolve().parents[2]


def function(source, signature):
    body = source.split(signature, 1)[1]
    return signature + body.split("\n}\n", 1)[0] + "\n}\n"


class CombatEffectBridgeTests(unittest.TestCase):
    def test_impacts_and_bomb_release_hide_native_pixels_only_when_owned(self):
        source = (ROOT / "injected_code.c").read_text()
        notify = function(source, "int\nnotify_custom_renderer_combat (Unit * source, unsigned int kind, int tile_x, int tile_y, int code)\n{")
        watch = function(source, "void\nwatch_custom_renderer_effect_anims (Animator * animator)\n{")
        hook = source.split("\t\tUnits_Image_Data_load_animated_effect (this, __, anim, effect_id);\n\t\t// A bombard hit", 1)[1]
        hook = "\t\t// A bombard hit" + hook.split("\t\treturn;\n\t}", 1)[0]
        program = r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cstdio>
#include <cstring>
#define __ 0
enum AnimatedEffect {AE_Hit=3,AE_Hit2=4,AE_Hit3=5,AE_Hit5=6,AE_Miss=7,AE_WaterMiss=8,AE_Smolder=9};
// 32-bit Civ III layout: the effect record's FLC sits 3 ints after its start.
struct AnimationSummary {int vtable,direction,queued_anim_type,tile_x,tile_y;};
struct FLC_Animation {int vtable;AnimationSummary summary;int Last;};
struct Unit {struct {int ID,UnitTypeID,CivID;} Body;};
#include <cstdint>
// Civ III stores the effect vector bounds as 32-bit ints; pointer-sized here.
struct Animator {Unit* Units2[4];int Units2_Count;std::intptr_t field_18E4[22];};
struct Map {int width;};struct BIC {Map Map;} bic,*p_bic_data=&bic;
struct LargeInteger {long long QuadPart;};
typedef LargeInteger LARGE_INTEGER;
struct State {struct {bool enable_custom_rendering;} current_config;
    c3x_renderer_unit_state_fn custom_renderer_unit_state;int custom_renderer_viewer_civ_id;
    long long custom_renderer_map_epoch,custom_renderer_viewer_epoch;LargeInteger custom_renderer_qpc_frequency;
    long long (*custom_renderer_visual_clock)();Unit* bombarding_unit;FLC_Animation* custom_renderer_bomb_anim;
    FLC_Animation* custom_renderer_drawn_anim;FLC_Animation* custom_renderer_declined_anim;} state,*is=&state;
bool Map_in_range(Map*,int,int x,int y){return x>=0&&y>=0&&x<100&&y<100;}
bool custom_renderer_tile_visible_at(int,int){return true;}
bool QueryPerformanceCounter(LargeInteger* v){v->QuadPart=777;return true;}
long long clock_now(){return 5000;}
static c3x_renderer_unit_state_v1 facts[8];static int fact_count=0,answer=C3X_RENDERER_RESULT_OK;
int renderer(c3x_renderer_unit_state_v1 const* f){facts[fact_count++]=*f;return answer;}
static int failures=0;
#define CHECK(x) do{if(!(x)){std::printf("FAIL line %d: %s\n",__LINE__,#x);++failures;}}while(0)
''' + notify + watch + r'''
struct Record {int V[3];FLC_Animation flc;};
void load_hook(FLC_Animation* anim,int effect_id){
''' + hook + r'''
}
int main(){
    state.current_config.enable_custom_rendering=true;state.custom_renderer_unit_state=renderer;
    state.custom_renderer_viewer_civ_id=1;state.custom_renderer_map_epoch=3;state.custom_renderer_viewer_epoch=4;
    state.custom_renderer_qpc_frequency.QuadPart=1000;state.custom_renderer_visual_clock=clock_now;
    Unit gun{{42,7,2}},bomber{{43,8,2}};
    // Impact accepted: only Last's low byte clears; the fact names source, tile, effect.
    state.bombarding_unit=&gun;
    Record hit{{12,14,AE_Hit3},{}};hit.flc.Last=0x00ABCD01;
    load_hook(&hit.flc,AE_Hit3);
    CHECK(fact_count==1&&facts[0].kind==C3X_RENDERER_UNIT_STATE_IMPACT&&facts[0].unit_id==42&&
          facts[0].tile_x==12&&facts[0].tile_y==14&&facts[0].action==AE_Hit3&&facts[0].visible==1&&
          facts[0].map_epoch==3&&facts[0].viewer_epoch==4&&facts[0].presentation_time_ticks==5000&&
          facts[0].struct_size==sizeof(c3x_renderer_unit_state_v1));
    CHECK(hit.flc.Last==0x00ABCD00);
    // Declined, unrelated and config-off loads keep native pixels.
    answer=C3X_RENDERER_RESULT_SUPERSEDED;Record miss{{12,16,AE_WaterMiss},{}};miss.flc.Last=1;
    load_hook(&miss.flc,AE_WaterMiss);CHECK(fact_count==2&&miss.flc.Last==1);
    answer=C3X_RENDERER_RESULT_OK;Record smoke{{1,1,AE_Smolder},{}};smoke.flc.Last=1;
    load_hook(&smoke.flc,AE_Smolder);CHECK(fact_count==2&&smoke.flc.Last==1);
    Record stray{{1,1,AE_Hit},{}};stray.flc.Last=1;
    load_hook(&stray.flc,AE_Hit2);CHECK(fact_count==2&&stray.flc.Last==1);
    // An unknown source (city defenses, other strike paths) is still drawn: no 2D remains.
    state.bombarding_unit=nullptr;Record orphan{{3,5,AE_Hit},{}};orphan.flc.Last=1;
    load_hook(&orphan.flc,AE_Hit);CHECK(fact_count==3&&facts[2].unit_id==-1&&facts[2].tile_x==3&&orphan.flc.Last==0);
    state.current_config.enable_custom_rendering=false;state.bombarding_unit=&gun;
    Record off{{1,1,AE_Hit},{}};off.flc.Last=1;load_hook(&off.flc,AE_Hit);CHECK(fact_count==3&&off.flc.Last==1);
    state.current_config.enable_custom_rendering=true;

    // Bomb release: hidden, then revealed once; reported with tile and heading.
    Animator a{};a.Units2[0]=&bomber;a.Units2_Count=1;state.bombarding_unit=nullptr;
    FLC_Animation bomb{};bomb.Last=0;bomb.summary.tile_x=20;bomb.summary.tile_y=22;bomb.summary.direction=5;
    FLC_Animation sam{};sam.Last=1;sam.summary.tile_x=30;sam.summary.tile_y=32;   // visible from the start: standalone
    FLC_Animation* list[2]={&sam,&bomb};
    a.field_18E4[3]=reinterpret_cast<std::intptr_t>(list);a.field_18E4[4]=reinterpret_cast<std::intptr_t>(list+2);
    fact_count=0;
    watch_custom_renderer_effect_anims(&a);
    CHECK(fact_count==1&&facts[0].kind==C3X_RENDERER_UNIT_STATE_STANDALONE_EFFECT&&facts[0].tile_x==30&&
          facts[0].tile_y==32&&sam.Last==0&&bomb.Last==0);
    bomb.Last=1;watch_custom_renderer_effect_anims(&a);
    CHECK(fact_count==2&&facts[1].kind==C3X_RENDERER_UNIT_STATE_BOMB_RELEASE&&facts[1].unit_id==43&&
          facts[1].tile_x==20&&facts[1].tile_y==22&&facts[1].action==5&&bomb.Last==0);
    watch_custom_renderer_effect_anims(&a);CHECK(fact_count==2);
    // Declined (effects unavailable): the native FLC stays and is not reported every frame.
    answer=C3X_RENDERER_RESULT_SUPERSEDED;bomb.Last=0;watch_custom_renderer_effect_anims(&a);bomb.Last=1;
    watch_custom_renderer_effect_anims(&a);watch_custom_renderer_effect_anims(&a);CHECK(fact_count==3&&bomb.Last==1);
    // An emptied effect list forgets the decline.
    a.field_18E4[4]=a.field_18E4[3];watch_custom_renderer_effect_anims(&a);
    CHECK(state.custom_renderer_declined_anim==nullptr&&state.custom_renderer_drawn_anim==nullptr);
    std::printf(failures?"FAILED %d\n":"PASS combat effect bridge\n",failures);
    return failures!=0;
}
'''
        try:
            output = run_host(program, "")
        except subprocess.CalledProcessError as failure:
            self.fail(failure.stdout)
        self.assertIn("PASS combat effect bridge", output)


if __name__ == "__main__":
    unittest.main()
