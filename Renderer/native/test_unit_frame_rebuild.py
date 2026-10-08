"""Map unit frame sprites are rebuilt only when their animation changes.

Civ III's map animator ticks every animating unit (`FUN_004f08f0` calls
`FLC_Animation::tick`, 0x402620, at 0x4F0AA2, and for the unit each one refers
to at +0x1D0 at 0x4F0AF0). Each tick destroys and
re-creates the unit's JGL frame sprite. Under the game's compatibility layers
each JGL destroy/create costs a shimmed critical-section call, which filled
Civ III's thread on the busy 1498 AD save (performance review, October 7,
section 7). Custom rendering draws map unit bodies in 3D and reads only the
frame's FLC and size, so the rebuild runs only when the FLC changes, or for
the selected unit, whose frame the unit panel shows.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class UnitFrameRebuildTests(unittest.TestCase):
    def test_map_unit_tick_skips_unchanged_frames_only_with_custom_rendering(self):
        source = (ROOT / 'injected_code.c').read_text()
        patch = source.split('#ifdef FLC_Animation_tick\n', 1)[1].split('\n#endif', 1)[0]
        # Injected code is C; `this` is an ordinary parameter name there.
        patch = patch.replace('this', 'self')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#define __fastcall
struct Flic_Anim_Info {void* Flic_Anim_Data;};
struct Animation_Info {Flic_Anim_Info** Animations;};
struct Unit {int id;};
struct FLC_Frame_Image {Flic_Anim_Info* Flic_Info;};
struct AnimationSummary {int current_anim_type;};
struct FLC_Animation {AnimationSummary summary;FLC_Frame_Image Frame_1;Animation_Info* Animation_Info;Unit* Unit;};
struct Main_Screen_Form {Unit* Current_Unit;} screen;Main_Screen_Form* p_main_screen_form=&screen;
struct State {struct {bool enable_custom_rendering=false;} current_config;} state;State* is=&state;
int ticks=0;
void FLC_Animation_tick(FLC_Animation*,int edx,int direction,int frame){assert(edx==7&&direction==0&&frame==-1);++ticks;}
''' + patch + r'''
int main(){
 int data=0;Flic_Anim_Info run={&data},idle={&data},fidget={nullptr};
 Flic_Anim_Info* animations[4]={&idle,&run,&fidget,nullptr};Animation_Info info={animations};
 Unit unit={1},selected={2};FLC_Animation animation={{1},{&run},&info,&unit};
 // Config-off: vanilla runs every time, with unchanged arguments.
 patch_FLC_Animation_tick_map_unit(&animation,7,0,-1);patch_FLC_Animation_tick_map_unit(&animation,7,0,-1);
 assert(ticks==2);
 state.current_config.enable_custom_rendering=true;screen.Current_Unit=&selected;
 // Same FLC already decoded: no sprite rebuild.
 patch_FLC_Animation_tick_map_unit(&animation,7,0,-1);assert(ticks==2);
 // A changed animation rebuilds, so the FLC and size the renderer reads stay current.
 animation.summary.current_anim_type=0;patch_FLC_Animation_tick_map_unit(&animation,7,0,-1);assert(ticks==3);
 // The selected unit always ticks: the unit panel shows its frame.
 animation.summary.current_anim_type=1;screen.Current_Unit=&unit;
 patch_FLC_Animation_tick_map_unit(&animation,7,0,-1);assert(ticks==4);
 // Like vanilla, an animation without loaded data (or none) draws the default
 // one (index 1); a frame already holding it is current.
 screen.Current_Unit=&selected;animation.summary.current_anim_type=2;
 patch_FLC_Animation_tick_map_unit(&animation,7,0,-1);assert(ticks==4);
 animation.summary.current_anim_type=3;patch_FLC_Animation_tick_map_unit(&animation,7,0,-1);assert(ticks==4);
 animation.Frame_1.Flic_Info=&idle;patch_FLC_Animation_tick_map_unit(&animation,7,0,-1);assert(ticks==5);
 // Missing animation data falls back to vanilla.
 animation.Animation_Info=nullptr;
 patch_FLC_Animation_tick_map_unit(&animation,7,0,-1);assert(ticks==6);
}
''')

    def test_patch_table_targets_only_the_map_animator_unit_tick(self):
        rows = [line for line in (ROOT / 'civ_prog_objects.csv').read_text().splitlines() if 'FLC_Animation_tick' in line]
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0].startswith('define, 0x402620,'))
        for row, site in zip(rows[1:], ('0x4F0AA2', '0x4F0AF0')):
            self.assertTrue(row.startswith(f'repl call, {site},') and '"FLC_Animation_tick_map_unit"' in row)


if __name__ == '__main__':
    unittest.main()
