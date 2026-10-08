"""Execute the injected volcano-effect suppression after Civ III's native spawn."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class VolcanoEffectSuppressionTests(unittest.TestCase):
    def test_custom_rendering_hides_native_volcano_animation_and_keeps_its_state(self):
        # The patch's ordinary path: the unchanged native spawn, then the
        # renderer-only suppression (the custom-animation branch above it runs
        # only with custom rendering off).
        source = (ROOT / 'injected_code.c').read_text()
        body = source.split('patch_Tile_spawn_animated_effect (Tile * this, int edx, enum AnimatedEffect effect, '
                            'int tile_x, int tile_y, bool randomize_start_frame, enum direction dummy_dir)\n{', 1)[1]
        tail = '\tTile_spawn_animated_effect (this, __, effect, tile_x, tile_y, randomize_start_frame, dummy_dir);\n' + \
            body.split('\treturn;\n\t}\n\tTile_spawn_animated_effect (this, __, effect, tile_x, tile_y, '
                       'randomize_start_frame, dummy_dir);\n', 1)[1].split('\n}\n', 1)[0]
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#include <initializer_list>
enum AnimatedEffect {AE_Disorder=0,AE_Smolder=9,AE_Eruption=10,AE_Other=11};
enum direction {DIR_SW=1};
struct FLC_Animation {int Last;};
struct Tile_Animated_Effect {int V[3];FLC_Animation flc_animation;};
struct Tile {struct {Tile_Animated_Effect* active_tile_effect;} Body;};
struct State {struct {bool enable_custom_rendering;} current_config;} state,*is=&state;
constexpr int __=0;
Tile_Animated_Effect spawned;int calls=0;AnimatedEffect seen;int seen_x,seen_y;bool seen_random;direction seen_dir;
void Tile_spawn_animated_effect(Tile* t,int,AnimatedEffect e,int x,int y,bool r,direction d){
 ++calls;seen=e;seen_x=x;seen_y=y;seen_random=r;seen_dir=d;
 spawned=Tile_Animated_Effect{{x,y,e},{1}};t->Body.active_tile_effect=&spawned;
}
void tail(Tile* t,AnimatedEffect effect,int tile_x,int tile_y,bool randomize_start_frame,direction dummy_dir){
 Tile* self=t;
#define this self
''' + tail + r'''
#undef this
}
int main(){
 for(bool on:{false,true})for(AnimatedEffect e:{AE_Smolder,AE_Eruption,AE_Other}){
  state.current_config.enable_custom_rendering=on;Tile t{{nullptr}};calls=0;
  tail(&t,e,7,9,true,DIR_SW);
  // Civ III always spawns the effect with unchanged arguments; it keeps
  // ticking and carries the volcano state the renderer captures.
  assert(calls==1 && seen==e && seen_x==7 && seen_y==9 && seen_random && seen_dir==DIR_SW);
  assert(t.Body.active_tile_effect==&spawned && spawned.V[2]==e);
  // Only custom rendering hides the native smoke/lava animation, only for volcanoes.
  bool hidden=on && (e==AE_Smolder || e==AE_Eruption);
  assert(spawned.flc_animation.Last==(hidden?0:1));
 }
}
''')


if __name__ == '__main__':
    unittest.main()
