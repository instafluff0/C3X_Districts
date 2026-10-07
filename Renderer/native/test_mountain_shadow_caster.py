"""Mountains cast shadows on the ground again, and hills keep theirs.

Vertices of the joined relief grid (around any mountain) share the mountain
caster tag. Its coverage channel used to be the mountain footprint; it was
later given to hill support, and the caster kept reading it: hills cast, but
every mountain fragment away from a hill was discarded, so mountains cast no
shadow while hills beside them did.

The caster now keeps a fragment when either the hill-support cutoff passes or
the vertex rises above the ground beneath it (world.z minus world.w - 1, the
rise the visible mountain shader uses). The almost-flat collar still casts
nothing. This test compiles the actual vertex and pixel lines as C++; the old
rule casts no mountain, and a rise-only rule drops hill shadows.
"""
import re
import subprocess
import unittest
from pathlib import Path

from Renderer.native.native_cpp_test import run_cpp

SOURCE = Path(__file__).resolve().parents[2]


def caster(name='render_core'):
    return (SOURCE / 'Renderer/native' / name / 'source_caster.hlsl').read_text()


def program(rule='current'):
    text = caster()
    vertex = re.search(r'if\(i\.material>=42 && i\.material<=43\.001\)o\.boundary=([^;]+);', text)
    pixel = re.search(r' if\(!volcano_body\)clip\(([^;]+)\);return i\.depth;', text)
    rise = vertex[1] if vertex else 'i.material'
    cutoff = pixel[1] if pixel else '-1'
    if rule == 'hill-support-only':  # the rule that silenced mountains
        cutoff = 'smoothstep(.08,.72,i.coverage)-.45'
    elif rule == 'rise-only':  # a fix that silenced hills instead
        cutoff = 'i.boundary-.012'
    return r'''
#include <algorithm>
#include <cstdio>
struct float4 {float x,y,z,w;};
float max(float a,float b){return std::max(a,b);}
float smoothstep(float a,float b,float x){float t=std::clamp((x-a)/(b-a),0.f,1.f);return t*t*(3-2*t);}
struct Input {float material;float4 world;float coverage;float boundary;};
// Returns true where a relief-grid fragment enters the shadow field.
bool casts(Input i){
 i.boundary=i.material;
 if(i.material>=42 && i.material<=43.001)i.boundary=''' + rise + r''';
 return (''' + cutoff + r''')>=0;
}
int main(){
 // A mountain body away from hills: no hill support, well above its ground.
 for(float rise:{.05f,.2f,.6f,1.f})for(float ground:{0.f,.02f,.15f})
  if(!casts({42.9f,{3.f,4.f,ground+2.5f/112+rise,1+ground},0.f,0.f})){std::puts("MOUNTAIN_CASTS_NOTHING");return 3;}
 // A hill inside the same grid: no mountain rise, full hill support.
 for(float ground:{.05f,.15f,.25f})
  if(!casts({42.9f,{3.f,4.f,ground+2.5f/112,1+ground},.9f,0.f})){std::puts("HILL_CASTS_NOTHING");return 5;}
 // The flat collar that cross-fades into the ground stays out of the field.
 for(float ground:{0.f,.02f,.15f})
  if(casts({42.9f,{3.f,4.f,ground+2.5f/112+.004f,1+ground},.1f,0.f})){std::puts("COLLAR_CASTS");return 4;}
 std::puts("PASS relief caster: mountains and hills cast, the flat collar does not");
}
'''


class MountainShadowCasterTests(unittest.TestCase):
    def test_mountains_and_hills_cast_and_the_collar_does_not(self):
        run_cpp(program())

    def test_reading_hill_support_alone_casts_no_mountain(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            run_cpp(program('hill-support-only'))
        self.assertEqual(failure.exception.returncode, 3)

    def test_reading_rise_alone_casts_no_hill(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            run_cpp(program('rise-only'))
        self.assertEqual(failure.exception.returncode, 5)

    def test_generated_casters_carry_the_rule(self):
        for line in ('if(i.material>=42 && i.material<=43.001)o.boundary=i.world.z-max(0,i.world.w-1)-2.5/112;',
                     'if(!volcano_body)clip(max(smoothstep(.08,.72,i.coverage)-.45,i.boundary-.012));return i.depth;'):
            self.assertIn(line, caster())
            for name in ('city_fidelity', 'environment_refresh'):
                self.assertIn(line, caster(name), name)


if __name__ == '__main__':
    unittest.main()
