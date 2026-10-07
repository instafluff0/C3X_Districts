"""Bridge piers and walls below the waterline stay underwater.

Route bridge models continue below their base: the medieval arch about a
quarter of its height, the industrial and modern piers about half. A bridge
rests on its lower bank, the river's waterline. The river used to write a
depth about 27 units nearer than the ground, which hid that underwater part;
once it stopped writing depth (so banks no longer cover bridges, farms or
routes), the walls showed through the shallow carved channel and the bridges
looked tall and off-centre.

VSSharedFeature now carries a bridge vertex's height above its base as
q6_world.w = 2 + h / 128 (other features keep 1), and PSIntegratedFeature clips
bridge fragments below the base. This test compiles the actual rigid_point and
both shader lines as C++: parts below the base clip, parts above stay, and
non-bridge features never clip. Without the change the underwater wall shows.
"""
import re
import subprocess
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def program(mutated=False):
    geometry = (ROOT / 'Renderer/native/render_core/rigid_instance_geometry.hlsl').read_text()
    point = geometry[geometry.index('struct RigidPoint'):]
    point = point[:point.index('\n}') + 2]
    vertex = (ROOT / 'Renderer/native/render_core/rigid_feature.hlsl').read_text()
    carry = re.search(r' if\(bridge\)o\.q6_world\.w=([^;]+);', vertex)
    pixel = between((ROOT / 'Renderer/native/render_core/terrain_scene.hlsl').read_text(),
                    'float4 PSIntegratedFeature(FeaturePixelInput input) : SV_TARGET', 'return PSFeature')
    clip = re.search(r'if \(input\.q6_world\.w > ([0-9.]+)\) clip\(([^;]+)\);', pixel)
    carried = '1' if mutated or not carry else carry[1]
    return r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#define precise
struct float2 {float x,y;};
struct float3 {float x,y,z;float3(float a=0,float b=0,float c=0):x(a),y(b),z(c){}};
struct float4 {float x,y,z,w;};
float3 operator/(float3 a,float s){return {a.x/s,a.y/s,a.z/s};}
using std::sqrt;
struct RigidInput {float3 source_position,source_normal;float2 source_uv;float4 place0,place1,projection,placement_view;};
''' + point + r'''
struct Pixel {float4 q6_world;};
// Returns false where the pixel shader discards the fragment.
bool kept(Pixel input){
''' + (f' if (input.q6_world.w > {clip[1]}) {{ if(({clip[2].replace("input.", "input.")})<0) return false; }}\n' if clip else '') + r'''
 return true;
}
bool shown(float material,float source_z){
 RigidInput i{};i.source_position=float3(.01f,.002f,source_z);i.source_normal=float3(0,0,1);
 i.place0={20,30,.5f,.5f};i.place1={1,0,4.2f,0};i.projection={20,30,128,1080};i.placement_view={0,0,0,material};
 RigidPoint p=rigid_point(i);Pixel o{{p.world.x,p.world.y,p.world.z,1}};
 bool bridge=material>12.5 && material<20.5;
 if(bridge)o.q6_world.w=''' + carried + r''';
 return kept(o);
}
int main(){
 for(float material:{13.f,14.f,17.f,20.f}){
  // The medieval arch's walls reach -0.012 below its base; the industrial piers -0.04.
  for(float z:{-.04f,-.012f,-.004f})if(shown(material,z)){std::puts("UNDERWATER_SHOWN");return 3;}
  for(float z:{0.f,.004f,.02f,.05f})assert(shown(material,z));
 }
 // Other rigid features (resources, sites, farms) keep every part.
 for(float material:{8.f,12.f,21.f,21.0235f,30.f})for(float z:{-.04f,0.f,.03f})assert(shown(material,z));
 std::puts("PASS bridge waterline: parts below the base clip, the arch above stays");
}
'''


class BridgeWaterlineTests(unittest.TestCase):
    def test_parts_below_the_bridge_base_stay_underwater(self):
        run_cpp(program())

    def test_without_the_waterline_the_walls_show(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            run_cpp(program(mutated=True))
        self.assertEqual(failure.exception.returncode, 3)

    def test_generated_feature_shaders_carry_the_clip(self):
        for name in ('city_fidelity', 'environment_refresh'):
            shader = (ROOT / 'Renderer/native' / name / 'feature.hlsl').read_text()
            self.assertIn('if (input.q6_world.w > 1.5) clip(input.q6_world.w - 2.0 + 0.002);', shader, name)
        rigid = (ROOT / 'Renderer/native/city_fidelity/rigid_feature.hlsl').read_text()
        self.assertIn('if(bridge)o.q6_world.w=2+p.position.z/128;', rigid)


if __name__ == '__main__':
    unittest.main()
