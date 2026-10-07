"""River banks never cover the objects standing beside them.

The river surface (ground layer kind 9) drew its whole bank band with a depth
pulled 0.025*reserved.x (about 27 units at 1080 rows) toward the camera, and
wrote that depth. Its ground basis also weighted height far less than the
natural terrain's (0.35 against 1.73 depth units per unit of relief), so the
bias had to be large. Bridges, farms, routes and object shadows drawn after
the river failed the depth test inside the bank band; trees drawn before it
were painted over wherever they stood within the bias.

Now the generated hydrology shaders define C3X_RIVER_NATURAL_DEPTH: river
vertices sort on the natural basis (flat row plus 0.0016*H per unit of
relief) with a small bias, and the river draws with the test-only decal depth
state, like routes, in the Lab and in the game's fresh pipeline. This test
compiles translated_depth from
integrated_terrain.hlsl as C++. With the define, the river must sort just in
front of the natural ground beneath it at every elevation, and behind an
object standing six rows in front of it. Without the define (the old code),
those checks must fail.
"""
import re
import subprocess
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def program(define):
    adapter = (ROOT / 'Renderer/native/integrated_terrain.hlsl').read_text()
    depth = between(adapter, 'float translated_depth(IntegratedVertexInput input, bool feature)',
                    '\n// Match the rasterizer')
    projection = between((ROOT / 'Renderer/native/render_core/world_projection.hlsl').read_text(),
                         'float3 project_world_content(', '\n// Resource bodies')
    return ('#define C3X_RIVER_NATURAL_DEPTH 1\n' if define else '') + r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
using std::floor;
struct float2 {float x,y;};
struct float3 {float x,y,z;float3(float a=0,float b=0,float c=0):x(a),y(b),z(c){}
 float3(float2 v,float c):x(v.x),y(v.y),z(c){}float2 xy()const{return {x,y};}};
struct float4 {float x,y,z,w;};
float2 operator*(float2 a,float s){return {a.x*s,a.y*s};}
float3 operator*(float3 a,float s){return {a.x*s,a.y*s,a.z*s};}
template<class T>T clamp(T v,T lo,T hi){return v<lo?lo:v>hi?hi:v;}
inline float max(float a,float b){return a>b?a:b;}
inline float min(float a,float b){return a<b?a:b;}
struct IntegratedVertexInput {float3 position;float surface_kind;float4 q6_world;};
float c3x_viewport_depth_translation=0;
float2 c3x_viewport_reserved{1080,0};
''' + depth + '\n' + re.sub(r'\.xy\b', '.xy()', projection) + r'''
constexpr float tile=128,rows=1080,relief_pixels=tile/224*.82f;
float stored(float pixel_depth){return clamp(.5f-(floor(pixel_depth*256+.5f)/256)/16384.f,.001f,.999f);}
// Natural terrain (VSNative and project_world_content kind 1), no layer bias.
float ground(float u,float v,float h){
 float4 projection{20,30,tile,rows};
 return stored(project_world_content(float3(),float3(20+u,30+1-v,(h+2.5f)/112),projection,1).z);
}
// A river vertex as ground_compiler.h emits it: y = G - lift, z = G + .75 lift.
float river(float u,float v,float h){
 float row=(u+v)*tile*.25f,lift=h*relief_pixels;
 IntegratedVertexInput input{float3((u-v)*tile*.5f,row-lift,row+.75f*lift),9,{20+u,30+1-v,(h+2.5f)/112,1}};
 return translated_depth(input,false);
}
int main(){
 for(float h:{-1.1f,0.f,3.f,12.f,40.f})for(float u:{.1f,.5f,.83f})for(float v:{.2f,.6f}){
  float water=river(u,v,h),beneath=ground(u,v,h);
  // In front of the coplanar ground it covers, by a few depth units only.
  float lead=(beneath-water)*16384;
  std::printf("h=%.1f u=%.2f v=%.2f lead=%.2f\n",h,u,v,lead);
  if(!(lead>=2 && lead<=6))return 4;
  // Objects drawn after the river only meet the ground's depth. One drawn
  // before it (a tree) whose ground point is six rows in front of a river
  // pixel it overlaps on screen stays in front; the old bias covered 27.
  float step=6/(tile*.25f);
  if(!(ground(u+step*.5f,v+step*.5f,h)<water)){std::puts("OBJECT_COVERED");return 3;}
 }
 std::puts("PASS river natural depth: small lead over the ground, never over objects in front");
}
'''


class RiverDepthTests(unittest.TestCase):
    def test_river_sorts_on_the_natural_basis_with_a_small_lead(self):
        run_cpp(program(True))

    def test_old_river_depth_covers_objects_beside_the_bank(self):
        # Confirm the check: the old bias and basis compile and fail it.
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            run_cpp(program(False))
        self.assertIn(failure.exception.returncode, (3, 4))

    def test_generated_hydrology_enables_the_natural_river_basis(self):
        generator = (ROOT / 'Renderer/native/render_core/generate_shaders.py').read_text()
        self.assertIn('#define C3X_RIVER_NATURAL_DEPTH 1', generator)
        for name in ('source_fidelity', 'environment_refresh', 'city_fidelity'):
            shader = (ROOT / 'Renderer/native' / name / 'hydrology.hlsl').read_text()
            self.assertIn('#define C3X_RIVER_NATURAL_DEPTH 1', shader, name)
            self.assertIn('float4 q6_world : TEXCOORD14;', between(
                shader, 'struct IntegratedVertexInput', '};'), name)

    def test_river_never_writes_depth(self):
        # Drawn with the routes' test-only decal state between water and shadows.
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        block = between(source, 'if (!draw(geometry_bed) || !draw(geometry_water))return false;',
                        'bool routes_drawn=draw(geometry_route);')
        before = block[:block.index('draw(geometry_river)')]
        self.assertIn('OMSetDepthStencilState(natural.decal_depth,0);', before)
        self.assertEqual(source.count('draw(geometry_river)'), 1)
        # The game's draw_layer gives the river the routes' decal state, and
        # the overlay layer's depth-only water prepass no longer draws it.
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        decal = re.search(r'else if \(\(([^)]*)\) && renderer\.natural\.decal_depth\)', fresh)
        self.assertIsNotNone(decal)
        self.assertIn('layer==geometry_river', decal[1])
        strip = between(fresh, 'bool write_overlay_strip(', 'bool write_slot(')
        self.assertNotIn('draw_layer(water,geometry_river', strip)


if __name__ == '__main__':
    unittest.main()
