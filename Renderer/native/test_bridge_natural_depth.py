"""Road bridges draw whole over the rivers they span.

In-game, bridges over rivers showed only their parapet arcs (performance
review 4y). The river surface (ground layer kind 9) sorts 0.025*reserved.x
nearer than its height in translated_depth, so it covers its bed and banks.
At about 1.9 depth units per unit of height, every bridge part lower than about
16 units lost to it. Rigid features also wrote depth on the feature basis,
which weights ground height far less than natural surfaces do, so raised
rivers hid even more. The Lab's river sits at ground level, and its bridges
drew whole.

VSSharedFeature now gives bridge materials (13-20) the natural height-depth
basis, as resource_natural_depth does for resource bodies, and a layer bias
slightly larger than the river's. This test compiles the actual shader source
as C++ with a small vector shim. A deck just above the river must sort in
front of the biased river at any elevation. Raising the ground must shift
bridge and river depth equally. Other rigid materials keep the feature basis.
"""
import re
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


class BridgeNaturalDepthTests(unittest.TestCase):
    def test_bridge_depth_follows_natural_surfaces_at_any_ground_height(self):
        native = ROOT / 'Renderer/native/render_core'
        # Through farm_kit_material, which the rigid shader also calls.
        projection = between((native / 'world_projection.hlsl').read_text(),
                             'float3 project_world_content(', '\nfloat resource_natural_depth(')
        geometry = (native / 'rigid_instance_geometry.hlsl').read_text()
        point = between(geometry, 'struct RigidPoint', '\n// b1') if '\n// b1' in geometry else \
            geometry[geometry.index('struct RigidPoint'):]
        shader = (native / 'rigid_feature.hlsl').read_text()
        depth = between(shader, ' float3 position=project_world_content(p.position,p.world,i.projection,2);',
                        ' return o;')
        depth = '\n'.join(line for line in depth.splitlines()
                           if 'o.position.xy=' not in line and 'q6_world' not in line)
        # The generated hydrology shaders sort the river on the natural basis
        # (C3X_RIVER_NATURAL_DEPTH) with this layer bias.
        adapter = (ROOT / 'Renderer/native/integrated_terrain.hlsl').read_text()
        bias = re.search(r'#ifdef C3X_RIVER_NATURAL_DEPTH.*?river_bias = ([0-9.]+);', adapter, re.S)[1]
        self.assertIn('append_ground_layer(destination.river_vertices, 9.0f,',
                      (ROOT / 'Renderer/native/source_fidelity/ground_compiler.h').read_text())
        program = '#define RIVER_BIAS ' + bias + 'f\n' + r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#define precise
using std::floor;using std::sqrt;using std::abs;
inline float frac(float v){return v-std::floor(v);}
struct float2 {float x,y;};
struct float3 {float x,y,z;float3(float a=0,float b=0,float c=0):x(a),y(b),z(c){}
 float3(float2 v,float c):x(v.x),y(v.y),z(c){}float2 xy()const{return {x,y};}};
struct float4 {float x,y,z,w;};
float2 operator*(float2 a,float s){return {a.x*s,a.y*s};}
float3 operator*(float3 a,float s){return {a.x*s,a.y*s,a.z*s};}
float3 operator/(float3 a,float s){return {a.x/s,a.y/s,a.z/s};}
template<class T>T clamp(T v,T lo,T hi){return v<lo?lo:v>hi?hi:v;}
inline double max(double a,double b){return a>b?a:b;}
struct RigidInput {float3 source_position,source_normal;float4 place0,place1,projection,placement_view;};
float2 c3x_viewport_reserved{1192,0};
''' + re.sub(r'\.xy\b', '.xy()', projection) + '\n' + point + r'''
struct Output {float4 position;};
float rigid_depth(RigidInput i){
 RigidPoint p=rigid_point(i);Output o{};
''' + depth + r'''
 return o.position.z;
}
// A natural surface (terrain or water) at world height g, kind 1.
float natural_depth(float g,float4 projection,float wx,float wy){
 float3 z=project_world_content(float3(),float3(wx,wy,(g+2.5f)/112),projection,1);
 return clamp(.5f-(floor(z.z*256+.5f)/256)/16384.f,.001f,.999f);
}
RigidInput part(float material,float ground,float height){
 RigidInput i{};i.source_position=float3(.05f,-.03f,height);i.source_normal=float3(0,0,1);
 i.place0={20,30,.5f,.5f};i.place1={1,0,1,ground};i.projection={20,30,128,1192};
 i.placement_view={0,0,0,material};return i;
}
// The river surface (ground layer kind 9) at its true height, with its
// translated_depth layer bias.
float river_depth(float g,float4 projection,float wx,float wy){
 return std::fmax(.001f,natural_depth(g,projection,wx,wy)-RIVER_BIAS*c3x_viewport_reserved.x/16384.f);
}
int main(){
 float4 projection={20,30,128,1192};
 // A deck or arch a little above the river it spans sorts in front of the
 // river's biased surface, at any elevation; before, everything below about
 // 16 height units lost to it.
 for(float material:{13.f,16.f,18.f,20.f})for(float g:{0.f,12.f,40.f})for(float height:{.004f,.02f,.06f}){
  RigidInput deck=part(material,g,height);RigidPoint at=rigid_point(deck);
  float river=river_depth(g,projection,at.world.x,at.world.y);
  assert(rigid_depth(deck)<river);
 }
 for(float height:{.02f,.06f,.12f}){
  // A bridge deck part and the river beneath it, both raised by 40 units.
  float bridge=rigid_depth(part(13,0,height))-rigid_depth(part(13,40,height));
  float river=natural_depth(0,projection,20.5f,30.5f)-natural_depth(40,projection,20.5f,30.5f);
  std::printf("height=%.2f bridge_shift=%.7f river_shift=%.7f\n",height,bridge,river);
  assert(std::fabs(bridge-river)<2e-6f);
  for(float material:{14.f,17.f,18.f,19.f,20.f})
   assert(std::fabs(rigid_depth(part(material,0,height))-rigid_depth(part(material,40,height))-river)<2e-6f);
  // A railroad tunnel's portal shares the bridge slots, marked .0035 as farm
  // kit props are. It sorts on the same natural basis without the river's
  // bias, so mountain rock a little above a part hides it; with the bias its
  // block showed through the rock (a bridge part there stays in front).
  for(float material:{14.0035f,16.0035f})for(float g:{0.f,40.f}){
   RigidInput tunnel=part(material,g,height);RigidPoint at=rigid_point(tunnel);float h=at.world.z*112-2.5f;
   float rock=natural_depth(h+3,projection,at.world.x,at.world.y);
   std::printf("tunnel=%.7f rock=%.7f bridge=%.7f\n",rigid_depth(tunnel),rock,rigid_depth(part(std::floor(material),g,height)));
   assert(rock<rigid_depth(tunnel) && rigid_depth(part(std::floor(material),g,height))<rock);
  }
  // Other rigid materials keep the feature basis: raised ground moves them less.
  for(float material:{8.f,12.f,21.f,30.f}){
   float shift=rigid_depth(part(material,0,height))-rigid_depth(part(material,40,height));
   assert(shift>0 && shift<river*.5f);
  }
 }
 std::puts("PASS bridge natural depth: equal shift with natural surfaces at every elevation");
}
'''
        run_cpp(program)


if __name__ == '__main__':
    unittest.main()
