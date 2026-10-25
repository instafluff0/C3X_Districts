"""Relief, trees and units take the same clock light as cities and water.

The gameplay `frame_illumination` (Q6_GAMEPLAY_NIGHT) lifts night ambient and
weights moonlight x2.2*.82. Source (BeautyFrame) receivers built their light
from the raw environment instead, so at midnight relief, mountains, trees and
units got about half the light of cities, water and rivers beside them. This
oracle runs the production HLSL lighting function as C++ on flat ground under
the shared key light over every half hour and season: the direct and ambient
light that `lighting::source_response` hands BeautyFrame receivers, relative
to noon, must equal frame_illumination's.
"""
import pathlib
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.source_fidelity.prepare import function

ROOT = pathlib.Path(__file__).resolve().parents[2]
NATIVE = ROOT / "Renderer/native"
RECEIVER = NATIVE / "city_fidelity/hydrology.hlsl"
BUILDERS = (ROOT / "Renderer/lab/shared/natural/data.h", NATIVE / "unit_body_renderer.h",
            ROOT / "Renderer/sandbox/direct_units.h")


def program():
    return r'''
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "Renderer/native/scene_lighting.h"
using namespace c3x_renderer;
struct float3{float x,y,z;float3(float a,float b,float c):x(a),y(b),z(c){}float operator[](int i)const{return i==0?x:i==1?y:z;}};
float3 operator+(float3 a,float3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
float3 operator-(float3 a,float3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
float3 operator*(float3 a,float s){return {a.x*s,a.y*s,a.z*s};}
float3 operator*(float3 a,float3 b){return {a.x*b.x,a.y*b.y,a.z*b.z};}
float3& operator*=(float3& a,float s){a=a*s;return a;}
float dot(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
float saturate(float v){return v<0?0:v>1?1:v;}
float lerp(float a,float b,float t){return a+(b-a)*t;}
float3 vec(float const* v){return {v[0],v[1],v[2]};}
// Production cbuffer values, as fresh_pipeline.h update_environment fills them.
float3 environment_sun_direction{0,0,1},environment_moon_direction{0,0,1};
float3 environment_sun_color{0,0,0},environment_moon_color{0,0,0},environment_ambient_color{0,0,0};
float environment_sun_intensity,environment_moon_intensity,environment_shadow_strength,environment_night_activation;
#define Q6_GAMEPLAY_NIGHT 1
''' + function(RECEIVER.read_text(), "frame_illumination") + r'''
void bind(EnvironmentState const& e){
 auto light=lighting::key_light(e);
 environment_sun_direction=environment_moon_direction=vec(light.direction.data());
 environment_sun_color=vec(e.sun_color);environment_moon_color=vec(e.moon_color);
 environment_ambient_color=vec(e.ambient_color);
 environment_sun_intensity=e.sun_intensity;environment_moon_intensity=e.moon_intensity;
 environment_shadow_strength=e.shadow_strength;environment_night_activation=e.night_activation;
}
int main(){try{
 float3 up{0,0,1};
 bind(evaluate_environment(12,0));
 float3 noon_direct=frame_illumination(up,1,0),noon_ambient=frame_illumination(up,1,1)-noon_direct;
 unsigned samples=0;float midnight=0;
 for(int season=0;season<4;++season)for(int step=0;step<48;++step){
  auto e=evaluate_environment(step*.5f,season);bind(e);
  float3 direct=frame_illumination(up,1,0),ambient=frame_illumination(up,1,1)-direct;
  auto source=lighting::source_response(e);
  for(int i=0;i<3;++i){
   float q6_direct=direct[i]/noon_direct[i],q6_ambient=ambient[i]/noon_ambient[i];
   float source_direct=source.strength*source.color[i];
   if(std::fabs(source_direct-q6_direct)>2e-3f*std::fmax(1.f,q6_direct) ||
      std::fabs(source.ambient[i]-q6_ambient)>2e-3f*std::fmax(1.f,q6_ambient)){
    std::printf("hour %.1f season %d channel %d direct %.4f/%.4f ambient %.4f/%.4f\n",step*.5f,season,i,
                source_direct,q6_direct,source.ambient[i],q6_ambient);
    throw std::runtime_error("source receivers' clock light differs from frame_illumination");
   }
  }
  if(step==0 && season==0)midnight=source.strength;
  ++samples;
 }
 if(midnight<=0)throw std::runtime_error("no direct moonlight at midnight");
 std::printf("PASS source night response: %u samples; midnight direct %.3f of noon\n",samples,midnight);
 return 0;
}catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}}
'''


class SourceNightResponseTests(unittest.TestCase):
    def test_source_receivers_match_gameplay_illumination(self):
        run_cpp(program(), sources=("Renderer/native/environment_runtime.cpp",), timeout=60)

    def test_every_beauty_frame_builder_uses_the_shared_response(self):
        for path in BUILDERS:
            text = path.read_text()
            self.assertTrue("source_response(" in text, path.name)
            self.assertFalse("noon.ambient_color" in text or "noon.sun_color" in text, path.name)


if __name__ == "__main__":
    unittest.main()
