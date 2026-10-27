"""Every shadow matches noon's, and unit and resource shadows match world shadows.

World shadows (terrain, relief, forests, cities, static resources) remove the
shadow-strength share of direct light in the production `frame_illumination`
and keep its ambient. Dynamic footprints blend a dark layer instead. A fixed
opacity left them visibly lighter at noon (0.40 against 0.58), while a unit
limb in its body's self-shadow lost all direct light, darker than any world
shadow. This oracle runs the production HLSL lighting function, the generated
resource-shadow alpha and the unit self-shadow visibility as C++ over every
half hour and season. Unit and resource footprints must remove the same share
of open-ground light, and self-shadowed limbs the same share of direct light.
"""
import pathlib
import re
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.source_fidelity.prepare import function

ROOT = pathlib.Path(__file__).resolve().parents[2]
NATIVE = ROOT / "Renderer/native"
RECEIVER = NATIVE / "city_fidelity/hydrology.hlsl"
RESOURCE_SHADOWS = (NATIVE / "city_fidelity/resource_shadow.hlsl",
                    NATIVE / "city_fidelity/feature.hlsl")
FRESH = ROOT / "Renderer/sandbox/fresh_pipeline.h"
DIRECT = ROOT / "Renderer/sandbox/direct_units.h"
UNIT_SHADER = NATIVE / "environment_refresh/unit_shader.h"
UNIT_BODIES = NATIVE / "unit_body_renderer.h"


def resource_alpha(text):
    """The generated animated-resource shadow alpha statements."""
    branch = text.index("Animated resources project their current posed source triangles.")
    start = text.index("clip(coverage - 0.08);", branch) + len("clip(coverage - 0.08);")
    end = text.index("clip(alpha - 0.004);", start)
    return text[start:end]


def unit_visibility(text):
    """The generated unit material's self-shadow visibility statement."""
    start = text.index("float visibility=")
    return text[start:text.index(";", start) + 1]


def program():
    illumination = function(RECEIVER.read_text(), "frame_illumination")
    providers = "\n".join(
        "float resource_" + str(index) + "(float coverage){\n" + resource_alpha(path.read_text()) +
        "\n return alpha;\n}\n" for index, path in enumerate(RESOURCE_SHADOWS))
    return r'''
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "Renderer/native/scene_lighting.h"
using namespace c3x_renderer;
struct float3{float x,y,z;float3(float a,float b,float c):x(a),y(b),z(c){}};
float3 operator+(float3 a,float3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
float3 operator*(float3 a,float s){return {a.x*s,a.y*s,a.z*s};}
float3& operator*=(float3& a,float s){a=a*s;return a;}
float dot(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
float saturate(float v){return v<0?0:v>1?1:v;}
float lerp(float a,float b,float t){return a+(b-a)*t;}
float smoothstep(float a,float b,float v){float t=saturate((v-a)/(b-a));return t*t*(3-2*t);}
float3 vec(float const* v){return {v[0],v[1],v[2]};}
// Production cbuffer values, as fresh_pipeline.h update_environment fills them.
float3 environment_sun_direction{0,0,1},environment_moon_direction{0,0,1};
float3 environment_sun_color{0,0,0},environment_moon_color{0,0,0},environment_ambient_color{0,0,0};
float environment_sun_intensity,environment_moon_intensity,environment_shadow_strength,environment_night_activation;
#define Q6_GAMEPLAY_NIGHT 1
using lighting::c3x_dynamic_shadow_opacity;
struct float4{float x,y,z,w;} View{0,0,0,0};
float unit_self_visibility(float occluded){''' + unit_visibility(UNIT_SHADER.read_text()) + r''' return visibility;}
''' + illumination + providers + r'''
int main(){try{
 float noon=0;unsigned samples=0;
 for(int season=0;season<4;++season)for(int step=0;step<48;++step){
  auto e=evaluate_environment(step*.5f,season);auto light=lighting::key_light(e);
  environment_sun_direction=environment_moon_direction=vec(light.direction.data());
  environment_sun_color=vec(e.sun_color);environment_moon_color=vec(e.moon_color);
  environment_ambient_color=vec(e.ambient_color);
  environment_sun_intensity=e.sun_intensity;environment_moon_intensity=e.moon_intensity;
  environment_shadow_strength=e.shadow_strength;environment_night_activation=e.night_activation;
  float3 luma{.2126f,.7152f,.0722f},up{0,0,1};
  float world=1-dot(frame_illumination(up,0,1),luma)/dot(frame_illumination(up,1,1),luma);
  float unit=lighting::dynamic_shadow_opacity(e);
  float resource[2]={resource_0(1),resource_1(1)};
  if(step==24 && season==0)noon=world;
  // Every hour's shadow removes noon's share of open-ground light (user,
  // 2026-10-27); sky-lit nights once left 0.08, invisible on grass.
  if(std::fabs(world-.5835f)>.005f){
   std::printf("hour %.1f season %d world shadow %.4f strength %.3f\n",step*.5f,season,world,e.shadow_strength);
   throw std::runtime_error("a shadow is not as dark as noon's");
  }
  // The unit beauty frame carries the strength in View.w; self-shadow is
  // direct light only, so it stops at 1.
  View.w=std::min(e.shadow_strength,1.f);
  for(float occluded:{.5f,1.f}){
   float limb=unit_self_visibility(occluded);
   float direct=dot(frame_illumination(up,1-occluded,0),luma)/dot(frame_illumination(up,1,0),luma);
   if(std::fabs(limb-direct)>2e-4f){
    std::printf("hour %.1f season %d occluded %.1f limb %.4f world %.4f\n",step*.5f,season,occluded,limb,direct);
    throw std::runtime_error("self-shadowed unit limb differs from a world shadow");
   }
  }
  if(std::fabs(unit-world)>2e-4f || std::fabs(resource[0]-world)>2e-4f || std::fabs(resource[1]-world)>2e-4f){
   std::printf("hour %.1f season %d world %.4f unit %.4f resource %.4f/%.4f\n",step*.5f,season,world,unit,resource[0],resource[1]);
   throw std::runtime_error("dynamic ground shadow differs from a world shadow");
  }
  ++samples;
 }
 if(noon<.5f)throw std::runtime_error("noon world shadow unexpectedly weak");
 std::printf("PASS dynamic shadow opacity: %u samples; noon removes %.3f of open-ground light\n",samples,noon);
 return 0;
}catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}}
'''


class DynamicShadowOpacityTests(unittest.TestCase):
    def test_unit_and_resource_shadows_match_world_shadows(self):
        run_cpp(program(), sources=("Renderer/native/environment_runtime.cpp",), timeout=60)

    def test_oracle_uses_production_light_bindings(self):
        fresh = FRESH.read_text()
        # The receiver's sun and moon directions are both the shared key light.
        self.assertIn("std::copy(light.direction.begin(),light.direction.end(),settings.light_direction);", fresh)
        self.assertIn("std::copy(light.direction.begin(),light.direction.end(),settings.moon_direction);", fresh)
        self.assertIn("settings.shadow_strength=environment.shadow_strength;", fresh)
        for path in (RECEIVER, *RESOURCE_SHADOWS):
            self.assertRegex(path.read_text(), re.compile(r"^#define Q6_GAMEPLAY_NIGHT 1$", re.M), path.name)
        direct = DIRECT.read_text()
        self.assertIn("placement_values[14]=c3x_renderer::lighting::dynamic_shadow_opacity(environment);", direct)
        # Every unit beauty frame (live and preview) fills the strength.
        self.assertEqual(direct.count("beauty_values[15]=std::min(environment.shadow_strength,1.f);"), 2)
        self.assertEqual(direct.count("beauty_values[14]=.469979f;"), 2)
        self.assertIn("beauty[15]=std::min(environment.shadow_strength,1.f);", UNIT_BODIES.read_text())
        shader = UNIT_SHADER.read_text()
        self.assertEqual(shader.count("(i,albedo,n,visibility)") + shader.count("emission,visibility)"), 2)
        self.assertNotIn("1-occluded)", shader)


if __name__ == "__main__":
    unittest.main()
