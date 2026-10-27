"""Autumn's sunrise and sunset stay golden without one strong brown.

Autumn turned all turf about 20 degrees toward orange and the gold sunrise and
sunset added about 15 more, so grassland and plains shared one brown (hue gap
near 0 against Summer's 7-10; user, 2026-10-26). Autumn now shifts plains
(straw) further than grassland (green turf), and at the sun's warmest pulls
the sun partway back toward noon's colour with a little less display
saturation. Noon and the other seasons keep their light.
"""
import pathlib
import re
import unittest

from Renderer.native.native_cpp_test import run_cpp

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTUMN = ROOT / "Renderer/lab/shared/seasons/autumn.hlsl"

PROGRAM = r'''
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include "Renderer/native/environment_runtime.h"
using namespace c3x_renderer;
float balance(EnvironmentState const& e){return e.sun_color[2]/e.sun_color[0];}
int main(){try{
 auto noon_summer=evaluate_environment(12.036f,0),noon_autumn=evaluate_environment(12.036f,1);
 for(int i=0;i<3;++i)if(std::fabs(noon_summer.sun_color[i]-noon_autumn.sun_color[i])>1e-6f)
  throw std::runtime_error("Autumn changed noon's sun");
 if(std::fabs(noon_summer.saturation-noon_autumn.saturation)>1e-6f)throw std::runtime_error("Autumn changed noon's saturation");
 for(float hour:{6.f,18.55f}){
  auto summer=evaluate_environment(hour,0),autumn=evaluate_environment(hour,1);
  std::printf("%.2f summer b/r %.3f sat %.3f; autumn b/r %.3f sat %.3f\n",hour,balance(summer),summer.saturation,
              balance(autumn),autumn.saturation);
  // Closer to noon's balance, but still a warm, golden sun.
  if(!(balance(autumn)>balance(summer)+.08f && balance(autumn)<balance(noon_summer)-.1f))
   throw std::runtime_error("Autumn's low sun is not softened toward noon");
  if(!(autumn.saturation<summer.saturation*.95f))throw std::runtime_error("Autumn's golden hour keeps full saturation");
  if(std::fabs(autumn.sun_intensity-summer.sun_intensity)>1e-6f)throw std::runtime_error("Autumn changed the sun's strength");
 }
 for(int step=0;step<96;++step){
  float hour=step*.25f;
  auto summer=evaluate_environment(hour,0),spring=evaluate_environment(hour,3),autumn=evaluate_environment(hour,1);
  for(int i=0;i<3;++i)if(spring.sun_color[i]!=summer.sun_color[i])throw std::runtime_error("Spring's sun changed");
  if(spring.saturation!=summer.saturation)throw std::runtime_error("Spring's saturation changed");
  if(balance(autumn)>balance(noon_summer)+1e-4f)throw std::runtime_error("Autumn's sun turned cooler than noon");
 }
 std::puts("PASS Autumn golden hour");return 0;
}catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}}
'''


def define(name, text):
    found = re.search(r"#define " + name + r" (-?[\d.]+)", text)
    assert found, name + " is missing from autumn.hlsl"
    return float(found.group(1))


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


class AutumnGoldenHourTests(unittest.TestCase):
    def test_autumn_softens_its_golden_hour_only(self):
        run_cpp(PROGRAM, sources=("Renderer/native/environment_runtime.cpp",), timeout=60)

    def test_grassland_keeps_more_of_its_hue_than_plains(self):
        text = AUTUMN.read_text()
        self.assertIn("lch.z += (lerp(AUTUMN_STRAW_SHIFT, AUTUMN_GRASS_SHIFT, smoothstep(100, 110, hue)) + "
                      "patch * AUTUMN_PATCH_HUE) * gate;", text)
        straw, grass = define("AUTUMN_STRAW_SHIFT", text), define("AUTUMN_GRASS_SHIFT", text)
        # Mean ground hues in Summer (village view, noon): plains 101, grassland 111.
        plains = straw + (grass - straw) * smoothstep(100, 110, 101)
        meadow = straw + (grass - straw) * smoothstep(100, 110, 111)
        self.assertLessEqual(plains, -18, "plains no longer turn tawny")
        self.assertLessEqual(meadow, -8, "grassland no longer turns autumnal")
        self.assertGreaterEqual(meadow - plains, 8, "grassland and plains turn the same brown")


if __name__ == "__main__":
    unittest.main()
