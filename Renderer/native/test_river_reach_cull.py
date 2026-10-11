"""River triangles beyond every shader variant's bank reach are left out (review 72).

The river layer is each river tile's whole ground grid (2,048 triangles at
32 subdivisions), and the river shader discards every pixel beyond its bank
only after its texture work: on the busy save the river layer was about 40%
of an idle frame's GPU time. A triangle whose three vertices are all beyond
the reach draws nothing, so the compiler leaves it out.
"""
import math
import random
import unittest
from Renderer.lab.platform import ROOT

COMPILER = ROOT / "Renderer/native/source_fidelity/ground_compiler.h"
SHADER = ROOT / "Renderer/native/city_fidelity/hydrology.hlsl"


def continuous_discards(d, aa, noise, sediment, gravel, shore):
    """hydrology.hlsl q3_water_material, continuous rivers: the clip before shading."""
    def smoothstep(e0, e1, x):
        t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    water_width = 5.8 + noise * 1.1
    bank_width = water_width + 2.5 + noise * 1.4 + (sediment - .5) * 1.3
    bank_feather = 1.8 + sediment * .7
    bank_edge = d + (gravel - .5) * .7 + shore * 1.1
    bank = 1 - smoothstep(bank_width - bank_feather, bank_width + .6 + aa, bank_edge)
    grit_reach = smoothstep(water_width + .4, water_width + 1.5, d) * \
        (1 - smoothstep(bank_width + .5, bank_width + 4.5, d))
    return max(bank, grit_reach) - .001 < 0  # bank*outlet and grit*others only shrink


class RiverReachCullTests(unittest.TestCase):
    def test_the_reach_covers_every_river_variant(self):
        shader = SHADER.read_text()
        # Constants the bound is derived from; a change here must revisit it.
        for text in ("float water_width=5.8+noise*1.1;",
                     "float bank_width=water_width+2.5+noise*1.4+(sediment_noise-.5)*1.3;",
                     "float bank_edge_distance=distance_pixels+(gravel_noise-.5)*.7+shore_noise*1.1;",
                     "float bank=1-smoothstep(bank_width-bank_feather,bank_width+.6+aa,bank_edge_distance);",
                     "*(1-smoothstep(bank_width+.5,bank_width+4.5,distance_pixels));",
                     "clip(max(bank,grit_spill)-.001);",
                     "float bank=1-smoothstep(6.0,7.4,distance_pixels);clip(bank-.001);",
                     "float bank=1-smoothstep(6.0,9.0,distance_pixels);clip(bank-.001);",
                     "float bank_width = 10.2 + bank_noise * 2.6 + mouth_weight * 6.0 +",
                     "junction_weight * 2.0 + source_weight * 1.2;"):
            self.assertIn(text, shader)
        # The legacy path's widest bank is 20.7; continuous rivers discard at
        # 11.7 + aa and, for grit, 14.7, for any noise sample.
        rng = random.Random(7)
        for _ in range(20000):
            aa = rng.uniform(.35, 12)
            d = max(21.7, 12.7 + aa) + rng.uniform(0, 3)
            self.assertTrue(continuous_discards(d, aa, rng.uniform(-.5, .5), rng.random(), rng.random(), rng.uniform(-.5, .5)))
        # The bound is not loose by more than its one-pixel margin.
        self.assertFalse(continuous_discards(11.0 + .35, .35, .5, 1, 0, -.5))

    def test_the_compiler_culls_with_the_triangle_gradient_at_the_smallest_zoom(self):
        source = COMPILER.read_text()
        self.assertIn("bool cull=layer==9.f && input.cull_river_reach && input.fidelity_profile;", source)
        self.assertIn("if(!(least>=21.7f))return true;", source)
        self.assertIn("float aa=std::max(.35f,2.f*(std::fabs(gx)+std::fabs(gy)));", source)
        self.assertIn("return !(least>=std::max(21.7f,12.7f+aa));", source)
        # World ground projects exactly as project_world_content does.
        self.assertIn("y=(p.world_x-p.world_y+1.f)*width*.25f-h*(width/224.f*.82f);", source)
        shader = SHADER.read_text()
        self.assertIn("float base=(dx-dy+1)*width*.25;", shader)
        self.assertIn("return float3((dx+dy)*width*.5,base-h*(width/224*.82),", shader)
        self.assertIn("SceneProjection", (ROOT / "Renderer/native/scene_projection.h").read_text())
        self.assertIn("minimum=.5f", (ROOT / "Renderer/native/scene_projection.h").read_text())
        # The control keeps whole grids, and the flag is part of the recipe key.
        self.assertIn("add(input.cull_river_reach);", (ROOT / "Renderer/native/world_preparation.h").read_text())
        self.assertIn("C3X_RENDERER_RIVER_CULL_CONTROL", (ROOT / "Renderer/native/c3x_renderer.cpp").read_text())


if __name__ == '__main__':
    unittest.main()
