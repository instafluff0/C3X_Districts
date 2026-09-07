import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[5]
FIXTURES = ROOT / "Renderer/terrain_lab/v2/fixtures/relief"
SYSTEM = ROOT / "Renderer/terrain_lab/v2/systems/relief"
SHADERS = ROOT / "Renderer/terrain_lab/v2/shaders/relief"


class BeautyTerrainContractTests(unittest.TestCase):
    def test_fixture_is_mac_fast_path_quality_study(self):
        fixture = json.loads((FIXTURES / "beauty-terrain.fixture.json").read_text())
        self.assertEqual([1536, 1024], fixture["viewport"])
        self.assertEqual(4, fixture["settings"]["samples"])
        self.assertEqual(16, fixture["settings"]["anisotropy"])
        self.assertEqual(2, fixture["settings"]["render_scale"])
        self.assertEqual(-1.0, fixture["settings"]["mip_bias"])
        self.assertEqual("Renderer/packs/Civ5EnvironmentSkin", fixture["packs"]["terrain"])
        self.assertEqual("Renderer/packs/DecalsNormalized", fixture["packs"]["decals"])
        self.assertIn("hill_clutter", fixture["isolations"])

    def test_source_material_and_hill_recipe_are_explicit(self):
        source = (SYSTEM / "beauty_terrain.cpp").read_text()
        self.assertNotIn("#import <Metal", source)
        self.assertNotIn("#include <d3d11.h>", source)
        for stem in (
            "grassland", "grasshill_top", "plains", "plainshill_top", "tundra_blend"
        ):
            self.assertIn(stem + "_base_color.dds", source)
            self.assertIn(stem + "_height.dds", source)
            self.assertIn(stem + "_specular.dds", source)
        self.assertIn("hills/standard/height_lod0.dds", source)
        self.assertIn("base_color_c996c6a9d015eebe.dds", source)
        self.assertIn("height_31eb0f0117ea3beb.dds", source)
        self.assertIn("weighted_cells[7] = {0, 0, 0, 1, 1, 2, 2}", source)

    def test_each_hill_has_a_unique_stable_seed_and_variable_density(self):
        source = (SYSTEM / "beauty_terrain.cpp").read_text()
        block = source.split("constexpr std::array<Hill, 6> hills", 1)[1].split("}};", 1)[0]
        seeds = re.findall(r"0x[0-9a-f]+u", block)
        rockiness = [float(value) for value in re.findall(r", (0\.\d+|1\.00)f, 0x", block)]
        self.assertEqual(6, len(seeds))
        self.assertEqual(6, len(set(seeds)))
        self.assertGreater(max(rockiness) - min(rockiness), 0.5)
        self.assertIn("random01(state) * 6.283185307f", source)
        self.assertIn("0.90f + 0.20f * random01(state)", source)
        self.assertIn("biome_coordinate(hill_field, hill.x, hill.y) > 1.48f", source)

    def test_shader_keeps_source_and_inference_distinct(self):
        shader = (SHADERS / "beauty_terrain.hlsl").read_text()
        self.assertIn("an inferred Lab response", shader)
        self.assertIn("HillDecalColor.Sample", shader)
        self.assertIn("GrassHillColor.Sample", shader)


if __name__ == "__main__":
    unittest.main()
