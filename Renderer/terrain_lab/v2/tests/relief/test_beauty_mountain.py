import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[5]
FIXTURES = ROOT / "Renderer/terrain_lab/v2/fixtures/relief"
SYSTEM = ROOT / "Renderer/terrain_lab/v2/systems/relief"


class BeautyMountainContractTests(unittest.TestCase):
    def test_focused_fixture_uses_source_assets_and_backend_neutral_packet(self):
        fixture = json.loads((FIXTURES / "beauty-mountain.fixture.json").read_text())
        self.assertEqual([1536, 1024], fixture["viewport"])
        self.assertEqual(4, fixture["settings"]["samples"])
        self.assertEqual(16, fixture["settings"]["anisotropy"])
        self.assertEqual("Renderer/packs/Civ5EnvironmentSkin", fixture["packs"]["terrain"])
        self.assertEqual(["civ6.mountain"], fixture["references"])
        source = (SYSTEM / "beauty_mountain.cpp").read_text()
        self.assertNotIn("#import <Metal", source)
        self.assertNotIn("#include <d3d11.h>", source)
        self.assertIn('const std::string pack = "Renderer/packs/Civ5EnvironmentSkin/"', source)
        for channel in ("mtn_base", "mtn_top", "mtn_snow"):
            self.assertIn(channel + "_base_color.dds", source)
            self.assertIn(channel + "_height.dds", source)
            self.assertIn(channel + "_specular.dds", source)
        self.assertIn("grid = 256", source)

    def test_baseline_and_candidate_share_geometry_and_shader(self):
        beauty = json.loads((SYSTEM / "beauty_mountain.module.json").read_text())
        baseline = json.loads((SYSTEM / "beauty_mountain_baseline.module.json").read_text())
        self.assertEqual(beauty["source"], baseline["source"])
        self.assertEqual(beauty["shader"], baseline["shader"])
        self.assertNotEqual(beauty["id"], baseline["id"])


if __name__ == "__main__":
    unittest.main()
