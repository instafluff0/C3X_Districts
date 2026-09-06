#!/usr/bin/env python3
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class L21CompleteSceneContractTests(unittest.TestCase):
    def test_complete_modes_and_alternate_skin_are_frozen(self) -> None:
        source = (ROOT / "Renderer/terrain_lab/terrain_lab.cpp").read_text(encoding="utf-8")
        runner = (ROOT / "Renderer/terrain_lab/RUN_L21.bat").read_text(encoding="utf-8")
        for mode in ("complete_noon", "complete_sunset", "complete_midnight",
                     "complete_sunrise", "complete_zoom2", "complete_no_units",
                     "complete_no_borders"):
            self.assertIn(mode, source)
            self.assertIn(mode.removeprefix("complete_"), runner)
        self.assertIn("Civ5EnvironmentSkin", runner)
        self.assertIn("l20_units_192.csv", runner)
        self.assertIn("C3X_L21_MODES", runner)
        self.assertIn("for /l %%R in (1,1,3)", runner)
        self.assertIn("ping -n 16 127.0.0.1", runner)
        self.assertNotIn("launch", runner.lower())

    def test_territory_and_raised_object_shadow_contract(self) -> None:
        source = (ROOT / "Renderer/terrain_lab/terrain_lab.cpp").read_text(encoding="utf-8")
        shader = (ROOT / "Renderer/terrain_lab/terrain_lab.hlsl").read_text(encoding="utf-8")
        self.assertIn("build_lab_territory_owners", source)
        self.assertIn("add_territory_boundary", source)
        self.assertIn("owner_a == owner_b", source)
        self.assertIn("surface_kind = 13.0f", source)
        self.assertIn("one narrow ribbon in its own main color", source)
        self.assertIn("14.0f", source)
        self.assertIn("Static raised map objects use the same projected source-mesh method", shader)
        self.assertIn("raised_map_object_weight", shader)
        self.assertIn("One owner color only", shader)

    def test_final_scene_keeps_192_tile_authoritative_viewport(self) -> None:
        fixture = ROOT / "Renderer/preview/out/terrain_lab/test_biq_l13_rivers_192.csv"
        header = fixture.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(["16", "12", "192"], header[1:4])


if __name__ == "__main__":
    unittest.main()
