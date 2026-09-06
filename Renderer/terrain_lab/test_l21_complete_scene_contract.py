#!/usr/bin/env python3
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class L21CompleteSceneContractTests(unittest.TestCase):
    def test_complete_modes_and_alternate_skin_are_frozen(self) -> None:
        source = (ROOT / "Renderer/terrain_lab/terrain_lab.cpp").read_text(encoding="utf-8")
        runner = (ROOT / "Renderer/terrain_lab/RUN_L21.bat").read_text(encoding="utf-8")
        for mode in ("complete_noon", "complete_sunset", "complete_midnight",
                     "complete_sunrise", "complete_zoom2", "complete_no_units"):
            self.assertIn(mode, source)
            self.assertIn(mode.removeprefix("complete_"), runner)
        self.assertIn("Civ5EnvironmentSkin", runner)
        self.assertIn("l20_units_192.csv", runner)
        self.assertNotIn("launch", runner.lower())

    def test_final_scene_keeps_192_tile_authoritative_viewport(self) -> None:
        fixture = ROOT / "Renderer/preview/out/terrain_lab/test_biq_l13_rivers_192.csv"
        header = fixture.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(["16", "12", "192"], header[1:4])


if __name__ == "__main__":
    unittest.main()
