#!/usr/bin/env python3
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class L19BInfrastructureContractTests(unittest.TestCase):
    def test_source_bundles_and_fixture_are_present(self) -> None:
        pack = ROOT / "Renderer/packs/TileObjectsNormalized"
        for name in ("fortification_runtime.bin", "airfield_runtime.bin",
                     "ground_state_runtime.bin"):
            payload = (pack / name).read_bytes()
            self.assertTrue(payload.startswith(b"C3XVEG1\0"))
            self.assertGreater(len(payload), 2000)
        fixture = ROOT / "Renderer/terrain_lab/fixtures/l19b_infrastructure_192.csv"
        self.assertGreater(len(fixture.read_bytes()), 600)
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         hashlib.sha256(fixture.read_bytes()).hexdigest())

    def test_lab_modes_keep_rejected_and_transient_effects_out(self) -> None:
        source = (ROOT / "Renderer/terrain_lab/terrain_lab.cpp").read_text(encoding="utf-8")
        shader = (ROOT / "Renderer/terrain_lab/terrain_lab.hlsl").read_text(encoding="utf-8")
        runner = (ROOT / "Renderer/terrain_lab/RUN_L19B.bat").read_text(encoding="utf-8")
        for mode in ("infrastructure_noon", "infrastructure_night",
                     "infrastructure_zoom2", "infrastructure_no_infrastructure",
                     "infrastructure_fortifications_only", "infrastructure_airfields_only",
                     "infrastructure_strategic_only", "infrastructure_damage_only"):
            self.assertIn(mode, source)
            self.assertIn(mode, runner)
        self.assertIn("pollution_weight", shader)
        self.assertNotIn("IMP_SILO", runner)
        self.assertNotIn("radar_observatory_body", runner)
        self.assertNotIn("smoke", runner.lower())
        self.assertNotIn("explosion", runner.lower())


if __name__ == "__main__":
    unittest.main()
