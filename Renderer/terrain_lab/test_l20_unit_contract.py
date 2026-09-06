#!/usr/bin/env python3
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class L20UnitContractTests(unittest.TestCase):
    def test_source_bundles_and_fixture_are_present(self) -> None:
        pack = ROOT / "Renderer/packs/UnitFamilyLab"
        for slug in ("archer", "swordsman", "infantry", "fighter", "galley", "worker"):
            payload = (pack / f"unit_{slug}_runtime.bin").read_bytes()
            self.assertTrue(payload.startswith(b"C3XVEG1\0"))
            self.assertGreater(len(payload), 10000)
        compound = ROOT / "Renderer/packs/CompoundUnitLab"
        for slug in ("horseman", "catapult", "tank", "great_general_classical"):
            payload = (compound / f"unit_{slug}_runtime.bin").read_bytes()
            self.assertTrue(payload.startswith(b"C3XVEG1\0"))
            self.assertGreater(len(payload), 10000)
        fixture = ROOT / "Renderer/terrain_lab/fixtures/l20_units_192.csv"
        self.assertGreater(len(fixture.read_bytes()), 700)

    def test_lab_modes_and_owner_animation_contract(self) -> None:
        source = (ROOT / "Renderer/terrain_lab/terrain_lab.cpp").read_text(encoding="utf-8")
        shader = (ROOT / "Renderer/terrain_lab/terrain_lab.hlsl").read_text(encoding="utf-8")
        runner = (ROOT / "Renderer/terrain_lab/RUN_L20.bat").read_text(encoding="utf-8")
        for mode in ("units_noon", "units_night", "units_zoom2", "units_no_units",
                     "units_only", "units_turntable", "units_actions"):
            self.assertIn(mode, source)
            self.assertIn(mode, runner)
        self.assertIn("unit_team_mask", shader)
        self.assertIn("progress_milli", source)
        self.assertIn("army_commander_plus_member", (
            ROOT / "Renderer/terrain_lab/build_l20_unit_scenario.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("C3X_LAB_COMPOUND_UNITS", runner)
        self.assertNotIn("launch", runner.lower())


if __name__ == "__main__":
    unittest.main()
