#!/usr/bin/env python3
import json
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
        self.assertIn("unit_team_strength", shader)
        self.assertIn("unit_color_ramp", shader)
        self.assertIn("unit_source_value", shader)
        self.assertIn("unit_source_tint_code", shader)
        self.assertIn("unit_neutral_tint", shader)
        self.assertIn("unit_style_fraction", shader)
        self.assertIn("source_tint_code) * 0.0001f", source)
        self.assertIn('asset.id.rfind(":n")', source)
        self.assertNotIn("albedo * 0.28 + unit_color * 0.72", shader)
        self.assertIn("unit_weight * 1.10", shader)
        self.assertIn("unit_weight * 0.22", shader)
        self.assertIn("make_projected_feature_shadow_vertex", source)
        self.assertIn("project_shadow_mesh", source)
        self.assertIn("shadows, unit_output[instance.kind], 6.0f, 1.10f, true", source)
        self.assertIn("surface_kind > 10.5", shader)
        self.assertIn("progress_milli", source)
        self.assertIn("army_commander_plus_member", (
            ROOT / "Renderer/terrain_lab/build_l20_unit_scenario.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("facing = 4 if kind == 7", (
            ROOT / "Renderer/terrain_lab/build_l20_unit_scenario.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("C3X_LAB_COMPOUND_UNITS", runner)
        self.assertIn("C3X_L20_MODES", runner)
        self.assertNotIn("launch", runner.lower())

    def test_canonical_owner_color_authoring_is_component_local(self) -> None:
        overrides = json.loads((
            ROOT / "Renderer/tools/asset_compiler/unit_family_owner_color_overrides.json"
        ).read_text(encoding="utf-8"))["overrides"]
        self.assertEqual("none", overrides["unit/catapult_operator/hair"]["mode"])
        self.assertEqual(0.82, overrides["unit/swordsman/shield"]["strength"])
        self.assertNotIn("unit/tank_vehicle/teamcolor", overrides)
        tank_component = json.loads((
            ROOT / "Renderer/packs/CompoundUnitLab/units/components/tank_vehicle_teamcolor.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual("TeamColor", tank_component["role"])
        self.assertEqual("USE_CIV_COLOR", tank_component["tint"])
        self.assertEqual("solid_color", tank_component["owner_color"]["mode"])
        self.assertEqual("constant_one", tank_component["owner_color"]["mask_source"])
        tank_recipe = json.loads((
            ROOT / "Renderer/packs/CompoundUnitLab/units/tank_composition.json"
        ).read_text(encoding="utf-8"))
        tank_team_color = next(
            component
            for component in tank_recipe["nodes"]["vehicle"]["components"]
            if component["role"] == "TeamColor"
        )
        self.assertEqual("tankAll", tank_team_color["attachment_bone"])
        self.assertEqual(0.35, overrides["unit/infantry/armor"]["strength"])
        self.assertEqual(
            "authored_mask",
            overrides["unit/great_general_classical_rider/armor"]["mode"],
        )
        runtime = (
            ROOT / "Renderer/tools/asset_compiler/build_l20_unit_runtime.py"
        ).read_text(encoding="utf-8")
        compound = (
            ROOT / "Renderer/tools/asset_compiler/build_l20_compound_unit_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def owner_color_style", runtime)
        self.assertIn("def source_tint_style", runtime)
        self.assertIn('"Vehicle_Woodland": 6', runtime)
        self.assertIn('"Infantry_European": 5', runtime)
        self.assertIn("OWNER_COLOR_OVERRIDES.get(asset_id", runtime)
        self.assertIn("owner_color_style(record[\"asset\"]", compound)
        self.assertIn("source_tint_style(document)", compound)
        self.assertIn('record.get("attachment_bone"', compound)


if __name__ == "__main__":
    unittest.main()
