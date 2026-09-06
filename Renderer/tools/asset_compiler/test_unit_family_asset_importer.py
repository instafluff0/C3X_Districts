from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.unit_family_asset_importer import (
    _initial_entry,
    load_action_contract,
    load_owner_color_contract,
    load_owner_color_overrides,
    load_strategy,
)


class UnitFamilyAssetImporterTests(unittest.TestCase):
    def test_strategy_spans_unique_non_warrior_archetypes_and_basic_actions(self) -> None:
        strategy = load_strategy()
        self.assertEqual(
            ["archer", "swordsman", "infantry", "fighter", "galley"],
            [unit["slug"] for unit in strategy["units"]],
        )
        self.assertEqual(5, len({unit["archetype"] for unit in strategy["units"]}))
        self.assertEqual(
            44,
            sum(
                len(unit["actions"]) + len(unit.get("additional_actions", {}))
                for unit in strategy["units"]
            ),
        )
        self.assertEqual(8, strategy["runtime"]["direction_count"])
        self.assertEqual("unit_body_only", strategy["runtime"]["body_ownership"])
        self.assertEqual("not_enabled", strategy["runtime_integration"])
        fighter = next(unit for unit in strategy["units"] if unit["slug"] == "fighter")
        self.assertEqual("idle", fighter["additional_actions"]["fidget"]["alias"])
        self.assertEqual("turn_right", fighter["additional_actions"]["defend"]["alias"])

    def test_basic_action_contract_aliases_attack_slots_and_keeps_defend_event_driven(self) -> None:
        contract = load_action_contract()
        self.assertEqual(
            {"idle", "fidget", "move", "fortify", "attack", "defend", "victory", "death"},
            set(contract["actions"]),
        )
        self.assertEqual(
            ["ATTACK1", "ATTACK2", "ATTACK3"],
            contract["actions"]["attack"]["civ3_slots"],
        )
        self.assertEqual([], contract["actions"]["defend"]["civ3_slots"])
        self.assertEqual("not_enabled", contract["runtime_integration"])

    def test_import_profiles_may_reuse_an_archetype_for_arbitrary_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            strategy = load_strategy()
            strategy["units"][1]["archetype"] = strategy["units"][0]["archetype"]
            path = Path(temporary) / "strategy.json"
            path.write_text(json.dumps(strategy), encoding="utf-8")
            loaded = load_strategy(path)
            self.assertEqual(
                loaded["units"][0]["archetype"], loaded["units"][1]["archetype"]
            )

    def test_initial_entry_requires_a_unique_package_string(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "unit.blp"
            package.write_bytes(b"Duplicate\0Unique\0Duplicate\0")
            self.assertEqual("Unique", _initial_entry(package, ["Duplicate", "Unique"]))
            with self.assertRaisesRegex(ValueError, "no unique initialization entry"):
                _initial_entry(package, ["Duplicate"])

    def test_owner_color_contract_selects_loaded_scenario_palette_at_runtime(self) -> None:
        contract = load_owner_color_contract()
        self.assertFalse(contract["asset_conversion"]["bake_owner_variants"])
        self.assertEqual(
            "display_color_table_id", contract["runtime_selection"]["instance_selector"]
        )
        self.assertTrue(
            contract["runtime_selection"]["do_not_assume_owner_equals_display_civ"]
        )
        self.assertEqual([64, 32], [contract["gpu_lut"]["width"], contract["gpu_lut"]["height"]])
        self.assertEqual(
            "effective_tables_already_loaded_by_civ3",
            contract["scenario_policy"]["source"],
        )
        self.assertEqual(
            "update_instance_selector_only",
            contract["invalidation"]["owner_or_display_identity_change"],
        )
        self.assertEqual(
            "per_material_component_not_per_unit_code",
            contract["authoring"]["scope"],
        )
        self.assertEqual(
            "unit_name_or_unit_type_branches_in_runtime_shader",
            contract["authoring"]["forbidden"],
        )
        self.assertEqual(
            "needs_pack_authoring_override", contract["coverage_gate"]["below_threshold"]
        )

    def test_owner_color_overrides_are_generic_component_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "owner_colors.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "c3x.unit_owner_color_overrides.v0",
                        "overrides": {
                            "unit/modded_scout/cape": {
                                "mode": "authored_mask",
                                "mask_source": "constant_one",
                                "strength": 0.45,
                                "representative_palette_index": 6,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            overrides = load_owner_color_overrides(path)
            self.assertEqual(0.45, overrides["unit/modded_scout/cape"]["strength"])


if __name__ == "__main__":
    unittest.main()
