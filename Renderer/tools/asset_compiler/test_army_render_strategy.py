from __future__ import annotations

import unittest

from Renderer.tools.asset_compiler.army_render_strategy import compose_army, load_strategy


class ArmyRenderStrategyTests(unittest.TestCase):
    def test_profiles_use_dedicated_great_general_art(self) -> None:
        strategy = load_strategy()
        self.assertEqual(
            {"UNIT_GREAT_GENERAL", "UNIT_GREAT_GENERAL_MODERN"},
            {profile["source_unit"] for profile in strategy["era_profiles"].values()},
        )
        self.assertFalse(strategy["commander_presentation"]["whole_body_owner_tint"])

    def test_loaded_army_composes_arbitrary_member_and_commander(self) -> None:
        result = compose_army({
            "army_id": 41,
            "era": "industrial",
            "army_anchor": [140, 88],
            "army_action": "idle",
            "army_direction": 3,
            "displayed_member": {
                "unit_id": 99,
                "unit_type": "scenario/clockwork_elephant",
                "anchor": [100, 88],
                "action": "attack",
                "direction": 3,
            },
        })
        self.assertEqual(["displayed_member", "commander"], [child["role"] for child in result["children"]])
        self.assertEqual(
            {"unit_type": "scenario/clockwork_elephant"},
            result["children"][0]["asset_selector"],
        )
        self.assertEqual("unit/army/commander/modern_foot", result["children"][1]["asset_id"])
        self.assertEqual(1, result["retained_parent_hud_instances"])

    def test_empty_army_is_commander_only(self) -> None:
        result = compose_army({
            "army_id": 7,
            "era": "ancient",
            "army_anchor": [40, 40],
            "army_action": "idle",
            "army_direction": 0,
        })
        self.assertEqual(["commander"], [child["role"] for child in result["children"]])

    def test_displayed_member_change_replaces_only_member_child_identity(self) -> None:
        base = {
            "army_id": 8,
            "era": "modern",
            "army_anchor": [200, 100],
            "army_action": "defend",
            "army_direction": 5,
        }
        first = compose_army({**base, "displayed_member": {
            "unit_id": 11, "unit_type": "PRTO_Infantry", "anchor": [160, 100],
            "action": "defend", "direction": 5,
        }})
        second = compose_army({**base, "displayed_member": {
            "unit_id": 12, "unit_type": "PRTO_Tank", "anchor": [160, 100],
            "action": "attack", "direction": 5,
        }})
        self.assertNotEqual(first["children"][0]["instance_id"], second["children"][0]["instance_id"])
        self.assertEqual(first["children"][1]["instance_id"], second["children"][1]["instance_id"])


if __name__ == "__main__":
    unittest.main()
