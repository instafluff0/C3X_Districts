from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.tile_object_asset_importer import (
    DEFAULT_STRATEGY,
    _dependency_asset_id,
    barbarian_variant_for,
    goody_variant_for_bucket,
    load_strategy,
)
from Renderer.tools.asset_compiler.build_tile_object_runtime import root_bindings


class TileObjectAssetImporterTests(unittest.TestCase):
    def test_checked_strategy_preserves_native_hut_and_colony_selectors(self) -> None:
        strategy = load_strategy(DEFAULT_STRATEGY)
        self.assertEqual(8, len(strategy["goody_hut"]["runtime"]["bucket_to_variant"]))
        self.assertEqual(
            list(range(4)),
            sorted(era for item in strategy["colony"]["eras"] for era in item["civ3_eras"]),
        )
        self.assertEqual("Colony_Body.OwnerID", strategy["colony"]["runtime"]["owner_source"])
        self.assertTrue(strategy["colony"]["runtime"]["territory_owner_is_not_colony_owner"])
        barbarian = strategy["barbarian_camp"]
        self.assertEqual(
            ["VIL_BAR_01", "VIL_BAR_IND"],
            [entry for stage in barbarian["source_stages"] for entry in stage["source_entries"]],
        )
        self.assertEqual("preindustrial", barbarian["runtime"]["default_stage"])
        self.assertEqual("none", barbarian["runtime"]["owner_color"])
        self.assertEqual(
            "preserve_authoritative_civ3_resource_visibility",
            barbarian["runtime"]["resource_policy"],
        )
        infrastructure = strategy["infrastructure"]
        self.assertEqual(
            {"fortress", "barricade", "airfield", "outpost"},
            set(infrastructure["families"]),
        )
        self.assertEqual(5, len(infrastructure["source_assets"]))
        self.assertEqual(
            {"radar_tower", "pollution", "crater", "victory_location"},
            set(infrastructure["l19b_promoted_families"]),
        )
        self.assertTrue(all(
            len(family["civ3_era_assets"]) == 4
            for family in infrastructure["families"].values()
        ))

    def test_all_hut_buckets_resolve_to_compiled_semantic_assets(self) -> None:
        strategy = load_strategy(DEFAULT_STRATEGY)
        resolved = [goody_variant_for_bucket(strategy, bucket) for bucket in range(8)]
        self.assertEqual(3, len(set(resolved)))
        self.assertTrue(all(value.startswith("tile_object/goody_hut/") for value in resolved))
        with self.assertRaisesRegex(ValueError, "outside"):
            goody_variant_for_bucket(strategy, 8)

    def test_colony_cannot_use_territory_owner_or_hide_the_resource(self) -> None:
        strategy = copy.deepcopy(load_strategy(DEFAULT_STRATEGY))
        strategy["colony"]["runtime"]["owner_source"] = "Tile.Territory_OwnerID"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strategy.json"
            path.write_text(json.dumps(strategy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "colony body"):
                load_strategy(path)

    def test_dependency_ids_are_stable_and_source_neutral(self) -> None:
        first = _dependency_asset_id("Base/private.blp", "Named_Source_Component")
        self.assertEqual(first, _dependency_asset_id("Base/private.blp", "Named_Source_Component"))
        self.assertRegex(first, r"^tile_object/component/[0-9a-f]{16}$")
        self.assertNotIn("Source", first)

    def test_barbarian_camp_selection_is_stable_and_stage_specific(self) -> None:
        strategy = load_strategy(DEFAULT_STRATEGY)
        primitive = barbarian_variant_for(strategy, "preindustrial", 71, 222, 45)
        self.assertEqual(primitive, barbarian_variant_for(strategy, "preindustrial", 71, 222, 45))
        self.assertEqual(
            "tile_object/barbarian_camp/preindustrial/vil_bar_01",
            primitive,
        )
        self.assertEqual(
            "tile_object/barbarian_camp/industrial_optional/vil_bar_ind",
            barbarian_variant_for(strategy, "industrial_optional", 71, 222, 45),
        )
        with self.assertRaisesRegex(ValueError, "stage"):
            barbarian_variant_for(strategy, "missing", 71, 222, 45)

    def test_barbarian_camp_cannot_inherit_owner_color_or_hide_resources(self) -> None:
        strategy = copy.deepcopy(load_strategy(DEFAULT_STRATEGY))
        strategy["barbarian_camp"]["runtime"]["owner_color"] = "territory_owner"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strategy.json"
            path.write_text(json.dumps(strategy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not inherit"):
                load_strategy(path)

    def test_runtime_bundle_keeps_barbarian_camps_separate_from_colonies(self) -> None:
        catalog = {
            "goody_hut": {"variants": ["hut/a"]},
            "barbarian_camp": {
                "stages": [
                    {"variants": ["camp/primitive"]},
                    {"variants": ["camp/industrial"]},
                ]
            },
            "colony": {
                "eras": [
                    {"variants": ["colony/a", "colony/b", "colony/c"]},
                    {"variants": ["unused"]},
                    {"variants": ["colony/d", "colony/e", "colony/f"]},
                ]
            },
        }
        bindings = root_bindings(catalog)
        self.assertEqual(
            ["barbarian_camp_0", "barbarian_camp_1"],
            [role for role, _asset in bindings if role.startswith("barbarian_camp_")],
        )
        self.assertTrue(all(
            not asset.startswith("colony/")
            for role, asset in bindings
            if role.startswith("barbarian_camp_")
        ))

    def test_infrastructure_family_cannot_reference_an_uncompiled_asset(self) -> None:
        strategy = copy.deepcopy(load_strategy(DEFAULT_STRATEGY))
        strategy["infrastructure"]["families"]["outpost"]["civ3_era_assets"][0] = (
            "infrastructure/outpost/missing"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strategy.json"
            path.write_text(json.dumps(strategy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "four Civ III eras"):
                load_strategy(path)


if __name__ == "__main__":
    unittest.main()
