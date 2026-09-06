from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.tile_object_asset_importer import (
    DEFAULT_STRATEGY,
    _dependency_asset_id,
    goody_variant_for_bucket,
    load_strategy,
)


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
