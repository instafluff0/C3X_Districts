from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.palace_asset_importer import (
    DEFAULT_STRATEGY,
    _dependency_asset_id,
    _palace_asset_id,
    load_strategy,
)


class PalaceAssetImporterTests(unittest.TestCase):
    def test_checked_strategy_is_source_neutral_and_offline_only(self) -> None:
        strategy = load_strategy(DEFAULT_STRATEGY)
        inventory = strategy["inventory"]
        self.assertEqual(48, inventory["expected_standard_bindings"])
        self.assertEqual(47, inventory["expected_unique_standard_assets"])
        self.assertFalse(inventory["include_scenarios"])
        runtime = strategy["runtime_selection"]
        self.assertEqual("authoritative_civ3_is_capital", runtime["capital_source"])
        self.assertEqual("provenance_only", runtime["source_selectors_are"])
        self.assertFalse(runtime["hard_coded_civ6_civilization_ids"])
        self.assertEqual("not_enabled", strategy["runtime_integration"])

    def test_root_ids_distinguish_same_named_assets_in_different_content(self) -> None:
        first = _palace_asset_id(
            "DLC/Expansion1/Platforms/Windows/BLPs/landmarks/cities/creepalace.blp",
            "DIS_CTY_CREE_Palace",
        )
        second = _palace_asset_id(
            "DLC/Expansion2/Platforms/Windows/BLPs/landmarks/cities/creepalace.blp",
            "DIS_CTY_CREE_Palace",
        )
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^city/palace/root/[0-9a-f]{16}$")
        self.assertRegex(second, r"^city/palace/root/[0-9a-f]{16}$")
        self.assertNotIn("CREE", first)

    def test_dependency_ids_are_stable_and_hide_source_names(self) -> None:
        first = _dependency_asset_id("Base/private.blp", "SourcePalaceComponent")
        self.assertEqual(first, _dependency_asset_id("Base/private.blp", "SourcePalaceComponent"))
        self.assertRegex(first, r"^city/palace_component/[0-9a-f]{16}$")
        self.assertNotIn("SourcePalaceComponent", first)

    def test_strategy_rejects_civ6_runtime_selectors(self) -> None:
        strategy = copy.deepcopy(load_strategy(DEFAULT_STRATEGY))
        strategy["runtime_selection"]["source_selectors_are"] = "runtime_ids"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strategy.json"
            path.write_text(json.dumps(strategy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime selectors"):
                load_strategy(path)


if __name__ == "__main__":
    unittest.main()
