from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.improvement_asset_importer import (
    DEFAULT_STRATEGY,
    _asset_id,
    _graph_summary,
    load_strategy,
)


class ImprovementAssetImporterTests(unittest.TestCase):
    def test_checked_strategy_covers_civ3_eras_and_irrigation_masks(self) -> None:
        strategy = load_strategy(DEFAULT_STRATEGY)
        self.assertEqual(
            list(range(4)),
            sorted(era for item in strategy["mine"]["eras"] for era in item["civ3_eras"]),
        )
        self.assertEqual(
            list(range(4)),
            sorted(era for item in strategy["farm"]["eras"] for era in item["civ3_eras"]),
        )
        self.assertEqual(4, strategy["farm"]["runtime"]["civ3_adjacency_bits"])
        self.assertEqual(16, strategy["farm"]["runtime"]["civ3_adjacency_masks"])
        self.assertIn("default", {item["id"] for item in strategy["farm"]["crop_styles"]})

    def test_incomplete_era_coverage_is_rejected(self) -> None:
        strategy = copy.deepcopy(load_strategy(DEFAULT_STRATEGY))
        strategy["farm"]["eras"][0]["civ3_eras"] = []
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strategy.json"
            path.write_text(json.dumps(strategy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cover Civ III eras"):
                load_strategy(path)

    def test_asset_id_is_stable_and_hides_source_names(self) -> None:
        first = _asset_id("Base/private.blp", "IMP_Mine_Source_Name")
        self.assertEqual(first, _asset_id("Base/private.blp", "IMP_Mine_Source_Name"))
        self.assertRegex(first, r"^improvement/component/[0-9a-f]{16}$")
        self.assertNotIn("Mine", first)

    def test_graph_summary_counts_unique_map_visual_terminals(self) -> None:
        report = {
            "graphs": [
                {
                    "graph_id": "improvement/mine",
                    "nodes": [{}, {}],
                    "terminals": [
                        {"scope": "map_visual", "package_path": "a.blp", "entry": "A", "class": "TileBase"},
                        {"scope": "map_visual", "package_path": "a.blp", "entry": "A", "class": "TileBase"},
                        {"scope": "audio", "package_path": "a.blp", "entry": "S", "class": "Sound"},
                    ],
                }
            ]
        }
        summary = _graph_summary(report, "improvement/mine")
        self.assertEqual(2, summary["nodes"])
        self.assertEqual(2, summary["terminals"])
        self.assertEqual(1, summary["unique_assets"])
        self.assertEqual({"TileBase": 2}, summary["classes"])


if __name__ == "__main__":
    unittest.main()
