from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import city_adjunct_asset_importer as importer


class CityAdjunctAssetImporterTests(unittest.TestCase):
    def test_default_mapping_has_capital_and_complete_wall_roles(self) -> None:
        mapping = importer.load_mapping()
        self.assertEqual(len(mapping["assets"]), 33)
        self.assertEqual(
            mapping["capital_probe"]["status"],
            "composition_marker_not_terminal_asset",
        )
        for era in ("ancient", "medieval", "industrial"):
            roles = {
                item["role"]
                for item in mapping["assets"]
                if item["kind"] == "wall_piece" and item["era"] == era
            }
            self.assertEqual(roles, {"half", "segment", "gate", "tower"})
        self.assertEqual(mapping["kit_usage"]["modern_barricade"]["role"],
                         "non_city_candidate")
        self.assertNotIn("modern_barricade", mapping["era_selection"].values())
        self.assertEqual(mapping["derived_kits"]["modern_clean"]["source_kit"],
                         "modern_barricade")
        self.assertEqual(mapping["derived_kits"]["modern_low"]["source_kit"],
                         "modern_clean")
        self.assertEqual(mapping["era_selection"],
                         {"0": "ancient", "1": "medieval", "2": "tsikhe", "3": "modern_low"})
        self.assertEqual(mapping["population_wall_rule"], "town_only")

    def test_rejects_duplicate_runtime_asset_ids(self) -> None:
        mapping = importer.load_mapping()
        mapping["assets"][1]["asset_id"] = mapping["assets"][0]["asset_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                importer.load_mapping(path)


if __name__ == "__main__":
    unittest.main()
