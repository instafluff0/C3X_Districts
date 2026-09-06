from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import analytic_light_importer as importer


class AnalyticLightImporterTests(unittest.TestCase):
    def test_mapping_covers_twelve_production_lights_and_explicit_exclusions(self) -> None:
        mapping = importer.load_mapping()
        self.assertEqual(12, len(mapping["lights"]))
        self.assertEqual(4, len(mapping["excluded_source_entries"]))
        self.assertEqual(
            16,
            len(
                {item["source_entry"] for item in mapping["lights"]}
                | {item["source_entry"] for item in mapping["excluded_source_entries"]}
            ),
        )
        self.assertEqual(
            {"architectural_glow", "beacon", "vehicle_light", "natural_glow"},
            {item["family"] for item in mapping["lights"]},
        )

    def test_mapping_rejects_duplicate_runtime_ids(self) -> None:
        mapping = importer.load_mapping()
        mapping["lights"][1]["asset_id"] = mapping["lights"][0]["asset_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                importer.load_mapping(path)


if __name__ == "__main__":
    unittest.main()
