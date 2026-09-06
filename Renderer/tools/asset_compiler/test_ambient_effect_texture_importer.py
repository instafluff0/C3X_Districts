from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import ambient_effect_texture_importer as importer


class AmbientEffectTextureImporterTests(unittest.TestCase):
    def test_mapping_uses_generic_unique_asset_ids(self) -> None:
        mapping = importer.load_mapping(importer.DEFAULT_MAPPING)
        ids = [item["asset_id"] for item in mapping["textures"]]
        self.assertEqual(len(ids), 13)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("effect/fire/torch_sheet", ids)
        self.assertIn("effect/smoke/cloud", ids)
        self.assertEqual(5, sum(value.startswith("effect/pollution/") for value in ids))

    def test_rejects_duplicate_source_entries(self) -> None:
        mapping = {
            "schema": "c3x.source_ambient_effect_texture_mapping.v0",
            "source_root": "source",
            "textures": [
                {"source_entry": "Same", "asset_id": "effect/a", "usage": "mask"},
                {"source_entry": "Same", "asset_id": "effect/b", "usage": "mask"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                importer.load_mapping(path)


if __name__ == "__main__":
    unittest.main()
