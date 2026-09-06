from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import combat_effect_texture_importer as importer


class CombatEffectTextureImporterTests(unittest.TestCase):
    def test_mapping_covers_conventional_and_nuclear_effect_families(self) -> None:
        mapping = importer.load_mapping(importer.DEFAULT_MAPPING)
        ids = [item["asset_id"] for item in mapping["textures"]]
        self.assertEqual(22, len(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("effect/combat/projectile/artillery_shell", ids)
        self.assertIn("effect/combat/impact/water_wave", ids)
        self.assertIn("effect/combat/nuclear/scorch", ids)

    def test_rejects_source_specific_runtime_id(self) -> None:
        mapping = {
            "schema": "c3x.source_combat_effect_texture_mapping.v0",
            "source_root": "Base/Platforms/Windows/BLPs/SHARED_DATA",
            "textures": [
                {"source_entry": "Source", "asset_id": "civ6/effect", "usage": "mask"}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source-independent"):
                importer.load_mapping(path)


if __name__ == "__main__":
    unittest.main()
