from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.generic_terrain_element_compiler import load_mapping
from Renderer.tools.asset_compiler.terrain_relief_builder import (
    decode_terrain_element_record,
    fnv1a32,
)


class GenericTerrainElementCompilerTests(unittest.TestCase):
    def test_mapping_rejects_duplicate_normalized_ids(self) -> None:
        mapping = {
            "schema": "c3x.source_terrain_element_mapping.v0",
            "source_package": "DLC/Test/TerrainElementSet.blp",
            "elements": [
                {"source_entry": "SOURCE_A", "asset_id": "terrain/feature/test"},
                {"source_entry": "SOURCE_B", "asset_id": "terrain/feature/test"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
                load_mapping(path)

    def test_typed_record_preserves_present_and_absent_channels(self) -> None:
        name = "ART_DEF_TERRAIN_ELEMENT_FEATURE_TEST"
        raw = bytearray(144)
        struct.pack_into("<I", raw, 0x40, fnv1a32(name))
        struct.pack_into("<3I", raw, 0x48, 0, 10, 11)
        struct.pack_into("<3I", raw, 0x54, 0, 20, 21)
        struct.pack_into("<3I", raw, 0x60, 0, 0, 0)
        struct.pack_into("<3I", raw, 0x6C, 0, 0, 0)
        struct.pack_into("<3f", raw, 0x78, 0.25, 2.0, 18.0)
        struct.pack_into("<2I", raw, 0x84, 512, 512)
        resources = {
            10: {"index": 10, "name": "height0", "width": 256, "height": 256},
            11: {"index": 11, "name": "height1", "width": 128, "height": 128},
            20: {"index": 20, "name": "blend0", "width": 256, "height": 256},
            21: {"index": 21, "name": "blend1", "width": 128, "height": 128},
        }
        element = decode_terrain_element_record(bytes(raw), name, resources)
        self.assertEqual({"height", "blend"}, set(element["channels"]))
        self.assertEqual([512, 512], element["grid_dimensions"])
        self.assertEqual(18.0, element["parameters"]["height_scale"])

    def test_typed_record_rejects_half_present_lod_pair(self) -> None:
        name = "ART_DEF_TERRAIN_ELEMENT_FEATURE_TEST"
        raw = bytearray(144)
        struct.pack_into("<I", raw, 0x40, fnv1a32(name))
        struct.pack_into("<3I", raw, 0x48, 0, 10, 0)
        struct.pack_into("<3f", raw, 0x78, 0.0, 0.0, 10.0)
        struct.pack_into("<2I", raw, 0x84, 512, 512)
        with self.assertRaisesRegex(ValueError, "LOD vector is malformed"):
            decode_terrain_element_record(bytes(raw), name, {})

    def test_typed_record_rejects_wrong_lod_dimensions(self) -> None:
        name = "ART_DEF_TERRAIN_ELEMENT_FEATURE_TEST"
        raw = bytearray(144)
        struct.pack_into("<I", raw, 0x40, fnv1a32(name))
        struct.pack_into("<3I", raw, 0x48, 0, 10, 11)
        struct.pack_into("<3f", raw, 0x78, 0.0, 0.0, 10.0)
        struct.pack_into("<2I", raw, 0x84, 512, 512)
        resources = {
            10: {"index": 10, "name": "height0", "width": 512, "height": 512},
            11: {"index": 11, "name": "height1", "width": 128, "height": 128},
        }
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            decode_terrain_element_record(bytes(raw), name, resources)


if __name__ == "__main__":
    unittest.main()
