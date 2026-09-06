from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.compound_landmark_importer import (
    MATERIAL_TEXTURE_SLOTS,
    _attachment_semantic,
    _decode_terrain_edit,
    _normalize_geometry,
    decode_granny_bone,
    decode_skin_influences,
    decode_state_mask,
    load_mapping,
    validate_bind_pose,
)


class CompoundLandmarkImporterTests(unittest.TestCase):
    def test_checked_in_infrastructure_probe_has_five_unique_roots(self) -> None:
        mapping = load_mapping(Path(__file__).with_name("infrastructure_source_sets.json"))
        assets = [asset for package in mapping["packages"] for asset in package["assets"]]
        self.assertEqual(5, len(assets))
        self.assertEqual(5, len({asset["asset_id"] for asset in assets}))
        self.assertEqual(
            {"IMP_Fort_Medieval_Base", "IMP_Fort_Industrial_Base", "IMP_Airstrip",
             "IMP_Airstrip_Tower", "VIL_BAR_IND_Tower"},
            {asset["source_entry"] for asset in assets},
        )

    def test_terrain_edit_policy_rejects_unknown_modes_before_reading(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported compound terrain-edit policy"):
            _decode_terrain_edit(None, 1, "interpret")

    def test_city_material_and_attachment_extensions_are_generic(self) -> None:
        self.assertEqual((0x3C, "Generic_Emissive", False), MATERIAL_TEXTURE_SLOTS["emissive"])
        self.assertEqual("smoke", _attachment_semantic("ATT_DIS_CTY_Smoke_Chimney"))
        self.assertEqual("flame", _attachment_semantic("ATT_DIS_CTY_Torch"))
        self.assertEqual("night_light", _attachment_semantic("ATT_DIS_CTY_Lantern"))
        self.assertEqual("unresolved", _attachment_semantic("ATT_DIS_CTY_Unknown"))

    def test_state_mask_preserves_compound_state_selection(self) -> None:
        states = ["construction", "pillaged", "unbuilt", "unworked", "worked"]
        self.assertEqual(
            ["construction", "unbuilt", "unworked", "worked"],
            decode_state_mask(0b11101, states),
        )
        self.assertEqual(["pillaged"], decode_state_mask(0b00010, states))
        with self.assertRaisesRegex(ValueError, "outside the state table"):
            decode_state_mask(0b100000, states)

    def test_skin_influences_are_compacted_and_normalized(self) -> None:
        vertex = bytearray(32)
        vertex[12:16] = bytes((3, 7, 9, 0))
        vertex[16:20] = bytes((128, 64, 63, 0))
        skin = decode_skin_influences(bytes(vertex), 0, 10)
        self.assertEqual([3, 7, 9], skin["bone_indices"])
        self.assertAlmostEqual(1.0, sum(skin["bone_weights"]))
        with self.assertRaisesRegex(ValueError, "sum to 255"):
            decode_skin_influences(bytes(32), 0, 10)
        vertex[12:16] = bytes((10, 0, 0, 0))
        vertex[16:20] = bytes((255, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "invalid bone"):
            decode_skin_influences(bytes(vertex), 0, 10)

    def test_unit_adapter_can_normalize_nonzero_quantized_weight_sums(self) -> None:
        vertex = bytearray(32)
        vertex[12:16] = bytes((2, 5, 0, 0))
        vertex[16:20] = bytes((128, 126, 0, 0))
        with self.assertRaisesRegex(ValueError, "sum to 255"):
            decode_skin_influences(bytes(vertex), 0, 10)
        skin = decode_skin_influences(
            bytes(vertex), 0, 10, normalize_nonzero_weights=True
        )
        self.assertEqual(254, skin["source_weight_sum"])
        self.assertAlmostEqual(1.0, sum(skin["bone_weights"]))

    def test_rigid_vertex_profile_cannot_be_misread_as_skin(self) -> None:
        vertex_bytes = bytearray(3 * 24)
        for index, position in enumerate(((0, 0, 0), (1, 0, 0), (0, 1, 0))):
            struct.pack_into("<3e", vertex_bytes, index * 24, *position)
            struct.pack_into("<2e", vertex_bytes, index * 24 + 8, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "proven skin vertex profile"):
            _normalize_geometry(
                bytes(vertex_bytes),
                struct.pack("<3H", 0, 1, 2),
                {"format": 0x315CFCD9, "stride": 24, "count": 3},
                {"bytes_per_index": 2, "count": 3},
                {
                    "first_index": 0,
                    "index_count": 3,
                    "base_vertex": 0,
                    "vertex_count": 3,
                },
                1.0,
                1,
            )
        with self.assertRaisesRegex(ValueError, "declared vertex range"):
            _normalize_geometry(
                bytes(vertex_bytes),
                struct.pack("<3H", 0, 1, 2),
                {"format": 0x315CFCD9, "stride": 24, "count": 3},
                {"bytes_per_index": 2, "count": 3},
                {
                    "first_index": 0,
                    "index_count": 3,
                    "base_vertex": 0,
                    "vertex_count": 2,
                },
                1.0,
                None,
            )

    def test_degenerate_source_triangles_are_omitted_when_valid_geometry_remains(self) -> None:
        vertex_bytes = bytearray(3 * 24)
        for index, position in enumerate(((0, 0, 0), (1, 0, 0), (0, 1, 0))):
            struct.pack_into("<3e", vertex_bytes, index * 24, *position)
            struct.pack_into("<2e", vertex_bytes, index * 24 + 8, 0.0, 0.0)
        mesh, evidence = _normalize_geometry(
            bytes(vertex_bytes),
            struct.pack("<6H", 0, 0, 0, 0, 1, 2),
            {"format": 0x315CFCD9, "stride": 24, "count": 3},
            {"bytes_per_index": 2, "count": 6},
            {
                "first_index": 0,
                "index_count": 6,
                "base_vertex": 0,
                "vertex_count": 3,
            },
            1.0,
            None,
        )
        self.assertEqual(mesh["topology"]["indices"], [0, 1, 2])
        self.assertEqual(evidence["source_triangles"], 2)
        self.assertEqual(evidence["omitted_degenerate_triangles"], 1)

    def test_bone_normalizes_spatial_values_but_not_rotation(self) -> None:
        raw = bytearray(164)
        struct.pack_into("<iI", raw, 0x08, -1, 0)
        struct.pack_into("<3f", raw, 0x10, 100.0, -50.0, 25.0)
        struct.pack_into("<4f", raw, 0x1C, 0.0, 0.0, 0.0, 1.0)
        struct.pack_into("<9f", raw, 0x2C, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        inverse = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0, -100.0, 50.0, -25.0, 1.0]
        struct.pack_into("<16f", raw, 0x50, *inverse)
        struct.pack_into("<f", raw, 0x90, 10.0)
        bone = decode_granny_bone(bytes(raw), "root", 0, 1, 100.0)
        self.assertEqual([1.0, -0.5, 0.25], bone["rest"]["position"])
        self.assertEqual([-1.0, 0.5, -0.25], bone["inverse_bind_matrix"][12:15])
        self.assertEqual([0.0, 0.0, 0.0, 1.0], bone["rest"]["orientation"])
        self.assertEqual(0.1, bone["lod_error"])
        self.assertAlmostEqual(0.0, validate_bind_pose([bone]))

    def test_mapping_rejects_duplicate_runtime_ids(self) -> None:
        mapping = {
            "schema": "c3x.source_compound_landmark_mapping.v0",
            "packages": [{
                "source_package": "Base/package.blp",
                "shared_data": "Base/shared",
                "source_units_per_tile": 100,
                "assets": [
                    {"source_entry": "A", "asset_id": "resource/test"},
                    {"source_entry": "B", "asset_id": "resource/test"},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
                load_mapping(path)

    def test_mapping_accepts_ordered_shared_texture_roots(self) -> None:
        mapping = {
            "schema": "c3x.source_compound_landmark_mapping.v0",
            "packages": [{
                "source_package": "DLC/package.blp",
                "shared_data": ["DLC/SHARED_DATA", "Base/SHARED_DATA"],
                "source_units_per_tile": 100,
                "assets": [{"source_entry": "Bridge", "asset_id": "route/bridge/test"}],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            self.assertEqual(mapping, load_mapping(path))


if __name__ == "__main__":
    unittest.main()
