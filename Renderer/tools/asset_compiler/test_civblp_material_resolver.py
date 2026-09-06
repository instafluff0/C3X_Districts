#!/usr/bin/env python3
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import civblp_material_resolver as resolver


class CivblpMaterialResolverTests(unittest.TestCase):
    def make_record(
        self,
        dxgi_format: int = 78,
        width: int = 16,
        height: int = 8,
        mip_count: int = 3,
        relative_offset: int = 64,
    ) -> bytes:
        record = bytearray(104)
        block_bytes = resolver.FORMAT_INFO[dxgi_format][2]
        byte_count = resolver.expected_bc_bytes(width, height, mip_count, block_bytes)
        struct.pack_into("<QQ", record, 0x20, relative_offset, byte_count)
        struct.pack_into("<6H", record, 0x58, dxgi_format, height, width, 1, 1, mip_count)
        return bytes(record)

    def test_every_material_class_has_a_distinct_role(self) -> None:
        self.assertEqual(
            resolver.ROLE_BY_TEXTURE_CLASS,
            {
                "Terrain_BaseColor": "base_color",
                "Terrain_Heightmap": "height",
                "Terrain_Spec": "specular",
                "Terrain_FOWColor": "fow_color",
            },
        )

    def test_metadata_preserves_format_and_color_space(self) -> None:
        srgb = resolver.decode_texture_metadata(self.make_record(78), 0x58)
        linear = resolver.decode_texture_metadata(self.make_record(80), 0x58)

        self.assertEqual((srgb["format"]["name"], srgb["format"]["color_space"]), ("BC3_UNORM_SRGB", "srgb"))
        self.assertEqual((linear["format"]["name"], linear["format"]["color_space"]), ("BC4_UNORM", "linear"))
        self.assertEqual((srgb["width"], srgb["height"], srgb["mip_count"]), (16, 8, 3))

    def test_layout_inference_uses_all_texture_records(self) -> None:
        records = [self.make_record(78, relative_offset=64), self.make_record(80, relative_offset=512)]

        metadata_offset, metadata_evidence = resolver.infer_texture_metadata_offset(records)
        offset_field, size_field, storage_evidence = resolver.infer_embedded_resource_fields(
            records, metadata_offset, 4096
        )

        self.assertEqual(metadata_offset, 0x58)
        self.assertEqual((offset_field, size_field), (0x20, 0x28))
        self.assertEqual(metadata_evidence[0]["validated_records"], 2)
        self.assertEqual(storage_evidence[0]["validated_records"], 2)

    def test_embedded_resource_inference_rejects_bad_size(self) -> None:
        record = bytearray(self.make_record())
        struct.pack_into("<Q", record, 0x28, 1)

        with self.assertRaisesRegex(ValueError, "embedded-resource layout"):
            resolver.infer_embedded_resource_fields([bytes(record)], 0x58, 4096)

    def test_expected_bc_size_includes_each_declared_mip(self) -> None:
        self.assertEqual(resolver.expected_bc_bytes(8, 8, 2, 16), 80)
        self.assertEqual(resolver.expected_bc_bytes(8, 8, 2, 8), 40)


if __name__ == "__main__":
    unittest.main()
