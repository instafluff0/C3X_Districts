from __future__ import annotations

import struct
import unittest

from Renderer.tools.asset_compiler import terrain_relief_builder


class TerrainReliefBuilderTests(unittest.TestCase):
    def test_fnv_matches_installed_resource_evidence(self) -> None:
        self.assertEqual(0x9C7C0EAA, terrain_relief_builder.fnv1a32("TER_Hills_Standard_Element_1"))
        self.assertEqual(0x07A76DB4, terrain_relief_builder.fnv1a32("Mountain_Single_01_HM_0"))

    def test_r8_dds_is_bounded_and_source_independent(self) -> None:
        pixels = bytes(range(16))
        dds = terrain_relief_builder.make_r8_dds(4, 4, pixels)
        self.assertEqual(b"DDS ", dds[:4])
        self.assertEqual(124, struct.unpack_from("<I", dds, 4)[0])
        self.assertEqual((4, 4), struct.unpack_from("<II", dds, 12))
        self.assertEqual(b"DX10", dds[84:88])
        self.assertEqual(61, struct.unpack_from("<I", dds, 128)[0])
        self.assertEqual(pixels, dds[148:])

    def test_r8_dds_rejects_mismatched_payload(self) -> None:
        with self.assertRaises(ValueError):
            terrain_relief_builder.make_r8_dds(4, 4, b"short")

    def test_discrete_region_dds_uses_r8_uint(self) -> None:
        dds = terrain_relief_builder.make_r8_dds(2, 2, bytes((0, 51, 102, 153)), 62)
        self.assertEqual(62, struct.unpack_from("<I", dds, 128)[0])

    def test_channel_summary_distinguishes_discrete_region_data(self) -> None:
        summary = terrain_relief_builder.summarize_channel(
            bytes((0, 0, 51, 51, 102, 102, 153, 153, 0, 0, 51, 51, 102, 102, 153, 153)),
            4,
            4,
        )
        self.assertEqual(4, summary["unique_values"])
        self.assertEqual([0, 51, 102, 153], summary["values"])
        self.assertEqual(153, summary["maximum"])

    def test_lod_relationship_requires_correlated_two_to_one_pair(self) -> None:
        high = bytes((
            0, 0, 10, 10,
            0, 0, 10, 10,
            20, 20, 30, 30,
            20, 20, 30, 30,
        ))
        relationship = terrain_relief_builder.compare_lod_pair(high, 4, bytes((0, 10, 20, 30)), 2)
        self.assertEqual("lower_resolution_lod", relationship["interpretation"])
        self.assertEqual("high", relationship["confidence"])
        self.assertAlmostEqual(1.0, relationship["box_downsample_correlation"])
        self.assertAlmostEqual(0.0, relationship["box_downsample_mean_absolute_error"])

    def test_authored_relief_inventory_preserves_source_families_and_channels(self) -> None:
        self.assertEqual(5, len(terrain_relief_builder.MOUNTAIN_FAMILIES["standard"]))
        self.assertEqual(4, len(terrain_relief_builder.MOUNTAIN_FAMILIES["desert"]))
        self.assertEqual(("HM", "HBLEND", "ID"), terrain_relief_builder.MOUNTAIN_CHANNELS)
        self.assertEqual(
            {"standard", "continental", "continental_plains", "continental_snow"},
            set(terrain_relief_builder.HILL_FAMILIES),
        )


if __name__ == "__main__":
    unittest.main()
