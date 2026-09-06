import json
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.generic_decal_compiler import (
    DEFAULT_SPEC,
    decode_decal_descriptor,
    load_mapping,
    read_artdef_group,
)


def _field(parameter: str, tag: str, value: str) -> str:
    attribute = f' text="{value}"' if tag in ("m_EntryName", "m_Value") else ""
    content = "" if attribute else value
    return (
        f"<Element><{tag}{attribute}>{content}</{tag}>"
        f'<m_ParamName text="{parameter}"/></Element>'
    )


class GenericDecalCompilerTests(unittest.TestCase):
    def test_default_mapping_covers_active_land_and_water_surface_clutter(self) -> None:
        mapping = load_mapping(DEFAULT_SPEC)
        groups = {group["group_id"]: group for group in mapping["groups"]}
        expected_counts = {
            "terrain/water/ocean_surface": 9,
            "terrain/grassland/surface": 11,
            "terrain/plains/surface": 8,
            "terrain/grassland_hills/surface": 3,
        }
        for group_id, count in expected_counts.items():
            self.assertIn(group_id, groups)
            self.assertEqual(count, len(groups[group_id]["assets"]))
        ocean_sources = {
            asset["source_asset"]
            for asset in groups["terrain/water/ocean_surface"]["assets"]
        }
        self.assertEqual(
            {f"TER_Ocean_Decal{index:02d}" for index in range(1, 6)} |
            {f"TER_Coast_Decal{index:02d}" for index in range(1, 5)},
            ocean_sources,
        )

    def test_mapping_requires_safe_unique_generic_ids(self) -> None:
        mapping = {
            "schema": "c3x.source_decal_mapping.v0",
            "source_units_per_tile": 12,
            "sources": {
                "package": "Base/source.blp",
                "shared_data": "Base/shared",
                "artdef": "Base/source.artdef",
            },
            "groups": [{
                "group_id": "terrain/test/decals",
                "artdef_set": "TEST",
                "collection": "Plants",
                "assets": [
                    {"source_asset": "SourceA", "asset_id": "terrain/test/decal_01"},
                    {"source_asset": "SourceB", "asset_id": "terrain/test/decal_01"},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate normalized decal ID"):
                load_mapping(path)

    def test_artdef_placement_maps_source_name_to_generic_id(self) -> None:
        values = "".join((
            _field("Asset", "m_EntryName", "SourceDecal"),
            _field("Scale", "m_fValue", "1.5"),
            _field("Count", "m_nValue", "3"),
            _field("ScaleVariation", "m_fValue", "0.2"),
            _field("ShowDecal", "m_bValue", "true"),
            _field("AllowOverlap", "m_bValue", "true"),
        ))
        xml = (
            "<Root><Element><m_ChildCollections><Element>"
            '<m_CollectionName text="Plants"/>'
            f"<Element><m_Fields><m_Values>{values}</m_Values></m_Fields></Element>"
            "</Element></m_ChildCollections>"
            '<m_Name text="TEST_SET"/></Element></Root>'
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.artdef"
            path.write_text(xml, encoding="utf-8")
            placements, report = read_artdef_group(
                path,
                "TEST_SET",
                "Plants",
                {"SourceDecal": "terrain/test/decal_01"},
            )
        self.assertEqual("terrain/test/decal_01", placements[0]["asset"])
        self.assertNotIn("source_asset", placements[0])
        self.assertTrue(placements[0]["show_decal"])
        self.assertEqual(["SourceDecal"], report["mapped_sources"])

    def test_descriptor_normalizes_bounds_and_omits_optional_class_mismatch(self) -> None:
        raw = bytearray(108)
        struct.pack_into("<8f", raw, 0x14, -6.0, -3.0, 6.0, 3.0, -4.0, -2.0, 4.0, 2.0)
        struct.pack_into("<4I", raw, 0x50, 2, 4, 6, 8)
        entries = {
            2: {"index": 2, "name": "Base", "class": "Decal_BaseColor"},
            4: {"index": 4, "name": "Height", "class": "Decal_Heightmap"},
            6: {"index": 6, "name": "Default", "class": "Decal_BaseColor"},
            8: {"index": 8, "name": "Fog", "class": "Decal_FOWColor"},
        }
        descriptor = decode_decal_descriptor(bytes(raw), entries.__getitem__, 12.0)
        self.assertEqual([-0.5, -0.25, 0.5, 0.25], descriptor["footprint_bounds"])
        self.assertEqual({"base_color", "height", "fog_color"}, set(descriptor["textures"]))
        self.assertEqual("class_mismatch", descriptor["texture_slots"]["specular"]["status"])

    def test_descriptor_rejects_required_class_mismatch(self) -> None:
        raw = bytearray(108)
        struct.pack_into("<8f", raw, 0x14, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0)
        struct.pack_into("<4I", raw, 0x50, 1, 2, 3, 4)
        entries = {
            1: {"index": 1, "name": "Wrong", "class": "Decal_Spec"},
            2: {"index": 2, "name": "Height", "class": "Decal_Heightmap"},
            3: {"index": 3, "name": "Spec", "class": "Decal_Spec"},
            4: {"index": 4, "name": "Fog", "class": "Decal_FOWColor"},
        }
        with self.assertRaisesRegex(ValueError, "Required decal base_color"):
            decode_decal_descriptor(bytes(raw), entries.__getitem__, 12.0)

    def test_descriptor_profile_can_admit_base_only_decal(self) -> None:
        raw = bytearray(108)
        struct.pack_into("<8f", raw, 0x14, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0)
        struct.pack_into("<4I", raw, 0x50, 1, 2, 3, 4)
        entries = {
            1: {"index": 1, "name": "Base", "class": "Decal_BaseColor"},
            2: {"index": 2, "name": "Default", "class": "Generic_BaseColor"},
            3: {"index": 3, "name": "Default", "class": "Generic_BaseColor"},
            4: {"index": 4, "name": "Fog", "class": "Decal_FOWColor"},
        }
        descriptor = decode_decal_descriptor(
            bytes(raw), entries.__getitem__, 100.0, required_roles=("base_color",)
        )
        self.assertEqual({"base_color", "fog_color"}, set(descriptor["textures"]))


if __name__ == "__main__":
    unittest.main()
