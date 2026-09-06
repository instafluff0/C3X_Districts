from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import civblp_probe
from Renderer.tools.asset_compiler import clutter_blp_extractor as extractor


def allocation(
    stripe: int,
    target: int,
    size: int,
    count: int,
    type_pointer: int,
    parent: int = 0,
) -> bytes:
    return struct.pack(
        "<BB4sHIII4xQQ",
        stripe,
        0,
        b"\x00" * 4,
        parent,
        target,
        size,
        count,
        0,
        type_pointer,
    )


class StaticAllocationTableTests(unittest.TestCase):
    def test_finds_largest_bounded_table_after_reflected_marker(self) -> None:
        marker = civblp_probe.ENTRY_MAP_TYPE
        table = b"".join(
            (
                allocation(0, 0, 24, 3, 3),
                allocation(0, 24, 8, 1, 3),
                allocation(1, 0, 5, 5, 3),
            )
        )
        package = b"prefix" + marker + b"padding" + table + b"\x00" * 80
        offset, decoded = extractor.find_static_allocation_table(package)
        self.assertEqual(offset, len(b"prefix" + marker + b"padding"))
        self.assertEqual(len(decoded), 3)
        self.assertEqual(decoded[0]["size"], 24)

    def test_rejects_table_with_out_of_range_type_pointer(self) -> None:
        marker = civblp_probe.ENTRY_MAP_TYPE
        table = b"".join(
            (
                allocation(0, 0, 24, 3, 99),
                allocation(0, 24, 8, 1, 1),
                allocation(1, 0, 5, 5, 1),
            )
        )
        with self.assertRaisesRegex(ValueError, "Could not locate"):
            extractor.find_static_allocation_table(marker + table + b"\x00" * 40)


class MeshProfileTests(unittest.TestCase):
    def test_decodes_complete_non_degenerate_profile(self) -> None:
        positions = ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 2.0))
        uvs = ((0.0, 1.0), (1.0, 1.0), (0.5, 0.0))
        vertex_format = 0x6679B170
        stride = extractor.VERTEX_PROFILES[vertex_format]["stride"]
        vertices = bytearray(3 * stride)
        for index, (position, uv) in enumerate(zip(positions, uvs)):
            base = index * stride
            struct.pack_into("<eee", vertices, base, *position)
            struct.pack_into("<ee", vertices, base + extractor.UV0_OFFSET, *uv)
        indices = struct.pack("<3H", 0, 1, 2)
        mesh, evidence = extractor.normalize_mesh(
            bytes(vertices),
            indices,
            {
                "format": vertex_format,
                "stride": stride,
                "count": 3,
            },
            {"bytes_per_index": 2, "count": 3},
            {
                "first_index": 0,
                "index_count": 3,
                "base_vertex": 0,
                "vertex_count": 3,
            },
            "feature.test.triangle",
        )
        self.assertEqual(evidence["triangles"], 1)
        self.assertEqual(mesh["bounds"]["minimum"], [-0.08333333, -0.08333333, 0.0])
        self.assertEqual(mesh["bounds"]["maximum"], [0.08333333, 0.08333333, 0.16666667])
        self.assertEqual(mesh["topology"]["indices"], [0, 1, 2])
        for vertex in mesh["vertices"]:
            length = sum(value * value for value in vertex["normal"]) ** 0.5
            self.assertAlmostEqual(length, 1.0, places=6)

    def test_decodes_24_byte_vegetation_profile_and_half_uvs(self) -> None:
        vertex_format = 0x315CFCD9
        stride = extractor.VERTEX_PROFILES[vertex_format]["stride"]
        vertices = bytearray(3 * stride)
        for index, (position, uv) in enumerate(
            zip(
                ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 4.0)),
                ((0.125, 0.25), (0.5, 0.75), (0.875, 1.0)),
            )
        ):
            base = index * stride
            struct.pack_into("<eee", vertices, base, *position)
            struct.pack_into("<ee", vertices, base + extractor.UV0_OFFSET, *uv)
        mesh, evidence = extractor.normalize_mesh(
            bytes(vertices),
            struct.pack("<3H", 0, 1, 2),
            {"format": vertex_format, "stride": stride, "count": 3},
            {"bytes_per_index": 2, "count": 3},
            {
                "first_index": 0,
                "index_count": 3,
                "base_vertex": 0,
                "vertex_count": 3,
            },
            "feature.test.compact",
        )
        self.assertEqual(evidence["unique_uv0"], 3)
        self.assertEqual(mesh["vertices"][0]["uv0"], [0.125, 0.25])
        self.assertEqual(mesh["vertices"][2]["uv0"], [0.875, 1.0])

    def test_rejects_unknown_vertex_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported vertex profile"):
            extractor.normalize_mesh(
                b"\x00" * 96,
                struct.pack("<3H", 0, 1, 2),
                {"format": 1, "stride": 32, "count": 3},
                {"bytes_per_index": 2, "count": 3},
                {
                    "first_index": 0,
                    "index_count": 3,
                    "base_vertex": 0,
                    "vertex_count": 3,
                },
                "feature.test.invalid",
            )

    def test_preserves_intentional_wrapping_uvs_when_enabled(self) -> None:
        vertex_format = 0x315CFCD9
        stride = extractor.VERTEX_PROFILES[vertex_format]["stride"]
        vertices = bytearray(3 * stride)
        for index, (position, uv) in enumerate(
            zip(
                ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 4.0)),
                ((-0.25, 0.25), (1.5, 0.75), (0.875, 2.0)),
            )
        ):
            base = index * stride
            struct.pack_into("<eee", vertices, base, *position)
            struct.pack_into("<ee", vertices, base + extractor.UV0_OFFSET, *uv)
        mesh, evidence = extractor.normalize_mesh(
            bytes(vertices),
            struct.pack("<3H", 0, 1, 2),
            {"format": vertex_format, "stride": stride, "count": 3},
            {"bytes_per_index": 2, "count": 3},
            {"first_index": 0, "index_count": 3, "base_vertex": 0, "vertex_count": 3},
            "feature.test.wrapping",
            True,
        )
        self.assertEqual(mesh["coordinate_system"]["uv0_address_mode"], "wrap")
        self.assertEqual(evidence["uv0_address_mode"], "wrap")
        self.assertEqual(mesh["vertices"][0]["uv0"], [-0.25, 0.25])


class ArtDefPlacementTests(unittest.TestCase):
    def test_alternate_leafy_specs_are_generic_and_collision_free(self) -> None:
        combined = extractor.FEATURE_SPECS + extractor.ALTERNATE_FEATURE_SPECS
        self.assertEqual(17, len(extractor.ALTERNATE_FEATURE_SPECS))
        self.assertEqual(len(combined), len({item["source_name"] for item in combined}))
        self.assertEqual(len(combined), len({item["manifest_key"] for item in combined}))
        for item in extractor.ALTERNATE_FEATURE_SPECS:
            self.assertTrue(item["manifest_key"].startswith("feature/"))
            self.assertNotIn("civv", item["manifest_key"])

    def test_reads_required_placement_fields_from_all_groups(self) -> None:
        def clutter_set(name: str, asset: str) -> str:
            return f"""
            <Element><m_ChildCollections><Element><m_CollectionName text="Plants"/>
              <Element><m_Fields><m_Values>
                <Element><m_EntryName text="{asset}"/><m_ParamName text="Asset"/></Element>
                <Element><m_fValue>0.800000</m_fValue><m_ParamName text="Scale"/></Element>
                <Element><m_nValue>3</m_nValue><m_ParamName text="Count"/></Element>
                <Element><m_fValue>0.100000</m_fValue><m_ParamName text="ScaleVariation"/></Element>
                <Element><m_bValue>true</m_bValue><m_ParamName text="AllowOverlap"/></Element>
              </m_Values></m_Fields></Element>
            </Element></m_ChildCollections><m_Name text="{name}"/></Element>
            """

        document = "<Root>" + "".join(
            clutter_set(set_name, group + "_asset")
            for group, set_name in extractor.ARTDEF_GROUPS.items()
        ) + "</Root>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Clutter.artdef"
            path.write_text(document, encoding="utf-8")
            result = extractor.read_artdef_placements(path)
        self.assertEqual(set(result), set(extractor.ARTDEF_GROUPS))
        self.assertEqual(result["jungle"][0]["scale"], 0.8)
        self.assertEqual(result["jungle"][0]["count"], 3)
        self.assertTrue(result["jungle"][0]["allow_overlap"])


class RuntimeBundleTests(unittest.TestCase):
    def test_writes_source_independent_asset_and_placement_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meshes").mkdir()
            (root / "materials").mkdir()
            (root / "textures").mkdir()
            (root / "textures" / "forest.dds").write_bytes(b"generic-texture")
            (root / "meshes" / "pine.json").write_text(
                '{"vertices":[{"position":[0,0,0],"normal":[0,0,1],"uv0":[0,0]}],'
                '"topology":{"indices":[0,0,0]}}',
                encoding="utf-8",
            )
            (root / "materials" / "pine.json").write_text(
                '{"base_color":{"texture":"textures/forest.dds"}}', encoding="utf-8"
            )
            placement = {
                "asset": "feature/forest/pine_01",
                "scale": 0.5,
                "scale_variation": 0.1,
                "count": 7,
                "min_count": 0,
                "priority": 3,
                "allow_overlap": True,
                "show_decal": True,
                "is_center_model": False,
                "width": 0.0,
                "low_end_reduction": 0.0,
            }
            manifest = {
                "assets": {
                    "feature/forest/pine_01": {
                        "mesh": "meshes/pine.json",
                        "material": "materials/pine.json",
                    }
                },
                "features": {"forest": {"placements": [placement]}},
            }
            evidence = extractor.write_runtime_bundle(root, manifest)
            data = (root / extractor.RUNTIME_BUNDLE).read_bytes()
        self.assertEqual(data[:8], b"C3XVEG1\0")
        self.assertEqual(struct.unpack_from("<IIII", data, 8), (1, 1, 1, 1))
        self.assertEqual(evidence["textures"], 1)
        self.assertNotIn(b"civilization", data.lower())


if __name__ == "__main__":
    unittest.main()
