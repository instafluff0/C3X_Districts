#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import terrain_geometry_resolver as resolver


TERRAINS_XML = """<?xml version="1.0"?>
<AssetObjects..ArtDefSet><m_RootCollections><Element>
<m_CollectionName text="Terrain"/><Element><m_Fields/><m_ChildCollections>
<Element><m_CollectionName text="TerrainType"/><Element><m_Fields><m_Values>
<Element><m_Value text="Flat"/><m_ParamName text="XrefName"/></Element>
</m_Values></m_Fields></Element></Element>
<Element><m_CollectionName text="TerrainSubType"/><Element><m_Fields><m_Values>
<Element><m_Value text="Grass"/><m_ParamName text="XrefName"/></Element>
</m_Values></m_Fields></Element></Element>
</m_ChildCollections><m_Name text="TERRAIN_GRASS"/></Element>
</Element></m_RootCollections></AssetObjects..ArtDefSet>"""


def blp_value(entry: str, xlp_class: str, parameter: str) -> str:
    return f"""<Element class="AssetObjects..BLPEntryValue">
<m_EntryName text="{entry}"/><m_XLPClass text="{xlp_class}"/>
<m_XLPPath text="Set.xlp"/><m_BLPPackage text="terrain/Set"/>
<m_LibraryName text="{xlp_class}"/><m_ParamName text="{parameter}"/>
</Element>"""


STYLE_XML = f"""<?xml version="1.0"?>
<AssetObjects..ArtDefSet><m_RootCollections>
<Element><m_CollectionName text="StandardFlat"/><Element><m_Fields><m_Values>
{blp_value(resolver.DEFAULT_MATERIAL, "TerrainMaterial", "GrasslandMtl")}
{blp_value("ART_DEF_TERRAIN_ELEMENT_CONTINENTAL_HILL_GRASSLAND", "TerrainElement", "GrasslandElement")}
</m_Values></m_Fields><m_ChildCollections/><m_Name text="Default"/></Element></Element>
<Element><m_CollectionName text="StandardHills"/><Element><m_Fields><m_Values>
{blp_value("ART_DEF_TERRAIN_ELEMENT_HILL", "TerrainElement", "HillElement")}
</m_Values></m_Fields><m_ChildCollections/><m_Name text="Default"/></Element></Element>
</m_RootCollections></AssetObjects..ArtDefSet>"""


class TerrainGeometryResolverTests(unittest.TestCase):
    def write_artdefs(self, base: Path) -> tuple[Path, Path]:
        artdefs = base / "ArtDefs"
        artdefs.mkdir(parents=True)
        terrains = artdefs / "Terrains.artdef"
        style = artdefs / "TerrainStyle.artdef"
        terrains.write_text(TERRAINS_XML, encoding="utf-8")
        style.write_text(STYLE_XML, encoding="utf-8")
        return terrains, style

    def write_blp(self, path: Path, metadata: bytes, entry_count: int = 1) -> None:
        package_offset = 28
        big_data_offset = package_offset + len(metadata)
        payload = b"payload"
        header = bytearray(28)
        header[:6] = b"CIVBLP"
        struct.pack_into(
            "<H5I",
            header,
            6,
            2,
            package_offset,
            len(metadata),
            big_data_offset,
            entry_count,
            big_data_offset + len(payload),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(header) + metadata + payload)

    def test_artdef_chain_selects_flat_material_and_separate_relief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            terrains, style = self.write_artdefs(Path(tmp))
            result = resolver.resolve_artdef_chain(terrains, style)

        self.assertEqual((result["terrain_type"], result["terrain_subtype"]), ("Flat", "Grass"))
        self.assertEqual(result["style_collection"], "StandardFlat")
        self.assertEqual(result["material_reference"]["parameter_name"], "GrasslandMtl")
        self.assertEqual(result["authored_relief_reference"]["parameter_name"], "GrasslandElement")

    def test_artdef_chain_rejects_non_flat_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            terrains, style = self.write_artdefs(Path(tmp))
            terrains.write_text(TERRAINS_XML.replace('text="Flat"', 'text="Hills"'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot represent terrain type Hills"):
                resolver.resolve_artdef_chain(terrains, style)

    def test_normalized_patch_has_valid_topology_normals_and_uvs(self) -> None:
        mesh = resolver.make_flat_patch()

        self.assertEqual(resolver.validate_normalized_mesh(mesh), [])
        self.assertEqual(mesh["topology"]["indices"], [0, 1, 2, 0, 2, 3])
        self.assertEqual(mesh["provenance"]["source_format_dependency"], None)

    def test_mesh_validator_rejects_reversed_winding(self) -> None:
        mesh = resolver.make_flat_patch()
        mesh["topology"]["indices"][:3] = [0, 2, 1]

        self.assertIn(
            "triangle winding is not counter-clockwise around +Z",
            resolver.validate_normalized_mesh(mesh),
        )

    def test_end_to_end_report_is_deterministic_and_source_agnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.write_artdefs(base)
            terrain_dir = base / "Platforms" / "Windows" / "BLPs" / "terrain"
            self.write_blp(
                terrain_dir / "TerrainAssetSet_Base.blp",
                b"BLP::IndexBufferEntry\x00FGXModel::ContainerDesc::Mesh\x00",
                2,
            )
            self.write_blp(
                terrain_dir / "TerrainElementSet_Base.blp",
                b"BLP::VertexBufferEntry\x00",
                3,
            )
            self.write_blp(
                terrain_dir / "TerrainMaterialSet_Base.blp",
                resolver.DEFAULT_MATERIAL.encode("ascii") + b"\x00",
                4,
            )
            mesh = resolver.make_flat_patch()
            first = resolver.build_report(base, mesh)
            second = resolver.build_report(base, mesh)

        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first["selection"]["mode"], "procedural_flat_grid")
        self.assertEqual(first["selection"]["uv_domain"], "per_tile_unit_square")
        self.assertEqual(first["loose_geometry_inventory"]["file_count"], 0)
        self.assertFalse(any("Civ VI" in key for key in mesh))


if __name__ == "__main__":
    unittest.main()
