#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3x_asset_compiler as compiler


class AssetCompilerTests(unittest.TestCase):
    def make_civbig(self, width: int = 8, height: int = 8, mip_count: int = 2, dxgi_format: int = 78) -> bytes:
        payload_bytes = compiler.expected_texture_bytes(width, height, mip_count, dxgi_format)
        header = bytearray(compiler.CIVBIG_HEADER_SIZE)
        header[:8] = b"CIVBIG\x00\x00"
        struct.pack_into("<I", header, 8, payload_bytes)
        struct.pack_into("<6H", header, 32, 1, mip_count, dxgi_format, width, height, 1)
        return bytes(header) + bytes((index % 251 for index in range(payload_bytes))) + b"padding"

    def test_extract_civbig_builds_dx10_dds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "texture"
            output = Path(tmp) / "texture.dds"
            source.write_bytes(self.make_civbig())

            info = compiler.extract_civbig_to_dds(source, output)
            data = output.read_bytes()

            self.assertEqual(info["width"], 8)
            self.assertEqual(info["height"], 8)
            self.assertEqual(info["mip_count"], 2)
            self.assertEqual(info["dxgi_format"], 78)
            self.assertEqual(info["trailing_padding_bytes"], 7)
            self.assertEqual(data[:4], b"DDS ")
            self.assertEqual(data[84:88], b"DX10")
            self.assertEqual(struct.unpack_from("<I", data, 128)[0], 78)
            self.assertEqual(len(data), compiler.DDS_DX10_HEADER_SIZE + info["payload_bytes"])

    def test_extracts_linear_water_texture_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for dxgi_format, bytes_per_pixel in ((10, 8), (11, 8), (35, 4)):
                with self.subTest(dxgi_format=dxgi_format):
                    source = Path(tmp) / f"source_{dxgi_format}"
                    output = Path(tmp) / f"output_{dxgi_format}.dds"
                    source.write_bytes(self.make_civbig(8, 4, 3, dxgi_format))
                    info = compiler.extract_civbig_to_dds(source, output)
                    data = output.read_bytes()
                    self.assertEqual(info["format_name"], compiler.DXGI_FORMAT_NAMES[dxgi_format])
                    self.assertEqual(struct.unpack_from("<I", data, 128)[0], dxgi_format)
                    self.assertEqual(struct.unpack_from("<I", data, 20)[0], 8 * bytes_per_pixel)
                    self.assertTrue(struct.unpack_from("<I", data, 8)[0] & 0x8)
                    self.assertFalse(struct.unpack_from("<I", data, 8)[0] & 0x80000)

    def test_civbig_rejects_wrong_payload_size(self) -> None:
        data = bytearray(self.make_civbig())
        struct.pack_into("<I", data, 8, 1)
        with self.assertRaisesRegex(ValueError, "does not match"):
            compiler.parse_civbig_header(bytes(data))

    def test_discover_and_build_prototype_from_minimal_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "Base"
            (base / "ArtDefs").mkdir(parents=True)
            (base / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA").mkdir(parents=True)
            (base / "ArtDefs" / "Terrains.artdef").write_text(
                "<Asset><m_Name text=\"Terrain_Grass_Hills\"/></Asset>",
                encoding="utf-8",
            )
            (base / "Civ6.dep").write_text(
                "<Element text=\"Terrains.artdef\"/>",
                encoding="utf-8",
            )
            (base / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA" / "TEXTURE_Terrain_Grass_Hills").write_bytes(b"fake")

            report = compiler.discover(base)
            self.assertEqual(report["schema"], "c3x.civ6.discovery.v0")
            self.assertEqual(report["blp_tree"]["file_count"], 1)
            self.assertTrue(report["blp_tree"]["candidate_assets"])

            pack = Path(tmp) / "pack"
            report_path = Path(tmp) / "report.json"
            compiler.build_prototype(base, report_path, pack)
            manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "c3x.asset_pack.v0")
            self.assertEqual(len(manifest["relief"]["mountains"]["variants"]), 5)
            self.assertTrue((pack / "materials" / "grassland.json").exists())

    def test_import_loose_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "grass.png").write_bytes(b"not really an image")
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "schema": "c3x.loose_source.v0",
                        "name": "TestLoose",
                        "terrains": {
                            "grassland": {
                                "albedo": "grass.png",
                                "preview_color": [1, 2, 3],
                            },
                            "plains": {
                                "preview_color": [4, 5, 6],
                            },
                        },
                        "relief": {
                            "mountains": {
                                "variants": [
                                    {
                                        "id": "missing_model",
                                        "model": "models/missing.glb",
                                        "preview_height": 1.2,
                                    }
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            pack = Path(tmp) / "pack"
            manifest = compiler.build_from_loose(source_manifest, pack)

            self.assertEqual(manifest["schema"], "c3x.asset_pack.v0")
            self.assertEqual(manifest["name"], "TestLoose")
            self.assertEqual(manifest["terrains"]["grassland"]["preview_color"], [1, 2, 3])
            self.assertEqual(manifest["relief"]["mountains"]["variants"][0]["model"], "models/missing.glb")
            self.assertTrue((pack / "grass.png").exists())
            self.assertTrue(any("Missing loose asset" in item for item in manifest["diagnostics"]))

    def test_build_grassland_poc_indexes_artdef_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "Base"
            (base / "ArtDefs").mkdir(parents=True)
            (base / "Platforms" / "Windows" / "BLPs" / "terrain").mkdir(parents=True)
            (base / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA").mkdir(parents=True)
            (base / "ArtDefs" / "TerrainStyle.artdef").write_text(
                """
                <Element class="AssetObjects..BLPEntryValue">
                  <m_EntryName text="ART_DEF_TERRAIN_MATERIAL_GRASSLAND"/>
                  <m_XLPClass text="TerrainMaterial"/>
                  <m_XLPPath text="TerrainMaterialSet_Base.xlp"/>
                  <m_BLPPackage text="terrain/TerrainMaterialSet_Base"/>
                  <m_LibraryName text="TerrainMaterial"/>
                  <m_ParamName text="GrasslandMtl"/>
                </Element>
                """,
                encoding="utf-8",
            )
            (base / "Platforms" / "Windows" / "BLPs" / "terrain" / "TerrainMaterialSet_Base.blp").write_bytes(
                b"CIVBLP\x00\x00ART_DEF_TERRAIN_MATERIAL_GRASSLAND\x00TextureEntry\x00"
            )
            (base / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA" / "TEXTURE_Terrain_Grass_Hills").write_bytes(
                b"CIVBIG"
            )
            (base / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA" / "TEXTURE_TER_Grass_Decal_B").write_bytes(
                self.make_civbig()
            )

            pack = Path(tmp) / "pack"
            manifest = compiler.build_grassland_poc(base, pack)
            mapping = json.loads((pack / "civ3_tile_art_map.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["name"], "Civ6GrasslandPOC")
            self.assertEqual(manifest["terrains"]["grassland"]["civ6_reference"], "ART_DEF_TERRAIN_MATERIAL_GRASSLAND")
            self.assertEqual(mapping["rules"][0]["match"]["square_type"], 2)
            self.assertTrue((pack / "civ6_grassland_sources.json").exists())
            self.assertTrue((pack / "textures" / "grassland_decal_b.dds").exists())


if __name__ == "__main__":
    unittest.main()
