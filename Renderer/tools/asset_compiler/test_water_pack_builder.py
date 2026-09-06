from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Renderer.tools.asset_compiler import c3x_asset_compiler
from Renderer.tools.asset_compiler import grassland_pack_builder
from Renderer.tools.asset_compiler import water_pack_builder


class WaterPackBuilderTests(unittest.TestCase):
    def make_civbig(self, dxgi_format: int = 78) -> bytes:
        width = height = 4
        payload_bytes = c3x_asset_compiler.expected_texture_bytes(width, height, 1, dxgi_format)
        header = bytearray(c3x_asset_compiler.CIVBIG_HEADER_SIZE)
        header[:8] = b"CIVBIG\0\0"
        struct.pack_into("<I", header, 8, payload_bytes)
        struct.pack_into("<6H", header, 32, 1, 1, dxgi_format, width, height, 1)
        return bytes(header) + bytes(payload_bytes)

    def test_extracts_height_edit_blob_to_r8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "height_blob"
            target = Path(temporary) / "height.dds"
            width, height = 4, 2
            header = bytearray(512)
            header[:8] = b"CIVBIG\0\0"
            struct.pack_into("<I", header, 8, len(header) + width * height * 2 - 440)
            struct.pack_into("<II", header, 24, width, height)
            header[40:61] = b"Terrain_EditHeightmap"
            pixels = bytes(range(width * height))
            source.write_bytes(bytes(header) + b"".join(bytes((0, value)) for value in pixels))
            info = water_pack_builder.extract_height_edit_blob(source, target)
            data = target.read_bytes()
            self.assertEqual((4, 2, 61), (info["width"], info["height"], info["dxgi_format"]))
            self.assertEqual(pixels, data[148:])

    def test_compiles_complete_generic_water_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "Assets"
            shared = assets / "Base" / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA"
            artdefs = assets / "Base" / "ArtDefs"
            shared.mkdir(parents=True)
            artdefs.mkdir(parents=True)
            for source_name in water_pack_builder.SOURCE_TEXTURES.values():
                (shared / source_name).write_bytes(self.make_civbig())
            optional_role, optional_relative = next(iter(water_pack_builder.OPTIONAL_TEXTURES.items()))
            optional_source = assets / optional_relative
            optional_source.parent.mkdir(parents=True, exist_ok=True)
            optional_source.write_bytes(self.make_civbig())
            for name in ("Water.artdef", "Wave.artdef"):
                (artdefs / name).write_text(
                    '<Element class="AssetObjects..FloatValue"><m_fValue>4.0</m_fValue>'
                    '<m_ParamName text="Example"/></Element>', encoding="utf-8"
                )
            relief_package = root / "relief.blp"
            relief_package.write_bytes(b"fixture")
            pack = root / "pack"
            report = root / "out" / "report.json"

            def fake_relief(_source: Path, output: Path):
                output.mkdir(parents=True, exist_ok=True)
                return {"schema": "c3x.water_relief_compile.v0", "compiled_texture_count": 20}

            with mock.patch.object(
                water_pack_builder.terrain_relief_builder,
                "extract_water_relief_resources",
                side_effect=fake_relief,
            ):
                result = water_pack_builder.compile_water_pack(
                    assets, relief_package, pack, report
                )

            self.assertEqual(len(water_pack_builder.SOURCE_TEXTURES) + 1, result["texture_count"])
            self.assertEqual(1, result["optional_texture_count"])
            self.assertEqual(20, result["relief_texture_count"])
            catalog = json.loads((pack / "water" / "catalog.json").read_text(encoding="utf-8"))
            profiles = json.loads((pack / "water" / "profiles.json").read_text(encoding="utf-8"))
            self.assertEqual(set(water_pack_builder.SOURCE_TEXTURES) | {optional_role}, set(catalog["textures"]))
            self.assertEqual(2, len(profiles["surface"]["large_lean"]))
            self.assertTrue((pack / "textures" / "water" / "effects" / "crash_foam.dds").is_file())
            self.assertEqual([], grassland_pack_builder.validate_runtime_independence(pack))


if __name__ == "__main__":
    unittest.main()
