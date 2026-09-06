#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.preview import render_textured_patch
from Renderer.tools.asset_compiler import c3x_asset_compiler
from Renderer.tools.asset_compiler import grassland_pack_builder as builder
from Renderer.tools.asset_compiler import terrain_geometry_resolver


class GrasslandPackBuilderTests(unittest.TestCase):
    def rgb565(self, r: int, g: int, b: int) -> int:
        return ((r * 31 // 255) << 11) | ((g * 63 // 255) << 5) | (b * 31 // 255)

    def bc3_block(self, color: tuple[int, int, int]) -> bytes:
        encoded = self.rgb565(*color)
        return bytes((255, 255)) + b"\x00" * 6 + struct.pack("<HHI", encoded, encoded, 0)

    def make_dds(self) -> tuple[bytes, dict]:
        payload = b"".join(
            self.bc3_block(color)
            for color in ((40, 100, 30), (80, 150, 45), (115, 175, 60), (55, 125, 35))
        )
        info = {
            "width": 8,
            "height": 8,
            "mip_count": 1,
            "dxgi_format": 78,
            "payload_bytes": len(payload),
            "format_name": "BC3_UNORM_SRGB",
            "color_space": "srgb",
            "logical_name": "SYNTHETIC_GRASS",
        }
        return c3x_asset_compiler.make_dds_dx10_header(info) + payload, info

    def write_synthetic_pack(self, root: Path) -> Path:
        dds, info = self.make_dds()
        mesh_path = root / "source" / "mesh.json"
        dds_path = root / "source" / "texture.dds"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(
            json.dumps(terrain_geometry_resolver.make_flat_patch(), indent=2) + "\n",
            encoding="utf-8",
        )
        dds_path.write_bytes(dds)
        pack = root / "pack"
        builder.build_normalized_pack(mesh_path, dds_path, info, pack)
        return pack / "manifest.json"

    def test_bc3_sampler_reads_distinct_blocks(self) -> None:
        dds, _info = self.make_dds()
        texture = render_textured_patch.DdsBc3Texture(dds)

        self.assertNotEqual(texture.sample(0.1, 0.1), texture.sample(0.75, 0.1))
        self.assertEqual(texture.sample(0.1, 0.1), texture.sample(1.1, 0.1))

    def test_extract_embedded_payload_writes_standard_dds(self) -> None:
        dds, info = self.make_dds()
        payload = dds[c3x_asset_compiler.DDS_DX10_HEADER_SIZE :]
        package_offset = 28
        package_metadata = b"metadata"
        big_data_offset = package_offset + len(package_metadata)
        header = bytearray(28)
        header[:6] = b"CIVBLP"
        struct.pack_into(
            "<H5I",
            header,
            6,
            2,
            package_offset,
            len(package_metadata),
            big_data_offset,
            1,
            big_data_offset + len(payload),
        )
        binding = {
            "roles": [
                {
                    "role": "base_color",
                    "status": "resolved",
                    "logical_name": "SYNTHETIC_GRASS",
                    "metadata": {
                        "width": 8,
                        "height": 8,
                        "mip_count": 1,
                        "format": {
                            "dxgi": 78,
                            "name": "BC3_UNORM_SRGB",
                            "color_space": "srgb",
                            "block_bytes": 16,
                        },
                    },
                    "storage": {
                        "mode": "embedded_blp_big_data",
                        "relative_offset": 0,
                        "absolute_file_offset": big_data_offset,
                        "bytes": len(payload),
                        "bounds_valid": True,
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "synthetic.blp"
            output = root / "normalized.dds"
            package.write_bytes(bytes(header) + package_metadata + payload)
            result = builder.extract_embedded_base_color(package, binding, output)
            normalized = output.read_bytes()

        self.assertEqual(normalized[:4], b"DDS ")
        self.assertEqual(normalized[c3x_asset_compiler.DDS_DX10_HEADER_SIZE :], payload)
        self.assertEqual(result["dds_sha256"], builder.hashlib.sha256(normalized).hexdigest())
        self.assertEqual(result["payload_bytes"], info["payload_bytes"])

    def test_pack_is_source_agnostic_and_rejects_escaping_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self.write_synthetic_pack(root)
            pack_root = manifest_path.parent
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(builder.validate_runtime_independence(pack_root), [])
            self.assertEqual(
                manifest["assets"]["terrain/grassland/base"]["mesh"],
                "meshes/flat_terrain_patch.json",
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                render_textured_patch.safe_pack_path(pack_root, "../outside.dds")
            with self.assertRaisesRegex(ValueError, "must be relative"):
                render_textured_patch.safe_pack_path(pack_root, r"C:\outside.dds")

    def test_same_pack_renders_deterministically_at_required_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self.write_synthetic_pack(Path(tmp))
            small = render_textured_patch.render_pack(manifest_path, 640, 480, 4)
            repeated = render_textured_patch.render_pack(manifest_path, 640, 480, 4)
            large = render_textured_patch.render_pack(manifest_path, 1024, 768, 4)

        self.assertEqual(small.pixels, repeated.pixels)
        self.assertGreater(small.non_background_pixels(), 10000)
        self.assertGreater(large.non_background_pixels(), 10000)
        self.assertGreater(len(set(small.pixels)), 4)

    def test_png_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self.write_synthetic_pack(root)
            canvas = render_textured_patch.render_pack(manifest_path, 320, 240, 2)
            first = root / "first.png"
            second = root / "second.png"
            render_textured_patch.write_png(canvas, first)
            render_textured_patch.write_png(canvas, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
