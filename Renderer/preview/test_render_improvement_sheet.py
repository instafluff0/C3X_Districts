from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.preview.render_improvement_sheet import _point, render_sheet
from Renderer.preview.render_textured_patch import decode_bc3_alpha
from Renderer.tools.asset_compiler.c3x_asset_compiler import make_dds_dx10_header


class ImprovementSheetTests(unittest.TestCase):
    def test_synthetic_matrix_is_complete_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("components", "meshes", "materials", "textures"):
                (root / directory).mkdir()
            mesh = {
                "schema": "c3x.normalized_mesh.v0",
                "vertices": [
                    {"position": [-0.4, -0.3, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 0.0]},
                    {"position": [0.4, -0.3, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 0.0]},
                    {"position": [0.0, 0.3, 0.5], "normal": [0.0, 0.0, 1.0], "uv0": [0.5, 1.0]},
                ],
                "topology": {"primitive": "triangles", "indices": [0, 1, 2]},
            }
            (root / "meshes/improvement.json").write_text(json.dumps(mesh), encoding="utf-8")
            header = make_dds_dx10_header(
                {"width": 4, "height": 4, "mip_count": 1, "dxgi_format": 72, "payload_bytes": 8}
            )
            (root / "textures/base.dds").write_bytes(header + struct.pack("<HHI", 0xFFFF, 0xFFFF, 0))
            material = {
                "schema": "c3x.material.v0",
                "channels": {
                    "base_color": {
                        "texture": "textures/base.dds",
                        "format": "BC1_UNORM_SRGB",
                        "address_u": "clamp",
                        "address_v": "clamp",
                    }
                },
            }
            (root / "materials/improvement.json").write_text(json.dumps(material), encoding="utf-8")
            component = {
                "schema": "c3x.compound_landmark.v0",
                "components": {
                    "geometry": ["meshes/improvement.json"],
                    "materials": ["materials/improvement.json"],
                    "skeletons": [],
                },
                "draw_bindings": [{"geometry": 0, "material": 0, "states": ["worked"]}],
                "attachment_points": [],
            }
            (root / "components/improvement.json").write_text(json.dumps(component), encoding="utf-8")
            asset_id = "improvement/component/test"
            eras = [
                {"id": "preindustrial", "civ3_eras": [0, 1], "building_pieces": [asset_id, asset_id], "tile_bases": [asset_id]},
                {"id": "industrial", "civ3_eras": [2], "building_pieces": [asset_id, asset_id], "tile_bases": [asset_id]},
                {"id": "modern", "civ3_eras": [3], "building_pieces": [asset_id, asset_id], "tile_bases": [asset_id]},
            ]
            catalog = {
                "mine": {"eras": [{"id": "all", "variants": [asset_id] * 6}]},
                "farm": {
                    "eras": eras,
                    "crop_styles": [
                        {"id": "default", "pieces": [asset_id, asset_id]},
                        {"id": "grain", "pieces": [asset_id]},
                        {"id": "maize", "pieces": [asset_id]},
                    ],
                },
            }
            (root / "improvement_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            manifest = {
                "schema": "c3x.asset_pack.v0",
                "improvement_catalog": "improvement_catalog.json",
                "assets": {asset_id: {"landmark": "components/improvement.json"}},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            first, evidence = render_sheet(manifest_path, 1200, 800)
            repeated, repeated_evidence = render_sheet(manifest_path, 1200, 800)
        self.assertEqual(first.pixels, repeated.pixels)
        self.assertEqual(evidence, repeated_evidence)
        self.assertEqual(12, evidence["day_night_pairs"])
        self.assertEqual(12, len(evidence["cells"]))
        self.assertGreater(first.non_background_pixels(), 1000)

    def test_point_transform_applies_translation_only_to_positions(self) -> None:
        matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 5.0, 6.0, 1.0]
        self.assertEqual([5.0, 7.0, 9.0], _point([1.0, 2.0, 3.0], matrix, True))
        self.assertEqual([1.0, 2.0, 3.0], _point([1.0, 2.0, 3.0], matrix, False))

    def test_bc3_alpha_uses_both_interpolation_modes(self) -> None:
        descending = bytes([255, 0]) + (2).to_bytes(6, "little") + bytes(8)
        ascending = bytes([0, 255]) + (7).to_bytes(6, "little") + bytes(8)
        self.assertEqual(218, decode_bc3_alpha(descending, 0, 0))
        self.assertEqual(255, decode_bc3_alpha(ascending, 0, 0))


if __name__ == "__main__":
    unittest.main()
