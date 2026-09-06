from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.preview.render_route_doodad_sheet import render_sheet
from Renderer.tools.asset_compiler.c3x_asset_compiler import make_dds_dx10_header


class RouteDoodadSheetTests(unittest.TestCase):
    def test_synthetic_bridge_sheet_is_deterministic_and_nonblank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("bodies", "meshes", "materials", "textures"):
                (root / directory).mkdir()
            mesh = {
                "schema": "c3x.normalized_mesh.v0",
                "vertices": [
                    {"position": [-0.4, -0.1, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 0.0]},
                    {"position": [0.4, -0.1, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 0.0]},
                    {"position": [0.4, 0.1, 0.1], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 1.0]},
                    {"position": [-0.4, 0.1, 0.1], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 1.0]},
                ],
                "topology": {"primitive": "triangles", "indices": [0, 1, 2, 0, 2, 3]},
            }
            (root / "meshes/bridge.json").write_text(json.dumps(mesh), encoding="utf-8")
            payload = struct.pack("<HHI", 0xFFFF, 0x001F, 0)
            header = make_dds_dx10_header(
                {"width": 4, "height": 4, "mip_count": 1, "dxgi_format": 72, "payload_bytes": 8}
            )
            (root / "textures/bridge.dds").write_bytes(header + payload)
            material = {
                "schema": "c3x.material.v0",
                "channels": {
                    "base_color": {
                        "texture": "textures/bridge.dds",
                        "address_u": "clamp",
                        "address_v": "clamp",
                    }
                },
            }
            (root / "materials/bridge.json").write_text(json.dumps(material), encoding="utf-8")
            body = {
                "schema": "c3x.compound_landmark.v0",
                "components": {
                    "geometry": ["meshes/bridge.json"],
                    "materials": ["materials/bridge.json"],
                },
                "draw_bindings": [
                    {"geometry": 0, "material": 0, "states": ["worked", "pillaged"]}
                ],
            }
            (root / "bodies/bridge.json").write_text(json.dumps(body), encoding="utf-8")
            manifest = {
                "schema": "c3x.asset_pack.v0",
                "assets": {
                    "route/bridge/test": {
                        "type": "compound_landmark",
                        "landmark": "bodies/bridge.json",
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            first = render_sheet(manifest_path, 320, 480)
            repeated = render_sheet(manifest_path, 320, 480)
        self.assertEqual(first.pixels, repeated.pixels)
        self.assertGreater(first.non_background_pixels(), 1000)


if __name__ == "__main__":
    unittest.main()
