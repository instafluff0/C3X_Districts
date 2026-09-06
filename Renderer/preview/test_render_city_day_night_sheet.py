from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.preview.render_city_day_night_sheet import render_sheet
from Renderer.tools.asset_compiler.c3x_asset_compiler import make_dds_dx10_header


class CityDayNightSheetTests(unittest.TestCase):
    def test_synthetic_city_matrix_is_complete_deterministic_and_emissive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("components", "meshes", "materials", "textures"):
                (root / directory).mkdir()
            mesh = {
                "schema": "c3x.normalized_mesh.v0",
                "vertices": [
                    {"position": [-0.3, -0.2, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 0.0]},
                    {"position": [0.3, -0.2, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 0.0]},
                    {"position": [0.0, 0.2, 0.5], "normal": [0.0, 0.0, 1.0], "uv0": [0.5, 1.0]},
                ],
                "topology": {"primitive": "triangles", "indices": [0, 1, 2]},
            }
            (root / "meshes/city.json").write_text(json.dumps(mesh), encoding="utf-8")
            header = make_dds_dx10_header(
                {"width": 4, "height": 4, "mip_count": 1, "dxgi_format": 72, "payload_bytes": 8}
            )
            (root / "textures/base.dds").write_bytes(header + struct.pack("<HHI", 0xFFFF, 0xFFFF, 0))
            (root / "textures/glow.dds").write_bytes(header + struct.pack("<HHI", 0xFC00, 0xFC00, 0))
            material = {
                "schema": "c3x.material.v0",
                "channels": {
                    "base_color": {"texture": "textures/base.dds"},
                    "emissive": {"texture": "textures/glow.dds"},
                },
            }
            (root / "materials/city.json").write_text(json.dumps(material), encoding="utf-8")
            component = {
                "schema": "c3x.compound_landmark.v0",
                "components": {"geometry": ["meshes/city.json"], "materials": ["materials/city.json"]},
                "draw_bindings": [{"geometry": 0, "material": 0, "states": ["worked"]}],
            }
            (root / "components/city.json").write_text(json.dumps(component), encoding="utf-8")
            eras = [{"civ3_era": index, "id": f"era{index}"} for index in range(4)]
            styles = []
            pools = {}
            for style_index in range(5):
                era_pools = {}
                for era in eras:
                    pool = f"city/pool/style{style_index}/{era['id']}"
                    era_pools[era["id"]] = pool
                    pools[pool] = {"components": ["city/component/test"]}
                styles.append(
                    {"civ3_culture_group": style_index, "id": f"style{style_index}", "era_pools": era_pools}
                )
            (root / "city_catalog.json").write_text(
                json.dumps({"eras": eras, "styles": styles, "pools": pools}), encoding="utf-8"
            )
            manifest = {
                "schema": "c3x.asset_pack.v0",
                "city_catalog": "city_catalog.json",
                "assets": {
                    "city/component/test": {
                        "type": "compound_landmark",
                        "landmark": "components/city.json",
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            first, evidence = render_sheet(manifest_path, 1200, 900)
            repeated, repeated_evidence = render_sheet(manifest_path, 1200, 900)
        self.assertEqual(first.pixels, repeated.pixels)
        self.assertEqual(evidence, repeated_evidence)
        self.assertEqual(20, len(evidence["cells"]))
        self.assertEqual(20, evidence["emissive_cells"])
        self.assertGreater(first.non_background_pixels(), 1000)


if __name__ == "__main__":
    unittest.main()
