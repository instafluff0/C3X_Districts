#!/usr/bin/env python3
import copy
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.definitions import definition_parser
from Renderer.preview.render_textured_patch import BACKGROUND, write_png
from Renderer.scenes import scene_contract
from Renderer.standalone import whole_viewport_renderer as renderer
from Renderer.tools.asset_compiler import c3x_asset_compiler


class WholeViewportRendererTests(unittest.TestCase):
    def rgb565(self, red: int, green: int, blue: int) -> int:
        return ((red * 31 // 255) << 11) | ((green * 63 // 255) << 5) | (blue * 31 // 255)

    def bc3_block(self, color: tuple[int, int, int]) -> bytes:
        encoded = self.rgb565(*color)
        return bytes((255, 255)) + b"\x00" * 6 + struct.pack("<HHI", encoded, encoded, 0)

    def mesh(self, overlap: bool = False) -> dict:
        vertices = [
            {"position": [-0.5, -0.5, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 1.0]},
            {"position": [0.5, -0.5, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 1.0]},
            {"position": [0.5, 0.5, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 0.0]},
            {"position": [-0.5, 0.5, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 0.0]},
        ]
        indices = [0, 1, 2, 0, 2, 3]
        if overlap:
            # Submit the raised triangle first.  The later base triangles overlap
            # it in screen space and must fail the depth test.
            vertices.extend(
                [
                    {"position": [-0.24, -0.24, 0.18], "normal": [0.0, 0.0, 1.0], "uv0": [0.75, 0.25]},
                    {"position": [0.24, -0.24, 0.18], "normal": [0.0, 0.0, 1.0], "uv0": [0.95, 0.25]},
                    {"position": [0.0, 0.24, 0.18], "normal": [0.0, 0.0, 1.0], "uv0": [0.85, 0.05]},
                ]
            )
            indices = [4, 5, 6] + indices
        return {
            "schema": "c3x.normalized_mesh.v0",
            "topology": {"primitive": "triangles", "indices": indices},
            "vertices": vertices,
        }

    def write_fixture(self, root: Path, overlap: bool = False):
        pack = root / "pack"
        (pack / "meshes").mkdir(parents=True)
        (pack / "materials").mkdir()
        (pack / "textures").mkdir()
        (pack / "meshes" / "terrain.json").write_text(
            json.dumps(self.mesh(overlap)), encoding="utf-8"
        )
        (pack / "materials" / "grassland.json").write_text(
            json.dumps(
                {
                    "schema": "c3x.material.v0",
                    "base_color": {
                        "texture": "textures/grassland.dds",
                        "format": "BC3_UNORM_SRGB",
                        "color_space": "srgb",
                        "uv_channel": "uv0",
                    },
                }
            ),
            encoding="utf-8",
        )
        colors = ((48, 112, 36), (82, 148, 48), (116, 180, 68), (61, 128, 42))
        payload = b"".join(self.bc3_block(color) for color in colors)
        dds_info = {
            "width": 8,
            "height": 8,
            "mip_count": 1,
            "dxgi_format": 78,
            "payload_bytes": len(payload),
        }
        (pack / "textures" / "grassland.dds").write_bytes(
            c3x_asset_compiler.make_dds_dx10_header(dds_info) + payload
        )
        (pack / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "c3x.asset_pack.v0",
                    "name": "SyntheticTerrain",
                    "assets": {
                        "terrain/grassland/base": {
                            "type": "terrain",
                            "mesh": "meshes/terrain.json",
                            "material": "materials/grassland.json",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        definitions_path = root / "default.custom_rendering.txt"
        definitions_path.write_text(
            """#Profile
id = default
terrain = replace
features = civ3
roads = civ3
rivers = civ3
improvements = civ3
resources = civ3
cities = civ3
units = civ3
effects = civ3
missing_asset = fallback
environment = earthlike
#Pack
id = synthetic
path = mod:pack
#Asset
id = grassland.base
pack = synthetic
asset = terrain/grassland/base
scale = 1.0
#Rule
id = terrain.grassland
category = terrain
priority = 100
terrain_type = grassland
asset = grassland.base
variant_selection = coordinate-hash
replacement = replace
#Environment
id = earthlike
day_night_source = c3x
season_source = c3x
sunrise_hour = 6
sunset_hour = 18
sun_azimuth_degrees = 135
noon_sun_color = 255, 244, 220
midnight_ambient_color = 22, 30, 52
night_exposure = 0.35
seasonal_materials = true
""",
            encoding="utf-8",
        )
        definitions = definition_parser.parse_definition_file(
            definitions_path, "default", root, root
        )
        catalog = definition_parser.merge_layers([("default", definitions)])
        loader = renderer.PackAssetLoader(catalog, mod_root=root, scenario_root=root)
        return catalog, loader

    def scene(self, *, width: int = 640, height: int = 480, hour: int = 12, season: str = "summer"):
        scene = scene_contract.load_scene(
            Path("Renderer/samples/scenes/grassland_viewport.scene.json")
        )
        scene = copy.deepcopy(scene)
        scene["viewport"]["width_px"] = width
        scene["viewport"]["height_px"] = height
        scene["viewport"]["map_rect_px"]["width"] = width
        scene["viewport"]["map_rect_px"]["height"] = height - 64
        scene["environment"]["hour"] = hour
        scene["environment"]["season"] = season
        scene["scene_id"] = scene_contract.scene_identifier(scene)
        return scene

    def mean_drawn_luminance(self, frame: renderer.RenderFrame) -> float:
        drawn = [
            sum(pixel) / 3.0
            for pixel, owner in zip(frame.canvas.pixels, frame.owner_buffer)
            if owner is not None
        ]
        return sum(drawn) / len(drawn)

    def test_recorded_scene_renders_deterministically_at_two_sizes_and_preserves_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, loader = self.write_fixture(root)
            scene = self.scene()
            device = renderer.WholeViewportRenderer(640, 480)
            first = device.render(scene, catalog, loader)
            repeated = device.render(scene, catalog, loader)
            first_png = root / "first.png"
            repeated_png = root / "repeated.png"
            write_png(first.canvas, first_png)
            write_png(repeated.canvas, repeated_png)
            first_png_bytes = first_png.read_bytes()
            repeated_png_bytes = repeated_png.read_bytes()

            larger = device.render(self.scene(width=800, height=600), catalog, loader)
            larger_repeated = device.render(self.scene(width=800, height=600), catalog, loader)
            larger_png = root / "larger.png"
            larger_repeated_png = root / "larger_repeated.png"
            write_png(larger.canvas, larger_png)
            write_png(larger_repeated.canvas, larger_repeated_png)
            larger_png_bytes = larger_png.read_bytes()
            larger_repeated_png_bytes = larger_repeated_png.read_bytes()

        self.assertEqual(first.canvas.pixels, repeated.canvas.pixels)
        self.assertEqual(first_png_bytes, repeated_png_bytes)
        self.assertEqual(larger.canvas.pixels, larger_repeated.canvas.pixels)
        self.assertEqual(larger_png_bytes, larger_repeated_png_bytes)
        self.assertGreater(first.stats["non_background_pixels"], 10000)
        self.assertGreater(larger.stats["non_background_pixels"], 10000)
        self.assertEqual(first.stats["rendered_instances"], 4)
        self.assertEqual(first.stats["fallback_instances"], 3)
        for instance_id in first.stats["rendered_ids"]:
            self.assertEqual(first.stats["anchor_owners"][instance_id], instance_id)
        map_bottom = scene["viewport"]["map_rect_px"]["height"]
        self.assertTrue(
            all(pixel == BACKGROUND for pixel in first.canvas.pixels[map_bottom * first.canvas.width :])
        )

    def test_depth_buffer_keeps_raised_geometry_when_base_is_submitted_later(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, loader = self.write_fixture(root, overlap=True)
            frame = renderer.WholeViewportRenderer(640, 480).render(self.scene(), catalog, loader)

        self.assertGreater(frame.stats["pixels_depth_rejected"], 0)
        self.assertIn(0, frame.primitive_buffer)
        raised_depths = [
            depth
            for depth, primitive in zip(frame.depth_buffer, frame.primitive_buffer)
            if primitive == 0
        ]
        self.assertTrue(raised_depths)
        self.assertTrue(all(math.isfinite(depth) for depth in raised_depths))

    def test_hour_and_season_change_material_response_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, loader = self.write_fixture(root)
            device = renderer.WholeViewportRenderer(640, 480)
            noon = device.render(self.scene(hour=12, season="summer"), catalog, loader)
            midnight = device.render(self.scene(hour=0, season="summer"), catalog, loader)
            winter = device.render(self.scene(hour=12, season="winter"), catalog, loader)

        self.assertNotEqual(noon.canvas.pixels, midnight.canvas.pixels)
        self.assertNotEqual(noon.canvas.pixels, winter.canvas.pixels)
        self.assertGreater(self.mean_drawn_luminance(noon), self.mean_drawn_luminance(midnight) * 4)
        self.assertNotEqual(noon.stats["lighting"]["season_tint"], winter.stats["lighting"]["season_tint"])

    def test_create_resize_recreate_and_teardown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, loader = self.write_fixture(root)
            device = renderer.WholeViewportRenderer(640, 480)
            self.assertEqual((device.state, device.generation), ("ready", 1))
            device.render(self.scene(), catalog, loader)
            self.assertEqual(device.generation, 1)
            resized = device.render(self.scene(width=800, height=600), catalog, loader)
            self.assertEqual((resized.canvas.width, resized.canvas.height), (800, 600))
            self.assertEqual(device.generation, 2)
            device.close()
            device.close()
            self.assertEqual(device.state, "closed")
            self.assertEqual((device.depth_buffer, device.owner_buffer), ([], []))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                device.render(self.scene(), catalog, loader)

    def test_pack_root_escape_and_missing_payload_become_safe_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, loader = self.write_fixture(root)
            (root / "pack" / "meshes" / "terrain.json").unlink()
            frame = renderer.WholeViewportRenderer(640, 480).render(self.scene(), catalog, loader)

        self.assertEqual(frame.stats["rendered_instances"], 0)
        self.assertEqual(frame.stats["fallback_instances"], 7)
        self.assertIn("grassland.base", frame.stats["asset_availability_errors"])
        self.assertEqual(frame.stats["non_background_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
