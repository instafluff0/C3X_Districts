#!/usr/bin/env python3
import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.scenes import scene_contract
from Renderer.standalone import test_whole_viewport_renderer as renderer_tests
from Renderer.tools import render_fixture_matrix as matrix


class FixtureMatrixTests(unittest.TestCase):
    def synthetic_inputs(self, root: Path):
        helper = renderer_tests.WholeViewportRendererTests(methodName="runTest")
        catalog, loader = helper.write_fixture(root)
        scene = helper.scene()
        references = matrix.validate_reference_catalog(
            json.loads(
                Path("Renderer/samples/validation/reference_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        inputs = {
            "scene": {"scope": "test", "path": "grassland_viewport.scene.json", "sha256": "scene-fixture"},
            "definitions": [{"layer": "default", "scope": "test", "path": "default.custom_rendering.txt", "sha256": "definition-fixture"}],
            "reference_catalog": {"scope": "test", "path": "reference_metadata.json", "sha256": "reference-fixture"},
        }
        return scene, catalog, loader, references, inputs

    def snapshot(self, root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_full_two_size_four_hour_four_season_matrix_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene, catalog, loader, references, inputs = self.synthetic_inputs(root)
            output = root / "matrix"
            kwargs = {
                "scene_label": "grassland_viewport",
                "input_records": inputs,
                "references": references,
                "viewports": ((384, 320), (512, 384)),
                "hours": (0, 6, 12, 18),
                "seasons": ("summer", "fall", "winter", "spring"),
                "thumbnail_size": (80, 60),
            }
            first = matrix.render_fixture_matrix(scene, catalog, loader, output, **kwargs)
            first_files = self.snapshot(output)
            repeated = matrix.render_fixture_matrix(scene, catalog, loader, output, **kwargs)
            repeated_files = self.snapshot(output)

        self.assertEqual(first, repeated)
        self.assertEqual(first_files, repeated_files)
        self.assertEqual(first["summary"]["cell_count"], 32)
        self.assertEqual(first["summary"]["comparison_count"], 16)
        self.assertTrue(first["summary"]["passed"])
        self.assertEqual(len([name for name in first_files if name.startswith("images/")]), 32)
        self.assertEqual(set(first_files) - {name for name in first_files if name.startswith("images/")}, {"contact_sheet.png", "manifest.json"})
        self.assertEqual({cell["renderer_generation"] for cell in first["cells"]}, {1, 2})
        self.assertTrue(all(cell["metrics"]["passed"] for cell in first["cells"]))
        self.assertTrue(all(comparison["passed"] for comparison in first["comparisons"]))
        parsed_manifest = json.loads(first_files["manifest.json"])
        self.assertEqual(first_files["manifest.json"], matrix.canonical_bytes(parsed_manifest))
        self.assertEqual(parsed_manifest["inputs"]["loaded_assets"][0]["logical_asset_id"], "terrain/grassland/base")
        self.assertEqual(
            {reference["kind"] for reference in parsed_manifest["inputs"]["references"]["references"]},
            {"structural_regression", "art_direction"},
        )
        contact = first_files["contact_sheet.png"]
        self.assertEqual(contact[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack_from(">II", contact, 16), (104 + 4 + 4 * 84, 18 + 4 + 8 * 64))

    def test_metrics_cover_mapping_bounds_depth_anchors_luminance_and_color(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene, catalog, loader, references, inputs = self.synthetic_inputs(root)
            result = matrix.render_fixture_matrix(
                scene,
                catalog,
                loader,
                root / "matrix",
                scene_label="metrics",
                input_records=inputs,
                references=references,
                viewports=((640, 480),),
                hours=(0, 12),
                seasons=("summer", "winter"),
                thumbnail_size=(64, 48),
            )

        cell = result["cells"][0]
        self.assertEqual(
            set(cell["metrics"]),
            {"passed", "checks", "mapping", "bounds", "anchors", "depth", "color"},
        )
        self.assertEqual(cell["metrics"]["mapping"]["coverage_basis_points"], 10000)
        self.assertEqual(cell["metrics"]["mapping"]["excluded_civ3_owned_instances"], 3)
        self.assertEqual(cell["metrics"]["bounds"]["outside_map_rect_pixels"], 0)
        self.assertEqual(cell["metrics"]["anchors"]["misses"], 0)
        self.assertEqual(cell["metrics"]["depth"]["invalid_values"], 0)
        self.assertGreater(cell["metrics"]["color"]["unique_colors"], 1)
        self.assertEqual(len(cell["metrics"]["color"]["luminance_histogram_16"]), 16)
        self.assertEqual(
            {comparison["kind"] for comparison in result["comparisons"]},
            {"time_response", "season_response"},
        )

    def test_reference_contract_rejects_cross_engine_pixel_gates_and_inferred_hours(self) -> None:
        reference = {
            "schema": matrix.REFERENCE_SCHEMA,
            "references": [
                {
                    "id": "external.reference",
                    "kind": "art_direction",
                    "source": "external screenshot",
                    "availability": "local",
                    "comparison_mode": "exact_hash",
                    "time_basis": "lighting_phase_only",
                    "phases": ["noon"],
                    "purpose": "test",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "pixel-equality"):
            matrix.validate_reference_catalog(reference)
        reference["references"][0]["comparison_mode"] = "qualitative"
        reference["references"][0]["exact_hour"] = 12
        with self.assertRaisesRegex(ValueError, "lighting phases"):
            matrix.validate_reference_catalog(reference)

    def test_viewport_reframe_preserves_projection_and_scene_validation(self) -> None:
        scene = scene_contract.load_scene(
            Path("Renderer/samples/scenes/grassland_viewport.scene.json")
        )
        resized = matrix.resize_scene(scene, 1024, 768)
        shift_x = resized["projection"]["origin_px"]["x"] - scene["projection"]["origin_px"]["x"]
        shift_y = resized["projection"]["origin_px"]["y"] - scene["projection"]["origin_px"]["y"]
        self.assertEqual((shift_x, shift_y), (192, 144))
        for before, after in zip(scene["tiles"], resized["tiles"]):
            self.assertEqual(after["anchor_px"]["x"] - before["anchor_px"]["x"], shift_x)
            self.assertEqual(after["anchor_px"]["y"] - before["anchor_px"]["y"], shift_y)
        self.assertEqual(resized["viewport"]["map_rect_px"], {"x": 0, "y": 0, "width": 1024, "height": 704})
        self.assertEqual(scene_contract.validate_scene(resized), resized)

    def test_matrix_argument_parsers_are_strict_and_deduplicate(self) -> None:
        self.assertEqual(matrix.parse_viewports("640x480, 640X480,800x600"), ((640, 480), (800, 600)))
        self.assertEqual(matrix.parse_hours("0,6,0,18"), (0, 6, 18))
        self.assertEqual(matrix.parse_seasons("summer,autumn,fall"), ("summer", "fall"))
        with self.assertRaisesRegex(ValueError, "WIDTHxHEIGHT"):
            matrix.parse_viewports("wide")
        with self.assertRaisesRegex(ValueError, "outside"):
            matrix.parse_hours("24")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            matrix.parse_seasons("monsoon")


if __name__ == "__main__":
    unittest.main()
