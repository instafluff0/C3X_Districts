#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import write_png
from Renderer.scenes import scene_contract
from Renderer.standalone import test_whole_viewport_renderer as renderer_tests
from Renderer.tools import process_scene_export_batch as batch
from Renderer.tools import render_fixture_matrix as matrix


class SceneExportBatchTests(unittest.TestCase):
    def plan(self, scene_name: str = "populated_viewport.scene.json") -> dict:
        return {
            "schema": batch.PLAN_SCHEMA,
            "batch_id": "integration-smoke",
            "source": {
                "id": "synthetic-map",
                "kind": "synthetic",
                "label": "Synthetic populated map",
                "artifact": None,
            },
            "fixtures": [
                {
                    "id": "populated_viewport",
                    "scene": scene_name,
                    "required_categories": ["terrain", "resource", "city", "unit"],
                    "in_game_evidence": {
                        "screenshot": None,
                        "layer_checks": {name: "pending" for name in batch.LAYER_CHECKS},
                    },
                }
            ],
        }

    def synthetic_inputs(self, root: Path):
        helper = renderer_tests.WholeViewportRendererTests(methodName="runTest")
        catalog, loader = helper.write_fixture(root)
        scene = helper.scene()
        references_path = Path("Renderer/samples/validation/reference_metadata.json")
        references = matrix.validate_reference_catalog(
            json.loads(references_path.read_text(encoding="utf-8"))
        )
        return helper, catalog, loader, scene, references

    def snapshot(self, root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_plan_enforces_fixture_and_screenshot_names(self) -> None:
        plan = self.plan("wrong.scene.json")
        with self.assertRaisesRegex(ValueError, "canonical filename"):
            batch.validate_plan(plan)

        plan = self.plan()
        evidence = plan["fixtures"][0]["in_game_evidence"]
        evidence["screenshot"] = "wrong.png"
        with self.assertRaisesRegex(ValueError, "canonical filename"):
            batch.validate_plan(plan)

    def test_plan_does_not_allow_layer_claims_without_a_screenshot(self) -> None:
        plan = self.plan()
        plan["fixtures"][0]["in_game_evidence"]["layer_checks"]["fog"] = "pass"
        with self.assertRaisesRegex(ValueError, "stay pending"):
            batch.validate_plan(plan)

    def test_batch_canonicalizes_inventories_renders_and_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog, loader, scene, references = self.synthetic_inputs(root)
            plan_path = root / "populated-plan.json"
            scene_path = root / "populated_viewport.scene.json"
            scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
            plan_path.write_text(json.dumps(self.plan(), indent=2), encoding="utf-8")
            output = root / "output"
            kwargs = {
                "mod_root": root,
                "definition_records": [
                    {
                        "layer": "default",
                        "scope": "mod",
                        "path": "default.custom_rendering.txt",
                        "sha256": "synthetic-definition",
                    }
                ],
                "reference_record": {
                    "scope": "test",
                    "path": "reference_metadata.json",
                    "sha256": "synthetic-references",
                },
                "viewports": ((384, 320),),
                "hours": (0, 12),
                "seasons": ("summer", "winter"),
                "thumbnail_size": (64, 48),
            }
            first = batch.process_export_batch(
                plan_path, catalog, loader, references, output, **kwargs
            )
            first_files = self.snapshot(output)
            repeated = batch.process_export_batch(
                plan_path, catalog, loader, references, output, **kwargs
            )
            repeated_files = self.snapshot(output)

        self.assertEqual(first, repeated)
        self.assertEqual(first_files, repeated_files)
        self.assertTrue(first["summary"]["offline_passed"])
        self.assertFalse(first["summary"]["full_m5_2_evidence_passed"])
        self.assertEqual(first["summary"]["matched_in_game_evidence_count"], 0)
        fixture = first["fixtures"][0]
        self.assertEqual(fixture["scene_validation"]["missing_required_categories"], [])
        self.assertEqual(
            {key: fixture["scene_validation"]["category_counts"][key] for key in ("terrain", "resource", "city", "unit")},
            {"terrain": 4, "resource": 1, "city": 1, "unit": 1},
        )
        self.assertEqual(
            first_files["canonical_scenes/populated_viewport.scene.json"],
            scene_contract.canonical_json(scene).encode("utf-8"),
        )
        self.assertIn("fixtures/populated_viewport/contact_sheet.png", first_files)
        self.assertIn("full M5.2 evidence: PENDING", first_files["contact_sheet.html"].decode("utf-8"))
        self.assertEqual(
            first_files["report.json"], matrix.canonical_bytes(json.loads(first_files["report.json"]))
        )

    def test_missing_required_category_fails_only_the_offline_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog, loader, scene, references = self.synthetic_inputs(root)
            plan = self.plan()
            plan["fixtures"][0]["required_categories"].append("effect")
            plan_path = root / "plan.json"
            (root / "populated_viewport.scene.json").write_text(
                scene_contract.canonical_json(scene), encoding="utf-8"
            )
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            report = batch.process_export_batch(
                plan_path,
                catalog,
                loader,
                references,
                root / "output",
                mod_root=root,
                definition_records=[],
                reference_record={"scope": "test", "path": "references", "sha256": "test"},
                viewports=((384, 320),),
                hours=(12,),
                seasons=("summer",),
                thumbnail_size=(32, 24),
            )

        self.assertFalse(report["summary"]["offline_passed"])
        self.assertEqual(
            report["fixtures"][0]["scene_validation"]["missing_required_categories"],
            ["effect"],
        )

    def test_matched_screenshot_requires_all_layer_reviews_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "plan.json"
            screenshot = root / "populated_viewport__ingame.png"
            write_png(Canvas(2, 2), screenshot)
            evidence = self.plan()["fixtures"][0]["in_game_evidence"]
            evidence["screenshot"] = screenshot.name
            evidence["layer_checks"] = {name: "pass" for name in batch.LAYER_CHECKS}
            record = batch._evidence_record(plan_path, "populated_viewport", evidence, root)
            self.assertTrue(record["passed"])
            self.assertEqual(record["status"], "reviewed")
            evidence["layer_checks"]["fog"] = "fail"
            record = batch._evidence_record(plan_path, "populated_viewport", evidence, root)
            self.assertFalse(record["passed"])
            self.assertEqual(record["status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
