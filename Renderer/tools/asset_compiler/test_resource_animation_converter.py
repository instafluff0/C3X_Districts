from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import resource_animation_converter as converter
from Renderer.tools.asset_compiler.test_normalized_animation import build_clip_bytes


class ResourceAnimationConverterTests(unittest.TestCase):
    def test_validate_only_checks_normalized_outputs_and_binding_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            pack = root / "pack"
            raw_clip = raw / "environment_clutter" / "0001_idle.fgx"
            normalized_clip = pack / "animations/resources/environment_clutter/0001_idle.c3anim"
            raw_clip.parent.mkdir(parents=True)
            normalized_clip.parent.mkdir(parents=True)
            raw_clip.write_bytes(b"raw fixture")
            normalized_clip.write_bytes(build_clip_bytes())
            extraction = {
                "schema": "c3x.resource_animation_extract.v0",
                "unique_clips": [
                    {
                        "source_package": "environment/clutter",
                        "name": "Idle",
                        "table_index": 1,
                        "raw_fgx": "environment_clutter/0001_idle.fgx",
                        "normalized_clip": "animations/resources/environment_clutter/0001_idle.c3anim",
                        "translation_scale": 1.0 / 12.0,
                    }
                ],
            }
            extraction_path = root / "extract.json"
            report_path = root / "conversion.json"
            extraction_path.write_text(json.dumps(extraction), encoding="utf-8")

            report = converter.convert_resource_animations(
                extraction_path,
                raw,
                pack,
                report_path,
                root / "unused.bat",
                validate_only=True,
            )

        self.assertEqual(report["summary"]["clips"], 1)
        self.assertEqual(report["summary"]["body_profiles_pending"], 1)
        self.assertEqual(report["clips"][0]["frame_count"], 3)
        self.assertEqual(
            report["clips"][0]["binding_status"],
            "model_aware_pose_cache_required",
        )

    def test_rejects_unsafe_report_paths(self) -> None:
        document = {
            "schema": "c3x.resource_animation_extract.v0",
            "unique_clips": [
                {
                    "source_package": "environment/clutter",
                    "table_index": 1,
                    "raw_fgx": "../escape.fgx",
                    "normalized_clip": "animations/clip.c3anim",
                    "translation_scale": 1.0 / 12.0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extract.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                converter.load_extract_report(path)


if __name__ == "__main__":
    unittest.main()
