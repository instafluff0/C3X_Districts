from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.test_compound_unit_asset_importer import _skeleton
from Renderer.tools.asset_compiler.test_normalized_animation import build_clip_bytes
from Renderer.tools.asset_compiler.unit_family_pose_cache_builder import build_family_pose_caches


class UnitFamilyPoseCacheBuilderTests(unittest.TestCase):
    def test_discovers_units_and_deduplicates_alias_clip_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            (pack / "units").mkdir()
            (pack / "components").mkdir()
            (pack / "skeletons").mkdir()
            (pack / "animations").mkdir()
            (pack / "skeletons/body.json").write_text(json.dumps(_skeleton("body")), encoding="utf-8")
            (pack / "components/body.json").write_text(
                json.dumps({"binding_mode": "vertex_skin", "skeleton": "skeletons/body.json"}),
                encoding="utf-8",
            )
            (pack / "animations/idle.c3anim").write_bytes(build_clip_bytes())
            recipe = {
                "schema": "c3x.unit_recipe.v0",
                "unit_id": "unit/arbitrary",
                "components": [{"asset": "unit/arbitrary/body"}],
                "actions": {
                    "idle": "animation/unit/arbitrary/idle",
                    "fidget": "animation/unit/arbitrary/fidget",
                },
            }
            (pack / "units/arbitrary.json").write_text(json.dumps(recipe), encoding="utf-8")
            base = {
                "clip": "animations/idle.c3anim",
                "binding_status": "validated_raw_clip_name_binding",
            }
            manifest = {
                "schema": "c3x.unit_pack.v0",
                "name": "UnitFamilyLab",
                "units": {"unit/arbitrary": {"recipe": "units/arbitrary.json"}},
                "assets": {"unit/arbitrary/body": {"component": "components/body.json"}},
                "animations": {
                    "animation/unit/arbitrary/idle": base,
                    "animation/unit/arbitrary/fidget": {**base, "alias_of": "animation/unit/arbitrary/idle"},
                },
            }
            (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = build_family_pose_caches(pack)
            self.assertEqual(1, report["unique_component_pose_caches"])
            self.assertEqual(2, report["skinned_component_action_bindings"])
            result = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                result["animations"]["animation/unit/arbitrary/idle"]["pose_caches"],
                result["animations"]["animation/unit/arbitrary/fidget"]["pose_caches"],
            )


if __name__ == "__main__":
    unittest.main()
