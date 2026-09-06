from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.compound_unit_pose_cache_builder import build_pose_caches
from Renderer.tools.asset_compiler.test_compound_unit_asset_importer import _skeleton
from Renderer.tools.asset_compiler.test_normalized_animation import build_clip_bytes


class CompoundUnitPoseCacheBuilderTests(unittest.TestCase):
    def test_bakes_once_and_reuses_cache_for_action_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            (pack / "units").mkdir()
            (pack / "skeletons").mkdir()
            (pack / "animations").mkdir()
            (pack / "skeletons/node.json").write_text(json.dumps(_skeleton("node")), encoding="utf-8")
            (pack / "animations/idle.c3anim").write_bytes(build_clip_bytes())
            animation_id = "animation/unit/test/node/idle"
            action = {
                "loop": True,
                "timeline": "shared_normalized_phase",
                "node_clips": {"node": animation_id},
            }
            recipe = {
                "schema": "c3x.unit_composition.v0",
                "unit_id": "unit/test",
                "root_node": "node",
                "nodes": {"node": {"skeleton": "skeletons/node.json"}},
                "joints": [],
                "actions": {"idle": action, "fidget": {**action, "loop": False}},
            }
            (pack / "units/test.json").write_text(json.dumps(recipe), encoding="utf-8")
            manifest = {
                "schema": "c3x.unit_pack.v0",
                "units": {"unit/test": {"recipe": "units/test.json", "type": "compound"}},
                "animations": {
                    animation_id: {
                        "clip": "animations/idle.c3anim",
                        "binding_status": "validated_node_local_raw_clip",
                        "group_index": 0,
                    }
                },
            }
            (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = build_pose_caches(pack)
            self.assertEqual(1, report["unique_pose_caches"])
            self.assertEqual(2, report["logical_node_action_bindings"])
            result = json.loads((pack / "units/test.json").read_text(encoding="utf-8"))
            self.assertEqual(
                result["actions"]["idle"]["node_pose_caches"],
                result["actions"]["fidget"]["node_pose_caches"],
            )


if __name__ == "__main__":
    unittest.main()
