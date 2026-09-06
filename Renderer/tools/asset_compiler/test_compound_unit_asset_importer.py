from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.compound_unit_asset_importer import (
    IDENTITY,
    _validate_generic_recipe,
    compile_compound_units,
    load_source_sets,
)
from Renderer.tools.asset_compiler.compound_unit_composition_validator import (
    validate_compositions,
)
from Renderer.tools.asset_compiler.test_normalized_animation import build_clip_bytes
from Renderer.tools.asset_compiler.unit_member_resolver import ASSETS_ROOT


def _skeleton(asset_id: str) -> dict:
    return {
        "schema": "c3x.normalized_skeleton.v0",
        "asset_id": asset_id,
        "track_group": "Actor",
        "matrix_convention": "row_major_row_vector",
        "position_unit": "tile",
        "bones": [
            {
                "name": "Bone",
                "parent": -1,
                "local": {
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "scale_shear": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                },
                "inverse_bind_matrix": IDENTITY,
            }
        ],
    }


class CompoundUnitAssetImporterTests(unittest.TestCase):
    def test_source_sets_cover_four_generic_socket_families(self) -> None:
        source = load_source_sets()
        self.assertEqual(
            ["horseman", "great_general_classical", "catapult", "tank"],
            [item["slug"] for item in source["compositions"]],
        )
        self.assertEqual(
            ["RiderAttach", "RiderAttach", "Operator", "GunnerAttach"],
            [item["children"][0]["socket"] for item in source["compositions"]],
        )
        self.assertEqual(
            "recipe_data_only_no_unit_name_branches",
            source["instance_contract"]["runtime_dispatch"],
        )
        tank = source["compositions"][3]
        self.assertEqual(
            ["tankAll", "tankBody"],
            tank["parent"]["component_attachment_bones"]["TeamColor"],
        )
        general = source["compositions"][1]
        self.assertEqual("defend", general["actions"]["attack"]["alias"])
        self.assertEqual("idle", general["actions"]["victory"]["alias"])

    def test_generic_recipe_and_animation_validator_accept_arbitrary_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            (pack / "units").mkdir()
            (pack / "skeletons").mkdir()
            (pack / "animations").mkdir()
            for node in ("parent", "child"):
                (pack / "skeletons" / f"{node}.json").write_text(
                    json.dumps(_skeleton(f"unit/test/{node}/skeleton")), encoding="utf-8"
                )
                (pack / "animations" / f"{node}.c3anim").write_bytes(build_clip_bytes())
            recipe = {
                "schema": "c3x.unit_composition.v0",
                "unit_id": "unit/test",
                "root_node": "parent",
                "nodes": {
                    "parent": {
                        "components": [{"asset": "unit/test/parent"}],
                        "animation_driver": "unit/test/parent",
                        "skeleton": "skeletons/parent.json",
                    },
                    "child": {
                        "components": [{"asset": "unit/test/child"}],
                        "animation_driver": "unit/test/child",
                        "skeleton": "skeletons/child.json",
                    },
                },
                "joints": [
                    {
                        "parent": "parent",
                        "child": "child",
                        "socket": "modder_socket",
                        "parent_bone": "Bone",
                        "child_root_bone": "Bone",
                        "local_transform": IDENTITY,
                    }
                ],
                "actions": {
                    "idle": {
                        "loop": True,
                        "timeline": "shared_normalized_phase",
                        "node_clips": {
                            "parent": "animation/unit/test/parent/idle",
                            "child": "animation/unit/test/child/idle",
                        },
                    }
                },
                "instance_contract": {"hud": "one_retained_native_parent_hud"},
            }
            (pack / "units/test.json").write_text(json.dumps(recipe), encoding="utf-8")
            manifest = {
                "schema": "c3x.unit_pack.v0",
                "name": "CompoundUnitLab",
                "units": {"unit/test": {"recipe": "units/test.json", "type": "compound"}},
                "assets": {
                    "unit/test/parent": {},
                    "unit/test/child": {},
                },
                "animations": {
                    "animation/unit/test/parent/idle": {
                        "clip": "animations/parent.c3anim",
                        "loop": True,
                        "binding_status": "validated_node_local_raw_clip",
                        "group_index": 0,
                        "matched_tracks": 1,
                    },
                    "animation/unit/test/child/idle": {
                        "clip": "animations/child.c3anim",
                        "loop": True,
                        "binding_status": "validated_node_local_raw_clip",
                        "group_index": 0,
                        "matched_tracks": 1,
                    },
                },
                "composition_contract": {
                    "runtime_dispatch": "recipe_data_only_no_unit_name_branches"
                },
            }
            (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            _validate_generic_recipe(recipe, manifest, pack)
            report = validate_compositions(pack)
            self.assertEqual(0.0, report["units"]["unit/test"]["actions"]["idle"]["maximum_socket_separation"])

    @unittest.skipUnless(ASSETS_ROOT.is_dir(), "installed Civ VI source assets unavailable")
    def test_local_source_compiler_resolves_horse_and_tank_sockets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = compile_compound_units(
                ASSETS_ROOT,
                pack=root / "pack",
                report_path=root / "report.json",
                only_slugs={"horseman", "tank"},
            )
            self.assertEqual(2, report["outputs"]["compositions"])
            horse = json.loads((root / "pack/units/horseman_composition.json").read_text(encoding="utf-8"))
            tank = json.loads((root / "pack/units/tank_composition.json").read_text(encoding="utf-8"))
            self.assertEqual("RiderAttach", horse["joints"][0]["parent_bone"])
            self.assertEqual("GunnerAttach", tank["joints"][0]["parent_bone"])
            team_color = next(
                component
                for component in tank["nodes"]["vehicle"]["components"]
                if component["role"] == "TeamColor"
            )
            self.assertEqual("tankAll", team_color["attachment_bone"])
            self.assertEqual("atomic_complete_unit_body", horse["instance_contract"]["failure"])
            self.assertEqual(
                {"idle", "fidget", "move", "fortify", "attack", "defend", "death", "victory"},
                set(horse["actions"]),
            )


if __name__ == "__main__":
    unittest.main()
