#!/usr/bin/env python3
"""Validate converted unit-family clips against every normalized proof component."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation, normalized_skin


SOCKET_PROFILE = {
    "Root": {"bone": "Root", "status": "identity_or_matching_root"},
    "ArmBand": {"bone": "Lure", "status": "inferred_lab_profile"},
    "Hat": {"bone": "Head", "status": "inferred_lab_profile"},
    "WeaponPrimary": {"bone": "Inven_R_Hand", "status": "inferred_lab_profile"},
    "WeaponSecondary": {"bone": "Inven_L_Hand", "status": "inferred_lab_profile"},
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _best_group(clip: normalized_animation.Clip, bone_names: set[str]) -> tuple[int, set[str]]:
    ranked = sorted(
        (
            (len({track.name for track in group.tracks} & bone_names), -index, index)
            for index, group in enumerate(clip.groups)
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] == 0:
        raise ValueError("animation has no track group matching the component skeleton")
    index = ranked[0][2]
    return index, {track.name for track in clip.groups[index].tracks} & bone_names


def validate_family_actions(pack: Path) -> dict[str, Any]:
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.unit_pack.v0" or manifest.get("name") != "UnitFamilyLab":
        raise ValueError("unsupported unit-family pack")
    action_contract = json.loads((pack / manifest["action_contract"]).read_text(encoding="utf-8"))
    if action_contract.get("schema") != "c3x.unit_action_conversion.v0":
        raise ValueError("unit-family pack has no valid action-conversion contract")
    basic_actions = set(action_contract["actions"])
    components = {}
    for asset_id, asset in manifest["assets"].items():
        document = json.loads((pack / asset["component"]).read_text(encoding="utf-8"))
        skeleton = normalized_skin.load_skeleton(pack / document["skeleton"])
        components[asset_id] = (document, skeleton)
    unit_reports = {}
    for unit_id, unit_entry in manifest["units"].items():
        recipe = json.loads((pack / unit_entry["recipe"]).read_text(encoding="utf-8"))
        if not basic_actions.issubset(recipe["actions"]):
            raise ValueError(f"{unit_id} does not bind every basic action")
        driver_id = recipe["animation_driver"]
        driver_document, driver_skeleton = components[driver_id]
        driver_names = {bone["name"] for bone in driver_skeleton["bones"]}
        action_reports = {}
        for action, animation_id in recipe["actions"].items():
            animation = manifest["animations"].get(animation_id, {})
            expected_loop = action_contract["actions"].get(action, {}).get("playback") == "loop"
            if action in basic_actions and animation.get("loop") != expected_loop:
                raise ValueError(f"{animation_id} does not match the basic playback policy")
            alias_of = animation.get("alias_of")
            if alias_of:
                target = manifest["animations"].get(alias_of, {})
                if animation.get("clip") != target.get("clip"):
                    raise ValueError(f"{animation_id} does not reuse its aliased clip")
            relative = animation.get("clip")
            if not relative:
                raise ValueError(f"{animation_id} has not been converted")
            path = pack / relative
            clip = normalized_animation.load_clip(path)
            driver_group, driver_common = _best_group(clip, driver_names)
            if len(driver_common) < recipe["minimum_matching_tracks"]:
                raise ValueError(
                    f"{animation_id} matches only {len(driver_common)} driver tracks; "
                    f"expected {recipe['minimum_matching_tracks']}"
                )
            bindings = {}
            for component_record in recipe["components"]:
                asset_id = component_record["asset"]
                document, skeleton = components[asset_id]
                if document["binding_mode"] != "vertex_skin":
                    continue
                bone_names = {bone["name"] for bone in skeleton["bones"]}
                group_index, common = _best_group(clip, bone_names)
                sample_time = clip.duration * (0.37 if animation["loop"] else 0.61)
                pose = normalized_skin.sample_pose(
                    skeleton, clip, group_index, sample_time, animation["loop"]
                )
                world_matrices = normalized_skin.world_matrices(skeleton, pose)
                mesh_paths = document.get("meshes") or [document["mesh"]]
                sampled_vertices = 0
                for mesh_relative in mesh_paths:
                    mesh = normalized_skin.load_mesh(
                        pack / mesh_relative, len(skeleton["bones"])
                    )
                    normalized_skin.validate_rest_pose(mesh, skeleton)
                    positions = normalized_skin.skin_positions(
                        mesh, skeleton, world_matrices
                    )
                    if not all(
                        math.isfinite(value)
                        for position in positions
                        for value in position
                    ):
                        raise ValueError(
                            f"{animation_id}/{asset_id} produces non-finite skinning"
                        )
                    sampled_vertices += len(positions)
                bindings[asset_id] = {
                    "group_index": group_index,
                    "group_name": clip.groups[group_index].name,
                    "matched_tracks": len(common),
                    "mesh_parts": len(mesh_paths),
                    "sampled_vertices": sampled_vertices,
                    "rest_fallback_bones": sorted(bone_names - common),
                }
            animation["binding_status"] = "validated_raw_clip_name_binding"
            animation["driver_group_index"] = driver_group
            action_reports[action] = {
                "animation": animation_id,
                "clip": relative,
                "sha256": _sha256(path),
                "duration": clip.duration,
                "frame_count": clip.frame_count,
                "driver_matched_tracks": len(driver_common),
                "components": bindings,
            }
        unit_reports[unit_id] = {
            "archetype": recipe["archetype"],
            "domain": recipe["domain"],
            "actions": action_reports,
        }
    manifest["unit_binding"] = {
        "status": "validated_family_raw_clip",
        "sockets": SOCKET_PROFILE,
        "boundary": "raw curve proof; model-aware pose caches remain required before L20 promotion",
    }
    _write_json(manifest_path, manifest)
    return {
        "schema": "c3x.unit_family_action_binding_report.v0",
        "units": unit_reports,
        "sockets": SOCKET_PROFILE,
        "runtime_integration": "not_enabled",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_family_actions(args.pack)
        _write_json(args.report, report)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    actions = sum(len(unit["actions"]) for unit in report["units"].values())
    print(f"Validated {actions} actions across {len(report['units'])} unit families")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
