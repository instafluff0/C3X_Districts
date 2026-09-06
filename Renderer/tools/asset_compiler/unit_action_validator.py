#!/usr/bin/env python3
"""Validate and register normalized unit actions against compiled member rigs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation, normalized_pose_cache, normalized_skin


ACTIONS = ("idle", "move", "attack", "death")
CORE_BONES = {"Root", "Pelvis", "Spine1", "Ribcage", "Head", "LHand", "RHand", "LFoot", "RFoot"}
SOCKET_PROFILE = {
    "Hat": {
        "bone": "Head",
        "status": "inferred_lab_profile",
        "basis": "member attachment point plus matching animated humanoid rig bone",
    },
    "WeaponPrimary": {
        "bone": "Inven_R_Hand",
        "status": "inferred_lab_profile",
        "basis": "member attachment point plus right-hand inventory bone",
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_actions(pack: Path) -> dict[str, Any]:
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.unit_pack.v0":
        raise ValueError("unsupported unit-pack schema")
    components = {}
    for asset_id, asset in manifest["assets"].items():
        document = json.loads((pack / asset["component"]).read_text(encoding="utf-8"))
        skeleton = normalized_skin.load_skeleton(pack / document["skeleton"])
        components[asset_id] = (document, skeleton)
    body_document, body_skeleton = components["unit/warrior/body"]
    body_names = {bone["name"] for bone in body_skeleton["bones"]}
    for point, profile in SOCKET_PROFILE.items():
        if profile["bone"] not in body_names:
            raise ValueError(f"socket profile {point} references missing bone {profile['bone']}")
    action_records = {}
    binding_records = {}
    for action in ACTIONS:
        relative = f"animations/unit/warrior/{action}.c3anim"
        path = pack / relative
        clip = normalized_animation.load_clip(path)
        groups = [index for index, group in enumerate(clip.groups) if group.name == "Root"]
        if len(groups) != 1:
            raise ValueError(f"{action} clip does not have one Root track group")
        group_index = groups[0]
        track_names = {track.name for track in clip.groups[group_index].tracks}
        if not CORE_BONES.issubset(track_names & body_names):
            raise ValueError(f"{action} clip is missing required humanoid core bones")
        per_component = {}
        pose_caches = {}
        for asset_id, (document, skeleton) in components.items():
            if document["binding_mode"] != "vertex_skin":
                continue
            skeleton_names = {bone["name"] for bone in skeleton["bones"]}
            common = track_names & skeleton_names
            if len(common) < 30:
                raise ValueError(f"{action} has too few matching tracks for {asset_id}: {len(common)}")
            mesh = normalized_skin.load_mesh(pack / document["mesh"], len(skeleton["bones"]))
            normalized_skin.validate_rest_pose(mesh, skeleton)
            sampled = normalized_skin.sample_pose(skeleton, clip, group_index, clip.duration * 0.37, action in ("idle", "move"))
            positions = normalized_skin.skin_positions(mesh, skeleton, normalized_skin.world_matrices(skeleton, sampled))
            if not all(math.isfinite(value) for position in positions for value in position):
                raise ValueError(f"{action}/{asset_id} produces a non-finite skinned position")
            role = document["role"].lower()
            pose_relative = f"poses/unit/warrior/{role}_{action}.c3pose"
            pose_path = pack / pose_relative
            pose_cache = normalized_pose_cache.load_pose_cache(pose_path)
            normalized_pose_cache.validate_skeleton_binding(pose_cache, skeleton)
            if abs(pose_cache.duration - clip.duration) > 1.0e-5 or pose_cache.frame_count != clip.frame_count:
                raise ValueError(f"{action}/{asset_id} pose-cache timing disagrees with the clip")
            pose_caches[asset_id] = {"path": pose_relative, "sha256": _sha256(pose_path)}
            per_component[asset_id] = {
                "matched_tracks": len(common),
                "rest_fallback_bones": sorted(skeleton_names - track_names),
                "ignored_helper_tracks": sorted(track_names - skeleton_names),
                "binding_policy": "bind matching bone names; retain authored rest transforms for untracked mesh helpers",
            }
        action_records[action] = {
            "clip": relative,
            "group_index": group_index,
            "duration": clip.duration,
            "sample_rate": clip.sample_rate,
            "frame_count": clip.frame_count,
            "loop": action in ("idle", "move"),
            "sha256": _sha256(path),
            "pose_caches": pose_caches,
        }
        binding_records[action] = per_component
    manifest["animations"] = {
        f"animation/unit/warrior/{action}": record for action, record in action_records.items()
    }
    manifest["unit_binding"] = {
        "status": "validated_lab_cpu_skin",
        "sockets": SOCKET_PROFILE,
        "owner_tint": {
            "status": "preserved_unrendered",
            "semantic": "USE_CIV_COLOR",
            "reason": "source material tint-mask/channel semantics are not yet decoded",
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "schema": "c3x.unit_action_binding_report.v0",
        "unit": "unit/warrior",
        "actions": action_records,
        "component_bindings": binding_records,
        "sockets": SOCKET_PROFILE,
        "owner_tint": manifest["unit_binding"]["owner_tint"],
        "runtime_integration": "not_enabled",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_actions(args.pack)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.report, report)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Validated {len(report['actions'])} Warrior actions across 3 skinned components")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
