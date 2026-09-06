#!/usr/bin/env python3
"""Bake every skinned component in a generic unit-family pack into pose caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation, normalized_pose_cache, normalized_skin
from Renderer.tools.asset_compiler.unit_family_action_validator import _best_group


MAX_ABSOLUTE_WORLD_VALUE = 1024.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_family_pose_caches(pack: Path, report_path: Path | None = None) -> dict[str, Any]:
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.unit_pack.v0":
        raise ValueError("unsupported unit-family pack")

    components = {}
    for asset_id, asset in manifest.get("assets", {}).items():
        document = json.loads((pack / asset["component"]).read_text(encoding="utf-8"))
        if document.get("binding_mode") == "vertex_skin":
            components[asset_id] = normalized_skin.load_skeleton(pack / document["skeleton"])

    baked: dict[tuple[str, str], dict[str, Any]] = {}
    logical_bindings = 0
    recipes = []
    for unit_id, entry in manifest.get("units", {}).items():
        recipe_path = pack / entry["recipe"]
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        recipes.append((recipe_path, recipe))
        skinned_assets = [item["asset"] for item in recipe["components"] if item["asset"] in components]
        if not skinned_assets:
            raise ValueError(f"unit has no skinned components: {unit_id}")
        for action, animation_id in recipe["actions"].items():
            animation = manifest["animations"].get(animation_id)
            if animation is None or animation.get("binding_status") != "validated_raw_clip_name_binding":
                raise ValueError(f"unit-family animation is not validated: {animation_id}")
            clip_relative = animation.get("clip")
            clip = normalized_animation.load_clip(pack / clip_relative)
            pose_caches = {}
            for asset_id in skinned_assets:
                logical_bindings += 1
                key = (asset_id, clip_relative)
                record = baked.get(key)
                if record is None:
                    skeleton = components[asset_id]
                    group_index, common = _best_group(
                        clip, {bone["name"] for bone in skeleton["bones"]}
                    )
                    frames = []
                    maximum = 0.0
                    for frame in range(clip.frame_count):
                        time = frame / clip.sample_rate
                        pose = normalized_skin.sample_pose(skeleton, clip, group_index, time, False)
                        worlds = normalized_skin.world_matrices(skeleton, pose)
                        maximum = max(maximum, *(abs(value) for matrix in worlds for value in matrix))
                        if not math.isfinite(maximum) or maximum > MAX_ABSOLUTE_WORLD_VALUE:
                            raise ValueError(
                                f"unit-family pose exceeded normalized transform envelope: "
                                f"{animation_id}/{asset_id} ({maximum})"
                            )
                        frames.append(worlds)
                    asset_slug = re.sub(r"[^a-z0-9]+", "_", asset_id.lower()).strip("_")
                    relative = Path("poses") / Path(clip_relative).relative_to("animations")
                    relative = relative.with_suffix("") / f"{asset_slug}.c3pose"
                    output = pack / relative
                    cache = normalized_pose_cache.write_pose_cache(
                        output,
                        clip.duration,
                        clip.sample_rate,
                        tuple(bone["name"] for bone in skeleton["bones"]),
                        frames,
                    )
                    normalized_pose_cache.validate_skeleton_binding(cache, skeleton)
                    record = {
                        "path": relative.as_posix(),
                        "sha256": _sha256(output),
                        "group_index": group_index,
                        "matched_tracks": len(common),
                        "bones": len(skeleton["bones"]),
                        "frames": clip.frame_count,
                        "maximum_absolute_world_value": maximum,
                    }
                    baked[key] = record
                pose_caches[asset_id] = record
            animation["pose_caches"] = pose_caches
            animation["pose_cache_status"] = "validated_model_aware_world_matrices"

    manifest["unit_binding"] = {
        **manifest.get("unit_binding", {}),
        "status": "validated_family_model_aware_pose_cache",
        "boundary": "model-aware caches; visual calibration remains required before L20 promotion",
    }
    _write_json(manifest_path, manifest)
    report = {
        "schema": "c3x.unit_family_pose_cache_build.v0",
        "pack": str(pack),
        "units": len(recipes),
        "actions": sum(len(recipe["actions"]) for _, recipe in recipes),
        "skinned_component_action_bindings": logical_bindings,
        "unique_component_pose_caches": len(baked),
        "maximum_allowed_absolute_world_value": MAX_ABSOLUTE_WORLD_VALUE,
        "runtime_integration": "not_enabled",
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_family_pose_caches(args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Baked {report['unique_component_pose_caches']} family pose caches for "
        f"{report['actions']} actions / {report['skinned_component_action_bindings']} component bindings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
