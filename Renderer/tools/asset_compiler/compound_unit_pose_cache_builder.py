#!/usr/bin/env python3
"""Bake generic compound-unit node clips into model-aware C3X pose caches."""

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


MAX_ABSOLUTE_WORLD_VALUE = 1024.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_pose_caches(pack: Path, report_path: Path | None = None) -> dict[str, Any]:
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.unit_pack.v0":
        raise ValueError("unsupported unit pack")

    animation_nodes: dict[str, tuple[str, dict[str, Any]]] = {}
    recipes: list[tuple[Path, dict[str, Any]]] = []
    for unit_id, unit in manifest.get("units", {}).items():
        if unit.get("type") != "compound":
            continue
        recipe_path = pack / unit["recipe"]
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        recipes.append((recipe_path, recipe))
        for binding in recipe["actions"].values():
            pose_bindings = {}
            for node_id, animation_id in binding["node_clips"].items():
                prior = animation_nodes.setdefault(animation_id, (node_id, recipe["nodes"][node_id]))
                if prior[1]["skeleton"] != recipe["nodes"][node_id]["skeleton"]:
                    raise ValueError(f"animation is bound to multiple skeletons: {animation_id}")
                pose_bindings[node_id] = None
            binding["node_pose_caches"] = pose_bindings

    cache_reports = {}
    for animation_id, (node_id, node) in sorted(animation_nodes.items()):
        animation = manifest["animations"].get(animation_id)
        if animation is None or animation.get("binding_status") != "validated_node_local_raw_clip":
            raise ValueError(f"compound-unit animation is not validated: {animation_id}")
        clip = normalized_animation.load_clip(pack / animation["clip"])
        skeleton = normalized_skin.load_skeleton(pack / node["skeleton"])
        group_index = animation.get("group_index")
        if not isinstance(group_index, int):
            raise ValueError(f"compound-unit animation has no selected group: {animation_id}")
        frames = []
        maximum = 0.0
        for frame in range(clip.frame_count):
            time = frame / clip.sample_rate
            pose = normalized_skin.sample_pose(skeleton, clip, group_index, time, False)
            worlds = normalized_skin.world_matrices(skeleton, pose)
            maximum = max(maximum, *(abs(value) for matrix in worlds for value in matrix))
            if not math.isfinite(maximum) or maximum > MAX_ABSOLUTE_WORLD_VALUE:
                raise ValueError(
                    f"compound-unit pose exceeded normalized transform envelope: {animation_id} ({maximum})"
                )
            frames.append(worlds)
        relative = Path("poses") / Path(animation["clip"]).with_suffix(".c3pose")
        output = pack / relative
        cache = normalized_pose_cache.write_pose_cache(
            output,
            clip.duration,
            clip.sample_rate,
            tuple(bone["name"] for bone in skeleton["bones"]),
            frames,
        )
        binding_report = normalized_pose_cache.validate_skeleton_binding(cache, skeleton)
        animation.update(
            {
                "pose_cache": relative.as_posix(),
                "pose_cache_status": "validated_model_aware_world_matrices",
                "pose_cache_sha256": _sha256(output),
            }
        )
        cache_reports[animation_id] = {
            "node": node_id,
            "path": relative.as_posix(),
            "sha256": animation["pose_cache_sha256"],
            "maximum_absolute_world_value": maximum,
            **binding_report,
        }

    action_count = 0
    logical_bindings = 0
    for recipe_path, recipe in recipes:
        for binding in recipe["actions"].values():
            action_count += 1
            for node_id, animation_id in binding["node_clips"].items():
                binding["node_pose_caches"][node_id] = manifest["animations"][animation_id]["pose_cache"]
                logical_bindings += 1
        _write_json(recipe_path, recipe)
    _write_json(manifest_path, manifest)

    report = {
        "schema": "c3x.compound_unit_pose_cache_build.v0",
        "pack": str(pack),
        "units": len(recipes),
        "actions": action_count,
        "logical_node_action_bindings": logical_bindings,
        "unique_pose_caches": len(cache_reports),
        "maximum_allowed_absolute_world_value": MAX_ABSOLUTE_WORLD_VALUE,
        "caches": cache_reports,
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
        report = build_pose_caches(args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Baked {report['unique_pose_caches']} pose caches for "
        f"{report['actions']} actions / {report['logical_node_action_bindings']} node bindings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
