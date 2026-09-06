#!/usr/bin/env python3
"""Validate animated generic unit trees and parent-socket attachment math."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation, normalized_skin
from Renderer.tools.asset_compiler.unit_family_action_validator import _best_group


PHASES = (0.0, 0.37, 0.61, 1.0)
MAX_ABSOLUTE_WORLD_VALUE = 1024.0


def _multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        sum(a[row * 4 + inner] * b[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    )


def _translation(matrix: Sequence[float]) -> tuple[float, float, float]:
    return float(matrix[12]), float(matrix[13]), float(matrix[14])


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _sample_node(
    pack: Path,
    manifest: dict[str, Any],
    node: dict[str, Any],
    animation_id: str,
    phase: float,
    loop: bool,
) -> tuple[tuple[tuple[float, ...], ...], dict[str, Any]]:
    skeleton = normalized_skin.load_skeleton(pack / node["skeleton"])
    animation = manifest["animations"][animation_id]
    if animation.get("binding_status") != "validated_node_local_raw_clip":
        raise ValueError(f"compound-unit animation is not validated: {animation_id}")
    clip = normalized_animation.load_clip(pack / animation["clip"])
    names = {bone["name"] for bone in skeleton["bones"]}
    group_index, common = _best_group(clip, names)
    if group_index != animation["group_index"] or len(common) != animation["matched_tracks"]:
        raise ValueError(f"compound-unit animation binding drifted: {animation_id}")
    time = clip.duration * phase
    pose = normalized_skin.sample_pose(skeleton, clip, group_index, time, loop)
    worlds = normalized_skin.world_matrices(skeleton, pose)
    if not all(math.isfinite(value) for matrix in worlds for value in matrix):
        raise ValueError(f"compound-unit animation produced a non-finite matrix: {animation_id}")
    maximum_world_value = max(abs(value) for matrix in worlds for value in matrix)
    if maximum_world_value > MAX_ABSOLUTE_WORLD_VALUE:
        raise ValueError(
            f"compound-unit animation exceeded the normalized transform envelope: "
            f"{animation_id} ({maximum_world_value})"
        )
    return worlds, {
        "clip": animation["clip"],
        "duration": clip.duration,
        "frame_count": clip.frame_count,
        "matched_tracks": len(common),
        "maximum_world_value": maximum_world_value,
    }


def validate_compositions(pack: Path) -> dict[str, Any]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.unit_pack.v0" or manifest.get("name") != "CompoundUnitLab":
        raise ValueError("unsupported compound-unit pack")
    if manifest.get("composition_contract", {}).get("runtime_dispatch") != "recipe_data_only_no_unit_name_branches":
        raise ValueError("compound-unit pack does not use generic recipe dispatch")
    unit_reports = {}
    for unit_id, unit_entry in manifest["units"].items():
        recipe = json.loads((pack / unit_entry["recipe"]).read_text(encoding="utf-8"))
        nodes = recipe["nodes"]
        node_skeletons = {
            node_id: normalized_skin.load_skeleton(pack / node["skeleton"])
            for node_id, node in nodes.items()
        }
        bone_indices = {
            node_id: {bone["name"]: index for index, bone in enumerate(skeleton["bones"])}
            for node_id, skeleton in node_skeletons.items()
        }
        action_reports = {}
        for action, binding in recipe["actions"].items():
            phase_reports = []
            maximum_separation = 0.0
            duration_by_node = {}
            for phase in PHASES:
                local_worlds = {}
                clip_reports = {}
                for node_id, node in nodes.items():
                    local_worlds[node_id], clip_reports[node_id] = _sample_node(
                        pack, manifest, node, binding["node_clips"][node_id], phase, binding["loop"]
                    )
                    duration_by_node[node_id] = clip_reports[node_id]["duration"]
                final_worlds = {recipe["root_node"]: local_worlds[recipe["root_node"]]}
                joint_reports = []
                remaining = list(recipe["joints"])
                while remaining:
                    progress = False
                    for joint in list(remaining):
                        parent, child = joint["parent"], joint["child"]
                        if parent not in final_worlds:
                            continue
                        parent_socket = final_worlds[parent][bone_indices[parent][joint["parent_bone"]]]
                        target = _multiply(joint["local_transform"], parent_socket)
                        # The child coordinate frame—not its animated root
                        # bone—is socketed. Root motion remains meaningful for
                        # attacks, reactions, and deaths instead of being
                        # cancelled by an inverse transform every frame.
                        node_transform = target
                        final_worlds[child] = tuple(
                            _multiply(world, node_transform) for world in local_worlds[child]
                        )
                        actual = final_worlds[child][bone_indices[child][joint["child_root_bone"]]]
                        origin = _translation(node_transform)
                        separation = _distance(origin, _translation(target))
                        root_offset = _distance(_translation(actual), origin)
                        maximum_separation = max(maximum_separation, separation)
                        joint_reports.append(
                            {
                                "parent": parent,
                                "child": child,
                                "socket": joint["socket"],
                                "separation": separation,
                                "animated_child_root_offset": root_offset,
                                "target": list(_translation(target)),
                            }
                        )
                        remaining.remove(joint)
                        progress = True
                    if not progress:
                        raise ValueError(f"{unit_id}/{action} contains an unresolved node graph")
                if set(final_worlds) != set(nodes):
                    raise ValueError(f"{unit_id}/{action} did not compose every node")
                phase_reports.append({"phase": phase, "joints": joint_reports, "clips": clip_reports})
            if maximum_separation > 1.0e-6:
                raise ValueError(f"{unit_id}/{action} child separated from its socket")
            action_reports[action] = {
                "loop": binding["loop"],
                "duration_by_node": duration_by_node,
                "maximum_socket_separation": maximum_separation,
                "phases": phase_reports,
            }
        unit_reports[unit_id] = {
            "nodes": len(nodes),
            "joints": len(recipe["joints"]),
            "actions": action_reports,
            "instance_contract": recipe["instance_contract"],
        }
    return {
        "schema": "c3x.compound_unit_composition_validation.v0",
        "units": unit_reports,
        "phase_samples": list(PHASES),
        "maximum_allowed_socket_separation": 1.0e-6,
        "maximum_allowed_absolute_world_value": MAX_ABSOLUTE_WORLD_VALUE,
        "runtime_integration": "not_enabled",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_compositions(args.pack)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    actions = sum(len(unit["actions"]) for unit in report["units"].values())
    print(f"Validated {len(report['units'])} compound units / {actions} actions at {len(PHASES)} phases")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
