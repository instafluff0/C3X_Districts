#!/usr/bin/env python3
"""Strict generic skeleton/skinned-mesh reader with a CPU skinning proof."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation


SKELETON_SCHEMA = "c3x.normalized_skeleton.v0"
MESH_SCHEMA = "c3x.normalized_skinned_mesh.v0"


def _finite(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def load_skeleton(path: Path | str) -> dict[str, Any]:
    skeleton = json.loads(Path(path).read_text(encoding="utf-8"))
    if skeleton.get("schema") != SKELETON_SCHEMA:
        raise ValueError("unsupported normalized skeleton schema")
    if skeleton.get("matrix_convention") != "row_major_row_vector":
        raise ValueError("unsupported skeleton matrix convention")
    if skeleton.get("position_unit") != "tile":
        raise ValueError("skeleton positions are not normalized to tiles")
    bones = skeleton.get("bones")
    if not isinstance(bones, list) or not bones:
        raise ValueError("skeleton has no bones")
    names = set()
    for index, bone in enumerate(bones):
        name = bone.get("name")
        parent = bone.get("parent")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("skeleton contains an empty or duplicate bone name")
        names.add(name)
        if not isinstance(parent, int) or parent < -1 or parent >= index:
            raise ValueError(f"bone {index} has a non-canonical parent")
        local = bone.get("local", {})
        if len(_finite(local.get("position", ()), f"bone {index} position")) != 3:
            raise ValueError(f"bone {index} position must have three components")
        orientation = _finite(local.get("orientation", ()), f"bone {index} orientation")
        if len(orientation) != 4 or sum(value * value for value in orientation) <= 1.0e-12:
            raise ValueError(f"bone {index} orientation is invalid")
        if len(_finite(local.get("scale_shear", ()), f"bone {index} scale/shear")) != 9:
            raise ValueError(f"bone {index} scale/shear must have nine components")
        if len(_finite(bone.get("inverse_bind_matrix", ()), f"bone {index} inverse bind")) != 16:
            raise ValueError(f"bone {index} inverse-bind matrix must have sixteen components")
    return skeleton


def load_mesh(path: Path | str, bone_count: int) -> dict[str, Any]:
    mesh = json.loads(Path(path).read_text(encoding="utf-8"))
    if mesh.get("schema") != MESH_SCHEMA:
        raise ValueError("unsupported normalized skinned-mesh schema")
    vertices = mesh.get("vertices")
    indices = mesh.get("topology", {}).get("indices")
    if not isinstance(vertices, list) or not vertices:
        raise ValueError("skinned mesh has no vertices")
    if not isinstance(indices, list) or len(indices) < 3 or len(indices) % 3:
        raise ValueError("skinned mesh has an invalid triangle list")
    if any(not isinstance(index, int) or index < 0 or index >= len(vertices) for index in indices):
        raise ValueError("skinned mesh has an out-of-range topology index")
    for index, vertex in enumerate(vertices):
        for field, count in (("position", 3), ("normal", 3), ("uv0", 2)):
            if len(_finite(vertex.get(field, ()), f"vertex {index} {field}")) != count:
                raise ValueError(f"vertex {index} {field} has an invalid component count")
        joints = vertex.get("joints")
        weights = _finite(vertex.get("weights", ()), f"vertex {index} weights")
        if not isinstance(joints, list) or len(joints) != 4 or len(weights) != 4:
            raise ValueError(f"vertex {index} must have four joints and weights")
        if any(not isinstance(joint, int) or joint < 0 or joint >= bone_count for joint in joints):
            raise ValueError(f"vertex {index} references outside the skeleton")
        if any(weight < 0.0 or weight > 1.0 for weight in weights) or abs(sum(weights) - 1.0) > 1.0e-6:
            raise ValueError(f"vertex {index} weights do not sum to one")
    return mesh


def _multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        sum(a[row * 4 + inner] * b[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    )


def _local_matrix(transform: dict[str, Sequence[float]] | normalized_animation.Transform) -> tuple[float, ...]:
    if isinstance(transform, normalized_animation.Transform):
        position = transform.position
        orientation = transform.orientation
        scale_shear = transform.scale_shear
    else:
        position = transform["position"]
        orientation = transform["orientation"]
        scale_shear = transform["scale_shear"]
    x, y, z, w = orientation
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 1.0e-12:
        raise ValueError("pose contains a zero-length quaternion")
    x, y, z, w = x / length, y / length, z / length, w / length
    rotation = (
        1 - 2 * y * y - 2 * z * z, 2 * x * y + 2 * z * w, 2 * x * z - 2 * y * w, 0.0,
        2 * x * y - 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z + 2 * x * w, 0.0,
        2 * x * z + 2 * y * w, 2 * y * z - 2 * x * w, 1 - 2 * x * x - 2 * y * y, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    scale = (
        scale_shear[0], scale_shear[1], scale_shear[2], 0.0,
        scale_shear[3], scale_shear[4], scale_shear[5], 0.0,
        scale_shear[6], scale_shear[7], scale_shear[8], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    result = list(_multiply(scale, rotation))
    result[12:15] = position
    return tuple(result)


def world_matrices(
    skeleton: dict[str, Any], local_transforms: Sequence[dict[str, Sequence[float]] | normalized_animation.Transform] | None = None
) -> tuple[tuple[float, ...], ...]:
    bones = skeleton["bones"]
    if local_transforms is not None and len(local_transforms) != len(bones):
        raise ValueError("pose transform count does not match skeleton")
    worlds = []
    for index, bone in enumerate(bones):
        local = _local_matrix(bone["local"] if local_transforms is None else local_transforms[index])
        parent = bone["parent"]
        worlds.append(local if parent < 0 else _multiply(local, worlds[parent]))
    return tuple(worlds)


def skin_positions(
    mesh: dict[str, Any], skeleton: dict[str, Any], worlds: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    if len(worlds) != len(skeleton["bones"]):
        raise ValueError("world matrix count does not match skeleton")
    skin_matrices = [
        _multiply(bone["inverse_bind_matrix"], worlds[index])
        for index, bone in enumerate(skeleton["bones"])
    ]
    output = []
    for vertex in mesh["vertices"]:
        px, py, pz = vertex["position"]
        value = [0.0, 0.0, 0.0]
        for joint, weight in zip(vertex["joints"], vertex["weights"]):
            if weight == 0.0:
                continue
            matrix = skin_matrices[joint]
            transformed = (
                px * matrix[0] + py * matrix[4] + pz * matrix[8] + matrix[12],
                px * matrix[1] + py * matrix[5] + pz * matrix[9] + matrix[13],
                px * matrix[2] + py * matrix[6] + pz * matrix[10] + matrix[14],
            )
            for axis in range(3):
                value[axis] += transformed[axis] * weight
        output.append(tuple(value))
    return tuple(output)  # type: ignore[return-value]


def validate_rest_pose(mesh: dict[str, Any], skeleton: dict[str, Any]) -> dict[str, float]:
    worlds = world_matrices(skeleton)
    identity = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    matrix_error = max(
        abs(value - identity[component])
        for index, bone in enumerate(skeleton["bones"])
        for component, value in enumerate(_multiply(bone["inverse_bind_matrix"], worlds[index]))
    )
    skinned = skin_positions(mesh, skeleton, worlds)
    vertex_error = max(
        abs(value - vertex["position"][axis])
        for vertex, result in zip(mesh["vertices"], skinned)
        for axis, value in enumerate(result)
    )
    if matrix_error > 5.0e-5 or vertex_error > 5.0e-5:
        raise ValueError(
            f"rest-pose skinning does not reconstruct the source mesh: matrix={matrix_error} vertex={vertex_error}"
        )
    return {"maximum_matrix_error": matrix_error, "maximum_vertex_error": vertex_error}


def bind_clip(
    mesh: dict[str, Any], skeleton: dict[str, Any], clip: normalized_animation.AnimationClip, group_index: int
) -> dict[str, Any]:
    if not 0 <= group_index < len(clip.groups):
        raise ValueError("animation group index is out of range")
    names = {track.name for track in clip.groups[group_index].tracks}
    skeleton_names = {bone["name"] for bone in skeleton["bones"]}
    weighted = {joint for vertex in mesh["vertices"] for joint, weight in zip(vertex["joints"], vertex["weights"]) if weight > 0.0}
    missing_weighted = [skeleton["bones"][index]["name"] for index in sorted(weighted) if skeleton["bones"][index]["name"] not in names]
    unknown_tracks = sorted(names - skeleton_names)
    if missing_weighted or unknown_tracks:
        raise ValueError(
            f"animation binding mismatch: missing weighted bones={missing_weighted}, unknown tracks={unknown_tracks}"
        )
    return {
        "group_index": group_index,
        "tracks": len(names),
        "weighted_bones": len(weighted),
        "missing_weighted_bones": missing_weighted,
        "unknown_tracks": unknown_tracks,
    }


def sample_pose(
    skeleton: dict[str, Any], clip: normalized_animation.AnimationClip, group_index: int, time: float, loop: bool
) -> tuple[normalized_animation.Transform, ...]:
    if not math.isfinite(time):
        raise ValueError("sample time must be finite")
    local_time = time % clip.duration if loop else min(clip.duration, max(0.0, time))
    frame_position = local_time * (clip.frame_count - 1) / clip.duration
    first = min(clip.frame_count - 1, int(math.floor(frame_position)))
    second = min(clip.frame_count - 1, first + 1)
    alpha = frame_position - first
    tracks = {track.name: track for track in clip.groups[group_index].tracks}
    output = []
    for bone in skeleton["bones"]:
        track = tracks.get(bone["name"])
        if track is None:
            local = bone["local"]
            output.append(
                normalized_animation.Transform(
                    tuple(local["position"]), tuple(local["orientation"]), tuple(local["scale_shear"])
                )
            )
            continue
        local = bone["local"]
        position = normalized_animation._sample_channel(
            track.position, first, second, alpha, local["position"]
        )
        if track.orientation.mode == normalized_animation.IDENTITY:
            orientation = tuple(local["orientation"])
        else:
            try:
                orientation = normalized_animation._sample_orientation(
                    track.orientation, first, second, alpha
                )
            except ValueError:
                # A few shipped clips mix valid quaternion samples with a
                # missing all-zero sample. At that instant the model's rest
                # orientation is the only defined source-independent fallback.
                orientation = tuple(local["orientation"])
        scale_shear = normalized_animation._sample_channel(
            track.scale_shear, first, second, alpha, local["scale_shear"]
        )
        output.append(normalized_animation.Transform(position, orientation, scale_shear))
    return tuple(output)


def _bounds(positions: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    return {
        "minimum": [min(position[axis] for position in positions) for axis in range(3)],
        "maximum": [max(position[axis] for position in positions) for axis in range(3)],
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skeleton", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--time", type=float)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    skeleton = load_skeleton(args.skeleton)
    mesh = load_mesh(args.mesh, len(skeleton["bones"]))
    result: dict[str, Any] = {
        "bones": len(skeleton["bones"]),
        "vertices": len(mesh["vertices"]),
        "triangles": len(mesh["topology"]["indices"]) // 3,
        "rest_pose": validate_rest_pose(mesh, skeleton),
    }
    if args.clip is not None:
        clip = normalized_animation.load_clip(args.clip)
        result["binding"] = bind_clip(mesh, skeleton, clip, args.group)
        if args.time is not None:
            pose = sample_pose(skeleton, clip, args.group, args.time, args.loop)
            result["sampled_bounds"] = _bounds(
                skin_positions(mesh, skeleton, world_matrices(skeleton, pose))
            )
    elif args.time is not None:
        parser.error("--time requires --clip")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
