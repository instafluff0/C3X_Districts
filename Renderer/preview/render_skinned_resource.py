#!/usr/bin/env python3
"""Render deterministic timed contact sheets for normalized skinned resources."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.preview.render_feature_asset import DdsBc1Texture, draw_mesh
from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import BACKGROUND, safe_pack_path, write_png
from Renderer.tools.asset_compiler import normalized_animation, normalized_pose_cache, normalized_skin


class WrappingBc1Texture(DdsBc1Texture):
    def sample(self, u: float, v: float) -> tuple[int, int, int, int]:
        return super().sample(u % 1.0, v % 1.0)


def deformed_mesh(
    mesh: dict[str, Any],
    skeleton: dict[str, Any],
    clip: normalized_animation.AnimationClip,
    group_index: int,
    time: float,
) -> dict[str, Any]:
    pose = normalized_skin.sample_pose(skeleton, clip, group_index, time, True)
    positions = normalized_skin.skin_positions(
        mesh, skeleton, normalized_skin.world_matrices(skeleton, pose)
    )
    return {
        "topology": mesh["topology"],
        "vertices": [
            {"position": list(position), "normal": vertex["normal"], "uv0": vertex["uv0"]}
            for vertex, position in zip(mesh["vertices"], positions)
        ],
    }


def deformed_mesh_from_world_matrices(
    mesh: dict[str, Any], skeleton: dict[str, Any], matrices: tuple[tuple[float, ...], ...]
) -> dict[str, Any]:
    positions = normalized_skin.skin_positions(mesh, skeleton, matrices)
    return {
        "topology": mesh["topology"],
        "vertices": [
            {"position": list(position), "normal": vertex["normal"], "uv0": vertex["uv0"]}
            for vertex, position in zip(mesh["vertices"], positions)
        ],
    }


def draw_water(canvas: Canvas, center: tuple[int, int], scale: float) -> None:
    half_width = int(scale * 0.58)
    half_height = int(scale * 0.27)
    cx, cy = center
    canvas.fill_polygon(
        [(cx, cy - half_height), (cx + half_width, cy), (cx, cy + half_height), (cx - half_width, cy)],
        (42, 91, 125),
    )
    canvas.fill_polygon(
        [(cx, cy - half_height + 3), (cx + half_width - 6, cy), (cx, cy + half_height - 3), (cx - half_width + 6, cy)],
        (51, 113, 150),
    )


def render_contact_sheet(
    manifest_path: Path,
    resource_id: str,
    width: int,
    height: int,
    frame_count: int,
    allow_unvalidated: bool = False,
) -> tuple[Canvas, list[float]]:
    if width < 512 or height < 256 or frame_count < 2 or frame_count > 8:
        raise ValueError("contact sheet requires at least 512x256 and two to eight frames")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.resource_pack.v0":
        raise ValueError("unsupported resource-pack schema")
    resource = manifest["resources"][resource_id]
    animation_info = manifest["animations"][resource_id]
    pose_status = animation_info.get("pose_status")
    if pose_status not in ("validated_cpu_skin", "validated_model_aware_pose_cache") and not allow_unvalidated:
        raise ValueError(
            f"{resource_id} animated poses are not validated ({pose_status or 'unknown'}); "
            "use --allow-unvalidated only for diagnostic renders"
        )
    asset = manifest["assets"][resource["landmark_asset"]]
    root = manifest_path.parent
    skeleton = normalized_skin.load_skeleton(safe_pack_path(root, asset["skeleton"]))
    mesh = normalized_skin.load_mesh(safe_pack_path(root, asset["mesh"]), len(skeleton["bones"]))
    material = json.loads(safe_pack_path(root, asset["material"]).read_text(encoding="utf-8"))
    texture = WrappingBc1Texture.from_file(safe_pack_path(root, material["base_color"]["texture"]))
    clip = normalized_animation.load_clip(safe_pack_path(root, animation_info["clip"]))
    group_index = int(animation_info["group_index"])
    normalized_skin.validate_rest_pose(mesh, skeleton)
    normalized_skin.bind_clip(mesh, skeleton, clip, group_index)
    pose_cache = None
    if animation_info.get("pose_cache"):
        pose_cache = normalized_pose_cache.load_pose_cache(
            safe_pack_path(root, animation_info["pose_cache"])
        )
        normalized_pose_cache.validate_skeleton_binding(pose_cache, skeleton)

    canvas = Canvas(width, height, BACKGROUND)
    depth_buffer = [-math.inf] * (width * height)
    panel_width = width / frame_count
    scale = min(panel_width * 0.90, height * 0.75)
    center_y = int(height * 0.66)
    times = [clip.duration * index / frame_count for index in range(frame_count)]
    for index, time in enumerate(times):
        center = (int(panel_width * (index + 0.5)), center_y)
        draw_water(canvas, center, scale)
        posed_mesh = (
            deformed_mesh_from_world_matrices(mesh, skeleton, pose_cache.sample(time, True))
            if pose_cache is not None
            else deformed_mesh(mesh, skeleton, clip, group_index, time)
        )
        draw_mesh(
            canvas,
            depth_buffer,
            posed_mesh,
            texture,
            center,
            scale,
            math.radians(25.0),
        )
    return canvas, times


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--resource", choices=("resource/fish", "resource/whales"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="render unresolved pose semantics for diagnostic use only",
    )
    args = parser.parse_args(argv)
    try:
        canvas, times = render_contact_sheet(
            args.pack,
            args.resource,
            args.width,
            args.height,
            args.frames,
            args.allow_unvalidated,
        )
        write_png(canvas, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"Wrote {args.output} ({canvas.non_background_pixels()} drawn pixels; "
        f"times={','.join(f'{time:.3f}' for time in times)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
