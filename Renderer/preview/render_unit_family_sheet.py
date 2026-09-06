#!/usr/bin/env python3
"""Render static and raw-animation proof poses for representative unit families."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import write_png
from Renderer.preview.render_unit_turntable import (
    BACKGROUND,
    _load_components,
    _paste,
    _rigid_mesh,
    _skinned_mesh,
    draw_text,
)
from Renderer.preview.render_feature_asset import draw_mesh
from Renderer.tools.asset_compiler import normalized_animation, normalized_skin
from Renderer.tools.asset_compiler.unit_family_action_validator import SOCKET_PROFILE, _best_group


UNITS = ("archer", "swordsman", "infantry", "fighter")
POSES = ("rest", "idle", "move", "attack", "death")
BASIC_ACTIONS = ("idle", "fidget", "move", "fortify", "attack", "defend", "victory", "death")
ACTION_PHASE = {
    "idle": 0.37,
    "fidget": 0.61,
    "move": 0.37,
    "fortify": 1.0,
    "attack": 0.61,
    "defend": 0.61,
    "victory": 0.72,
    "death": 0.82,
}
ROTATION = math.radians(45.0)


def _pose_meshes(
    pack: Path,
    manifest: dict[str, Any],
    recipe: dict[str, Any],
    components: dict[str, dict[str, Any]],
    action: str,
) -> tuple[list[tuple[dict[str, Any], Any]], dict[str, Any]]:
    clip = None
    animation = None
    sample_time = 0.0
    if action != "rest":
        animation = manifest["animations"][recipe["actions"][action]]
        clip = normalized_animation.load_clip(pack / animation["clip"])
        sample_time = clip.duration * ACTION_PHASE[action]

    worlds: dict[str, Sequence[Sequence[float]]] = {}
    group_matches: dict[str, int] = {}
    for record in recipe["components"]:
        asset_id = record["asset"]
        component = components[asset_id]
        if component["document"]["binding_mode"] != "vertex_skin":
            continue
        skeleton = component["skeleton"]
        if clip is None:
            worlds[asset_id] = normalized_skin.world_matrices(skeleton)
            continue
        group_index, common = _best_group(
            clip, {bone["name"] for bone in skeleton["bones"]}
        )
        pose = normalized_skin.sample_pose(
            skeleton, clip, group_index, sample_time, animation["loop"]
        )
        worlds[asset_id] = normalized_skin.world_matrices(skeleton, pose)
        group_matches[asset_id] = len(common)

    driver_id = recipe["animation_driver"]
    driver = components[driver_id]
    driver_names = [bone["name"] for bone in driver["skeleton"]["bones"]]
    driver_worlds = worlds[driver_id]
    rendered = []
    for record in recipe["components"]:
        asset_id = record["asset"]
        component = components[asset_id]
        document = component["document"]
        if document["binding_mode"] == "vertex_skin":
            mesh = _skinned_mesh(component["mesh"], component["skeleton"], worlds[asset_id])
        else:
            point = document["attachment_point"]
            profile = SOCKET_PROFILE.get(point)
            if profile is None:
                raise ValueError(f"unit-family preview has no inferred socket for {point}")
            bone_name = profile["bone"]
            if bone_name not in driver_names:
                raise ValueError(f"unit-family driver has no inferred {point} bone {bone_name}")
            mesh = _rigid_mesh(
                component["mesh"],
                driver_worlds[driver_names.index(bone_name)],
                document["model_scale"],
            )
        rendered.append((mesh, component["texture"]))
    return rendered, {
        "action": action,
        "sample_time": sample_time,
        "duration": 0.0 if clip is None else clip.duration,
        "frame_count": 0 if clip is None else clip.frame_count,
        "matched_tracks": group_matches,
    }


def _projected_bounds(
    rendered: Sequence[tuple[dict[str, Any], Any]], model_scale: float, rotation: float = ROTATION
) -> tuple[float, float, float, float]:
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    points = []
    for mesh, _texture in rendered:
        for vertex in mesh["vertices"]:
            x, y, z = (value * model_scale for value in vertex["position"])
            rx = x * cosine - y * sine
            ry = x * sine + y * cosine
            points.append(((rx - ry) * 0.72, (rx + ry) * 0.36 - z))
    if not points:
        raise ValueError("unit-family pose contains no vertices")
    return (
        min(point[0] for point in points),
        max(point[0] for point in points),
        min(point[1] for point in points),
        max(point[1] for point in points),
    )


def _fit_cell(
    bounds: tuple[float, float, float, float], cell_width: int, cell_height: int
) -> tuple[float, tuple[int, int]]:
    minimum_x, maximum_x, minimum_y, maximum_y = bounds
    span_x = max(1.0e-6, maximum_x - minimum_x)
    span_y = max(1.0e-6, maximum_y - minimum_y)
    scale = min((cell_width - 24) / span_x, (cell_height - 50) / span_y)
    center = (
        int(round(cell_width / 2.0 - (minimum_x + maximum_x) * 0.5 * scale)),
        int(round(18.0 - minimum_y * scale)),
    )
    return scale, center


def render_family_sheet(
    pack: Path, output: Path, poses: Sequence[str] = POSES
) -> dict[str, Any]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "c3x.unit_pack.v0"
        or manifest.get("name") != "UnitFamilyLab"
        or manifest.get("unit_binding", {}).get("status") != "validated_family_raw_clip"
    ):
        raise ValueError("unit-family pack has not passed raw-clip validation")
    components = _load_components(pack, manifest, None, None)
    cell_width, cell_height = 226, 218
    left, top = 104, 58
    if not poses or any(pose != "rest" and pose not in BASIC_ACTIONS for pose in poses):
        raise ValueError("unit-family sheet requested an unsupported pose")
    width = left + cell_width * len(poses) + 12
    height = top + cell_height * len(UNITS) + 34
    canvas = Canvas(width, height, BACKGROUND)
    reports = {}
    for row, slug in enumerate(UNITS):
        unit_id = f"unit/{slug}"
        recipe = json.loads((pack / manifest["units"][unit_id]["recipe"]).read_text(encoding="utf-8"))
        member_scale = recipe["member"]["member_scale"] * recipe["member"]["variation_scale"]
        pose_reports = {}
        for column, action in enumerate(poses):
            rendered, pose_report = _pose_meshes(pack, manifest, recipe, components, action)
            bounds = _projected_bounds(rendered, member_scale)
            render_scale, center = _fit_cell(bounds, cell_width, cell_height)
            cell = Canvas(cell_width, cell_height, BACKGROUND)
            depth = [-math.inf] * (cell_width * cell_height)
            ground_y = cell_height - 17
            cell.fill_polygon(
                [
                    (cell_width // 2, ground_y - 16),
                    (cell_width // 2 + 55, ground_y),
                    (cell_width // 2, ground_y + 16),
                    (cell_width // 2 - 55, ground_y),
                ],
                (47, 61, 53),
            )
            for mesh, texture in rendered:
                draw_mesh(
                    cell,
                    depth,
                    mesh,
                    texture,
                    center,
                    render_scale,
                    ROTATION,
                    model_scale=member_scale,
                )
            _paste(canvas, cell, left + column * cell_width, top + row * cell_height)
            pose_reports[action] = {
                **pose_report,
                "fit_scale": render_scale,
                "projected_bounds": list(bounds),
            }
        reports[unit_id] = {
            "archetype": recipe["archetype"],
            "domain": recipe["domain"],
            "components": len(recipe["components"]),
            "poses": pose_reports,
        }

    title = (
        "NON-WARRIOR UNIT INTAKE - BASIC ACTION CONVERSION PROOF"
        if tuple(poses) == BASIC_ACTIONS
        else "NON-WARRIOR UNIT INTAKE - STATIC ART AND RAW CLIP PROOF"
    )
    draw_text(canvas, 14, 10, title, (232, 224, 194), 2)
    for column, action in enumerate(poses):
        draw_text(canvas, left + column * cell_width + 78, 36, action, (164, 203, 188), 1)
    for row, slug in enumerate(UNITS):
        draw_text(canvas, 9, top + row * cell_height + 98, slug, (164, 203, 188), 1)
    draw_text(
        canvas,
        14,
        height - 18,
        "OFFLINE L20 PREP - REAL SOURCE MESHES / RAW CURVES / INFERRED SOCKETS / EACH CELL AUTO-FIT",
        (190, 190, 180),
        1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_png(canvas, output)
    return {
        "schema": "c3x.unit_family_visual_proof.v0",
        "output": str(output),
        "width": width,
        "height": height,
        "units": reports,
        "poses": list(poses),
        "direction": "NE diagnostic at 45 degrees",
        "socket_status": "inferred_lab_profile",
        "animation_status": "raw_curve_sampling; model-aware pose caches remain required before L20",
        "cell_scaling": "per-pose auto-fit for source-art inspection; not a runtime scale decision",
        "runtime_integration": "not_enabled",
        "non_background_pixels": canvas.non_background_pixels(BACKGROUND),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--basic-actions", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = render_family_sheet(
            args.pack, args.output, BASIC_ACTIONS if args.basic_actions else POSES
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output} ({report['non_background_pixels']} rendered pixels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
