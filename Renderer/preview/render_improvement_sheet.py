#!/usr/bin/env python3
"""Render recursive normalized mine and farm proof compositions by day and night."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preview.render_city_day_night_sheet import _draw_ground, _draw_mesh
from preview.render_feature_asset import DdsBc1Texture
from preview.render_iso import Canvas
from preview.render_textured_patch import BACKGROUND, DdsBc3Texture, safe_pack_path, write_png
from Renderer.tools.asset_compiler.compound_landmark_importer import _matrix_multiply


IDENTITY = [1.0 if row == column else 0.0 for row in range(4) for column in range(4)]


class Bc3RgbaTexture:
    def __init__(self, texture: DdsBc3Texture) -> None:
        self.texture = texture

    def sample(self, u: float, v: float) -> tuple[int, int, int, int]:
        return self.texture.sample_rgba(u, v)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _texture(root: Path, channel: dict[str, Any]) -> Any:
    path = safe_pack_path(root, channel["texture"])
    format_name = channel.get("format", "")
    if format_name.startswith("BC1"):
        return DdsBc1Texture.from_file(
            path,
            "wrap" if channel.get("address_u") == "repeat" else channel.get("address_u", "clamp"),
            "wrap" if channel.get("address_v") == "repeat" else channel.get("address_v", "clamp"),
        )
    if format_name.startswith("BC3"):
        return Bc3RgbaTexture(DdsBc3Texture.from_file(path))
    raise ValueError(f"Improvement preview cannot sample {format_name}")


def _local_matrix(bone: dict[str, Any]) -> list[float]:
    x, y, z, w = bone["rest"]["orientation"]
    column = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]
    rotation = [column[col * 3 + row] for row in range(3) for col in range(3)]
    scale = bone["rest"]["scale_shear"]
    linear = [
        sum(rotation[row * 3 + inner] * scale[inner * 3 + col] for inner in range(3))
        for row in range(3)
        for col in range(3)
    ]
    return [
        linear[0], linear[1], linear[2], 0.0,
        linear[3], linear[4], linear[5], 0.0,
        linear[6], linear[7], linear[8], 0.0,
        *bone["rest"]["position"], 1.0,
    ]


def _skeleton_worlds(skeleton: dict[str, Any]) -> list[list[float]]:
    worlds = []
    for bone in skeleton["bones"]:
        local = _local_matrix(bone)
        parent = bone["parent"]
        worlds.append(local if parent < 0 else _matrix_multiply(local, worlds[parent]))
    return worlds


def _point(value: list[float], matrix: list[float], translate: bool) -> list[float]:
    x, y, z = value
    return [
        x * matrix[0] + y * matrix[4] + z * matrix[8] + (matrix[12] if translate else 0.0),
        x * matrix[1] + y * matrix[5] + z * matrix[9] + (matrix[13] if translate else 0.0),
        x * matrix[2] + y * matrix[6] + z * matrix[10] + (matrix[14] if translate else 0.0),
    ]


def _transform_mesh(mesh: dict[str, Any], matrix: list[float]) -> dict[str, Any]:
    return {
        **mesh,
        "vertices": [
            {
                **vertex,
                "position": _point(vertex["position"], matrix, True),
                "normal": _point(vertex["normal"], matrix, False),
            }
            for vertex in mesh["vertices"]
        ],
    }


def _decal_documents(root: Path, relative: str) -> list[dict[str, Any]]:
    document = _load_json(safe_pack_path(root, relative))
    if document.get("schema") == "c3x.decal.v0":
        return [document]
    if document.get("schema") == "c3x.decal_set.v0":
        return [_load_json(safe_pack_path(root, path)) for path in document["decals"]]
    raise ValueError("Improvement preview found an unsupported decal document")


def _decal_mesh(decal: dict[str, Any], matrix: list[float]) -> dict[str, Any]:
    left, top, right, bottom = decal["footprint"]["content_bounds_xy"]
    vertices = [
        {"position": [left, top, 0.002], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 0.0]},
        {"position": [right, top, 0.002], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 0.0]},
        {"position": [right, bottom, 0.002], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 1.0]},
        {"position": [left, bottom, 0.002], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 1.0]},
    ]
    return _transform_mesh(
        {"vertices": vertices, "topology": {"primitive": "triangles", "indices": [0, 1, 2, 0, 2, 3]}},
        matrix,
    )


def _collect_component(
    root: Path,
    manifest: dict[str, Any],
    asset_id: str,
    transform: list[float] | None = None,
    stack: tuple[str, ...] = (),
) -> list[tuple[dict[str, Any], Any, Any | None]]:
    if asset_id in stack or len(stack) >= 12:
        raise ValueError("Improvement preview component graph cycles or is too deep")
    transform = IDENTITY if transform is None else transform
    asset = manifest["assets"][asset_id]
    landmark = _load_json(safe_pack_path(root, asset["landmark"]))
    parts = []
    for binding in landmark["draw_bindings"]:
        if "worked" not in binding["states"]:
            continue
        mesh = _load_json(
            safe_pack_path(root, landmark["components"]["geometry"][binding["geometry"]])
        )
        material = _load_json(
            safe_pack_path(root, landmark["components"]["materials"][binding["material"]])
        )
        channels = material.get("channels", {})
        if "base_color" not in channels:
            continue
        parts.append(
            (
                _transform_mesh(mesh, transform),
                _texture(root, channels["base_color"]),
                _texture(root, channels["emissive"]) if "emissive" in channels else None,
            )
        )
    decal_path = landmark["components"].get("decal")
    if decal_path:
        for decal in _decal_documents(root, decal_path):
            base = decal.get("channels", {}).get("base_color")
            if isinstance(base, dict):
                parts.append((_decal_mesh(decal, transform), _texture(root, base), None))

    skeleton_worlds = [
        _skeleton_worlds(_load_json(safe_pack_path(root, path)))
        for path in landmark["components"]["skeletons"]
    ]
    for point in landmark["attachment_points"]:
        if point["binding_status"] != "resolved":
            continue
        skeleton_index = point["skeleton"]
        bone_index = point["bone"]
        child_transform = _matrix_multiply(
            skeleton_worlds[skeleton_index][bone_index], transform
        )
        parts.extend(
            _collect_component(
                root,
                manifest,
                point["component_asset"],
                child_transform,
                stack + (asset_id,),
            )
        )
    return parts


def _panel(
    canvas: Canvas,
    depth: list[float],
    root: Path,
    manifest: dict[str, Any],
    asset_ids: list[str],
    bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    parts = [part for asset_id in asset_ids for part in _collect_component(root, manifest, asset_id)]
    if not parts:
        raise ValueError("Improvement preview composition has no renderable parts")
    positions = [vertex["position"] for mesh, _base, _emissive in parts for vertex in mesh["vertices"]]
    extent_xy = max(
        max(value[0] for value in positions) - min(value[0] for value in positions),
        max(value[1] for value in positions) - min(value[1] for value in positions),
    )
    extent_z = max(value[2] for value in positions) - min(value[2] for value in positions)
    left, top, right, bottom = bounds
    panel_width, panel_height = right - left, bottom - top
    scale = min(panel_width * 0.30 / max(extent_xy, 0.1), panel_height * 0.58 / max(extent_z, extent_xy * 0.35, 0.1))
    for half, night in enumerate((False, True)):
        center = (int(left + (0.25 + half * 0.5) * panel_width), int(top + panel_height * 0.72))
        _draw_ground(canvas, center, min(panel_width * 0.42, panel_height * 0.62), night)
        for mesh, base, emissive in parts:
            _draw_mesh(canvas, depth, mesh, base, emissive, center, scale, math.radians(25.0), night)
    return {"assets": asset_ids, "draw_parts": len(parts), "emissive_parts": sum(item[2] is not None for item in parts)}


def render_sheet(manifest_path: Path, width: int = 1800, height: int = 900) -> tuple[Canvas, dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "c3x.asset_pack.v0":
        raise ValueError("Improvement preview requires a c3x.asset_pack.v0 manifest")
    root = manifest_path.parent
    catalog = _load_json(safe_pack_path(root, manifest["improvement_catalog"]))
    if width < 1200 or height < 800:
        raise ValueError("Improvement preview requires at least 1200x800 pixels")
    canvas = Canvas(width, height, BACKGROUND)
    depth = [-math.inf] * (width * height)
    cells = []
    mine_variants = [variant for era in catalog["mine"]["eras"] for variant in era["variants"]]
    for index, asset_id in enumerate(mine_variants):
        cells.append(
            {"kind": "mine", "index": index, **_panel(canvas, depth, root, manifest, [asset_id], (index * width / 6, 0, (index + 1) * width / 6, height / 3))}
        )
    default_crop = next(item for item in catalog["farm"]["crop_styles"] if item["id"] == "default")
    for index, era in enumerate(catalog["farm"]["eras"]):
        assets = [era["tile_bases"][0], era["building_pieces"][1], default_crop["pieces"][1]]
        cells.append(
            {"kind": "farm_era", "id": era["id"], **_panel(canvas, depth, root, manifest, assets, (index * width / 3, height / 3, (index + 1) * width / 3, 2 * height / 3))}
        )
    visible_crops = [item for item in catalog["farm"]["crop_styles"] if item["pieces"]][:3]
    preindustrial = catalog["farm"]["eras"][0]
    for index, crop in enumerate(visible_crops):
        crop_piece = crop["pieces"][1] if crop["id"] == "default" else crop["pieces"][0]
        assets = [preindustrial["tile_bases"][0], preindustrial["building_pieces"][1], crop_piece]
        cells.append(
            {"kind": "crop_style", "id": crop["id"], **_panel(canvas, depth, root, manifest, assets, (index * width / 3, 2 * height / 3, (index + 1) * width / 3, height))}
        )
    return canvas, {"cells": cells, "day_night_pairs": len(cells)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        canvas, evidence = render_sheet(args.manifest, args.width, args.height)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_png(canvas, args.output)
        report = {
            "schema": "c3x.improvement_preview.v0",
            "manifest": str(args.manifest),
            "output": str(args.output),
            "width": canvas.width,
            "height": canvas.height,
            "non_background_pixels": canvas.non_background_pixels(),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            **evidence,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Rendered {args.output}: {report['day_night_pairs']} day/night pairs, sha256={report['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
