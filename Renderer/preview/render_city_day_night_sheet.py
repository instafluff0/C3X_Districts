#!/usr/bin/env python3
"""Render the generic city style/era proof matrix in paired day and night views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview.render_feature_asset import DdsBc1Texture, edge, normalize, rotate_z
from preview.render_iso import Canvas
from preview.render_textured_patch import BACKGROUND, safe_pack_path, write_png


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _texture(root: Path, channel: dict[str, Any]) -> DdsBc1Texture:
    return DdsBc1Texture.from_file(
        safe_pack_path(root, channel["texture"]),
        "wrap" if channel.get("address_u") == "repeat" else channel.get("address_u", "clamp"),
        "wrap" if channel.get("address_v") == "repeat" else channel.get("address_v", "clamp"),
    )


def _draw_mesh(
    canvas: Canvas,
    depth_buffer: list[float],
    mesh: dict[str, Any],
    base: DdsBc1Texture,
    emissive: DdsBc1Texture | None,
    center: tuple[int, int],
    scale: float,
    rotation: float,
    night: bool,
) -> None:
    transformed = []
    for vertex in mesh["vertices"]:
        position = rotate_z(vertex["position"], rotation)
        transformed.append(
            {
                "screen": (
                    center[0] + (position[0] - position[1]) * scale * 0.72,
                    center[1] + (position[0] + position[1]) * scale * 0.36 - position[2] * scale,
                ),
                "depth": position[0] + position[1] + position[2] * 0.65,
                "normal": normalize(rotate_z(vertex["normal"], rotation)),
                "uv": vertex["uv0"],
            }
        )
    light = normalize((-0.45, -0.60, 1.0))
    indices = mesh["topology"]["indices"]
    for start in range(0, len(indices), 3):
        vertices = [transformed[index] for index in indices[start : start + 3]]
        points = [vertex["screen"] for vertex in vertices]
        area = edge(points[0], points[1], points[2])
        if abs(area) <= 1.0e-9:
            continue
        min_x = max(0, int(math.floor(min(point[0] for point in points))))
        max_x = min(canvas.width - 1, int(math.ceil(max(point[0] for point in points))))
        min_y = max(0, int(math.floor(min(point[1] for point in points))))
        max_y = min(canvas.height - 1, int(math.ceil(max(point[1] for point in points))))
        for pixel_y in range(min_y, max_y + 1):
            for pixel_x in range(min_x, max_x + 1):
                sample = (pixel_x + 0.5, pixel_y + 0.5)
                weights = (
                    edge(points[1], points[2], sample) / area,
                    edge(points[2], points[0], sample) / area,
                )
                weights = (weights[0], weights[1], 1.0 - weights[0] - weights[1])
                if min(weights) < -1.0e-9:
                    continue
                depth = sum(weights[index] * vertices[index]["depth"] for index in range(3))
                pixel_index = pixel_y * canvas.width + pixel_x
                if depth <= depth_buffer[pixel_index]:
                    continue
                u = sum(weights[index] * vertices[index]["uv"][0] for index in range(3))
                v = sum(weights[index] * vertices[index]["uv"][1] for index in range(3))
                color = base.sample(u, v)
                if color[3] < 128:
                    continue
                normal = normalize(
                    tuple(
                        sum(weights[index] * vertices[index]["normal"][axis] for index in range(3))
                        for axis in range(3)
                    )
                )
                diffuse = abs(sum(normal[axis] * light[axis] for axis in range(3)))
                shade = (0.13 + diffuse * 0.20) if night else (0.48 + diffuse * 0.62)
                output = [channel * shade for channel in color[:3]]
                if night and emissive is not None:
                    glow = emissive.sample(u, v)
                    if glow[3] >= 128:
                        output = [output[index] + glow[index] * 1.15 for index in range(3)]
                canvas.set_pixel(
                    pixel_x,
                    pixel_y,
                    tuple(max(0, min(255, int(round(channel)))) for channel in output),
                )
                depth_buffer[pixel_index] = depth


def _draw_ground(canvas: Canvas, center: tuple[int, int], scale: float, night: bool) -> None:
    half_width = int(scale * 0.55)
    half_height = int(scale * 0.25)
    canvas.fill_polygon(
        [
            (center[0], center[1] - half_height),
            (center[0] + half_width, center[1]),
            (center[0], center[1] + half_height),
            (center[0] - half_width, center[1]),
        ],
        (18, 26, 31) if night else (55, 68, 56),
    )


def _worked_parts(root: Path, landmark: dict[str, Any]) -> list[tuple[dict[str, Any], DdsBc1Texture, DdsBc1Texture | None]]:
    parts = []
    for binding in landmark["draw_bindings"]:
        if "worked" not in binding["states"]:
            continue
        mesh = _load_json(safe_pack_path(root, landmark["components"]["geometry"][binding["geometry"]]))
        material = _load_json(safe_pack_path(root, landmark["components"]["materials"][binding["material"]]))
        base_channel = material.get("channels", {}).get("base_color")
        if not isinstance(base_channel, dict):
            raise ValueError("City preview material has no base color")
        emissive_channel = material.get("channels", {}).get("emissive")
        parts.append(
            (
                mesh,
                _texture(root, base_channel),
                _texture(root, emissive_channel) if isinstance(emissive_channel, dict) else None,
            )
        )
    if not parts:
        raise ValueError("City component has no worked-state geometry")
    return parts


def render_sheet(manifest_path: Path, width: int = 2000, height: int = 1500) -> tuple[Canvas, dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "c3x.asset_pack.v0":
        raise ValueError("City preview requires a c3x.asset_pack.v0 manifest")
    root = manifest_path.parent
    catalog = _load_json(safe_pack_path(root, manifest["city_catalog"]))
    styles = sorted(catalog["styles"], key=lambda item: item["civ3_culture_group"])
    eras = sorted(catalog["eras"], key=lambda item: item["civ3_era"])
    if len(styles) != 5 or len(eras) != 4 or width < 1200 or height < 900:
        raise ValueError("City preview requires a 5-by-4 matrix and at least 1200x900 pixels")
    canvas = Canvas(width, height, BACKGROUND)
    depth_buffer = [-math.inf] * (width * height)
    cell_width = width / len(eras)
    cell_height = height / len(styles)
    cells = []
    for row, style in enumerate(styles):
        for column, era in enumerate(eras):
            pool_id = style["era_pools"][era["id"]]
            asset_id = catalog["pools"][pool_id]["components"][0]
            asset = manifest["assets"][asset_id]
            landmark = _load_json(safe_pack_path(root, asset["landmark"]))
            parts = _worked_parts(root, landmark)
            positions = [vertex["position"] for mesh, _base, _emissive in parts for vertex in mesh["vertices"]]
            extent_xy = max(
                max(value[0] for value in positions) - min(value[0] for value in positions),
                max(value[1] for value in positions) - min(value[1] for value in positions),
            )
            extent_z = max(value[2] for value in positions) - min(value[2] for value in positions)
            if extent_xy <= 0 or extent_z < 0:
                raise ValueError("City component has invalid bounds")
            model_scale = min(
                cell_width * 0.30 / extent_xy,
                cell_height * 0.64 / max(extent_z, extent_xy * 0.35),
            )
            for half, night in enumerate((False, True)):
                center = (
                    int(column * cell_width + (0.25 + half * 0.5) * cell_width),
                    int((row + 0.72) * cell_height),
                )
                _draw_ground(canvas, center, min(cell_width * 0.42, cell_height * 0.64), night)
                for mesh, base, emissive in parts:
                    _draw_mesh(
                        canvas,
                        depth_buffer,
                        mesh,
                        base,
                        emissive,
                        center,
                        model_scale,
                        math.radians(25.0),
                        night,
                    )
            cells.append(
                {
                    "culture_group": style["civ3_culture_group"],
                    "style": style["id"],
                    "civ3_era": era["civ3_era"],
                    "era": era["id"],
                    "pool": pool_id,
                    "component": asset_id,
                    "emissive_draw_parts": sum(emissive is not None for _mesh, _base, emissive in parts),
                }
            )
    return canvas, {"cells": cells, "emissive_cells": sum(item["emissive_draw_parts"] > 0 for item in cells)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--height", type=int, default=1500)
    args = parser.parse_args(argv)
    try:
        canvas, evidence = render_sheet(args.manifest, args.width, args.height)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_png(canvas, args.output)
        report = {
            "schema": "c3x.city_day_night_preview.v0",
            "manifest": str(args.manifest),
            "output": str(args.output),
            "width": canvas.width,
            "height": canvas.height,
            "non_background_pixels": canvas.non_background_pixels(),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            **evidence,
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Rendered {args.output}: {report['emissive_cells']}/20 emissive cells, "
        f"sha256={report['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
