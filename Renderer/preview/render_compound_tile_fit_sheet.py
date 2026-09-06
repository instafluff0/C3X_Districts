#!/usr/bin/env python3
"""Render an isolated eight-facing, two-zoom, day/night compound-asset sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.preview.render_city_day_night_sheet import _draw_mesh, _texture
from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import BACKGROUND, safe_pack_path, write_png
from Renderer.tools.asset_compiler.tile_fit_calibrator import FACINGS, ZOOMS, calibrate_compound_asset


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepared_mesh(mesh: dict[str, Any], translation: list[float], scale: float, z_ratio: float) -> dict[str, Any]:
    return {
        **mesh,
        "vertices": [
            {
                **vertex,
                "position": [
                    (vertex["position"][0] + translation[0]) * scale,
                    (vertex["position"][1] + translation[1]) * scale,
                    (vertex["position"][2] + translation[2]) * scale * z_ratio,
                ],
            }
            for vertex in mesh["vertices"]
        ],
    }


def render_sheet(
    manifest_path: Path,
    asset_id: str,
    cell_width: int = 180,
    cell_height: int = 180,
) -> tuple[Canvas, dict[str, Any]]:
    if cell_width < 150 or cell_height < 140:
        raise ValueError("tile-fit cells are too small")
    manifest = _load(manifest_path)
    root = manifest_path.parent.resolve()
    asset = manifest.get("assets", {}).get(asset_id)
    if not isinstance(asset, dict) or asset.get("type") != "compound_landmark":
        raise ValueError(f"pack has no compound landmark {asset_id}")
    landmark = _load(safe_pack_path(root, asset["landmark"]))
    # Isolated candidate inspection intentionally fills the admissible envelope;
    # the owning Lab still decides whether that enlargement is visually honest.
    calibration = calibrate_compound_asset(manifest_path, asset_id, allow_enlarge=True)
    parts = []
    for binding in landmark["draw_bindings"]:
        states = binding.get("states", [])
        if not set(states) & {"worked", "unworked", "built", "default"}:
            continue
        mesh = _load(safe_pack_path(root, landmark["components"]["geometry"][binding["geometry"]]))
        material = _load(safe_pack_path(root, landmark["components"]["materials"][binding["material"]]))
        channels = material.get("channels", {})
        base = channels.get("base_color")
        if not isinstance(base, dict):
            continue
        emissive = channels.get("emissive")
        parts.append((mesh, _texture(root, base), _texture(root, emissive) if isinstance(emissive, dict) else None))
    if not parts:
        raise ValueError("compound asset has no renderable default-state parts")
    canvas = Canvas(cell_width * len(FACINGS), cell_height * 4, BACKGROUND)
    cells = []
    calibration_cells = {(cell["facing"], cell["zoom"]): cell for cell in calibration["cells"]}
    translation = calibration["grounding"]["translation_tile"]
    for row, (zoom_id, night) in enumerate((("normal", False), ("normal", True), ("reduced", False), ("reduced", True))):
        zoom = ZOOMS[zoom_id]
        projection_scale = zoom["tile_pixels"][0] / (2.0 * 0.72)
        z_ratio = zoom["height_pixels_per_tile"] / projection_scale
        for column, facing in enumerate(FACINGS):
            center = (column * cell_width + cell_width // 2, row * cell_height + int(cell_height * 0.72))
            tile_width, tile_height = zoom["tile_pixels"]
            canvas.fill_polygon(
                [
                    (center[0], center[1] - tile_height // 2),
                    (center[0] + tile_width // 2, center[1]),
                    (center[0], center[1] + tile_height // 2),
                    (center[0] - tile_width // 2, center[1]),
                ],
                (18, 26, 31) if night else (55, 68, 56),
            )
            depth = [-math.inf] * (canvas.width * canvas.height)
            cell = calibration_cells[(facing, zoom_id)]
            for mesh, base, emissive in parts:
                prepared = _prepared_mesh(mesh, translation, cell["uniform_scale"], z_ratio)
                _draw_mesh(
                    canvas,
                    depth,
                    prepared,
                    base,
                    emissive,
                    center,
                    projection_scale,
                    math.radians(cell["yaw_degrees"]),
                    night,
                )
            cells.append({
                "facing": facing,
                "zoom": zoom_id,
                "night": night,
                "yaw_degrees": cell["yaw_degrees"],
                "uniform_scale": cell["uniform_scale"],
                "tile_pixels": zoom["tile_pixels"],
            })
    return canvas, {
        "schema": "c3x.compound_tile_fit_preview.v0",
        "asset_id": asset_id,
        "cells": cells,
        "calibration_hash": calibration["calibration_hash"],
        "recommended_yaw_degrees": calibration["recommended_yaw_degrees"],
        "approval": "pending_owning_lab",
        "runtime_activation": "not_enabled",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        canvas, report = render_sheet(args.manifest, args.asset)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_png(canvas, args.output)
        report.update({
            "output": str(args.output),
            "width": canvas.width,
            "height": canvas.height,
            "non_background_pixels": canvas.non_background_pixels(),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        })
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Rendered {len(report['cells'])} tile-fit cells at {args.output}; sha256={report['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
