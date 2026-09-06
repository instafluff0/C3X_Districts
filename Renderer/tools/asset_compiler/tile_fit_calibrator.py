#!/usr/bin/env python3
"""Derive deterministic Civ III tile-fit transforms for normalized compound assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = RENDERER_ROOT / "preview/out/calibration/tile_fit.json"
FACINGS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
ZOOMS = {
    "normal": {"tile_pixels": [128, 64], "height_pixels_per_tile": 54},
    "reduced": {"tile_pixels": [64, 32], "height_pixels_per_tile": 27},
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _safe(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("asset path escapes its pack")
    return path


def compound_mesh_paths(manifest_path: Path, asset_id: str) -> tuple[Path, list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset = manifest.get("assets", {}).get(asset_id)
    if not isinstance(asset, dict) or asset.get("type") != "compound_landmark":
        raise ValueError(f"pack has no compound landmark {asset_id}")
    root = manifest_path.parent.resolve()
    landmark_path = _safe(root, asset["landmark"])
    landmark = json.loads(landmark_path.read_text(encoding="utf-8"))
    if landmark.get("schema") != "c3x.compound_landmark.v0":
        raise ValueError("unsupported compound landmark schema")
    paths = [_safe(root, relative) for relative in landmark["components"].get("geometry", [])]
    if not paths:
        raise ValueError("compound landmark has no geometry")
    return landmark_path, paths


def mesh_positions(paths: Iterable[Path]) -> list[list[float]]:
    result = []
    for path in paths:
        mesh = json.loads(path.read_text(encoding="utf-8"))
        if mesh.get("schema") not in {"c3x.normalized_mesh.v0", "c3x.skinned_mesh.v0"}:
            raise ValueError("tile fit requires normalized mesh geometry")
        for vertex in mesh.get("vertices", []):
            position = vertex.get("position")
            if (
                not isinstance(position, list)
                or len(position) != 3
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in position)
            ):
                raise ValueError("normalized mesh contains an invalid position")
            result.append([float(v) for v in position])
    if not result:
        raise ValueError("tile fit received no vertices")
    return result


def _rotate(point: list[float], yaw_degrees: int) -> tuple[float, float, float]:
    angle = math.radians(yaw_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return point[0] * cosine - point[1] * sine, point[0] * sine + point[1] * cosine, point[2]


def _project(point: tuple[float, float, float], zoom: dict[str, Any]) -> tuple[float, float]:
    half_width = zoom["tile_pixels"][0] / 2.0
    half_height = zoom["tile_pixels"][1] / 2.0
    return (
        (point[0] - point[1]) * half_width,
        (point[0] + point[1]) * half_height - point[2] * zoom["height_pixels_per_tile"],
    )


def calibrate_positions(
    positions: list[list[float]],
    asset_id: str,
    footprint_fraction: float = 0.78,
    maximum_screen_height_tiles: float = 1.65,
    allow_enlarge: bool = False,
) -> dict[str, Any]:
    if not 0 < footprint_fraction <= 1 or maximum_screen_height_tiles <= 0:
        raise ValueError("tile-fit limits must be positive and bounded")
    minimum = [min(point[axis] for point in positions) for axis in range(3)]
    maximum = [max(point[axis] for point in positions) for axis in range(3)]
    center = [(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2]
    grounded = [[point[0] - center[0], point[1] - center[1], point[2] - minimum[2]] for point in positions]
    cells = []
    for facing_index, facing in enumerate(FACINGS):
        yaw = facing_index * 45
        rotated = [_rotate(point, yaw) for point in grounded]
        for zoom_id, zoom in ZOOMS.items():
            projected = [_project(point, zoom) for point in rotated]
            width = max(point[0] for point in projected) - min(point[0] for point in projected)
            height = max(point[1] for point in projected) - min(point[1] for point in projected)
            width_limit = zoom["tile_pixels"][0] * footprint_fraction
            height_limit = zoom["tile_pixels"][1] * maximum_screen_height_tiles
            scale = min(width_limit / max(width, 1e-9), height_limit / max(height, 1e-9))
            if not allow_enlarge:
                scale = min(1.0, scale)
            fitted = [(point[0] * scale, point[1] * scale) for point in projected]
            cells.append({
                "facing": facing,
                "civ3_direction": facing_index + 1,
                "yaw_degrees": yaw,
                "zoom": zoom_id,
                "uniform_scale": round(scale, 8),
                "screen_bounds_px": [
                    round(min(point[0] for point in fitted), 4),
                    round(min(point[1] for point in fitted), 4),
                    round(max(point[0] for point in fitted), 4),
                    round(max(point[1] for point in fitted), 4),
                ],
                "fits_limits": width * scale <= width_limit + 1e-6 and height * scale <= height_limit + 1e-6,
            })
    normal = [cell for cell in cells if cell["zoom"] == "normal"]
    recommended = max(normal, key=lambda cell: (cell["uniform_scale"], -cell["yaw_degrees"]))
    return {
        "schema": "c3x.tile_fit_calibration.v0",
        "asset_id": asset_id,
        "source_bounds_tile": [*minimum, *maximum],
        "grounding": {
            "translation_tile": [round(-center[0], 8), round(-center[1], 8), round(-minimum[2], 8)],
            "contact_plane": "minimum_source_z",
        },
        "limits": {
            "footprint_fraction": footprint_fraction,
            "maximum_screen_height_tiles": maximum_screen_height_tiles,
            "allow_enlarge": allow_enlarge,
        },
        "facings": list(FACINGS),
        "zooms": ZOOMS,
        "cells": cells,
        "recommended_yaw_degrees": recommended["yaw_degrees"],
        "recommended_uniform_scale": recommended["uniform_scale"],
        "calibration_hash": hashlib.sha256(_canonical(cells)).hexdigest(),
        "approval": "pending_owning_lab",
        "runtime_activation": "not_enabled",
    }


def calibrate_compound_asset(manifest_path: Path, asset_id: str, **kwargs: Any) -> dict[str, Any]:
    landmark, meshes = compound_mesh_paths(manifest_path, asset_id)
    result = calibrate_positions(mesh_positions(meshes), asset_id, **kwargs)
    result["inputs"] = {
        "manifest": str(manifest_path),
        "landmark": str(landmark),
        "meshes": [str(path) for path in meshes],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-enlarge", action="store_true")
    args = parser.parse_args()
    try:
        result = calibrate_compound_asset(args.manifest, args.asset, allow_enlarge=args.allow_enlarge)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        f"Calibrated {args.asset}: 8 facings x 2 zooms, "
        f"yaw={result['recommended_yaw_degrees']} scale={result['recommended_uniform_scale']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
