#!/usr/bin/env python3
"""Render normalized route bridge bodies without loading source-game formats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview.render_feature_asset import DdsBc1Texture, draw_ground, draw_mesh
from preview.render_iso import Canvas
from preview.render_textured_patch import BACKGROUND, safe_pack_path, write_png


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_draw(
    root: Path,
    landmark: dict[str, Any],
    state: str,
) -> tuple[dict[str, Any], DdsBc1Texture]:
    bindings = [item for item in landmark["draw_bindings"] if state in item["states"]]
    if len(bindings) != 1:
        raise ValueError(f"Route bridge state {state} requires exactly one draw binding")
    binding = bindings[0]
    mesh = _load_json(safe_pack_path(root, landmark["components"]["geometry"][binding["geometry"]]))
    if mesh.get("schema") != "c3x.normalized_mesh.v0":
        raise ValueError("Route bridge preview supports rigid normalized meshes only")
    material = _load_json(
        safe_pack_path(root, landmark["components"]["materials"][binding["material"]])
    )
    if material.get("schema") != "c3x.material.v0":
        raise ValueError("Route bridge material has an unsupported schema")
    base_color = material.get("channels", {}).get("base_color")
    if not isinstance(base_color, dict):
        raise ValueError("Route bridge material has no base color")
    address_u = "wrap" if base_color.get("address_u") == "repeat" else base_color.get(
        "address_u", "clamp"
    )
    address_v = "wrap" if base_color.get("address_v") == "repeat" else base_color.get(
        "address_v", "clamp"
    )
    texture = DdsBc1Texture.from_file(
        safe_pack_path(root, base_color["texture"]),
        address_u,
        address_v,
    )
    return mesh, texture


def render_sheet(manifest_path: Path, width: int = 1600, height: int = 900) -> Canvas:
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "c3x.asset_pack.v0":
        raise ValueError("Route bridge preview requires a c3x.asset_pack.v0 manifest")
    entries = sorted(
        (asset_id, asset)
        for asset_id, asset in manifest.get("assets", {}).items()
        if asset_id.startswith("route/bridge/") and asset.get("type") == "compound_landmark"
    )
    if not entries:
        raise ValueError("Route bridge pack has no bridge bodies")
    if width < len(entries) * 240 or height < 480:
        raise ValueError("Route bridge sheet is too small for its asset matrix")
    root = manifest_path.parent
    canvas = Canvas(width, height, BACKGROUND)
    depth_buffer = [-math.inf] * (width * height)
    panel_width = width / len(entries)
    panel_height = height / 2
    for column, (_asset_id, asset) in enumerate(entries):
        landmark = _load_json(safe_pack_path(root, asset["landmark"]))
        if landmark.get("schema") != "c3x.compound_landmark.v0":
            raise ValueError("Route bridge asset has an unsupported body schema")
        for row, state in enumerate(("worked", "pillaged")):
            mesh, texture = _state_draw(root, landmark, state)
            positions = [vertex["position"] for vertex in mesh["vertices"]]
            horizontal_extent = max(
                max(value[0] for value in positions) - min(value[0] for value in positions),
                max(value[1] for value in positions) - min(value[1] for value in positions),
            )
            vertical_extent = max(value[2] for value in positions) - min(
                value[2] for value in positions
            )
            if horizontal_extent <= 0 or vertical_extent < 0:
                raise ValueError("Route bridge mesh has invalid bounds")
            scale = min(
                panel_width * 0.70 / horizontal_extent,
                panel_height * 0.62 / max(vertical_extent, horizontal_extent * 0.35),
            )
            center = (
                int((column + 0.5) * panel_width),
                int((row + 0.72) * panel_height),
            )
            draw_ground(canvas, center, min(panel_width, panel_height) * 0.62)
            draw_mesh(
                canvas,
                depth_buffer,
                mesh,
                texture,
                center,
                scale,
                math.radians(25.0),
            )
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        canvas = render_sheet(args.manifest, args.width, args.height)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_png(canvas, args.output)
        png = args.output.read_bytes()
        report = {
            "schema": "c3x.route_doodad_preview.v0",
            "manifest": str(args.manifest),
            "output": str(args.output),
            "width": canvas.width,
            "height": canvas.height,
            "non_background_pixels": canvas.non_background_pixels(),
            "sha256": hashlib.sha256(png).hexdigest(),
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Rendered {args.output}: {report['non_background_pixels']} non-background pixels, "
        f"sha256={report['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
