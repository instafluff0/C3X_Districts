#!/usr/bin/env python3
"""Render one shared unit pack through several Civ III runtime palette rows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.preview.render_feature_asset import draw_mesh
from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import write_png
from Renderer.preview.render_unit_family_sheet import (
    ROTATION,
    _fit_cell,
    _pose_meshes,
    _projected_bounds,
)
from Renderer.preview.render_unit_turntable import BACKGROUND, _load_components, _paste, draw_text
from Renderer.preview.render_unit_turntable import (
    CIV_COLOR_GAIN,
    CIV_COLOR_NEUTRAL_FLOOR,
    CIV_COLOR_STRENGTH,
)
from Renderer.tools.asset_compiler.civ3_owner_palette_compiler import load_owner_color_table


DEFAULT_TABLE_IDS = (1, 4, 6, 10, 14, 20)


def manifest_unit_ids(manifest: dict[str, Any]) -> list[str]:
    unit_ids = sorted(manifest.get("units", {}))
    if not unit_ids or any(not unit_id.startswith("unit/") for unit_id in unit_ids):
        raise ValueError("unit pack contains no valid logical unit IDs")
    return unit_ids


def changed_pixel_count(neutral: Canvas, tinted: Canvas, delta_threshold: int) -> int:
    """Count screen pixels whose RGB changes enough to remain visually meaningful."""
    if neutral.width != tinted.width or neutral.height != tinted.height:
        raise ValueError("owner-color coverage canvases must have identical dimensions")
    if not isinstance(delta_threshold, int) or not 1 <= delta_threshold <= 255:
        raise ValueError("owner-color RGB delta threshold must be 1..255")
    return sum(
        max(abs(before[channel] - after[channel]) for channel in range(3)) >= delta_threshold
        for before, after in zip(neutral.pixels, tinted.pixels)
    )


def _render_cell(
    rendered: Sequence[tuple[dict[str, Any], Any]],
    member_scale: float,
    cell_width: int,
    cell_height: int,
    render_scale: float,
    center: tuple[int, int],
) -> Canvas:
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
    return cell


def render_owner_color_sheet(
    pack: Path, owner_palette_pack: Path, output: Path, table_ids: Sequence[int]
) -> dict[str, Any]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "c3x.unit_pack.v0"
        or manifest.get("unit_binding", {}).get("status") != "validated_family_raw_clip"
    ):
        raise ValueError("unit pack has not passed raw-clip validation")
    unit_ids = manifest_unit_ids(manifest)
    if not table_ids or len(set(table_ids)) != len(table_ids):
        raise ValueError("owner-color sheet needs unique palette table IDs")

    tables = [load_owner_color_table(owner_palette_pack, table_id) for table_id in table_ids]
    components_by_table = {
        table["color_table_id"]: _load_components(pack, manifest, None, table["colors"])
        for table in tables
    }
    neutral_components = _load_components(pack, manifest, None, None)
    contract = json.loads((pack / manifest["owner_color_contract"]).read_text(encoding="utf-8"))
    gate = contract["coverage_gate"]
    delta_threshold = gate["rgb_delta_threshold"]
    minimum = gate["minimum_changed_pixels"]
    normal_factor = gate["normal_scale_factor"]
    reduced_factor = gate["reduced_scale_factor"]
    cell_width, cell_height = 218, 218
    left, top = 104, 62
    width = left + cell_width * len(tables) + 12
    height = top + cell_height * len(unit_ids) + 38
    canvas = Canvas(width, height, BACKGROUND)
    units: dict[str, Any] = {}

    for row, unit_id in enumerate(unit_ids):
        slug = unit_id.removeprefix("unit/")
        recipe = json.loads((pack / manifest["units"][unit_id]["recipe"]).read_text(encoding="utf-8"))
        member_scale = recipe["member"]["member_scale"] * recipe["member"]["variation_scale"]
        cells = []
        neutral_rendered, _neutral_pose_report = _pose_meshes(
            pack, manifest, recipe, neutral_components, "idle"
        )
        for column, table in enumerate(tables):
            rendered, pose_report = _pose_meshes(
                pack, manifest, recipe, components_by_table[table["color_table_id"]], "idle"
            )
            bounds = _projected_bounds(rendered, member_scale)
            render_scale, center = _fit_cell(bounds, cell_width, cell_height)
            cell = _render_cell(
                rendered,
                member_scale,
                cell_width,
                cell_height,
                render_scale * normal_factor,
                center,
            )
            neutral_cell = _render_cell(
                neutral_rendered,
                member_scale,
                cell_width,
                cell_height,
                render_scale * normal_factor,
                center,
            )
            reduced_cell = _render_cell(
                rendered,
                member_scale,
                cell_width,
                cell_height,
                render_scale * reduced_factor,
                center,
            )
            neutral_reduced_cell = _render_cell(
                neutral_rendered,
                member_scale,
                cell_width,
                cell_height,
                render_scale * reduced_factor,
                center,
            )
            coverage = {
                "normal_changed_pixels": changed_pixel_count(
                    neutral_cell, cell, delta_threshold
                ),
                "reduced_changed_pixels": changed_pixel_count(
                    neutral_reduced_cell, reduced_cell, delta_threshold
                ),
            }
            _paste(canvas, cell, left + column * cell_width, top + row * cell_height)
            cells.append(
                {
                    "display_color_table_id": table["color_table_id"],
                    "display_color": table["display_color"],
                    "fit_scale": render_scale,
                    "sample_time": pose_report["sample_time"],
                    "coverage": coverage,
                }
            )
        best_normal = max(cell["coverage"]["normal_changed_pixels"] for cell in cells)
        best_reduced = max(cell["coverage"]["reduced_changed_pixels"] for cell in cells)
        units[unit_id] = {
            "cells": cells,
            "coverage_gate": {
                "normal_changed_pixels": best_normal,
                "reduced_changed_pixels": best_reduced,
                "status": (
                    "pass"
                    if best_normal >= minimum["normal"] and best_reduced >= minimum["reduced"]
                    else "needs_pack_authoring_override"
                ),
            },
        }

    draw_text(canvas, 14, 10, "RUNTIME CIV COLOR - ONE SHARED UNIT PACK", (232, 224, 194), 2)
    for column, table in enumerate(tables):
        label_x = left + column * cell_width + 64
        draw_text(canvas, label_x, 38, f"TABLE {table['color_table_id']:02d}", (164, 203, 188), 1)
        color = tuple(table["display_color"])
        canvas.fill_polygon(
            [(label_x - 17, 37), (label_x - 5, 37), (label_x - 5, 47), (label_x - 17, 47)],
            color,
        )
    for row, unit_id in enumerate(unit_ids):
        slug = unit_id.removeprefix("unit/")
        draw_text(canvas, 9, top + row * cell_height + 98, slug, (164, 203, 188), 1)
    draw_text(
        canvas,
        14,
        height - 19,
        "OFFLINE L20 PROOF - EXACT CIV3 COLOR / SOURCE-PRESERVING INFERRED TINT / RUNTIME ROW",
        (190, 190, 180),
        1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_png(canvas, output)
    return {
        "schema": "c3x.unit_owner_color_visual_proof.v0",
        "output": str(output),
        "width": width,
        "height": height,
        "palette_table_ids": list(table_ids),
        "units": units,
        "coverage_gate": {
            "measurement": gate["measurement"],
            "palette_sample_policy": gate["palette_sample_policy"],
            "rgb_delta_threshold": delta_threshold,
            "minimum_changed_pixels": minimum,
            "normal_scale_factor": normal_factor,
            "reduced_scale_factor": reduced_factor,
            "status": (
                "pass"
                if all(unit["coverage_gate"]["status"] == "pass" for unit in units.values())
                else "needs_pack_authoring_override"
            ),
        },
        "asset_reuse": "same_mesh_texture_skeleton_and_idle_clip_for_every_palette_column",
        "selection_policy": "runtime_display_color_table_id",
        "tint_calibration": {
            "mode": "source_preserving_linear_modulation",
            "strength": CIV_COLOR_STRENGTH,
            "neutral_floor": CIV_COLOR_NEUTRAL_FLOOR,
            "color_gain": CIV_COLOR_GAIN,
            "status": "provisional_pending_l20_visual_approval",
        },
        "runtime_integration": "not_enabled",
        "non_background_pixels": canvas.non_background_pixels(BACKGROUND),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--owner-palette-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--color-table-id", type=int, action="append", dest="table_ids")
    args = parser.parse_args(argv)
    try:
        report = render_owner_color_sheet(
            args.pack,
            args.owner_palette_pack,
            args.output,
            tuple(args.table_ids) if args.table_ids else DEFAULT_TABLE_IDS,
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
