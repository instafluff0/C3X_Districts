#!/usr/bin/env python3
"""Render all compiled Civ III owner-color tables as a deterministic lab sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import write_png
from Renderer.preview.render_unit_turntable import draw_text


BACKGROUND = (25, 31, 34)


def _rect(canvas: Canvas, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
    for row in range(y, y + height):
        for column in range(x, x + width):
            canvas.set_pixel(column, row, color)


def render_palette_sheet(pack: Path, output: Path) -> dict[str, object]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.owner_color_pack.v0":
        raise ValueError("owner-color pack has the wrong schema")
    document = json.loads((pack / manifest["color_tables"]).read_text(encoding="utf-8"))
    tables = document.get("tables", [])
    if document.get("schema") != "c3x.owner_color_tables.v0" or len(tables) != 32:
        raise ValueError("owner-color pack does not contain 32 tables")
    left, top, swatch, gap, row_height = 72, 42, 10, 1, 18
    width = left + 64 * (swatch + gap) + 18
    height = top + 32 * row_height + 30
    canvas = Canvas(width, height, BACKGROUND)
    draw_text(canvas, 12, 10, "CIV III OWNER COLOR TABLES - EXACT PALETTE ENTRIES 00-63", (232, 224, 194), 2)
    draw_text(canvas, left, 29, "PRIMARY RAMP 00-15", (164, 203, 188), 1)
    for row, table in enumerate(tables):
        if table.get("color_table_id") != row or len(table.get("colors", [])) != 64:
            raise ValueError("owner-color table ordering or size is invalid")
        y = top + row * row_height
        draw_text(canvas, 12, y + 1, f"TABLE {row:02d}", (190, 190, 180), 1)
        for column, color in enumerate(table["colors"]):
            x = left + column * (swatch + gap)
            _rect(canvas, x, y, swatch, 12, tuple(color))
            if column == document["display_color_index"]:
                for border_x in range(x - 1, x + swatch + 1):
                    canvas.set_pixel(border_x, y - 1, (255, 255, 255))
                    canvas.set_pixel(border_x, y + 12, (255, 255, 255))
                for border_y in range(y - 1, y + 13):
                    canvas.set_pixel(x - 1, border_y, (255, 255, 255))
                    canvas.set_pixel(x + swatch, border_y, (255, 255, 255))
    draw_text(canvas, 12, height - 17, "WHITE OUTLINE = CIV COLOR DISPLAY INDEX 06 / LAB ONLY", (190, 190, 180), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_png(canvas, output)
    return {
        "schema": "c3x.owner_color_palette_sheet_report.v0",
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "width": width,
        "height": height,
        "table_count": len(tables),
        "colors_per_table": 64,
        "display_color_index": document["display_color_index"],
        "primary_ramp": document["primary_ramp"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = render_palette_sheet(args.pack, args.output)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output} ({report['table_count']} owner-color tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
