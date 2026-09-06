#!/usr/bin/env python3
"""Render a labeled L8 contact sheet from a normalized vegetation pack.

This lab-only compositor uses Pillow for labels and layout.  The asset views
themselves come from render_feature_asset's source-agnostic mesh/DDS renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from render_feature_asset import BACKGROUND, render_feature


GROUPS = (
    ("forest", "FOREST - BASE"),
    ("forest_snow", "FOREST - SNOW"),
    ("jungle", "JUNGLE"),
)
SHEET_BACKGROUND = (*BACKGROUND, 255)
CELL_BACKGROUND = (31, 38, 40, 255)
CELL_BORDER = (66, 81, 72, 255)
TEXT = (224, 230, 219, 255)
MUTED = (157, 171, 158, 255)
ACCENT = (143, 181, 92, 255)


def asset_label(asset_id: str) -> str:
    parts = asset_id.split("/")
    return " / ".join(part.upper() for part in parts[1:])


def canvas_image(canvas) -> Image.Image:
    image = Image.new("RGB", (canvas.width, canvas.height))
    image.putdata(canvas.pixels)
    return image.convert("RGBA")


def render_sheet(manifest_path: Path, columns: int = 4) -> Image.Image:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.asset_pack.v0":
        raise ValueError("Unsupported asset-pack schema")

    cell_width = 400
    preview_height = 256
    label_height = 34
    cell_height = preview_height + label_height
    section_header = 48
    margin = 24
    footer = 52
    group_assets = []
    total_height = margin
    for group, title in GROUPS:
        variants = manifest.get("features", {}).get(group, {}).get("variants", [])
        if not variants:
            raise ValueError(f"Pack has no {group} variants")
        rows = (len(variants) + columns - 1) // columns
        group_assets.append((group, title, variants, rows))
        total_height += section_header + rows * cell_height + margin

    width = margin * 2 + columns * cell_width
    height = total_height + footer
    sheet = Image.new("RGBA", (width, height), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=24)
    label_font = ImageFont.load_default(size=15)
    footer_font = ImageFont.load_default(size=14)

    y = margin
    for _group, title, variants, rows in group_assets:
        draw.text((margin, y + 8), title, font=title_font, fill=ACCENT)
        draw.text(
            (width - margin, y + 14),
            f"{len(variants)} VERIFIED BODY VARIANTS",
            font=footer_font,
            fill=MUTED,
            anchor="ra",
        )
        y += section_header
        for index, asset_id in enumerate(variants):
            column = index % columns
            row = index // columns
            x = margin + column * cell_width
            cell_y = y + row * cell_height
            draw.rounded_rectangle(
                (x + 4, cell_y + 3, x + cell_width - 5, cell_y + cell_height - 5),
                radius=8,
                fill=CELL_BACKGROUND,
                outline=CELL_BORDER,
                width=1,
            )
            draw.text(
                (x + 16, cell_y + 10),
                asset_label(asset_id),
                font=label_font,
                fill=TEXT,
            )
            preview = canvas_image(
                render_feature(
                    manifest_path,
                    asset_id,
                    cell_width - 10,
                    preview_height,
                    model_scale_multiplier=1.65,
                )
            )
            sheet.alpha_composite(preview, (x + 5, cell_y + label_height))
        y += rows * cell_height + margin

    draw.line((margin, height - footer, width - margin, height - footer), fill=CELL_BORDER, width=1)
    draw.text(
        (margin, height - footer + 17),
        "TWO VIEWS PER ASSET  |  SHARED 1.65X VIEW ZOOM  |  SOURCE RELATIVE SCALE  |  NO PROXY ART",
        font=footer_font,
        fill=MUTED,
    )
    return sheet.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the L8 vegetation set contact sheet")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.columns < 1 or args.columns > 8:
            raise ValueError("Column count must be between 1 and 8")
        sheet = render_sheet(args.pack, args.columns)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.output)
        print(f"Wrote {args.output} ({sheet.width}x{sheet.height})")
        if args.thumbnail:
            args.thumbnail.parent.mkdir(parents=True, exist_ok=True)
            thumbnail = sheet.resize((sheet.width // 2, sheet.height // 2), Image.Resampling.BOX)
            thumbnail.save(args.thumbnail)
            print(f"Wrote {args.thumbnail} ({thumbnail.width}x{thumbnail.height})")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
