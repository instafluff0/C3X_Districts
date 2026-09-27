#!/usr/bin/env python3
"""Render the standard and supplemental wall kits around one fixed city.

The modern barricade is an audition from Zombie Defense, not a proposed
replacement for the default city perimeter.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from Renderer.lab.studies.cities.sheet import font, render_cell


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Renderer/lab/out/cities/wall-source-audit/wall-kit-comparison.png"
KITS = (
    ("ancient", "Ancient Walls"),
    ("medieval", "Castle"),
    ("industrial", "Star Fort (Renaissance)"),
    ("tsikhe", "Tsikhe (Georgia)"),
    ("pirate_medieval", "Pirates scenario castle"),
    ("modern_barricade", "Modern Tower Defense*"),
    ("modern_clean", "Modern Tower Defense, spikes removed*"),
    ("modern_low", "Modern Tower Defense, lower wall*"),
)
PAIR_OUT = OUT.with_name("modern-tower-defense-refinement.png")


def main() -> None:
    layouts = json.loads((ROOT / "Renderer/lab/out/cities/all-era-source-auditions/review/modern/default/layouts.json").read_text())
    design = next(item for item in layouts["designs"]
                  if item["culture"] == 1 and item["era"] == 1)
    cell = (650, 430)
    margin, header, label = 24, 84, 42
    canvas = Image.new("RGB", (margin * 2 + cell[0] * len(KITS),
                               header + cell[1] + label + 46), (34, 29, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), f"Civ VI wall source audit | same modern city, {len(KITS)} source kits",
              font=font(26), fill=(249, 240, 249))
    draw.text((margin, 51), "Civ VI Modern DEFAULT buildings | C3X City population | flat magenta | source wall geometry",
              font=font(16), fill=(206, 192, 213))
    for index, (kit, title) in enumerate(KITS):
        x = margin + index * cell[0]
        draw.text((x + 10, header + 8), title, font=font(18), fill=(250, 230, 247))
        image, _ = render_cell({**design, "wall_kit": kit}, 1, True, False,
                               cell=cell, tile_pixels=440)
        canvas.paste(image, (x, header + label))
    draw.text((margin, header + label + cell[1] + 12),
              "*Zombie Defense improvement; no source gate. Shown for comparison, not selected for city runtime.",
              font=font(16), fill=(224, 203, 225))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(OUT.relative_to(ROOT))

    pair_cell = (700, 460)
    pair = Image.new("RGB", (pair_cell[0] * 3 + 48, pair_cell[1] + 150), (34, 29, 42))
    draw = ImageDraw.Draw(pair)
    draw.text((24, 15), "Modern Tower Defense | original, spike-free, then lower masonry",
              font=font(27), fill=(249, 240, 249))
    draw.text((24, 52), "Same Civ VI Modern DEFAULT city, wall texture and camera | no texture stretching",
              font=font(17), fill=(206, 192, 213))
    for index, (kit, title) in enumerate((("modern_barricade", "Original source"),
                                           ("modern_clean", "Spikes removed"),
                                           ("modern_low", "Lower skirt removed"))):
        x = 24 + index * pair_cell[0]
        draw.text((x + 10, 84), title, font=font(21), fill=(250, 230, 247))
        image, _ = render_cell({**design, "wall_kit": kit}, 1, True, False,
                               cell=pair_cell, tile_pixels=490)
        pair.paste(image, (x, 118))
    draw.text((24, 588), "Zombie Defense improvement art; this is a Lab audition, not the selected city-wall kit.",
              font=font(16), fill=(224, 203, 225))
    pair.save(PAIR_OUT)
    print(PAIR_OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
