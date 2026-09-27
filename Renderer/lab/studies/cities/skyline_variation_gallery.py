"""Show seeded skyline and vegetation variation across late source families."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from Renderer.lab.studies.cities.all_source_pair_review import OUT
from Renderer.lab.studies.cities.medieval_family_review import slug
from Renderer.lab.studies.cities.sheet import font, render_cell


def build(era: str) -> Path:
    entries = json.loads((OUT / "review" / era / "index.json").read_text())
    cell = (560, 360)
    row_height = cell[1] + 40
    image = Image.new("RGB", (cell[0] * 3 + 140,
                              100 + len(entries) * 2 * row_height), (31, 25, 38))
    draw = ImageDraw.Draw(image)
    title = (f"{era.title()} city recipes | stable downtown, house and tree seeds"
             if era == "industrial" else
             "Modern source-art auditions | stable house and tree seeds")
    draw.text((20, 14), title,
              font=font(27), fill=(249, 236, 249))
    detail = ("Same family and population per row; downtown, houses and farm trees vary by seed"
              if era == "industrial" else
              "These raw Modern source families vary individual houses and farm trees")
    draw.text((20, 51), detail,
              font=font(17), fill=(202, 186, 204))
    for seed in range(3):
        draw.text((140 + seed * cell[0] + 12, 76), f"Seed {seed}",
                  font=font(18), fill=(249, 236, 249))
    for family_index, entry in enumerate(entries):
        family = entry["source_culture"]
        for seed in range(3):
            root = (OUT / "review" if seed == 0 else
                    OUT / "review-variants" / f"seed-{seed}")
            path = root / era / slug(family) / "layouts.json"
            layouts = json.loads(path.read_text())
            design = next(item for item in layouts["designs"]
                          if item.get("source_art_era") == entry["source_art_era"])
            for size in (1, 2):
                row = family_index * 2 + size - 1
                y = 100 + row * row_height
                if seed == 0:
                    draw.text((12, y + 12), family.removeprefix("CIVILIZATION_")[:14],
                              font=font(16), fill=(249, 236, 249))
                    draw.text((12, y + 34), "City" if size == 1 else "Metropolis",
                              font=font(15), fill=(202, 186, 204))
                image_cell, _ = render_cell(design, size, False, False,
                                            cell=cell, tile_pixels=390)
                image.paste(image_cell, (140 + seed * cell[0], y))
    suffix = "seeded-skyline-and-trees" if era == "industrial" else "seeded-source-art-and-trees"
    target = OUT / "review" / f"{era}-{suffix}.png"
    image.save(target)
    return target


if __name__ == "__main__":
    for era_name in ("industrial", "modern"):
        print(build(era_name))
