"""Compare seeded late-era city candidates for the five Civ III cultures.

This is an offline art audition. The source-family mapping remains editable
metadata, and the seed stands in for a stable per-city identity.
"""

from __future__ import annotations

import json

from PIL import Image, ImageDraw

from Renderer.lab.studies.cities.all_source_pair_review import FAMILIES, OUT, ROOT, palace_for
from Renderer.lab.studies.cities.medieval_family_review import generate
from Renderer.lab.studies.cities.medieval_family_review import slug
from Renderer.lab.studies.cities.sheet import font, render_cell, render_culture
from Renderer.lab.studies.cities.skyline_recipe import apply_skyline
from Renderer.lab.studies.cities.seeded_variants import apply_house_variation


STRATEGY = ROOT / "Renderer/tools/asset_compiler/city_render_strategy.json"
CELL = (460, 350)


def composition(style, era, seed, palaces):
    base_era = style.get("composition_base_by_era", {}).get(era, era)
    family = style["source_culture_by_era"][base_era]
    report_path, pack = FAMILIES[base_era]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pool = next(item for item in report["pools"]
                if item["source_culture"] == family)
    palace, _ = palace_for(pool, palaces)
    layouts, _ = generate(pool, pack, OUT / "flat-palaces", palace,
                          ROOT / "Renderer/lab/out/cities/medieval-art/farm-tree-pack",
                          use_curated=False, variation_seed=seed)
    design = next(item for item in layouts["designs"]
                  if item.get("source_art_era") == pool["source_art_era"])
    apply_skyline(design, era, seed)
    apply_house_variation(design, seed)
    if era == "modern":
        design["wall_kit"] = "modern_low"
    design["era_name"] = era.title()
    design["target_civ3_era"] = era
    design["composition_base_source_era"] = base_era
    return design, family


def build():
    strategy = json.loads(STRATEGY.read_text(encoding="utf-8"))
    styles = sorted(strategy["styles"], key=lambda entry: entry["civ3_culture_group"])
    palaces = json.loads((ROOT / "Renderer/packs/CityPalacesNormalized/palace_catalog.json")
                         .read_text(encoding="utf-8"))
    if len(styles) != 5:
        raise ValueError("Expected the five Civ III culture group mappings")
    left, top, caption, gap = 125, 94, 36, 10
    width = left + len(styles) * (CELL[0] + gap)
    height = top + 4 * (CELL[1] + caption + gap)
    image = Image.new("RGB", (width, height), (31, 25, 38))
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "Civ III culture candidates | seed 0 templates",
              font=font(27), fill=(249, 236, 249))
    draw.text((16, 49), "Modern retains each culture's Industrial base; alternate city seeds are reviewed separately",
              font=font(17), fill=(202, 186, 204))
    for column, style in enumerate(styles):
        x = left + column * (CELL[0] + gap)
        draw.text((x + 12, 75), style["id"].replace("_", " ").title(),
                  font=font(18), fill=(249, 236, 249))
        seed = 0
        for era_index, era in enumerate(("industrial", "modern")):
            design, family = composition(style, era, seed, palaces)
            recipe = OUT / "review" / "civ3-culture-compositions" / slug(style["id"]) / era
            recipe.mkdir(parents=True, exist_ok=True)
            (recipe / "design.json").write_text(json.dumps(design, indent=2) + "\n",
                                                encoding="utf-8")
            design["review_context"] = f"Civ III {era.title()} culture candidate"
            render_culture({"designs": [design], "styles": [style["id"].title()] * 5},
                           1, recipe / "sheet.png", only_era=1)
            for size in (1, 2):
                row = era_index * 2 + size - 1
                y = top + row * (CELL[1] + caption + gap)
                if column == 0:
                    draw.text((12, y + 8), era.title(), font=font(18),
                              fill=(249, 236, 249))
                    draw.text((12, y + 32), "City" if size == 1 else "Metropolis",
                              font=font(17), fill=(202, 186, 204))
                draw.text((x + 10, y + 8), f"{family} | seed {seed}",
                          font=font(17), fill=(249, 236, 249))
                cell, _ = render_cell(design, size, False, False,
                                      cell=CELL, tile_pixels=340)
                image.paste(cell, (x, y + caption))
    target = OUT / "review" / "civ3-five-culture-late-era-candidates.png"
    image.save(target)
    return target


def build_capitals():
    """Focus the review on the glass buildings' relationship to each palace."""
    strategy = json.loads(STRATEGY.read_text(encoding="utf-8"))
    styles = sorted(strategy["styles"], key=lambda entry: entry["civ3_culture_group"])
    left, top, caption, gap = 125, 100, 36, 10
    image = Image.new("RGB", (left + len(styles) * (CELL[0] + gap),
                              top + 2 * (CELL[1] + caption + gap)), (31, 25, 38))
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "Modern capitals | glass towers behind the palace",
              font=font(27), fill=(249, 236, 249))
    draw.text((16, 49), "City adds one tower; Metropolis adds a second distinct tower in the same rear arc",
              font=font(17), fill=(202, 186, 204))
    for column, style in enumerate(styles):
        x = left + column * (CELL[0] + gap)
        draw.text((x + 12, 77), style["id"].replace("_", " ").title(),
                  font=font(18), fill=(249, 236, 249))
        recipe = OUT / "review" / "civ3-culture-compositions" / slug(style["id"]) / "modern"
        design = json.loads((recipe / "design.json").read_text(encoding="utf-8"))
        for size in (1, 2):
            y = top + (size - 1) * (CELL[1] + caption + gap)
            if column == 0:
                draw.text((12, y + 8), "City" if size == 1 else "Metropolis",
                          font=font(18), fill=(249, 236, 249))
            draw.text((x + 10, y + 8), f"{design['culture_name']} | seed 0",
                      font=font(17), fill=(249, 236, 249))
            cell, _ = render_cell(design, size, False, True,
                                  cell=CELL, tile_pixels=340)
            image.paste(cell, (x, y + caption))
    target = OUT / "review" / "modern-capital-downtown-comparison.png"
    image.save(target)
    return target


if __name__ == "__main__":
    print(build())
    print(build_capitals())
