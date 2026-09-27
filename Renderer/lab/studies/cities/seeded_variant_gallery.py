"""Index three offline city recipe candidates and preview a few side by side."""

from __future__ import annotations

import json

from PIL import Image, ImageDraw

from Renderer.lab.studies.cities.all_source_pair_review import OUT, ROOT
from Renderer.lab.studies.cities.civ3_late_culture_gallery import STRATEGY, composition
from Renderer.lab.studies.cities.medieval_family_review import slug
from Renderer.lab.studies.cities.sheet import font, render_cell


ERAS = ("ancient", "classical", "industrial", "modern", "future", "unspecified")
SEEDS = (0, 1, 2)
CELL = (560, 390)


def index_variants():
    pairs = []
    for era in ERAS:
        by_seed = {}
        for seed in SEEDS:
            root = OUT / ("review" if seed == 0 else f"review-variants/seed-{seed}")
            entries = json.loads((root / era / "index.json").read_text(encoding="utf-8"))
            by_seed[seed] = {(entry["source_culture"], entry["source_art_era"]): entry
                             for entry in entries}
        if not all(set(by_seed[seed]) == set(by_seed[0]) for seed in SEEDS[1:]):
            raise ValueError(f"Incomplete {era} city variant family set")
        for culture, source_era in sorted(by_seed[0]):
            variants = []
            for seed in SEEDS:
                root = ("review" if seed == 0 else f"review-variants/seed-{seed}")
                variants.append({"seed": seed,
                                 "layout": f"{root}/{era}/{slug(culture)}/layouts.json",
                                 "house_changes": by_seed[seed][(culture, source_era)]
                                                  .get("seeded_house_changes", {})})
            pairs.append({"era_audition": era, "source_culture": culture,
                          "source_art_era": source_era, "variants": variants})
    by_family = {(pair["era_audition"], pair["source_culture"]): pair
                 for pair in pairs}
    strategy = json.loads(STRATEGY.read_text(encoding="utf-8"))
    palaces = json.loads((ROOT / "Renderer/packs/CityPalacesNormalized/palace_catalog.json")
                         .read_text(encoding="utf-8"))
    mapped = []
    for style in sorted(strategy["styles"], key=lambda item: item["civ3_culture_group"]):
        for civ3_era, era in enumerate(("ancient", "medieval", "industrial", "modern")):
            base_era = style.get("composition_base_by_era", {}).get(era, era)
            family = style["source_culture_by_era"][base_era]
            variants = []
            if era == "modern":
                for seed in SEEDS:
                    design, _ = composition(style, era, seed, palaces)
                    path = (f"review-variants/civ3-mapped/{slug(style['id'])}/"
                            f"{era}/seed-{seed}/design.json")
                    destination = OUT / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(design, indent=2) + "\n",
                                           encoding="utf-8")
                    variants.append({"seed": seed, "design": path})
            else:
                source_era = "classical" if era == "medieval" else era
                variants = [entry.copy() for entry in by_family[(source_era, family)]["variants"]]
            mapped.append({"civ3_culture_group": style["civ3_culture_group"],
                           "culture_id": style["id"], "civ3_era": civ3_era,
                           "era": era, "source_family": family,
                           "variant_kind": ("composition_design" if era == "modern"
                                            else "source_layout"),
                           "variants": variants})
    manifest = {
        "schema": "c3x.lab.city_recipe_variants.v1",
        "status": "review_candidates_not_promoted",
        "gallery_seed": 0,
        "candidate_seeds": list(SEEDS),
        "future_selection": {"inputs": ["world_seed", "city_id", "map_x", "map_y"],
                             "rule": "stable_hash_mod_candidate_count"},
        "source_pairs": pairs,
        "civ3_candidates": mapped,
    }
    target = OUT / "review" / "seeded-variant-manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def build():
    samples = [
        ("ancient", "DEFAULT", 1, False, "Ancient\nDEFAULT\nCity base"),
        ("classical", "Mediterranean", 2, True, "Classical\nMediterranean\nMetropolis capital"),
        ("classical", "EastAsian", 2, False, "Classical\nEastAsian\nMetropolis base"),
        ("industrial", "RowHouse", 1, True, "Industrial\nRowHouse\nCity capital"),
        ("mapped-modern", "asian", 2, True, "Modern Asian\nMetropolis capital"),
    ]
    top, label_height, gap = 90, 34, 12
    image = Image.new("RGB", (155 + len(SEEDS) * (CELL[0] + gap),
                              top + len(samples) * (CELL[1] + label_height + gap)),
                      (31, 25, 38))
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "City recipe variety | same culture, era, size and capital state",
              font=font(27), fill=(249, 236, 249))
    draw.text((16, 50), "Seed 0 remains the main template gallery; alternate seeds change a few houses and trees",
              font=font(17), fill=(202, 186, 204))
    strategy = json.loads(STRATEGY.read_text(encoding="utf-8"))
    asian = next(style for style in strategy["styles"] if style["id"] == "asian")
    palaces = json.loads((ROOT / "Renderer/packs/CityPalacesNormalized/palace_catalog.json")
                         .read_text(encoding="utf-8"))
    for row, (era, family, size, capital, label) in enumerate(samples):
        y = top + row * (CELL[1] + label_height + gap)
        draw.multiline_text((12, y + 6), label, font=font(15),
                  fill=(249, 236, 249))
        for column, seed in enumerate(SEEDS):
            x = 155 + column * (CELL[0] + gap)
            draw.text((x + 10, y + 6), f"Seed {seed}", font=font(18),
                      fill=(249, 236, 249))
            if era == "mapped-modern":
                design, _ = composition(asian, "modern", seed, palaces)
            else:
                root = OUT / ("review" if seed == 0 else f"review-variants/seed-{seed}")
                layouts = json.loads((root / era / slug(family) / "layouts.json")
                                     .read_text(encoding="utf-8"))
                if era == "classical":
                    design = next(item for item in layouts["designs"]
                                  if (item["culture"], item["era"]) == (1, 1))
                else:
                    source_era = ("ARTERA_ANCIENT" if era == "ancient"
                                  else "ARTERA_INDUSTRIAL")
                    design = next(item for item in layouts["designs"]
                                  if item.get("source_art_era") == source_era)
            cell, _ = render_cell(design, size, False, capital,
                                  cell=CELL, tile_pixels=390)
            image.paste(cell, (x, y + label_height))
    target = OUT / "review" / "seeded-variant-examples.png"
    image.save(target)
    return target


if __name__ == "__main__":
    print(index_variants())
    print(build())
