#!/usr/bin/env python3
"""Review mixed Civ III culture recipes without changing the game art mapping.

Source-family and historical-layer choices live in JSON. This script resolves
them into ordinary normalized city instances; no source selector is needed by
the runtime renderer. The output is ignored Lab evidence, not promoted art.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.all_source_pair_review import OUT, ROOT
from Renderer.lab.studies.cities.ancient_trees import plant
from Renderer.lab.studies.cities.build_layouts import footprint
from Renderer.lab.studies.cities.medieval_family_review import slug
from Renderer.lab.studies.cities.seeded_variants import apply_house_variation
from Renderer.lab.studies.cities.sheet import font, render_cell, render_culture
from Renderer.lab.studies.cities.skyline_recipe import apply_skyline


RECIPE = Path(__file__).with_name("civ3_culture_recipe_candidates.json")
DESTINATION = OUT / "review" / "civ3-mixed-culture-candidates"
ERAS = ("ancient", "medieval", "industrial", "modern")
WALL_KITS = ("ancient", "medieval", "tsikhe", "modern_low")
CELL = (420, 300)


def _bounds(item: dict) -> list[float]:
    body = component(item["asset"], Path(item["pack"]))
    return footprint({"low": body["lo"], "high": body["hi"]}, item)


@lru_cache(None)
def source_design(source_era: str, family: str) -> dict:
    path = OUT / "review" / source_era / slug(family) / "layouts.json"
    layouts = json.loads(path.read_text(encoding="utf-8"))
    source_tag = "ARTERA_" + source_era.upper()
    matches = [item for item in layouts["designs"]
               if item.get("source_art_era") == source_tag]
    if len(matches) != 1:
        raise ValueError(f"Expected one {source_tag}/{family} design in {path}")
    return matches[0]


def _seed(*values: object) -> random.Random:
    digest = hashlib.sha256("|".join(str(value) for value in values).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _donors(source: dict, role: str, size: int, rng: random.Random) -> list[dict]:
    tier = source["tier_designs"][size]
    houses = [item for item in tier["houses"]
              if not item.get("skyline_role") and abs(item["rotation"]) <= .2]
    rng.shuffle(houses)
    ordered = ([tier["base_centerpiece"]] + houses
               if role in ("civic", "heritage") else houses)
    unique = {}
    for item in ordered:
        unique.setdefault(item["asset"], item)
    donors = list(unique.values())
    if role == "heritage":
        # Older rooflines have to remain legible beside modern buildings.
        # Prefer the taller usable source pieces, with seed-dependent ties.
        def visual_height(item):
            bounds = component(item["asset"], Path(item["pack"]))
            return (bounds["hi"][2] - bounds["lo"][2]) * item["scale"]
        donors.sort(key=lambda item: visual_height(item) + .02 * rng.random(),
                    reverse=True)
    return donors


def _placement(old: dict, donor: dict, role: str, size: int,
               occupied: list[dict]) -> dict | None:
    old_box = _bounds(old)
    for factor in (1.0, .95, .9, .85, .8):
        item = {"asset": donor["asset"], "pack": donor["pack"],
                "scale": round(donor["scale"] * factor, 6),
                "rotation": donor["rotation"], "offset": old["offset"][:],
                "recipe_role": role}
        bounds = _bounds(item)
        # Replace a plot's building, never extend the town or the larger
        # city's sprawl to make a historical landmark fit.
        if any(bounds[axis] < old_box[axis] - .09 for axis in (0, 1)) or any(
            bounds[axis] > old_box[axis] + .09 for axis in (2, 3)
        ):
            continue
        limit = (.5, .72, .79)[size]
        if any(abs(edge) > limit for edge in bounds):
            continue
        old_collisions = {index for index, other in enumerate(occupied)
                          if overlaps(old_box, _bounds(other))}
        if any(overlaps(bounds, _bounds(other)) and index not in old_collisions
               for index, other in enumerate(occupied)):
            continue
        return item
    return None


def _slots(houses: list[dict], start: int, region: str, size: int,
           rng: random.Random) -> list[int]:
    positions = [index for index in range(start, len(houses))
                 if not houses[index].get("skyline_role") and
                 not houses[index].get("recipe_role")]
    rng.shuffle(positions)
    def score(index):
        x, y = houses[index]["offset"]
        radius = (x * x + y * y) ** .5
        # The outskirts are an inhabited ring, not the isolated farthest
        # plots of an already sprawling Metropolis. Favor the screen-front
        # half so a retained landmark reads beside, not behind, downtown.
        return (abs(radius - (.38, .51, .61)[size]) +
                .17 * max(0.0, .3 - x - y) if region == "outer" else radius)
    positions.sort(key=score)
    return positions


def add_layer(design: dict, layer: dict, identity: str, variation_seed: int) -> dict:
    donor = source_design(layer["source_era"], layer["family"])
    counts = layer["counts"]
    if (len(counts) != 3 or counts != sorted(counts) or
            layer["role"] not in ("civic", "heritage", "older_roofs") or
            layer["region"] not in ("inner", "outer")):
        raise ValueError(f"Invalid historical layer for {identity}: {layer}")
    tiers = design["tier_designs"]
    placed = {}
    for key in ("houses", "capital_houses"):
        placed[key] = []
        for size, target in enumerate(counts):
            existing = sum(item.get("recipe_role") == layer["role"] and
                           item.get("recipe_source_family") == layer["family"]
                           for item in tiers[size][key])
            needed = target - existing
            if needed < 0:
                raise ValueError(f"Layer count shrank for {identity}/{key}/{size}")
            start = len(tiers[size - 1][key]) if size else 0
            rng = _seed(identity, layer["family"], layer["role"], key,
                        size, variation_seed)
            candidates = _donors(donor, layer["role"], size, rng)
            slots = _slots(tiers[size][key], start, layer["region"], size, rng)
            for _ in range(needed):
                found = False
                used = {item["asset"] for item in tiers[size][key]
                        if item.get("recipe_source_family") == layer["family"]}
                for slot in slots:
                    old = tiers[size][key][slot]
                    if old.get("recipe_role") or old.get("skyline_role"):
                        continue
                    for donor_item in candidates:
                        if donor_item["asset"] in used:
                            continue
                        # City buildings persist in the corresponding Metro
                        # plots, and Town buildings likewise persist on growth.
                        replacements = []
                        for larger in range(size, 3):
                            houses = tiers[larger][key]
                            if slot >= len(houses) or houses[slot]["offset"] != old["offset"]:
                                break
                            center = (tiers[larger]["palace"] if key == "capital_houses"
                                      else tiers[larger]["base_centerpiece"])
                            occupied = [center] + [item for index, item in enumerate(houses)
                                                   if index != slot]
                            item = _placement(houses[slot], donor_item, layer["role"],
                                              larger, occupied)
                            if item is None:
                                break
                            item["recipe_source_family"] = layer["family"]
                            replacements.append((larger, item))
                        if len(replacements) != 3 - size:
                            continue
                        for larger, item in replacements:
                            tiers[larger][key][slot] = item
                        found = True
                        break
                    if found:
                        break
                if not found:
                    raise ValueError(f"No safe {layer['role']} plot for {identity} "
                                     f"{key} tier {size}; requested {counts}")
            actual = sum(item.get("recipe_role") == layer["role"] and
                         item.get("recipe_source_family") == layer["family"]
                         for item in tiers[size][key])
            if actual != target:
                raise ValueError(f"Incomplete {layer['role']} layer for {identity} "
                                 f"{key} tier {size}: {actual}/{target}")
            placed[key].append(actual)
    design["houses"] = tiers[2]["houses"]
    design["population_counts"] = [len(tier["houses"]) for tier in tiers]
    return {"family": layer["family"], "role": layer["role"], "counts": placed}


def compose(profile: dict, era: str, variation_seed: int = 0) -> tuple[dict, list[dict]]:
    recipe = profile["eras"][era]
    base = recipe["base"]
    design = copy.deepcopy(source_design(base["source_era"], base["family"]))
    design["culture"] = profile["culture_group"]
    design["culture_name"] = profile["label"]
    design["era"] = ERAS.index(era)
    design["era_name"] = "Middle Ages" if era == "medieval" else era.title()
    design["target_civ3_era"] = era
    design["art_profile_id"] = profile["id"]
    design["recipe_base"] = base
    design["wall_kit"] = WALL_KITS[design["era"]]
    if era in ("industrial", "modern"):
        apply_skyline(design, era, variation_seed)
    if variation_seed:
        apply_house_variation(design, variation_seed)
    layers = []
    for layer in recipe["layers"]:
        layers.append(add_layer(design, layer, profile["id"], variation_seed))
    tree_pack = ROOT / "Renderer/lab/out/cities/medieval-art/farm-tree-pack"
    plant(design["tier_designs"], tree_pack, counts=(2, 5, 8), scale=3.1,
          seed=f"{profile['id']}|{era}|{variation_seed}")
    return design, layers


def build(config: Path = RECIPE, destination: Path = DESTINATION,
          variation_seed: int = 0, full_sheets: bool = True) -> Path:
    data = json.loads(config.read_text(encoding="utf-8"))
    if data.get("schema") != "c3x.lab.city_culture_recipes.v1":
        raise ValueError("Unsupported city recipe schema")
    profiles = sorted(data["profiles"], key=lambda item: item["culture_group"])
    if {item["culture_group"] for item in profiles} != set(range(5)):
        raise ValueError("City recipes must cover the five Civ III culture groups")
    if any(set(profile["eras"]) != set(ERAS) for profile in profiles):
        raise ValueError("Every culture profile needs four Civ III eras")
    output = destination / f"seed-{variation_seed}"
    output.mkdir(parents=True, exist_ok=True)
    designs = {}
    report = []
    for profile in profiles:
        culture_designs = []
        for era in ERAS:
            design, layers = compose(profile, era, variation_seed)
            culture_designs.append(design)
            report.append({"profile": profile["id"], "culture_group": profile["culture_group"],
                           "era": era, "base": design["recipe_base"], "layers": layers,
                           "building_counts": design["population_counts"]})
        designs[profile["id"]] = culture_designs
        folder = output / profile["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "layouts.json").write_text(json.dumps(culture_designs, indent=2) + "\n",
                                               encoding="utf-8")
        if full_sheets:
            render_culture({"styles": [item["label"] for item in profiles],
                            "designs": culture_designs},
                           profile["culture_group"], folder / "sheet.png")
        print(profile["id"], [entry["counts"] for row in report[-4:]
                              for entry in row["layers"]], flush=True)
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    left, top, row_height, column_width = 120, 100, CELL[1] + 42, CELL[0] + 8
    image = Image.new("RGB", (left + column_width * 5, top + row_height * 4),
                      (32, 26, 40))
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), f"Civ III mixed culture candidates | seed {variation_seed}",
              font=font(28), fill=(248, 235, 248))
    draw.text((16, 51), "Metropolis bases; full Town/City/Metro state sheets are saved per culture",
              font=font(17), fill=(207, 190, 209))
    for column, profile in enumerate(profiles):
        x = left + column * column_width
        draw.text((x + 8, 76), profile["label"], font=font(19), fill=(248, 235, 248))
        for row, era in enumerate(ERAS):
            y = top + row * row_height
            if column == 0:
                draw.text((12, y + 8), "Middle Ages" if era == "medieval" else era.title(),
                          font=font(17), fill=(248, 235, 248))
            cell, _ = render_cell(designs[profile["id"]][row], 2, False, False,
                                  cell=CELL, tile_pixels=300)
            image.paste(cell, (x, y + 30))
    target = output / "overview.png"
    image.save(target)
    if full_sheets:
        detail_cell = (620, 400)
        detail = Image.new("RGB", (1280, 5 * 445 + 84), (32, 26, 40))
        labels = ImageDraw.Draw(detail)
        labels.text((16, 10), f"Industrial and Modern outer-ring heritage | seed {variation_seed}",
                    font=font(26), fill=(248, 235, 248))
        labels.text((140, 50), "Industrial", font=font(18), fill=(248, 235, 248))
        labels.text((780, 50), "Modern", font=font(18), fill=(248, 235, 248))
        for row, profile in enumerate(profiles):
            y = 84 + row * 445
            labels.text((8, y + 4), profile["label"], font=font(16),
                        fill=(248, 235, 248))
            for column, era_index in enumerate((2, 3)):
                cell, _ = render_cell(designs[profile["id"]][era_index], 2, False,
                                      False, cell=detail_cell, tile_pixels=420)
                detail.paste(cell, (20 + column * 640, y + 30))
        detail.save(output / "late-era-detail.png")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overview-only", action="store_true")
    arguments = parser.parse_args()
    print(build(variation_seed=arguments.seed, full_sheets=not arguments.overview_only))
