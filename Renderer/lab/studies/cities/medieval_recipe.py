#!/usr/bin/env python3
"""Curated Mediterranean medieval flat-ground composition candidate."""

import argparse
import json
import math
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.studies.cities.build_layouts import footprint
from Renderer.lab.shared.cities.growth import overlaps


ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--flat-pack", type=Path, required=True)
    parser.add_argument("--flat-palace-pack", type=Path)
    parser.add_argument("--farm-tree-pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_report.read_text())
    by_name = {entry["entry"].removeprefix("DIS_CTY_RMED_"): entry["asset_id"]
               for entry in source["pools"][0]["selected"]}
    pack = args.flat_pack.as_posix()

    def part(name, scale, x, y):
        return {"asset": by_name[name], "pack": pack, "scale": scale,
                "rotation": 0.0, "offset": [x, y]}

    town = [
        part("Block_REC_002", 1.65, -.20, .23),
        part("Block_REC_003", 1.65, .20, .23),
        part("Bld_010", 3.3, -.35, -.04),
        part("Bld_08", 3.3, .35, -.04),
        part("Bld_02", 3.3, 0, .39),
    ]
    city = town + [
        part("Block_REC_002", 1.3, -.44, -.33),
        part("Block_REC_003", 1.3, .44, -.33),
        part("Bld_01", 3.3, -.49, .35),
        part("Bld_07", 3.3, .49, .35),
    ]
    metro = city + [
        part("Block_SQ_003", 1.6, -.66, -.06),
        part("Block_SQ_001", 1.6, .65, -.06),
        part("Bld_03", 3.3, -.27, .56),
        part("Bld_05", 3.3, .27, .56),
    ]
    capital_town = [
        part("Bld_010", 3.65, -.335, .17),
        part("Bld_08", 3.3, .335, .17),
        part("Bld_01", 3.3, -.29, .35),
        part("Bld_07", 3.3, .29, .35),
    ]
    capital_city = capital_town + [
        part("Block_REC_002", 1.3, -.44, -.33),
        part("Block_REC_003", 1.3, .44, -.33),
        part("Bld_02", 3.3, -.55, .18),
        part("Bld_01", 3.3, .53, .18),
    ]
    capital_metro = capital_city + [
        part("Block_SQ_003", 1.6, -.65, -.06),
        part("Block_SQ_001", 1.6, .65, -.06),
        part("Bld_04", 3.3, -.53, .49),
        part("Bld_05", 3.3, .53, .49),
    ]
    layouts = json.loads((ROOT / "Renderer/lab/studies/cities/layouts.json").read_text())
    design = next(d for d in layouts["designs"] if (d["culture"], d["era"]) == (2, 1))
    civic = part("Block_LG_SQ_001", 2.25, 0, -.13)
    capital_civic = part("Block_LG_SQ_001", 2.6, 0, -.20)
    palace = dict(design["palace"])
    if args.flat_palace_pack:
        palace["pack"] = args.flat_palace_pack.as_posix()
    palace["rotation"] = math.pi / 6
    palace["offset"] = [0, .26]
    tiers = []
    for houses, capital_houses in zip((town, city, metro),
                                      (capital_town, capital_city, capital_metro)):
        tiers.append({"houses": houses, "capital_houses": capital_houses,
                      "base_centerpiece": civic, "capital_centerpiece": capital_civic,
                      "palace": palace})
    if args.farm_tree_pack:
        tree_pack = args.farm_tree_pack.as_posix()
        tree_asset = "city/prop/source_farm_tree"
        tree_body = component(tree_asset, args.farm_tree_pack)
        # Search a short, curated list of planting sites rather than allowing
        # props to obscure a door, the civic roof, or the enclosing wall.
        sites = [(-.36, -.37), (.36, -.37), (-.39, .06), (.39, .06),
                 (-.52, -.12), (.52, -.12), (-.32, .52), (.32, .52),
                 (-.23, -.52), (.23, -.52), (0, -.45),
                 (-.55, -.39), (.55, -.39), (-.57, .27), (.57, .27), (0, .61),
                 (-.68, -.3), (.68, -.3), (-.69, .44), (.69, .44)]
        for size, tier in enumerate(tiers):
            for capital in (False, True):
                buildings = ((tier["capital_houses"] + [capital_civic, palace]) if capital else
                             (tier["houses"] + [civic]))
                occupied = [footprint({"low": component(i["asset"], Path(i["pack"]))["lo"],
                                       "high": component(i["asset"], Path(i["pack"]))["hi"]}, i)
                            for i in buildings]
                planted = []
                for x, y in sites:
                    prop = {"asset": tree_asset, "pack": tree_pack, "scale": 2.35,
                            "rotation": 0.0, "offset": [x, y], "surface": False}
                    box = footprint({"low": tree_body["lo"], "high": tree_body["hi"]}, prop)
                    radius = (.46, .62, .74)[size]
                    if any((abs(px) / radius) ** 6 + (abs(py) / radius) ** 6 > 1
                           for px in (box[0], box[2]) for py in (box[1], box[3])):
                        continue
                    if any(overlaps(box, previous) for previous in occupied):
                        continue
                    planted.append(prop)
                    occupied.append(box)
                    if len(planted) == (2, 4, 6)[size]:
                        break
                tier["capital_decorations" if capital else "decorations"] = planted
        for tier in tiers:
            tier["ground_cover"] = {
                "texture": str(args.flat_pack / "textures/compound/base_color_07a9d1cc9983722e.dds"),
                "normal_0": str(args.flat_pack / "textures/compound/normal_0_1afe4528aa33c1fd.dds"),
                "gloss": str(args.flat_pack / "textures/compound/gloss_0c8f690bcd7c93b1.dds"),
                "period": [.32, .32], "atlas_uv": [0, 0, 1, 1],
                "margin": .105, "feather": .045, "lane_half_width": .065,
            }
    design.update(population_counts=[len(town), len(city), len(metro)],
                  tier_designs=tiers, houses=metro, base_centerpiece=civic,
                  palace=palace, capital_replaces_centerpiece=False)
    for size, tier in enumerate(tiers):
        for capital in (False, True):
            instances = ((tier["capital_houses"] + [capital_civic, palace]) if capital else
                         (tier["houses"] + [civic]))
            boxes = []
            for inst in instances:
                body = component(inst["asset"], Path(inst["pack"]))
                box = footprint({"low": body["lo"], "high": body["hi"]}, inst)
                for prior, prior_box in boxes:
                    if overlaps(box, prior_box):
                        raise ValueError(f"overlapping city bodies: {size} {capital} "
                                         f"{inst['asset']} {prior['asset']}")
                radius = (.46, .62, .74)[size]
                if any((abs(x) / radius) ** 6 + (abs(y) / radius) ** 6 > 1
                       for x in (box[0], box[2]) for y in (box[1], box[3])):
                    raise ValueError(f"city body lies outside wall clearance: {size} "
                                     f"{capital} {inst['asset']}")
                if size == 0 and any(abs(value) > .5 for value in box):
                    raise ValueError(f"town body lies outside tile: {capital} {inst['asset']}")
                boxes.append((inst, box))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layouts, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
