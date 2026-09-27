#!/usr/bin/env python3
"""Compose grounded medieval city candidates from each culture's complete blocks."""

import argparse
import json
import math
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.build_layouts import STYLES, footprint, inside_wall
from Renderer.lab.studies.cities.medieval_density import fill
from Renderer.lab.studies.cities.ancient_trees import plant


ROOT = Path(__file__).resolve().parents[4]
MEDIEVAL_TREE_SITES = (
    (-.35, -.10), (.35, -.10), (-.35, .15), (.35, .15),
    (-.40, -.50), (.40, -.50), (-.45, -.45), (.45, -.45),
    (-.45, .40), (.45, .40), (-.50, -.60), (.50, -.60),
    (-.50, .50), (.50, .50), (-.20, -.60), (.20, -.60),
    (-.20, .60), (.20, .60), (-.55, -.25), (.55, -.25),
    (-.55, .25), (.55, .25))
# These are offline art decisions. The runtime reads only the resulting pack.
PROFILES = {
    "european": dict(digits=2, core=2, core_scale=2.5, core_y=.12,
                     flank=(1, 1), flank_scale=2.3, landmark="Bld_MD_A_01",
                     landmark_scale=3.0, infill="Bld_MD_A_02", infill_scale=1.5,
                     city_group=(1, 1), city_group_scale=(2.5, 2.5),
                     rear_group=1, rear_scale=2.3,
                     metro=(2, 3), front_rec=1,
                     side_single="Bld_SM_A_01", side_scale=2.5, side_x=.46,
                     palace_scale=18.0,
                     density=(('Bld_MD_A_03', 2.35), ('Bld_SM_A_01', 2.35),
                              ('Bld_SM_A_02', 2.25), ('Bld_MD_A_04', 2.35))),
    "asian": dict(digits=2, core=3, core_scale=2.4,
                  flank=(3, 3), flank_scale=2.3, landmark="Bld_03",
                  landmark_scale=3.0, infill="Bld_05", infill_scale=1.9,
                  city_group=(1, 3), city_group_scale=(2.5, 2.3),
                  rear_group=3, rear_scale=2.2,
                  metro=(2, 3), palace_scale=16.0,
                  density=(('Bld_06', 2.2), ('Bld_08', 1.8),
                           ('Bld_09', 2.0), ('Bld_11', 2.2))),
    "american": dict(digits=3, core=3, core_scale=2.15, core_y=.12,
                     flank=(1, 1), flank_scale=2.0, landmark="Bld_03",
                     landmark_scale=2.9, infill="Bld_09", infill_scale=1.6,
                     city_group=(1, 1), city_group_scale=(2.3, 2.3),
                     rear_group=1, rear_scale=2.2,
                     metro=(2, 3), palace_scale=16.0,
                     density=(('Bld_10', 2.2), ('Bld_11', 2.1),
                              ('Bld_12', 1.9), ('Bld_13', 1.8)),
                     density_counts=(0, 2, 4), capital_density_counts=(0, 2, 4)),
    "middle_eastern": dict(digits=3, core=2, core_scale=2.5,
                           flank=(2, 2), flank_scale=2.3, landmark="Bld03",
                           landmark_scale=2.5, infill="Bld02", infill_scale=1.8,
                           city_group=(2, 1), city_group_scale=(2.3, 2.6),
                           rear_group=2, rear_scale=2.2,
                           metro=(2, 3), palace_scale=18.0,
                           density=(('Bld04', 2.1), ('Bld05', 1.9),
                                    ('Bld06', 1.8), ('Bld07', 1.6))),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--culture", choices=tuple(PROFILES), required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--flat-pack", type=Path, required=True)
    parser.add_argument("--flat-palace-pack", type=Path, required=True)
    parser.add_argument("--farm-tree-pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = PROFILES[args.culture]
    report = json.loads(args.source_report.read_text())
    if report["pools"][0]["pool"] != f"city/pool/{args.culture}/medieval":
        raise ValueError("source pool does not match the selected culture")
    assets = {"Block_" + entry["entry"].split("_Block_", 1)[1].removeprefix("A_"):
              entry["asset_id"]
              for entry in report["pools"][0]["selected"]
              if "_Block_" in entry["entry"]}
    def source_asset(name):
        matches = [entry for entry in report["pools"][0]["selected"]
                   if entry["entry"].endswith("_" + name)]
        if len(matches) != 1:
            raise ValueError(f"medieval source asset is missing or ambiguous: {name}")
        return matches[0]["asset_id"]

    landmark_asset = source_asset(profile["landmark"])
    infill_asset = source_asset(profile["infill"])
    pack = args.flat_pack.as_posix()

    def block(group, number, scale, x, y):
        name = f"Block_{group}_{number:0{profile['digits']}d}"
        return {"asset": assets[name], "pack": pack, "scale": scale,
                "rotation": 0.0, "offset": [x, y]}

    def single(asset, scale, x, y):
        return {"asset": asset, "pack": pack, "scale": scale,
                "rotation": 0.0, "offset": [x, y]}

    def side(x):
        if "side_single" in profile:
            return single(source_asset(profile["side_single"]),
                          profile["side_scale"], x * profile["side_x"], -.20)
        return block("SQ", 3, 1.35, x * .45, -.20)

    civic = block("LG_SQ", profile["core"], profile["core_scale"],
                  0, profile.get("core_y", .10))
    town = [block("SQ", profile["flank"][0], profile["flank_scale"], -.24, -.26),
            block("SQ", profile["flank"][1], profile["flank_scale"], .24, -.26),
            single(landmark_asset, profile["landmark_scale"], 0, -.27),
            single(infill_asset, profile["infill_scale"], -profile.get("infill_x", .34), .02),
            single(infill_asset, profile["infill_scale"], profile.get("infill_x", .34), .02)]
    city = town + [block("SQ", profile["city_group"][0],
                         profile["city_group_scale"][0], -.42, .26),
                   block("SQ", profile["city_group"][1],
                         profile["city_group_scale"][1], .42, .26),
                   side(-1), side(1),
                   block("REC", 2, profile.get("rear_rec_scale", 1.4), 0, .46),
                   block("REC", profile.get("front_rec", 3),
                         profile.get("front_rec_scale", 1.2), 0, -.48)]
    metro = city + [block("WR", profile["metro"][0], 1.1, -.54, -.42),
                    block("WR", profile["metro"][1], 1.1, .54, -.42),
                    block("SQ", profile["rear_group"], profile["rear_scale"], -.32, .53),
                    block("SQ", profile["rear_group"], profile["rear_scale"], .32, .53)]

    layouts = json.loads((ROOT / "Renderer/lab/studies/cities/layouts.json").read_text())
    culture = tuple(s.lower().replace(" ", "_") for s in STYLES).index(args.culture)
    design = next(d for d in layouts["designs"] if (d["culture"], d["era"]) == (culture, 1))
    palace = dict(design["palace"])
    palace.update(pack=args.flat_palace_pack.as_posix(),
                  scale=profile["palace_scale"], rotation=math.pi / 6,
                  offset=[0, .10])
    def box(item):
        body = component(item["asset"], Path(item["pack"]))
        return footprint({"low": body["lo"], "high": body["hi"]}, item)

    palette = [(source_asset(name), scale) for name, scale in profile["density"]]
    tiers = []
    for size, original in enumerate((town, city, metro)):
        houses = fill(original, [civic], size,
                      profile.get("density_counts", (0, 5, 7))[size],
                      palette, single, box)
        capital_houses = fill(original, [palace], size,
                              profile.get("capital_density_counts", (0, 6, 8))[size],
                              palette, single, box)
        for capital, instances in ((False, houses + [civic]),
                                   (True, capital_houses + [palace])):
            boxes = []
            for inst in instances:
                body = component(inst["asset"], Path(inst["pack"]))
                bounds = footprint({"low": body["lo"], "high": body["hi"]}, inst)
                if not inside_wall(bounds, size, clearance=.015):
                    raise ValueError(f"outside wall clearance: {args.culture} {size} "
                                     f"{capital} {inst['asset']}")
                if size == 0 and any(abs(value) > .5 for value in bounds):
                    raise ValueError(f"town outside tile: {args.culture} {capital} "
                                     f"{inst['asset']}")
                for previous, prior in boxes:
                    if overlaps(bounds, prior):
                        raise ValueError(f"overlapping city bodies: {args.culture} {size} "
                                         f"{capital} {inst['asset']} {previous['asset']}")
                boxes.append((inst, bounds))
        tiers.append({"houses": houses, "capital_houses": capital_houses,
                      "base_centerpiece": civic,
                      "palace": palace})
    if args.farm_tree_pack:
        plant(tiers, args.farm_tree_pack, counts=(1, 3, 5),
              sites=MEDIEVAL_TREE_SITES, min_spacing=.16)
    design.update(grounding="terrain", slope_limit=64.0, vertical_metric=1.0,
                  population_counts=[len(tier["houses"]) for tier in tiers],
                  tier_designs=tiers, houses=tiers[-1]["houses"], base_centerpiece=civic,
                  palace=palace, capital_replaces_centerpiece=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layouts, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
