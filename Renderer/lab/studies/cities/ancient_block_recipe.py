#!/usr/bin/env python3
"""Compose dense, grounded ancient city auditions from complete source blocks."""

import argparse
import json
import math
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.build_layouts import STYLES, footprint, inside_wall
from Renderer.lab.studies.cities.ancient_trees import plant


ROOT = Path(__file__).resolve().parents[4]
# Culture selection is an offline art decision. Runtime sees the generic pack.
PROFILES = {
    "american": dict(prefix="DIS_CTY_AW_", family="A", core=2, flank=(1, 2),
                     landmark="Bld_A_04", infill="Bld_A_07", detail="Bld_A_09",
                     detail_scale=2.3, palace_scale=14.0),
    "european": dict(prefix="DIS_CTY_AE_", family="A", core=2, flank=(1, 2),
                     landmark="Bld_Tower_A_01", infill="Bld_SM_A_03",
                     detail="Bld_SM_A_01", detail_scale=2.3, palace_scale=14.0),
    "mediterranean": dict(prefix="DIS_CTY_AB_", family="", core=1, flank=(1, 3),
                          landmark="Bld_13", infill="Bld_06", infill_scale=1.7,
                          detail="Bld_10", detail_scale=2.0, palace_scale=15.0),
    "middle_eastern": dict(prefix="DIS_CTY_AE_", family="B", core=3, flank=(2, 3),
                           landmark="Bld_Tower_B_01", infill="Bld_SM_B_03",
                           detail="Bld_SM_B_01", detail_scale=2.3, palace_scale=14.0),
    "asian": dict(prefix="DIS_CTY_AW_", family="B", core=3, flank=(2, 3),
                  landmark="Bld_B_04", infill="Bld_B_09", detail="Bld_B_02",
                  detail_scale=2.3, palace_scale=14.0),
    "mapuche": dict(prefix="DIS_CTY_RMAP_", core=2, flank=(1, 2),
                    landmark="Bld_Tower_A_01", infill="Bld_SM_A_03",
                    detail="Bld_XSM_A_01", detail_scale=2.3,
                    palace_scale=14.0),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--culture", choices=tuple(STYLES[index].lower().replace(" ", "_")
                                                   for index in range(5)), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES),
                        help="Optional offline source-family art profile")
    parser.add_argument("--source-family", help="Select one source family from a multi-family report")
    parser.add_argument("--palace-asset", help="Matching normalized palace for this art profile")
    parser.add_argument("--farm-tree-pack", type=Path)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--flat-pack", type=Path, required=True)
    parser.add_argument("--flat-palace-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = PROFILES[args.profile or args.culture]
    report = json.loads(args.source_report.read_text())
    matches = [pool for pool in report["pools"]
               if (pool["source_culture"] == args.source_family if args.source_family
                   else pool["pool"] == f"city/pool/{args.culture}/ancient")]
    if len(matches) != 1:
        raise ValueError("source pool does not match the selected culture")
    pool = matches[0]
    by_name = {entry["entry"].removeprefix(profile["prefix"]): entry["asset_id"]
               for entry in pool["selected"]}
    pack = args.flat_pack.as_posix()

    def part(name, scale, x, y):
        return {"asset": by_name[name], "pack": pack, "scale": scale,
                "rotation": 0.0, "offset": [x, y]}

    def block(kind, number, scale, x, y):
        return part(f"Block_{kind}_{number:02d}", scale, x, y)

    # The town has a large complete core and two roof groups. Later tiers add
    # pieces at their authored size rather than growing the existing houses.
    civic = block("LG_SQ", profile["core"], 2.4, 0, .11)
    town = [block("SQ", profile["flank"][0], 2.1, -.25, -.28),
            block("SQ", profile["flank"][1], 2.1, .25, -.28),
            part(profile["landmark"], 3.7, 0, -.29),
            part(profile["infill"], profile.get("infill_scale", 2.3), -.32, .00),
            part(profile["infill"], profile.get("infill_scale", 2.3), .32, .00)]
    city = town + [block("SQ", 3, 1.7, -.46, .26),
                   block("SQ", 3, 1.7, .46, .26),
                   block("SQ", 1, 1.6, -.46, -.26),
                   block("SQ", 1, 1.6, .46, -.26),
                   block("REC", 2, 1.3, 0, .47),
                   block("REC", 3, 1.2, 0, -.50)]
    metro = city + [block("WR", 2, .7, -.52, -.50),
                    block("WR", 3, .7, .52, -.50),
                    block("SQ", 2, 1.3, -.34, .53),
                    block("SQ", 2, 1.3, .34, .53)]

    layouts = json.loads((ROOT / "Renderer/lab/studies/cities/layouts.json").read_text())
    culture = tuple(s.lower().replace(" ", "_") for s in STYLES).index(args.culture)
    design = next(d for d in layouts["designs"] if (d["culture"], d["era"]) == (culture, 0))
    palace = dict(design["palace"])
    if args.palace_asset:
        palace["asset"] = args.palace_asset
    palace.update(pack=args.flat_palace_pack.as_posix(),
                  scale=profile["palace_scale"], rotation=math.pi / 6,
                  offset=[0, .09])

    city_extension = city[5:]
    metro_extension = metro[11:]

    def fill(houses, targets, size, reserved=()):
        result = list(houses)
        for tx, ty in targets:
            choices = [(round(tx + dx * .02, 3), round(ty + dy * .02, 3))
                       for dx in range(-5, 6) for dy in range(-5, 6)]
            choices.sort(key=lambda xy: ((xy[0] - tx) ** 2 + (xy[1] - ty) ** 2,
                                         xy[0], xy[1]))
            for x, y in choices:
                inst = part(profile["detail"], profile["detail_scale"], x, y)
                body = component(inst["asset"], Path(inst["pack"]))
                box = footprint({"low": body["lo"], "high": body["hi"]}, inst)
                if not inside_wall(box, size, clearance=.02):
                    continue
                if size == 0 and any(abs(value) > .5 for value in box):
                    continue
                occupied = []
                for prior in result + list(reserved) + [civic, palace]:
                    prior_body = component(prior["asset"], Path(prior["pack"]))
                    occupied.append(footprint({"low": prior_body["lo"],
                                               "high": prior_body["hi"]}, prior))
                if any(overlaps(box, previous) for previous in occupied):
                    continue
                result.append(inst)
                break
        return result

    town = fill(town, [(-.30, .24), (.30, .24)], 0,
                city_extension + metro_extension)
    city = fill(town + city_extension,
                [(-.26, .44), (.26, .44), (-.34, -.46), (.34, -.46),
                 (-.52, .02), (.52, .02)], 1, metro_extension)
    metro = fill(city + metro_extension,
                 [(-.50, .43), (.50, .43), (-.14, .61), (.14, .61),
                  (-.60, -.12), (.60, -.12)], 2)
    tiers = []
    for size, houses in enumerate((town, city, metro)):
        for capital, instances in ((False, houses + [civic]),
                                   (True, houses + [palace])):
            boxes = []
            for inst in instances:
                body = component(inst["asset"], Path(inst["pack"]))
                box = footprint({"low": body["lo"], "high": body["hi"]}, inst)
                if not inside_wall(box, size, clearance=.015):
                    raise ValueError(f"outside wall: {args.culture} {size} "
                                     f"{capital} {inst['asset']} {box}")
                if size == 0 and any(abs(value) > .5 for value in box):
                    raise ValueError(f"town outside tile: {args.culture} {inst['asset']}")
                for previous, prior in boxes:
                    if overlaps(box, prior):
                        raise ValueError(f"overlap: {args.culture} {size} {capital} "
                                         f"{inst['asset']} {previous['asset']}")
                boxes.append((inst, box))
        tiers.append({"houses": houses, "base_centerpiece": civic,
                      "palace": palace})
    if args.farm_tree_pack:
        plant(tiers, args.farm_tree_pack)
    design.update(grounding="terrain", slope_limit=64.0, vertical_metric=1.0,
                  population_counts=[len(town), len(city), len(metro)],
                  tier_designs=tiers, houses=metro, base_centerpiece=civic,
                  palace=palace, capital_replaces_centerpiece=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layouts, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
