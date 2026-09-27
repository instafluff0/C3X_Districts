#!/usr/bin/env python3
"""Audition complete ancient blocks against the medieval Roman visual scale."""

import argparse
import json
import math
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.ancient_capital_infill import fill
from Renderer.lab.studies.cities.ancient_trees import plant
from Renderer.lab.studies.cities.build_layouts import STYLES, footprint, inside_wall


ROOT = Path(__file__).resolve().parents[4]
PROFILES = {
    "american": dict(prefix="DIS_CTY_RMAP_", core=2, core_scale=2.5,
                     palace_scale=14.0, landmark="Bld_MD_A_01",
                     infill="Bld_SM_A_02", gap_infill="Bld_SM_A_03"),
    "european": dict(prefix="DIS_CTY_AE_", core=2, core_scale=2.5,
                     palace_scale=14.0, landmark="Bld_MD_A_01",
                     infill="Bld_SM_A_02", capital_infill_scale=2.0),
    "mediterranean": dict(prefix="DIS_CTY_AB_", core=1, core_scale=2.8,
                          palace_scale=15.0, landmark="Bld_12",
                          infill="Bld_10", gap_infill="Bld_03"),
    "asian": dict(prefix="DIS_CTY_AW_", core=3, core_scale=2.7,
                  palace_scale=14.0, landmark="Bld_A_04",
                  infill="Bld_A_03", vertical_metric=.86),
    "cree": dict(prefix="DIS_CTY_CREE_", core=2, core_scale=2.4,
                 palace_scale=14.0, landmark="Bld_J", infill="Bld_E",
                 gap_infill="Bld_B"),
}
CAPITAL_INFILL_COUNTS = {
    "american": (1, 2, 2),
    "european": (1, 1, 1),
    "mediterranean": (3, 3, 3),
    "asian": (3, 3, 3),
    "cree": (1, 2, 2),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--culture", choices=tuple(s.lower().replace(" ", "_")
                                                    for s in STYLES), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-family")
    parser.add_argument("--flat-pack", type=Path, required=True)
    parser.add_argument("--flat-palace-pack", type=Path, required=True)
    parser.add_argument("--palace-asset")
    parser.add_argument("--farm-tree-pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile_name = args.profile or args.culture
    profile = PROFILES[profile_name]
    report = json.loads(args.source_report.read_text())
    pools = [pool for pool in report["pools"]
             if (pool["source_culture"] == args.source_family if args.source_family
                 else pool["pool"] == f"city/pool/{args.culture}/ancient")]
    if len(pools) != 1:
        raise ValueError("ambiguous ancient source family")
    assets = {entry["entry"].removeprefix(profile["prefix"]): entry["asset_id"]
              for entry in pools[0]["selected"]}

    def part(name, scale, x, y):
        return {"asset": assets[name], "pack": args.flat_pack.as_posix(),
                "scale": scale, "rotation": 0.0, "offset": [x, y]}

    def box(item):
        body = component(item["asset"], Path(item["pack"]))
        return footprint({"low": body["lo"], "high": body["hi"]}, item)

    layouts = json.loads((ROOT / "Renderer/lab/studies/cities/layouts.json").read_text())
    culture = tuple(s.lower().replace(" ", "_") for s in STYLES).index(args.culture)
    design = next(d for d in layouts["designs"] if (d["culture"], d["era"]) == (culture, 0))
    civic = part(f"Block_LG_SQ_{profile['core']:02d}", profile["core_scale"], 0, .11)
    palace = dict(design["palace"])
    if args.palace_asset:
        palace["asset"] = args.palace_asset
    palace.update(pack=args.flat_palace_pack.as_posix(),
                  scale=profile["palace_scale"], rotation=math.pi/6, offset=[0, .11])

    # Complete source blocks contain several full-size houses and preserve
    # their authored spacing and orientation. Growth adds blocks without
    # rescaling earlier roofs. These targets follow the medieval Roman recipe.
    requests = [
        [("Block_SQ_01", 2.55, -.24, -.27),
         ("Block_SQ_02", 2.55, .24, -.27),
         (profile["landmark"], 2.6, 0, -.31),
         (profile["infill"], 2.2, -.34, .08),
         (profile["infill"], 2.2, .34, .08)],
        [("Block_SQ_03", 2.55, -.43, .22),
         ("Block_SQ_01", 2.55, .43, .22),
         ("Block_SQ_02", 2.45, -.43, -.24),
         ("Block_SQ_03", 2.45, .43, -.24),
         ("Block_SQ_01", 2.4, 0, -.43),
         ("Block_SQ_02", 2.4, 0, .45),
         (profile["infill"], 2.3, -.52, .02),
         (profile["infill"], 2.3, .52, .02),
         (profile.get("gap_infill", profile["infill"]), 2.2, -.26, -.05),
         (profile.get("gap_infill", profile["infill"]), 2.2, .26, -.05),
         (profile.get("gap_infill", profile["infill"]), 2.2, -.20, .31),
         (profile.get("gap_infill", profile["infill"]), 2.2, .20, .31)],
        [("Block_SQ_02", 2.35, -.54, -.42),
         ("Block_SQ_01", 2.35, .54, -.42),
         ("Block_SQ_03", 2.35, -.54, .40),
         ("Block_SQ_02", 2.35, .54, .40),
         (profile.get("gap_infill", profile["infill"]), 2.2, .43, .25)],
    ]
    houses = []
    occupied = [box(civic), box(palace)]
    tiers = []
    for size, additions in enumerate(requests):
        for name, scale, tx, ty in additions:
            offsets = [(dx*.025, dy*.025) for dx in range(-10, 11)
                       for dy in range(-10, 11)]
            offsets.sort(key=lambda item: (item[0]**2+item[1]**2,
                                           abs(item[0]), abs(item[1])))
            candidates = (scale, *[trial for trial in (2.4, 2.25, 2.1, 1.9,
                                                        1.7, 1.5, 1.3)
                                   if trial < scale])
            for trial_scale in candidates:
                for dx, dy in offsets:
                    item = part(name, trial_scale, round(tx+dx, 3),
                                round(ty+dy, 3))
                    bounds = box(item)
                    if not inside_wall(bounds, size, clearance=.015):
                        continue
                    if size == 0 and any(abs(value) > .5 for value in bounds):
                        continue
                    if any(overlaps(bounds, prior) for prior in occupied):
                        continue
                    houses.append(item)
                    occupied.append(bounds)
                    break
                else:
                    continue
                break
            else:
                raise ValueError(f"{args.culture} tier {size}: no plot for {name}")
        tiers.append({"houses": list(houses), "base_centerpiece": civic,
                      "palace": palace})
    fill(tiers, profile.get("gap_infill", profile["infill"]),
         profile.get("capital_infill_scale", 2.2),
         CAPITAL_INFILL_COUNTS[profile_name], part, box)
    if args.farm_tree_pack:
        plant(tiers, args.farm_tree_pack)
    design.update(grounding="terrain", slope_limit=64.0,
                  vertical_metric=profile.get("vertical_metric", .648266978876),
                  population_counts=[len(tier["houses"]) for tier in tiers],
                  tier_designs=tiers, houses=houses, base_centerpiece=civic,
                  palace=palace, capital_replaces_centerpiece=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layouts, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
