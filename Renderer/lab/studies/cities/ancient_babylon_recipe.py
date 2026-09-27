#!/usr/bin/env python3
"""Audition a dense ancient Middle Eastern layout from single source houses."""

import argparse
import json
import math
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.build_layouts import footprint, inside_wall
from Renderer.lab.studies.cities.ancient_trees import plant
from Renderer.lab.studies.cities.ancient_capital_infill import fill


ROOT = Path(__file__).resolve().parents[4]
PALACE = "city/palace/root/c55a2797a1e95711"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--flat-pack", type=Path, required=True)
    parser.add_argument("--flat-palace-pack", type=Path, required=True)
    parser.add_argument("--farm-tree-pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.source_report.read_text())
    pool = next(pool for pool in report["pools"]
                if pool["source_culture"] == "CIVILIZATION_BABYLON_STK"
                and pool["source_art_era"] == "ARTERA_ANCIENT")
    names = {entry["entry"].removeprefix("DIS_CTY_RBAB_"): entry["asset_id"]
             for entry in pool["selected"]}

    def part(name, scale, x, y):
        return {"asset": names[name], "pack": args.flat_pack.as_posix(),
                "scale": scale, "rotation": 0.0, "offset": [x, y]}

    civic = part("Bld_25", 6.0, 0, .11)
    palace = {"asset": PALACE, "pack": args.flat_palace_pack.as_posix(),
              "scale": 12.0, "rotation": math.pi/6, "offset": [0, .11]}

    def box(instance):
        body = component(instance["asset"], Path(instance["pack"]))
        return footprint({"low": body["lo"], "high": body["hi"]}, instance)

    tiers = []
    houses = []
    requests = [
        [("Bld_21", 5.5, -.28, -.30), ("Bld_25", 5.4, .28, -.30),
         ("Bld_01", 3.8, -.33, .16), ("Bld_17", 3.8, .33, .16),
         ("Bld_11", 4.8, -.18, .34), ("Bld_12", 4.8, .18, .34)],
        [("Bld_11", 5.0, -.46, .30), ("Bld_12", 5.0, .46, .30),
         ("Bld_09", 5.5, 0, -.50), ("Bld_10", 5.5, 0, .50),
         ("Bld_01", 4.0, -.53, -.12), ("Bld_17", 4.0, .53, -.12),
         ("Bld_09", 4.0, -.18, -.48), ("Bld_10", 4.0, .18, -.48),
         ("Bld_09", 4.2, -.27, -.06), ("Bld_10", 4.2, .27, -.06),
         ("Bld_11", 4.0, -.20, .30), ("Bld_12", 4.0, .20, .30)],
        [("Bld_23", 4.7, -.58, -.43), ("Bld_24", 4.7, .58, -.43),
         ("Bld_21", 4.7, -.58, .40), ("Bld_25", 4.7, .58, .40),
         ("Bld_17", 3.7, -.29, .59), ("Bld_01", 3.7, .29, .59),
         ("Bld_09", 4.2, -.14, .55), ("Bld_10", 4.2, .14, .55)],
    ]
    reserved = [box(civic), box(palace)]
    for size, additions in enumerate(requests):
        for name, scale, tx, ty in additions:
            offsets = [(dx*.025, dy*.025) for dx in range(-12, 13)
                       for dy in range(-12, 13)]
            offsets.sort(key=lambda item: (item[0]**2+item[1]**2,
                                           abs(item[0]), abs(item[1])))
            for dx, dy in offsets:
                trial = part(name, scale, round(tx+dx, 3), round(ty+dy, 3))
                candidate = box(trial)
                if not inside_wall(candidate, size, clearance=.015):
                    continue
                if size == 0 and any(abs(value) > .5 for value in candidate):
                    continue
                if any(overlaps(candidate, previous) for previous in reserved):
                    continue
                houses.append(trial)
                reserved.append(candidate)
                break
            else:
                raise ValueError(f"No legal position for {name} in tier {size}")
        tiers.append({"houses": list(houses), "base_centerpiece": civic,
                      "palace": palace})
    fill(tiers, "Bld_09", 4.2, (1, 1, 1), part, box)
    if args.farm_tree_pack:
        plant(tiers, args.farm_tree_pack)
    layouts = json.loads((ROOT / "Renderer/lab/studies/cities/layouts.json").read_text())
    design = next(d for d in layouts["designs"] if (d["culture"], d["era"]) == (3, 0))
    design.update(grounding="terrain", slope_limit=64.0,
                  vertical_metric=.95,
                  population_counts=[len(tier["houses"]) for tier in tiers],
                  tier_designs=tiers, houses=houses, base_centerpiece=civic,
                  palace=palace, capital_replaces_centerpiece=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layouts, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
