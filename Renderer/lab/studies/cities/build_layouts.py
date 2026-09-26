#!/usr/bin/env python3
"""Freeze flat city arrangements per culture, era and population for Lab review.

This reads normalized local art. It never changes the runtime city pack.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import expanded, gap, overlaps


ROOT = Path(__file__).resolve().parents[4]
STYLES = ("American", "European", "Mediterranean", "Middle Eastern", "Asian")
ERAS = ("Ancient", "Medieval", "Industrial", "Modern")
COUNTS_BY_ERA = ((4, 6, 8), (3, 5, 7), (2, 4, 6), (2, 3, 5))
HEIGHTS_BY_ERA = ((48, 48, 48, 48, 42, 42, 38, 38),
                  (65, 65, 65, 55, 55, 50, 50),
                  (80, 80, 70, 70, 60, 60),
                  (100, 100, 90, 80, 80))
ERA_TARGETS = (
    ((-.30, .30), (-.10, .34), (.10, .34), (.30, .30),
     (-.39, .02), (.39, .02), (-.39, -.22), (.39, -.22)),
    ((-.28, .31), (0, .35), (.28, .31),
     (-.39, .02), (.39, .02), (-.39, -.22), (.39, -.22)),
    ((-.23, .32), (.23, .32), (-.39, .02), (.39, .02),
     (-.39, -.22), (.39, -.22)),
    ((-.23, .33), (.23, .33), (0, .38), (-.41, -.04), (.41, -.04)),
)
CULTURE_SHIFT = ((-.025, -.015), (.0, -.035), (.025, .015),
                 (-.015, .025), (.015, .0))
WALL_SCALE = 2.3
WALL_RADII = (.43, .58, .70)
WALL_SEGMENTS = (16, 20, 24)
WALL_ROUNDNESS = 6
TIER_SCALES = (1.0, 1.22, 1.42)
# The capital palace occupies the civic center's plot, so its entry stays visible.
PALACE_STYLES = (
    ("SouthAmerican", "SouthAmerican", "America", "America"),
    ("AncientEarth", "DEFAULT", "DEFAULT", "DEFAULT"),
    ("AncientBrick", "Mediterranean", "Mediterranean", "Mediterranean"),
    ("AncientEarth", "Mughal", "Mughal", "Mughal"),
    ("AncientWood", "EastAsian", "EastAsian", "EastAsian"),
)


def wall_curve(size: int, count: int) -> list[tuple[float, float, float]]:
    """Equally spaced positions/tangents on a low rounded-square perimeter."""
    radius = WALL_RADII[size]
    steps = 2048
    points = []
    distances = [0.0]
    for index in range(steps + 1):
        angle = math.pi/4 + index*math.tau/steps
        c, s = math.cos(angle), math.sin(angle)
        point = (radius*math.copysign(abs(c)**(2/WALL_ROUNDNESS), c),
                 radius*math.copysign(abs(s)**(2/WALL_ROUNDNESS), s))
        if points:
            distances.append(distances[-1]+math.dist(points[-1], point))
        points.append(point)
    result = []
    for slot in range(count):
        target = distances[-1]*slot/count
        index = next(i for i, distance in enumerate(distances) if distance >= target)
        if index == 0:
            x, y = points[0]
        else:
            blend = ((target-distances[index-1])/
                     (distances[index]-distances[index-1]))
            x = points[index-1][0]*(1-blend)+points[index][0]*blend
            y = points[index-1][1]*(1-blend)+points[index][1]*blend
        before = points[(index-1) % steps]
        after = points[min(index+1, steps)]
        tangent = math.atan2(after[1]-before[1], after[0]-before[0])
        result.append((x, y, tangent-math.pi/2))
    return result


def wall_instances(kit: str, size: int = 0) -> list[dict]:
    pack = "Renderer/packs/CityAdjunctsNormalized"
    catalog = json.loads((ROOT / pack / "city_adjunct_catalog.json").read_text())
    parts = catalog["walls"]["kits"][kit]
    segment, gate = (parts[role][0] for role in ("segment", "gate"))
    tower = next((asset for asset in parts["tower"] if asset.endswith("tower_small")),
                 parts["tower"][0])
    result = []
    ring = wall_curve(size, WALL_SEGMENTS[size])
    for sector, (x, y, rotation) in enumerate(ring):
        result.append({"asset": gate if sector == 0 else segment, "pack": pack,
                       "scale": WALL_SCALE,
                       "rotation": rotation+math.pi/2 if sector == 0 else rotation,
                       "offset": [x, y]})
    for sector in range(2, WALL_SEGMENTS[size], 4):
        x, y, rotation = ring[sector]
        result.append({"asset": tower, "pack": pack, "scale": 2.0,
                       "rotation": rotation, "offset": [x, y]})
    return result


def bounds(model: dict, scale: float, rotation: float) -> list[float]:
    low, high = model["low"], model["high"]
    c, s = math.cos(rotation), math.sin(rotation)
    dx, dy = ((high[i]-low[i])/2 for i in (0, 1))
    corners = [(x*c-y*s, x*s+y*c) for x in (-dx, dx) for y in (-dy, dy)]
    return [min(x for x, _ in corners)*scale, min(y for _, y in corners)*scale,
            max(x for x, _ in corners)*scale, max(y for _, y in corners)*scale]


def candidate_positions(target: tuple[float, float]):
    values = [round(i*.035, 4) for i in range(-22, 23)]
    return sorted(((x, y) for x in values for y in values
                   if (x-target[0])**2+(y-target[1])**2 <= .25**2),
                  key=lambda xy: ((xy[0]-target[0])**2+(xy[1]-target[1])**2,
                                  abs(xy[0]), abs(xy[1]), xy))


def footprint(model: dict, instance: dict) -> list[float]:
    rotation = instance["rotation"]
    if abs(rotation) > 1e-9:
        # A palace's already-rotated source AABB is not its footprint. Rotating
        # those AABB corners again invents an oversized square and pushes legal
        # buildings out of the tile. Use the actual transformed source points.
        source = component(instance["asset"], Path(instance["pack"]))
        center = [(source["lo"][axis]+source["hi"][axis])/2 for axis in (0, 1)]
        c, s = math.cos(rotation), math.sin(rotation)
        points = [(instance["scale"]*((vertex["position"][0]-center[0])*c-
                                       (vertex["position"][1]-center[1])*s),
                   instance["scale"]*((vertex["position"][0]-center[0])*s+
                                       (vertex["position"][1]-center[1])*c))
                  for mesh, material in source["parts"] if material["alpha_mode"] != "blend"
                  for vertex in mesh["vertices"]]
        b = [min(x for x, _ in points), min(y for _, y in points),
             max(x for x, _ in points), max(y for _, y in points)]
    else:
        b = bounds(model, instance["scale"], rotation)
    x, y = instance["offset"]
    return expanded([b[0]+x, b[1]+y, b[2]+x, b[3]+y], .004)


def inside_wall(box: list[float], size: int, clearance: float = .025) -> bool:
    radius = WALL_RADII[size]-clearance
    return all((abs(x)/radius)**WALL_ROUNDNESS +
               (abs(y)/radius)**WALL_ROUNDNESS <= 1
               for x in (box[0], box[2]) for y in (box[1], box[3]))


def design_instances(models: list[dict], cores: list[list[float]], factor: float,
                     targets: tuple[tuple[float, float], ...],
                     heights: tuple[int, ...], counts: tuple[int, int, int],
                     forced_size: int | None = None,
                     fixed_scales: tuple[float, ...] | None = None) -> list[dict]:
    chosen: list[dict] = []
    for slot, model in enumerate(models):
        ranked = []
        size = (forced_size if forced_size is not None else
                0 if slot < counts[0] else 1 if slot < counts[1] else 2)
        span = [model["high"][axis]-model["low"][axis] for axis in range(3)]
        target_scale = heights[slot]/(150*span[2])
        scale = (fixed_scales[slot] if fixed_scales is not None else
                 min(target_scale, (.22, .30, .36)[size]/max(span[:2])))*factor
        # Keep every authored facade in the same isometric viewing direction.
        # Moving a body is preferable to turning it across the street grid.
        for rotation in (0.0,):
            local = bounds(model, scale, rotation)
            for x, y in candidate_positions(targets[slot]):
                box = expanded([local[0]+x, local[1]+y, local[2]+x, local[3]+y], .006)
                if (not inside_wall(box, size) or
                    any(overlaps(box, core) for core in cores)):
                    continue
                if any(overlaps(box, item["box"]) for item in chosen):
                    continue
                if min(gap(box, b) for b in cores+[item["box"] for item in chosen]) > .40:
                    continue
                ranked.append(((x-targets[slot][0])**2+(y-targets[slot][1])**2,
                               x, y, box, rotation))
        if not ranked:
            raise ValueError(f"no wall-contained arrangement for {model['asset']} slot {slot}")
        _, x, y, box, rotation = min(ranked)
        chosen.append({"asset": model["asset"], "pack": model["pack"],
                       "scale": round(scale, 8), "rotation": round(rotation, 8),
                       "offset": [x, y], "box": box})
    return [{k: v for k, v in item.items() if k != "box"} for item in chosen]


def build() -> dict:
    catalog = json.loads((ROOT/"Renderer/packs/CityStudyAuxiliaryUV/city_catalog.json").read_text())
    selected = json.loads((ROOT/"Renderer/packs/CityFidelitySources/current/recipes.json").read_text())
    palace_catalog = json.loads((ROOT/"Renderer/packs/CityPalacesNormalized/palace_catalog.json").read_text())
    palace_ids = {}
    for record in palace_catalog["palaces"]:
        for selector in record["source_selectors"]:
            if selector["culture_root"] == "Culture":
                palace_ids[(selector["culture"], selector["era"])] = record["asset_id"]
    designs = []
    for culture in range(5):
        for era in range(4):
            pool = "city/pool/"+STYLES[culture].lower().replace(" ", "_")+"/"+ERAS[era].lower()
            block_ids = set(selected["blocks_by_pool"][pool])
            models = []
            for asset in catalog["pools"][pool]["components"]:
                if asset in block_ids:
                    continue
                body = component(asset, Path("Renderer/packs/CityStudyAuxiliaryUV"))
                models.append({"asset": asset, "pack": "Renderer/packs/CityStudyAuxiliaryUV",
                               "low": body["lo"], "high": body["hi"]})
            # Prefer complete street-facing bodies over the old height-sorted
            # list, whose earliest industrial entries were narrow facade slices.
            models.sort(key=lambda m: (-(m["high"][0]-m["low"][0])*
                                       (m["high"][1]-m["low"][1]), m["asset"]))
            center_model = models[0]
            # Tall, recognizable facades survive Civ III zoom better than
            # oversized low roofs. Keep one broad civic body, then choose
            # authored street bodies with useful height-to-footprint ratios.
            house_models = sorted(models[1:], key=lambda m: (
                max(m["high"][0]-m["low"][0], m["high"][1]-m["low"][1]) /
                (m["high"][2]-m["low"][2]), m["asset"]))[:COUNTS_BY_ERA[era][2]]
            if len(house_models) != COUNTS_BY_ERA[era][2]:
                raise ValueError(f"insufficient city bodies for {STYLES[culture]} {ERAS[era]}")
            base_centerpiece = {"asset": center_model["asset"], "pack": center_model["pack"],
                                "scale": round(min(55/(150*(center_model["high"][2]-center_model["low"][2])),
                                                   .44/max(center_model["high"][axis]-center_model["low"][axis]
                                                           for axis in (0, 1))), 8),
                                "rotation": 0.0, "offset": [0, -.16]}
            selector = PALACE_STYLES[culture][era]
            palace_id = palace_ids.get((selector, "ARTERA_ANCIENT")) or palace_ids[(selector, "DEFAULT")]
            palace = component(palace_id, Path("Renderer/packs/CityPalacesNormalized"))
            palace_model = {"low": palace["lo"], "high": palace["hi"]}
            palace_scale = min((72+era*7)/(150*(palace["hi"][2]-palace["lo"][2])),
                               .53/max(palace["hi"][axis]-palace["lo"][axis]
                                       for axis in (0, 1)))
            # Source palace platforms share a 30-degree offset from the city
            # grid. Restore tile-edge alignment before placing the houses.
            palace_instance = {"asset": palace_id,
                               "pack": "Renderer/packs/CityPalacesNormalized",
                               "scale": round(palace_scale, 8), "rotation": math.pi/6,
                               "offset": [0, -.16]}
            cores = [footprint(center_model, base_centerpiece),
                     footprint(palace_model, palace_instance)]
            # The ancient Asian proof is composed for a Civ III tile at game
            # zoom: a broad palace/civic rear court and an identifiable front
            # row, with later houses growing along the sides. The capital body
            # takes the civic body's place so neither is hidden behind it.
            replaces_centerpiece = True
            if culture == 4 and era == 0:
                # The original representative intake stopped at the darker A
                # houses. The authored B family uses the golden thatch atlas
                # region visible in the supplied Japanese city reference.
                audition_pack = "Renderer/packs/CityAncientWoodCandidates"
                house_ids = ("city/component/6a24fa4c3eae7164",  # B_05
                             "city/component/c4f54876637029c8",  # B_02
                             "city/component/6a24fa4c3eae7164",  # B_05
                             "city/component/6a24fa4c3eae7164",  # B_05
                             "city/component/2fe091000eb20825",  # B_10
                             "city/component/b343ea82f6041dfe",  # B_06
                             "city/component/fa7ad2bdac50f63c",  # B_07
                             "city/component/fc05d09b408d3779") # B_09
                house_models = []
                for asset in house_ids:
                    body = component(asset, Path(audition_pack))
                    house_models.append({"asset": asset, "pack": audition_pack,
                                         "low": body["lo"], "high": body["hi"]})
                center_asset = "city/component/fc05d09b408d3779" # B_09
                center_body = component(center_asset, Path(audition_pack))
                base_centerpiece = {"asset": center_asset, "pack": audition_pack,
                                    "scale": round(52/
                                                   (150*(center_body["hi"][2]-center_body["lo"][2])), 8),
                                    "rotation": 0.0, "offset": [0, -.17]}
                # Its authored platform is 30 degrees off the city grid. Turn
                # the intact body so its edges run parallel to the tile edges
                # and the zero-rotation B houses' street grid.
                palace_instance = {**palace_instance, "scale": 14.0,
                                   "rotation": math.pi/6, "offset": [0, -.16]}
                cores = [footprint({"low": center_body["lo"], "high": center_body["hi"]},
                                   base_centerpiece), footprint(palace_model, palace_instance)]
            shift_x, shift_y = CULTURE_SHIFT[culture]
            targets = tuple((x+shift_x, y+shift_y) for x, y in ERA_TARGETS[era])
            desired_heights = tuple(height*TIER_SCALES[2] for height in HEIGHTS_BY_ERA[era])
            fixed_scales = []
            for slot, model in enumerate(house_models):
                first_size = (0 if slot < COUNTS_BY_ERA[era][0] else
                              1 if slot < COUNTS_BY_ERA[era][1] else 2)
                span = [model["high"][axis]-model["low"][axis] for axis in range(3)]
                fixed_scales.append(min(desired_heights[slot]/(150*span[2]),
                                        (.28, .31, .36)[first_size]/max(span[:2])))
            for factor in (1.0, .94, .88, .82, .76, .70, .64, .58):
                try:
                    tier_designs = []
                    for size, count in enumerate(COUNTS_BY_ERA[era]):
                        spread = 1+(TIER_SCALES[size]-1)*.5
                        tier_targets = tuple((x*spread, y*spread)
                                             for x, y in targets[:count])
                        tier_houses = design_instances(
                            house_models[:count], cores, factor, tier_targets,
                            desired_heights[:count], COUNTS_BY_ERA[era], size,
                            tuple(fixed_scales[:count]))
                        tier_designs.append({"houses": tier_houses,
                                             "base_centerpiece": base_centerpiece,
                                             "palace": palace_instance})
                    break
                except ValueError:
                    if factor == .58:
                        raise ValueError(f"no fixed-scale layout for {STYLES[culture]} {ERAS[era]}")
            houses = tier_designs[2]["houses"]
            designs.append({"culture": culture, "culture_name": STYLES[culture],
                            "era": era, "era_name": ERAS[era],
                            "population_counts": list(COUNTS_BY_ERA[era]),
                            "tier_designs": tier_designs,
                            "base_centerpiece": base_centerpiece,
                            "houses": houses,
                            "palace": palace_instance,
                            "capital_replaces_centerpiece": replaces_centerpiece,
                            **({"vertical_metric": 1.0} if replaces_centerpiece else {}),
                            "wall_kit": ("ancient", "medieval", "industrial", "industrial")[era]})
    return {"schema": "c3x.lab.city_design.v1", "site": "flat grassland",
            "tile_xy": [-.5, -.5, .5, .5], "wall_perimeters": list(WALL_RADII),
            "styles": list(STYLES), "eras": list(ERAS), "designs": designs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("layouts.json"))
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"Wrote {len(result['designs'])} culture/era layouts to {args.output}")


if __name__ == "__main__":
    main()
