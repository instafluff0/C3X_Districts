#!/usr/bin/env python3
"""Build repeatable four-state Lab layouts for every imported medieval art family.

The five already curated Civ III mappings are reused unchanged. Other source
families receive a dimension-aware initial audition for visual comparison.
All source choices happen offline; the renderer sees only generic pack metadata.
"""

import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.ground import footprint_alignment
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.ancient_trees import plant
from Renderer.lab.studies.cities.build_layouts import ROOT, footprint, inside_wall


CURATED = {
    "Mediterranean": ("medieval-art", 2),
    "DEFAULT": ("medieval-european", 1),
    "EastAsian": ("medieval-asian", 4),
    "SouthAmerican": ("medieval-american", 0),
    "Mughal": ("medieval-middle_eastern", 3),
}
SITES = {
    0: [(-.25, .27), (.25, .27), (0, .34), (-.35, -.07), (.35, -.07)],
    1: [(-.44, .25), (.44, .25), (-.43, -.24), (.43, -.24),
        (0, .45), (0, -.48), (-.52, .02), (.52, .02),
        (-.23, .41), (.23, .41), (-.20, -.43), (.20, -.43)],
    2: [(-.56, -.43), (.56, -.43), (-.56, .38), (.56, .38),
        (0, .60), (0, -.60), (-.37, .56), (.37, .56),
        (-.59, -.07), (.59, -.07)],
}
FACADE_INFILL = {
    0: [(-.19, .12), (.19, .12)],
    1: [(-.24, -.18), (.24, -.18), (-.30, .07), (.30, .07),
        (-.18, .30), (.18, .30)],
    2: [(-.37, -.36), (.37, -.36), (-.30, -.10), (.30, -.10),
        (-.18, .22), (.18, .22), (-.42, .30), (.42, .30)],
}
OFFSETS = [(dx*.025, dy*.025) for dx in range(-12, 13)
           for dy in range(-12, 13)]
OFFSETS.sort(key=lambda p: (p[0]*p[0]+p[1]*p[1], abs(p[0]), abs(p[1])))


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.removeprefix("CIVILIZATION_").lower()).strip("-")


def box(item):
    body = component(item["asset"], Path(item["pack"]))
    return footprint({"low": body["lo"], "high": body["hi"]}, item)


def scale_for(asset, pack, height, span, maximum=18.0):
    body = component(asset, pack)
    rise = max(.001, body["hi"][2]-body["lo"][2])
    width = max(body["hi"][0]-body["lo"][0], body["hi"][1]-body["lo"][1], .001)
    return round(min(height/rise, span/width, maximum), 3)


@lru_cache(None)
def facing_correction(asset, pack):
    body = component(asset, pack)
    points = [vertex["position"][:2]
              for mesh, material in body["parts"] if material["alpha_mode"] != "blend"
              for vertex in mesh["vertices"]]
    correction = footprint_alignment(points)
    # Small hull asymmetries are authored details, not a different street
    # direction. Align larger baked rotations to the tile-edge facade grid;
    # the particular doorway side still needs visual source-art review.
    return round(correction, 8) if abs(correction) > math.radians(10) else 0.0


def instance(asset, pack, scale, x, y, rotation=None):
    if rotation is None:
        rotation = facing_correction(asset, pack)
    return {"asset": asset, "pack": pack.as_posix(), "scale": scale,
            "rotation": rotation, "offset": [x, y]}


def palace_for(family, catalog):
    exact = [p["asset_id"] for p in catalog["palaces"]
             if any(s["culture"] == family and s["era"] == "DEFAULT"
                    for s in p["source_selectors"])]
    if len(exact) != 1:
        raise ValueError(f"No unique source palace for {family}: {exact}")
    return exact[0]


def choices(pool, pack):
    entries = pool["selected"]
    family = pool["source_culture"]
    blocks = [p for p in entries if "_Block_" in p["entry"]]
    cores = [p for p in blocks if "LG_SQ" in p["entry"]]
    squares = [p for p in blocks if "_SQ_" in p["entry"] and "LG_SQ" not in p["entry"]]
    singles = [p for p in entries if "_Bld" in p["entry"] and "_Block_" not in p["entry"]]
    # These source blocks contain several houses with conflicting baked front
    # directions. Use isolated houses so the street-facing facade stays on a
    # SE/SW tile-edge side. Vietnam's generic-era pool has blocks only; the
    # same-culture Classical pool supplies its matching individual houses.
    if family == "CIVILIZATION_VIETNAM" and pool["source_art_era"] == "DEFAULT":
        source = ROOT / "Renderer/lab/out/cities/medieval-source-families"
        report = json.loads((source / "source-report-uv.json").read_text())
        related = next(p for p in report["pools"] if p["source_culture"] == family)
        singles = [{**entry, "pack": (source / "flat-pack-uv").as_posix(),
                    "rotation": 0.0}
                   for entry in related["selected"] if "_Bld_" in entry["entry"]]
        cores = [{**p, "rotation": 0.0} for p in blocks if "_Block_SQ_01" in p["entry"]]
        squares = []
    elif family in ("Vikings", "CIVILIZATION_MAORI") and pool["source_art_era"] == "DEFAULT":
        singles = [{**p, "rotation": 0.0} for p in singles]
        cores = [next(p for p in singles if p["entry"].endswith(
            "_Bld_A_05" if family == "Vikings" else "_Bldg_A"))]
        squares = []
    # Portugal and Vietnam's era-unspecified source pools expose complete
    # blocks but no isolated houses. Audition their smaller blocks as infill.
    if not singles:
        singles = [p for p in blocks if "LG_SQ" not in p["entry"]]
    if not singles:
        raise ValueError(f"No source buildings for {pool['source_culture']}")
    def dimensions(record):
        b = component(record["asset_id"], Path(record.get("pack", pack)))
        return ((b["hi"][0]-b["lo"][0])*(b["hi"][1]-b["lo"][1]),
                b["hi"][2]-b["lo"][2])
    singles.sort(key=lambda p: (dimensions(p)[0], dimensions(p)[1]), reverse=True)
    # Avoid the largest civic model as a repeated house and keep several roof
    # forms. The physical size calculation below chooses a uniform scale.
    ordinary = singles[1:14] if len(singles) >= 9 and singles[0] not in blocks else singles
    if family in ("Vikings", "CIVILIZATION_MAORI") and pool["source_art_era"] == "DEFAULT":
        ordinary = [p for p in ordinary if p["asset_id"] != cores[0]["asset_id"]]
    return (cores or squares or singles)[:3], squares[:3], ordinary


def place(assets, pack, occupied, size, target, height, span):
    for dx, dy in OFFSETS:
        x, y = round(target[0]+dx, 3), round(target[1]+dy, 3)
        for entry in assets:
            asset = entry["asset_id"]
            source_pack = Path(entry.get("pack", pack))
            nominal = scale_for(asset, source_pack, height, span)
            for factor in (1.0, .92, .84):
                item = instance(asset, source_pack, round(nominal*factor, 3), x, y,
                                entry.get("rotation"))
                bounds = box(item)
                if not inside_wall(bounds, size, clearance=.015):
                    continue
                if size == 0 and any(abs(value) > .5 for value in bounds):
                    continue
                if any(overlaps(bounds, other) for other in occupied):
                    continue
                occupied.append(bounds)
                return item
    return None


def generate(pool, flat_pack, palace_pack, palace_asset, tree_pack,
             use_curated=True):
    family = pool["source_culture"]
    layouts = json.loads((ROOT / "Renderer/lab/studies/cities/layouts.json").read_text())
    design = next(d for d in layouts["designs"] if (d["culture"], d["era"]) == (1, 1))
    layouts["styles"][1] = family.removeprefix("CIVILIZATION_")
    design["source_art_era"] = pool["source_art_era"]
    design["era_name"] = ("Middle Ages" if use_curated else
                          pool["source_art_era"].removeprefix("ARTERA_").title())
    design["culture_name"] = layouts["styles"][1]
    if not use_curated:
        design["review_context"] = "Civ VI source-art audition"
        design["wall_kit"] = ("ancient" if pool["source_art_era"] == "ARTERA_ANCIENT"
                              else "industrial" if pool["source_art_era"] in
                              ("ARTERA_INDUSTRIAL", "ARTERA_MODERN", "ARTERA_FUTURE")
                              else "medieval")
    if use_curated and family in CURATED:
        directory, culture = CURATED[family]
        path = ROOT / f"Renderer/lab/out/cities/{directory}/review-dense-layouts.json"
        authored = json.loads(path.read_text())
        source = next(d for d in authored["designs"] if (d["culture"], d["era"]) == (culture, 1))
        tiers = json.loads(json.dumps(source["tier_designs"]))
        for tier in tiers:
            for key in ("houses", "capital_houses", "decorations", "capital_decorations"):
                for item in tier.get(key, []):
                    if item["asset"] != "city/prop/source_farm_tree":
                        item["pack"] = flat_pack.as_posix()
            for key in ("base_centerpiece", "capital_centerpiece"):
                if key in tier:
                    tier[key]["pack"] = flat_pack.as_posix()
            tier["palace"]["pack"] = palace_pack.as_posix()
        copied = {key: source[key] for key in ("population_counts", "grounding", "slope_limit",
                                               "vertical_metric", "capital_replaces_centerpiece")
                  if key in source}
        design.update(**copied, tier_designs=tiers, houses=tiers[-1]["houses"],
                      base_centerpiece=tiers[0]["base_centerpiece"], palace=tiers[0]["palace"])
        return layouts, {"kind": "curated", "counts": [len(t["houses"]) for t in tiers]}

    cores, squares, singles = choices(pool, flat_pack)
    facing_profile = family in ("CIVILIZATION_VIETNAM", "Vikings",
                                "CIVILIZATION_MAORI") and pool["source_art_era"] == "DEFAULT"
    core_asset = cores[0]["asset_id"]
    core_pack = Path(cores[0].get("pack", flat_pack))
    core = instance(core_asset, core_pack, scale_for(core_asset, core_pack, .36, .43),
                    0, -.08, -math.pi/4 if facing_profile else cores[0].get("rotation"))
    palace = instance(palace_asset, palace_pack,
                      scale_for(palace_asset, palace_pack, .46, .42),
                      0, -.08, -math.pi/4 if facing_profile else 0.0)
    tiers = []
    for center, capital in ((core, False), (palace, True)):
        occupied = [box(center)]
        houses = []
        for size in range(3):
            targets = SITES[size] + (FACADE_INFILL[size] if facing_profile else [])
            for slot, target in enumerate(targets):
                if size == 0 and slot < 2 and squares:
                    palette, height, span = squares, .30, .23
                elif size > 0 and slot < 2 and squares:
                    palette, height, span = squares, .28, .22
                else:
                    rotation = (size*7+slot*3) % len(singles)
                    palette = singles[rotation:] + singles[:rotation]
                    height, span = .30, .19
                item = place(palette, flat_pack, occupied, size, target, height, span)
                if item is not None:
                    houses.append(item)
            if len(houses) < (4, 11, 17)[size]:
                raise ValueError(f"Sparse {family} {'capital' if capital else 'base'} tier {size}: {len(houses)}")
            if not capital:
                tiers.append({"houses": list(houses), "base_centerpiece": core,
                              "palace": palace})
            else:
                tiers[size]["capital_houses"] = list(houses)
    plant(tiers, tree_pack, counts=(1, 3, 5))
    design.update(grounding="terrain", slope_limit=64.0, vertical_metric=1.0,
                  population_counts=[len(t["houses"]) for t in tiers],
                  tier_designs=tiers, houses=tiers[-1]["houses"],
                  base_centerpiece=core, palace=palace,
                  capital_replaces_centerpiece=True)
    return layouts, {"kind": "automatic", "counts": [len(t["houses"]) for t in tiers],
                     "capital_counts": [len(t["capital_houses"]) for t in tiers],
                     "facade_source": ("same-culture Classical individual houses"
                                       if family == "CIVILIZATION_VIETNAM" and
                                       pool["source_art_era"] == "DEFAULT"
                                       else "same-pool individual houses"
                                       if family in ("Vikings", "CIVILIZATION_MAORI") and
                                       pool["source_art_era"] == "DEFAULT"
                                       else "source-pool components")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--flat-pack", type=Path, required=True)
    parser.add_argument("--palace-pack", type=Path, required=True)
    parser.add_argument("--farm-tree-pack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", help="Only one original ArtDef culture tag")
    args = parser.parse_args()
    report = json.loads(args.source_report.read_text())
    catalog = json.loads((ROOT / "Renderer/packs/CityPalacesNormalized/palace_catalog.json").read_text())
    results = []
    for pool in report["pools"]:
        family = pool["source_culture"]
        if args.family and family != args.family:
            continue
        palace = palace_for(family, catalog)
        layouts, detail = generate(pool, args.flat_pack, args.palace_pack,
                                   palace, args.farm_tree_pack)
        target = args.output_dir / slug(family) / "layouts.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(layouts, indent=2) + "\n")
        results.append({"family": family, "slug": slug(family),
                        "source_pool": pool["pool"], "palace": palace,
                        "layouts": target.as_posix(), **detail})
        print(family, detail, flush=True)
    if not results:
        raise ValueError("No matching medieval source family")
    (args.output_dir / "index.json").write_text(json.dumps(results, indent=2) + "\n")
    print(args.output_dir / "index.json")


if __name__ == "__main__":
    main()
