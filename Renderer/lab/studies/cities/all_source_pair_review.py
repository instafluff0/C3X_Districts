#!/usr/bin/env python3
"""Prepare one magenta city-variant sheet per installed city-art source pair.

These are source auditions, not assignments to Civ III cultures or eras. All
asset choices are made here, offline; nothing is added to game runtime code.
"""

import argparse
import json
from pathlib import Path

from Renderer.lab.studies.cities.build_layouts import ROOT
from Renderer.lab.studies.cities.ancient_trees import plant
from Renderer.lab.studies.cities.medieval_family_review import box, generate, place, scale_for, slug
from Renderer.lab.studies.cities.sheet import render_culture
from Renderer.lab.studies.cities.skyline_recipe import apply_skyline
from Renderer.lab.studies.cities.seeded_variants import apply_house_variation


OUT = ROOT / "Renderer/lab/out/cities/all-era-source-auditions"
FAMILIES = {
    "ancient": (ROOT / "Renderer/lab/out/cities/ancient-source-families/source-report.json",
                ROOT / "Renderer/lab/out/cities/ancient-source-families/foundation-free-pack"),
    "classical": (ROOT / "Renderer/lab/out/cities/medieval-source-families/source-report-uv.json",
                  ROOT / "Renderer/lab/out/cities/medieval-source-families/foundation-free-pack-uv-v2"),
    **{era: (OUT / era / "source-report.json", OUT / era / "foundation-free-pack")
       for era in ("industrial", "modern", "future", "unspecified")},
}

# These large source blocks contain baked-in, differently facing buildings.
# A single aligned civic building makes the tile-edge direction unambiguous.
CLASSICAL_CENTERS = {
    "Baltic": "DIS_CTY_RBAL_Bld_F",
    "Scottish": "DIS_CTY_RSCT_Bld_07",
    "CIVILIZATION_VIETNAM": "DIS_CTY_RVIE_Bld_07",
}


def classical_layout(source, pool, pack):
    layouts = json.loads(source.read_text())
    original_pack = "Renderer/lab/out/cities/medieval-source-families/flat-pack-uv"
    replacement = pack.relative_to(ROOT).as_posix()

    def repoint(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "pack" and value == original_pack:
                    node[key] = replacement
                else:
                    repoint(value)
        elif isinstance(node, list):
            for value in node:
                repoint(value)

    repoint(layouts)
    name = CLASSICAL_CENTERS.get(pool["source_culture"])
    if name:
        record = next(entry for entry in pool["selected"] if entry["entry"] == name)
        design = next(d for d in layouts["designs"] if (d["culture"], d["era"]) == (1, 1))
        old = design["base_centerpiece"]
        aligned = {**old, "asset": record["asset_id"], "pack": replacement,
                   "scale": scale_for(record["asset_id"], pack, .39, .33),
                   "rotation": 0.0}
        design["base_centerpiece"] = aligned
        for tier in design["tier_designs"]:
            tier["base_centerpiece"] = aligned.copy()
    if pool["source_culture"] == "America":
        # The source LG_SQ and SQ_01 blocks bake several houses with
        # conflicting street directions into one mesh. Rebuild their plots
        # from aligned same-family individual buildings. Every substitute
        # stays within the footprint of the block it replaces.
        records = {entry["entry"]: entry["asset_id"] for entry in pool["selected"]}
        def aligned_item(source, entry, scale, dx=0.0, dy=0.0):
            return {**source, "asset": records[entry], "pack": replacement,
                    "scale": scale, "rotation": 0.0,
                    "offset": [round(source["offset"][0] + dx, 3),
                               round(source["offset"][1] + dy, 3)]}
        design = next(d for d in layouts["designs"] if (d["culture"], d["era"]) == (1, 1))
        old_core = design["base_centerpiece"]
        core = aligned_item(old_core, "DIS_CTY_RE_Bld_MD_B_03", 2.85)
        square = records["DIS_CTY_RE_Block_B_SQ_01"]
        for tier in design["tier_designs"]:
            tier["base_centerpiece"] = core.copy()
            for key in ("houses", "capital_houses"):
                houses = []
                for item in tier[key]:
                    if item["asset"] == square:
                        inward = (-.025 if item["offset"][0] > .5 else
                                  .025 if item["offset"][0] < -.5 else 0.0)
                        houses.extend((aligned_item(item, "DIS_CTY_RE_Bld_MD_B_01", 2.18,
                                                    dx=inward, dy=-.045),
                                       aligned_item(item, "DIS_CTY_RE_Bld_XSM_B_01", 2.0,
                                                    dx=inward, dy=.065)))
                    else:
                        houses.append(item)
                if key == "houses":
                    for dx in (-.155, .155):
                        houses.append(aligned_item(old_core, "DIS_CTY_RE_Bld_SM_B_01",
                                                   2.0, dx=dx, dy=-.165))
                tier[key] = houses
        infill = ["DIS_CTY_RE_Bld_XSM_B_02", "DIS_CTY_RE_Bld_SM_B_02",
                  "DIS_CTY_RE_Bld_XSM_B_01", "DIS_CTY_RE_Bld_MD_B_02"]
        targets = [(-.16, .26), (.16, .26), (-.33, -.04), (.33, -.04),
                   (0, -.16)]
        for size, tier in enumerate(design["tier_designs"]):
            if size == 0:
                continue
            for key, center in (("houses", core), ("capital_houses", tier["palace"])):
                occupied = [box(item) for item in tier[key]] + [box(center)]
                for index, target in enumerate(targets):
                    choices = [dict(asset_id=records[entry], rotation=0.0)
                               for entry in infill[index % len(infill):] +
                               infill[:index % len(infill)]]
                    extra = place(choices, pack, occupied, size, target, .28, .16)
                    if extra is not None:
                        tier[key].append(extra)
        design["base_centerpiece"] = core
        design["houses"] = design["tier_designs"][-1]["houses"]
        design["population_counts"] = [len(tier["houses"])
                                       for tier in design["tier_designs"]]
    return layouts


def palace_for(pool, catalog):
    source = pool["source_culture"]
    era = pool["source_art_era"]
    for culture, art_era, status in ((source, era, "exact source pair"),
                                     (source, "DEFAULT", "same culture, generic era"),
                                     ("DEFAULT", "DEFAULT", "generic comparison palace")):
        matches = [p["asset_id"] for p in catalog["palaces"]
                   if any(s["culture"] == culture and s["era"] == art_era
                          for s in p["source_selectors"])]
        if len(matches) == 1:
            return matches[0], status
    raise ValueError(f"No palace available for {source} {era}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--family", help="One exact source culture tag")
    parser.add_argument("--render", action="store_true", help="Render full 12-cell sheets")
    parser.add_argument("--variation-seed", type=int, default=0,
                        help="Stable per-city placement seed for Lab comparisons")
    args = parser.parse_args()
    report_path, pack = FAMILIES[args.era]
    report = json.loads(report_path.read_text())
    catalog = json.loads((ROOT / "Renderer/packs/CityPalacesNormalized/palace_catalog.json").read_text())
    palace_pack = OUT / "flat-palaces"
    tree_pack = ROOT / "Renderer/lab/out/cities/medieval-art/farm-tree-pack"
    review_root = OUT / ("review" if args.variation_seed == 0 else
                         f"review-variants/seed-{args.variation_seed}")
    results = []
    for pool in report["pools"]:
        culture = pool["source_culture"]
        if args.family and culture != args.family:
            continue
        destination = review_root / args.era / slug(culture)
        destination.mkdir(parents=True, exist_ok=True)
        sheet = destination / "sheet.png"
        if args.era == "classical":
            source = (ROOT / "Renderer/lab/out/cities/medieval-source-families/review" /
                      slug(culture) / "layouts.json")
            layouts = classical_layout(source, pool, pack)
            design = next(d for d in layouts["designs"]
                          if (d["culture"], d["era"]) == (1, 1))
            plant(design["tier_designs"], tree_pack, counts=(2, 5, 8), scale=3.1,
                  seed=f"{pool['source_art_era']}|{culture}|{args.variation_seed}")
            changes = apply_house_variation(design, args.variation_seed)
            (destination / "layouts.json").write_text(json.dumps(layouts, indent=2) + "\n")
            if args.render:
                design["review_context"] = "Civ VI source-art audition"
                render_culture(layouts, 1, sheet, only_era=1)
            detail = {"kind": ("aligned individual composition" if culture == "America"
                               else "aligned Classical composition" if culture in CLASSICAL_CENTERS
                               else "existing Classical composition"),
                      "counts": ([len(tier["houses"]) for tier in design["tier_designs"]]
                      if args.render and culture == "America" else None),
                      "seeded_house_changes": changes}
        else:
            asset, palace_status = palace_for(pool, catalog)
            layouts, detail = generate(pool, pack, palace_pack, asset, tree_pack,
                                       use_curated=False,
                                       variation_seed=args.variation_seed)
            if args.era == "industrial":
                design = next(d for d in layouts["designs"]
                              if d.get("source_art_era") == pool["source_art_era"])
                apply_skyline(design, args.era, args.variation_seed)
                detail = {**detail, "skyline": [
                    {role: sum(item.get("skyline_role") == role
                               for item in tier["houses"])
                     for role in ("modern_infill", "skyscraper")}
                    for tier in design["tier_designs"]]}
            design = next(d for d in layouts["designs"]
                          if d.get("source_art_era") == pool["source_art_era"])
            changes = apply_house_variation(design, args.variation_seed)
            (destination / "layouts.json").write_text(json.dumps(layouts, indent=2) + "\n")
            if args.render:
                render_culture(layouts, 1, sheet, only_era=1)
            detail = {**detail, "palace_asset": asset,
                      "palace_status": palace_status, "seeded_house_changes": changes}
        results.append({"source_culture": culture,
                        "source_art_era": pool["source_art_era"],
                        "source_pool": pool["pool"],
                        "selected_components": len(pool["selected"]),
                        "rejected_components": pool["candidate_count"] - len(pool["selected"]),
                        "sheet": sheet.relative_to(OUT).as_posix(),
                        **detail})
        print(args.era, culture, detail, "rendered" if args.render else "layout", flush=True)
    index = review_root / args.era / "index.json"
    index.write_text(json.dumps(results, indent=2) + "\n")
    print(index)


if __name__ == "__main__":
    main()
