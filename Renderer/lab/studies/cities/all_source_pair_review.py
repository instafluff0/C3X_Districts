#!/usr/bin/env python3
"""Prepare one magenta city-variant sheet per installed city-art source pair.

These are source auditions, not assignments to Civ III cultures or eras. All
asset choices are made here, offline; nothing is added to game runtime code.
"""

import argparse
import json
from pathlib import Path

from Renderer.lab.studies.cities.build_layouts import ROOT
from Renderer.lab.studies.cities.medieval_family_review import generate, slug
from Renderer.lab.studies.cities.sheet import render_culture


OUT = ROOT / "Renderer/lab/out/cities/all-era-source-auditions"
FAMILIES = {
    "ancient": (ROOT / "Renderer/lab/out/cities/ancient-source-families/source-report.json",
                ROOT / "Renderer/lab/out/cities/ancient-source-families/no-ground-flat-pack"),
    "classical": (ROOT / "Renderer/lab/out/cities/medieval-source-families/source-report-uv.json",
                  ROOT / "Renderer/lab/out/cities/medieval-source-families/flat-pack-uv"),
    **{era: (OUT / era / "source-report.json", OUT / era / "flat-pack")
       for era in ("industrial", "modern", "future", "unspecified")},
}


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
    args = parser.parse_args()
    report_path, pack = FAMILIES[args.era]
    report = json.loads(report_path.read_text())
    catalog = json.loads((ROOT / "Renderer/packs/CityPalacesNormalized/palace_catalog.json").read_text())
    palace_pack = OUT / "flat-palaces"
    tree_pack = ROOT / "Renderer/lab/out/cities/medieval-art/farm-tree-pack"
    results = []
    for pool in report["pools"]:
        culture = pool["source_culture"]
        if args.family and culture != args.family:
            continue
        destination = OUT / "review" / args.era / slug(culture)
        destination.mkdir(parents=True, exist_ok=True)
        sheet = destination / "sheet.png"
        if args.era == "classical":
            source = (ROOT / "Renderer/lab/out/cities/medieval-source-families/review" /
                      slug(culture) / "layouts.json")
            if args.render:
                layouts = json.loads(source.read_text())
                design = next(d for d in layouts["designs"]
                              if (d["culture"], d["era"]) == (1, 1))
                design["review_context"] = "Civ VI source-art audition"
                render_culture(layouts, 1, sheet, only_era=1)
            detail = {"kind": "existing Classical composition", "counts": None}
        else:
            asset, palace_status = palace_for(pool, catalog)
            layouts, detail = generate(pool, pack, palace_pack, asset, tree_pack,
                                       use_curated=False)
            (destination / "layouts.json").write_text(json.dumps(layouts, indent=2) + "\n")
            if args.render:
                render_culture(layouts, 1, sheet, only_era=1)
            detail = {**detail, "palace_asset": asset, "palace_status": palace_status}
        results.append({"source_culture": culture,
                        "source_art_era": pool["source_art_era"],
                        "source_pool": pool["pool"],
                        "selected_components": len(pool["selected"]),
                        "rejected_components": pool["candidate_count"] - len(pool["selected"]),
                        "sheet": sheet.relative_to(OUT).as_posix(),
                        **detail})
        print(args.era, culture, detail, "rendered" if args.render else "layout", flush=True)
    index = OUT / "review" / args.era / "index.json"
    index.write_text(json.dumps(results, indent=2) + "\n")
    print(index)


if __name__ == "__main__":
    main()
