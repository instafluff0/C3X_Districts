#!/usr/bin/env python3
"""Make one local palace pack for the installed medieval source families."""

import argparse
import json
import os
import shutil
from pathlib import Path

from Renderer.lab.studies.cities.medieval_family_review import palace_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--all-palaces", action="store_true",
                        help="Prepare every normalized palace for source-era auditions")
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing Lab derivative; choose a new output path")
    report = json.loads(args.source_report.read_text()) if not args.all_palaces else None
    catalog = json.loads((args.source_pack / "palace_catalog.json").read_text())
    manifest = json.loads((args.source_pack / "manifest.json").read_text())
    selected = ({palace_for(pool["source_culture"], catalog) for pool in report["pools"]}
                if report else {palace["asset_id"] for palace in catalog["palaces"]})
    shutil.copytree(args.source_pack, args.output, copy_function=os.link)
    changes = []
    for asset in sorted(selected):
        source = args.source_pack / manifest["assets"][asset]["landmark"]
        path = args.output / manifest["assets"][asset]["landmark"]
        landmark = json.loads(source.read_text())
        worked = {binding["geometry"] for binding in landmark["draw_bindings"]
                  if "worked" in binding["states"]}
        removed = set()
        for index, name in enumerate(landmark["components"]["geometry"]):
            mesh = json.loads((args.source_pack / name).read_text())
            vertices = mesh["vertices"]
            levels = [vertex["position"][2] for vertex in vertices]
            if (index in worked and min(levels) < -.007 and max(levels) < .005):
                removed.add(index)
            elif (0 <= min(levels) <= .002 and
                  max(levels)-min(levels) < .002 and
                  all(vertex["normal"][2] > .95 for vertex in vertices)):
                removed.add(index)
        landmark["draw_bindings"] = [binding for binding in landmark["draw_bindings"]
                                      if binding["geometry"] not in removed]
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(landmark, indent=2) + "\n")
        os.replace(temporary, path)
        changes.append({"asset": asset, "removed_geometry": sorted(removed)})
    result = args.output / "medieval-family-flattening.json"
    result.write_text(json.dumps(changes, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
