#!/usr/bin/env python3
"""Make a disposable study pack without deep plinths or optional ground planes."""

import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omit-ground-planes", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing Lab derivative; choose a new output path")
    report = json.loads(args.report.read_text())
    manifest = json.loads((args.source_pack / "manifest.json").read_text())
    omissions = {}
    ground_planes = 0
    selected = {record["asset_id"]: record
                for pool in report["pools"] for record in pool["selected"]}
    for record in selected.values():
        if "_Block_" not in record["entry"]:
            continue
        path = args.source_pack / manifest["assets"][record["asset_id"]]["landmark"]
        landmark = json.loads(path.read_text())
        candidates = []
        for index, name in enumerate(landmark["components"]["geometry"]):
            mesh = json.loads((args.source_pack / name).read_text())
            levels = [vertex["position"][2] for vertex in mesh["vertices"]]
            if min(levels) < -.05 and max(levels) < .01:
                candidates.append(index)
        if not candidates:
            raise ValueError(f"unrecognized block plinth for {record['entry']}")
        # Some later art families split the same deep platform across two
        # meshes. Both belong to the source block's ground plane.
        omissions[record["asset_id"]] = set(candidates)
        if args.omit_ground_planes:
            for index, name in enumerate(landmark["components"]["geometry"]):
                if index in omissions[record["asset_id"]]:
                    continue
                mesh = json.loads((args.source_pack / name).read_text())
                vertices = mesh["vertices"]
                levels = [vertex["position"][2] for vertex in vertices]
                if (3 <= len(vertices) <= 6 and
                        0 <= min(levels) <= .002 and
                        max(levels) - min(levels) < .0005 and
                        all(vertex["normal"][2] > .95 for vertex in vertices)):
                    omissions[record["asset_id"]].add(index)
                    ground_planes += 1
    shutil.copytree(args.source_pack, args.output, copy_function=os.link)
    removed = []
    for record in selected.values():
        if record["asset_id"] not in omissions:
            continue
        path = args.output / manifest["assets"][record["asset_id"]]["landmark"]
        landmark = json.loads(path.read_text())
        landmark["draw_bindings"] = [binding for binding in landmark["draw_bindings"]
                                      if binding["geometry"] not in
                                      omissions[record["asset_id"]]]
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(landmark, indent=2) + "\n")
        os.replace(temporary, path)
        removed.append(record["entry"])
    print(json.dumps({"flat_blocks": len(removed),
                      "omitted_ground_planes": ground_planes,
                      "source_pools": [pool["pool"] for pool in report["pools"]]}))


if __name__ == "__main__":
    main()
