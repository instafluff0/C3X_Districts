#!/usr/bin/env python3
"""Make a small Lab copy of one palace without its below-ground plinth."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing Lab derivative; choose a new output path")
    manifest = json.loads((args.source_pack / "manifest.json").read_text())
    entry = manifest["assets"][args.asset]
    landmark = json.loads((args.source_pack / entry["landmark"]).read_text())
    removed = set()
    for index, name in enumerate(landmark["components"]["geometry"]):
        mesh = json.loads((args.source_pack / name).read_text())
        levels = [vertex["position"][2] for vertex in mesh["vertices"]]
        if min(levels) < -.007 and max(levels) < .001:
            removed.add(index)
    if len(removed) != 2:
        raise ValueError("unrecognized palace plinth geometry")
    landmark["draw_bindings"] = [binding for binding in landmark["draw_bindings"]
                                  if binding["geometry"] not in removed]
    files = set(landmark["components"]["geometry"] + landmark["components"]["materials"])
    for name in landmark["components"]["materials"]:
        material = json.loads((args.source_pack / name).read_text())
        for channel in material["channels"].values():
            if isinstance(channel, dict) and channel.get("texture"):
                files.add(channel["texture"])
    for name in sorted(files):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(args.source_pack / name, target)
    landmark_path = args.output / entry["landmark"]
    landmark_path.parent.mkdir(parents=True, exist_ok=True)
    landmark_path.write_text(json.dumps(landmark, indent=2) + "\n")
    manifest["assets"] = {args.asset: entry}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"asset": args.asset, "removed_geometry": sorted(removed),
                      "files": len(files)}))


if __name__ == "__main__":
    main()
