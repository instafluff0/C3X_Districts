#!/usr/bin/env python3
"""Make a Lab palace copy without its deep plinth or optional ground planes."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omit-ground-planes", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing Lab derivative; choose a new output path")
    manifest = json.loads((args.source_pack / "manifest.json").read_text())
    entry = manifest["assets"][args.asset]
    landmark = json.loads((args.source_pack / entry["landmark"]).read_text())
    worked = {binding["geometry"] for binding in landmark["draw_bindings"]
              if "worked" in binding["states"]}
    removed = set()
    for index, name in enumerate(landmark["components"]["geometry"]):
        mesh = json.loads((args.source_pack / name).read_text())
        vertices = mesh["vertices"]
        levels = [vertex["position"][2] for vertex in vertices]
        # Source palaces may have one or several separate ground slabs; their
        # finished tops can rise slightly above zero.
        if index in worked and min(levels) < -.007 and max(levels) < .005:
            removed.add(index)
        elif (args.omit_ground_planes and 0 <= min(levels) <= .002 and
              max(levels) - min(levels) < .002 and
              all(vertex["normal"][2] > .95 for vertex in vertices)):
            removed.add(index)
    if not removed:
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
