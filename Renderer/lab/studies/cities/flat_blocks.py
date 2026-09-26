#!/usr/bin/env python3
"""Make a disposable flat-ground study pack without source block plinths."""

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
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing Lab derivative; choose a new output path")
    report = json.loads(args.report.read_text())
    shutil.copytree(args.source_pack, args.output, copy_function=os.link)
    manifest = json.loads((args.output / "manifest.json").read_text())
    removed = []
    for record in report["pools"][0]["selected"]:
        if "_Block_" not in record["entry"]:
            continue
        path = args.output / manifest["assets"][record["asset_id"]]["landmark"]
        landmark = json.loads(path.read_text())
        first = landmark["components"]["geometry"][0]
        mesh = json.loads((args.output / first).read_text())
        levels = [vertex["position"][2] for vertex in mesh["vertices"]]
        if min(levels) > -.05 or max(levels) > .01:
            raise ValueError(f"unrecognized block plinth for {record['entry']}")
        landmark["draw_bindings"] = [binding for binding in landmark["draw_bindings"]
                                      if binding["geometry"] != 0]
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(landmark, indent=2) + "\n")
        os.replace(temporary, path)
        removed.append(record["entry"])
    print(json.dumps({"flat_blocks": len(removed), "source_pool": report["pools"][0]["pool"]}))


if __name__ == "__main__":
    main()
