#!/usr/bin/env python3
"""Replay each imported medieval family on the same captured test.biq site."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from Renderer.lab.studies.cities.build_layouts import ROOT


VARIANTS = ("town-base", "city-base", "city-capital", "city-both", "metro-both")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--palace-pack", type=Path, required=True)
    parser.add_argument("--farm-tree-pack", type=Path, required=True)
    parser.add_argument("--family", help="One ArtDef family tag")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-packs", action="store_true")
    args = parser.parse_args()
    gallery = ROOT / "Renderer/lab/out/cities/test-biq/gallery"
    for entry in json.loads(args.index.read_text()):
        if args.family and entry["family"] != args.family:
            continue
        directory = args.index.parent / entry["slug"]
        if args.wait_for_packs:
            deadline = time.monotonic()+1800
            while not ((directory / "pack/city.bin").is_file() and
                       (directory / "sheet/european-medieval.png").is_file()):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Family pack did not finish: {entry['family']}")
                time.sleep(2)
        prefix = "medieval-family-" + entry["slug"]
        report = directory / (prefix + "-replay.json")
        if args.resume and report.exists():
            records = json.loads(report.read_text())
            if (len(records) == len(VARIANTS) and
                all((gallery / item["image"]).exists() for item in records)):
                print("SKIP", entry["family"], flush=True)
                continue
        command = [sys.executable, str(ROOT / "Renderer/lab/studies/cities/render_candidate.py"),
                   "--candidate-pack", str(directory / "pack"),
                   "--source-pack", str(args.source_pack),
                   "--extra-source-pack", str(args.palace_pack),
                   "--extra-source-pack", str(args.farm_tree_pack),
                   "--name-prefix", prefix, "--culture", "european",
                   "--era", "medieval", "--site", "23,79"]
        for variant in VARIANTS:
            command.extend(("--variant", variant))
        subprocess.run(command, cwd=ROOT, check=True)
        print("PASS", entry["family"], flush=True)


if __name__ == "__main__":
    main()
