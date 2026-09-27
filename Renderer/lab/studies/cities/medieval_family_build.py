#!/usr/bin/env python3
"""Render all imported medieval family sheets and compile isolated map packs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from Renderer.lab.studies.cities.build_layouts import ROOT


STUDY = ROOT / "Renderer/lab/studies/cities"
FRAMES = ROOT / "Renderer/lab/shared/cities/prepare_normals.py"
COMPILER = ROOT / "Renderer/native/city_fidelity/prepare_pack.py"
SHEET = STUDY / "sheet.py"
TREE_FRAMES = ROOT / "Renderer/lab/out/cities/medieval-art/farm-tree-pack/frames.json"


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--flat-pack", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--only", choices=("frames", "packs", "sheets", "all"), default="all")
    parser.add_argument("--family", help="One ArtDef family tag")
    parser.add_argument("--resume", action="store_true", help="Skip complete family outputs")
    args = parser.parse_args()
    entries = json.loads(args.index.read_text())
    for entry in entries:
        if args.family and entry["family"] != args.family:
            continue
        directory = args.index.parent / entry["slug"]
        source_frames = directory / "source-frames-uv.json"
        palace_frames = directory / "palace-frames.json"
        merged_frames = directory / "frames-uv.json"
        if (args.resume and args.only == "all" and source_frames.exists() and
            (directory / "pack/city.bin").exists() and
            (directory / "sheet/european-medieval.png").exists()):
            print("SKIP", entry["family"], flush=True)
            continue
        if args.only in ("frames", "all", "packs"):
            if not source_frames.exists():
                run(FRAMES, "--pool", entry["source_pool"],
                    "--source-report", args.source_report, "--pack", args.flat_pack,
                    "--include-frame", "--output", source_frames)
            if not palace_frames.exists():
                run(FRAMES, "--palace", entry["palace"], "--include-frame",
                    "--output", palace_frames)
            meshes = {}
            for path in (source_frames, palace_frames, TREE_FRAMES):
                meshes.update(json.loads(path.read_text())["meshes"])
            merged_frames.write_text(json.dumps({"meshes": meshes},
                                                separators=(",", ":")) + "\n")
        if args.only in ("packs", "all"):
            run(COMPILER, "--lab-layouts", directory / "layouts.json",
                "--lab-focus", "european,medieval", "--lab-frames", merged_frames,
                "--output", directory / "pack")
        if args.only in ("sheets", "all"):
            run(SHEET, "--layouts", directory / "layouts.json",
                "--output-dir", directory / "sheet", "--culture", "european",
                "--era", "medieval")
        print("PASS", entry["family"], args.only, flush=True)


if __name__ == "__main__":
    main()
