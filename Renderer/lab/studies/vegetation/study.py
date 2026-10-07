#!/usr/bin/env python3
"""Before/after study of forest and jungle canopy beside routes, resources and sites.

Renders the vegetation Lab cases through the production renderer with a pinned
copy of the current candidate DLL, at gameplay and close-up zoom, then stitches
labelled side-by-side sheets with ffmpeg. Outputs are disposable and stay under
Renderer/lab/out/vegetation/.

    python3 Renderer/lab/studies/vegetation/study.py render before
    python3 Renderer/lab/studies/vegetation/study.py render after
    $C3X_RENDERER_PYTHON Renderer/lab/studies/vegetation/study.py sheet before after   # needs Pillow
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.lab.studies.vegetation import cases as vegetation_cases

OUT = ROOT / "Renderer/lab/out/vegetation"
CATEGORIES = ("forests", "jungles")
ZOOMS = (128, 256)


def selected(only):
    for category in CATEGORIES:
        for case in ("gameplay",) + vegetation_cases.cases(category):
            if not only or f"{category}/{case}" in only or case in only:
                yield category, case


def render(label, only=(), zooms=ZOOMS):
    renderer.prepare_sources(list(CATEGORIES))
    renderer.ensure_candidate(list(CATEGORIES))
    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    dll = root / "C3XRenderer.dll"
    shutil.copy2(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    images = []
    for category, case in selected(only):
        for zoom in zooms:
            entry = renderer.native_render(category, case, 12, zoom, root / category / case, candidate=dll)
            images.append({**entry, "category": category})
    (root / "render.json").write_text(json.dumps({"label": label, "dll_sha256": renderer.checksum(dll),
                                                  "images": images}, indent=2) + "\n")
    print(root / "render.json")


def sheet(before, after, only=(), crop=None):
    """One labelled side-by-side PNG per case and zoom (Pillow)."""
    from PIL import Image, ImageDraw
    out = OUT / f"compare-{before}-{after}"
    out.mkdir(parents=True, exist_ok=True)
    for category, case in selected(only):
        for zoom in ZOOMS:
            name = f"{case}-h12-z{zoom}.bmp"
            pair = [OUT / label / category / case / name for label in (before, after)]
            if not all(path.is_file() for path in pair):
                continue
            images = []
            for path in pair:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                if crop:
                    w, h = image.size
                    image = image.crop((int(w * crop[0]), int(h * crop[1]), int(w * crop[2]), int(h * crop[3])))
                images.append(image)
            w, h = images[0].size
            canvas = Image.new("RGB", (2 * w + 8, h + 30), "#1d2420")
            pen = ImageDraw.Draw(canvas)
            for index, (label, image) in enumerate(zip((before, after), images)):
                canvas.paste(image, (index * (w + 8), 30))
                pen.text((index * (w + 8) + 8, 8), f"{category} / {case} / tile {zoom} / {label}", fill="white")
            target = out / f"{category}-{case}-z{zoom}.png"
            canvas.save(target)
            print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--only", nargs="*", default=())
    run.add_argument("--zooms", nargs="*", type=int, default=list(ZOOMS))
    compare = sub.add_parser("sheet")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--only", nargs="*", default=())
    compare.add_argument("--crop", nargs=4, type=float, help="fractions: left top right bottom")
    args = parser.parse_args()
    if args.command == "render":
        render(args.label, args.only, tuple(args.zooms))
    else:
        sheet(args.before, args.after, args.only, args.crop)


if __name__ == "__main__":
    main()
