#!/usr/bin/env python3
"""Civ III connection-pattern roads over the infrastructure Lab fixtures.

Run with a Python that has Pillow (as for renderer.py gallery). Renders the
current candidate at overview, gameplay and close zooms. With
--before, the same scenes are rendered once more with the pattern pack hidden,
which selects the unchanged segment roads in the same DLL, and a labelled
before/after sheet is written. Outputs are disposable Lab previews.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.renderer import ensure_candidate, native_render, prepare_sources

OUT = ROOT / "Renderer/lab/out/roads/patterns"
PACK = ROOT / "Renderer/packs/RoutePatternsRuntime"


def render(label, cases, zooms, hour, center):
    images = {}
    for case in cases:
        for zoom in zooms:
            output = OUT / label / case
            image = output / f"{case}-h{hour:02}-z{zoom}.bmp"
            started = time.time()
            try:
                native_render("infrastructure", case, hour, zoom, output, center=center)
            except ValueError:
                # A close view can frame out the fixture's mine and farm; only
                # that ownership witness may fail, never the render itself.
                log = (output / "native.log").read_text(errors="replace")
                if not (image.is_file() and image.stat().st_mtime >= started and
                        "FAIL category object study" in log and " 0 fallback, output=" in log):
                    raise
            images[(case, zoom)] = image
    return images


def sheet(rows, destination):
    from PIL import Image, ImageDraw
    columns = max(len(row[1]) for row in rows)
    width, height = 640, 480
    canvas = Image.new("RGB", (columns * (width + 8) + 8, len(rows) * (height + 30) + 8), "#1c211f")
    pen = ImageDraw.Draw(canvas)
    for r, (title, images) in enumerate(rows):
        for c, (label, path) in enumerate(images):
            x, y = 8 + c * (width + 8), 8 + r * (height + 30)
            with Image.open(path) as image:
                canvas.paste(image.convert("RGB").resize((width, height)), (x, y + 22))
            pen.text((x + 4, y + 4), f"{title} / {label}", fill="white")
    canvas.save(destination)
    print(destination.relative_to(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="network,gameplay")
    parser.add_argument("--zooms", default="64,128,256")
    parser.add_argument("--hour", type=int, default=12)
    parser.add_argument("--center", default="16,16", help="raw tile x,y at the view center")
    parser.add_argument("--before", action="store_true", help="also render segment roads with the pack hidden")
    parser.add_argument("--no-river", action="store_true", help="strip fixture rivers (authored bridges)")
    parser.add_argument("--eras", action="store_true", help="render every road era (and its bridge) instead")
    args = parser.parse_args(argv)
    if args.no_river:
        import Renderer.renderer as dispatcher
        original = dispatcher.scene

        def scene(category, case, destination, *, world_size=32):
            original(category, case, destination, world_size=world_size)
            rows = destination.read_text().splitlines()
            destination.write_text("\n".join([rows[0]] + [",".join(row.split(",")[:-1] + ["0"])
                                                          for row in rows[1:]]) + "\n")
        dispatcher.scene = scene
    cases = args.cases.split(",")
    zooms = [int(z) for z in args.zooms.split(",")]
    prepare_sources(["infrastructure"])
    ensure_candidate(["infrastructure"])
    center = tuple(int(v) for v in args.center.split(","))
    if args.eras:
        rows = []
        for era, name in enumerate(("ancient", "medieval", "industrial", "modern")):
            os.environ["C3X_LAB_ROAD_ERA"] = str(era)
            try:
                images = render("era-" + name, cases, zooms, args.hour, center)
            finally:
                os.environ.pop("C3X_LAB_ROAD_ERA", None)
            rows.append((name, [(f"{case} tile {zoom}", images[(case, zoom)]) for case in cases for zoom in zooms]))
        sheet(rows, OUT / "eras.png")
        return 0
    after = render("after", cases, zooms, args.hour, center)
    rows = []
    if args.before:
        hidden = PACK.with_name(PACK.name + f".hidden-{os.getpid()}")
        PACK.rename(hidden)
        try:
            before = render("before", cases, zooms, args.hour, center)
        finally:
            hidden.rename(PACK)
        for case in cases:
            for zoom in zooms:
                rows.append((f"{case} tile {zoom}", [("segment roads (before)", before[(case, zoom)]),
                                                    ("Civ III patterns (after)", after[(case, zoom)])]))
    else:
        for case in cases:
            rows.append((case, [(f"tile {zoom}", after[(case, zoom)]) for zoom in zooms]))
    sheet(rows, OUT / ("compare.png" if args.before else "after.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
