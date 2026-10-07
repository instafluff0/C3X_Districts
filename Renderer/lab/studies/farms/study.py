#!/usr/bin/env python3
"""Before/after study of farms on every terrain and beside routes, resources and water.

Renders the farm Lab cases (Renderer/lab/studies/farms/cases.py) through the
production renderer with a pinned copy of the current candidate DLL: a
gameplay-zoom overview of each case plus 256-pixel close-ups of its key tiles.
Labelled comparison sheets need Pillow. Outputs are disposable and stay under
Renderer/lab/out/farm-study/.

    python3 Renderer/lab/studies/farms/study.py render before
    python3 Renderer/lab/studies/farms/study.py render after --farm-runtime "farm_runtime~kit.bin"
    $C3X_RENDERER_PYTHON Renderer/lab/studies/farms/study.py sheet before after
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
from Renderer.lab.studies.farms import cases as farm_cases

OUT = ROOT / "Renderer/lab/out/farm-study"
CATEGORY = "infrastructure"


def shots(only=(), close=True, centres=()):
    """(case, zoom, centre offset, name) for every requested render."""
    for case in farm_cases.CASES:
        if only and case not in only:
            continue
        yield case, 128, (0, 0), f"{case}-z128"
        if close:
            for dx, dy in farm_cases.CLOSE_UPS[case]:
                if centres and f"{dx},{dy}" not in centres:
                    continue
                yield case, 256, (dx, dy), f"{case}-z256-{dx}_{dy}"


def render(label, only=(), close=True, farm_runtime="", extra_env=None, hour=12, pinned=None, centres=()):
    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    dll = root / "C3XRenderer.dll"
    if pinned:
        # Re-render with an earlier pinned candidate (e.g. a lighting check).
        if pinned.resolve() != dll.resolve():
            shutil.copy2(pinned, dll)
    else:
        renderer.prepare_sources([CATEGORY])
        renderer.ensure_candidate([CATEGORY])
        shutil.copy2(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    images = []
    for case, zoom, (dx, dy), name in shots(only, close, centres):
        target = root / name
        env = dict(extra_env or {})
        if farm_cases.resources(CATEGORY, case):
            env["C3X_LAB_TILE_RESOURCES"] = farm_cases.resources(CATEGORY, case, (dx, dy))
        entry = renderer.native_render(CATEGORY, case, hour, zoom, target, candidate=dll,
                                       center=(16 + dx, 16 + dy), farm_runtime=farm_runtime,
                                       extra_env=env)
        images.append({**entry, "name": name})
    (root / "render.json").write_text(json.dumps({"label": label, "dll_sha256": renderer.checksum(dll),
                                                  "farm_runtime": farm_runtime or "farm_runtime.bin",
                                                  "images": images}, indent=2) + "\n")
    print(root / "render.json")


def image_of(label, case, name):
    found = sorted((OUT / label / name).glob(f"{case}-h*-z*.bmp"))
    return found[0] if found else OUT / label / name / "missing.bmp"


def sheet(labels, only=(), crop=None):
    """One labelled side-by-side PNG per shot across the given labels (Pillow)."""
    from PIL import Image, ImageDraw
    out = OUT / ("compare-" + "-".join(labels))
    out.mkdir(parents=True, exist_ok=True)
    for case, _zoom, _centre, name in shots(only):
        paths = [image_of(label, case, name) for label in labels]
        if not all(path.is_file() for path in paths):
            continue
        images = []
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
            if crop:
                w, h = image.size
                image = image.crop((int(w * crop[0]), int(h * crop[1]), int(w * crop[2]), int(h * crop[3])))
            images.append(image)
        w, h = images[0].size
        canvas = Image.new("RGB", (len(images) * (w + 8) - 8, h + 30), "#1d2420")
        pen = ImageDraw.Draw(canvas)
        for index, (label, image) in enumerate(zip(labels, images)):
            canvas.paste(image, (index * (w + 8), 30))
            pen.text((index * (w + 8) + 8, 8), f"{name} / {label}", fill="white")
        target = out / f"{name}.png"
        canvas.save(target)
        print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--only", nargs="*", default=())
    run.add_argument("--no-close", action="store_true", help="overview renders only")
    run.add_argument("--farm-runtime", default="", help="candidate farm kit file in ImprovementsNormalized")
    run.add_argument("--env", nargs="*", default=(), help="extra C3X_*=value renderer switches")
    run.add_argument("--hour", type=int, default=12)
    run.add_argument("--dll", type=Path, help="render with this pinned DLL instead of rebuilding")
    run.add_argument("--centres", default="", help='close-ups only at these offsets, e.g. --centres="-4,-6;4,6"')
    compare = sub.add_parser("sheet")
    compare.add_argument("labels", nargs="+")
    compare.add_argument("--only", nargs="*", default=())
    compare.add_argument("--crop", nargs=4, type=float, help="fractions: left top right bottom")
    args = parser.parse_args()
    if args.command == "render":
        render(args.label, args.only, not args.no_close, args.farm_runtime,
               dict(item.split("=", 1) for item in args.env), args.hour, args.dll,
               tuple(item for item in args.centres.split(";") if item))
    else:
        sheet(args.labels, args.only, args.crop)


if __name__ == "__main__":
    main()
