#!/usr/bin/env python3
"""Before/after city gallery through the production renderer.

Renders the city Lab cases (cases.py) with a pinned copy of the candidate
DLL and an optional candidate city pack (C3X_RENDERER_CITY_PACK). Outputs are
disposable and stay under Renderer/lab/out/city-study/. Sheets need Pillow.

    python3 Renderer/lab/studies/city_readability/study.py render before
    python3 Renderer/lab/studies/city_readability/study.py render after --pack CityCompositionLab
    $C3X_RENDERER_PYTHON Renderer/lab/studies/city_readability/study.py sheet before after
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
from Renderer.lab.studies.city_readability import cases as city_cases

OUT = ROOT / "Renderer/lab/out/city-study"
CATEGORY = "cities"
DEFAULT_CASES = ("city-ladder-european", "city-sites-industrial", "city-gameplay-industrial")


def shots(only=(), zooms=(128,)):
    for case in (only or DEFAULT_CASES):
        if case not in city_cases.CASES:
            raise ValueError("Unknown city case " + case)
        for zoom in zooms:
            yield case, zoom, f"{case}-z{zoom}"


def render(label, only=(), zooms=(128,), hour=12, pack="", pinned=None, extra_env=None,
           focus=None, view=None, suffix=""):
    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    dll = root / "C3XRenderer.dll"
    if pinned:
        if pinned.resolve() != dll.resolve():
            shutil.copy2(pinned, dll)
    elif not dll.is_file():
        # The cities asset job cannot rebuild CityCompositionRuntime since its
        # recipe intakes were retired; the DLL itself needs no category assets.
        renderer.ensure_candidate([])
        shutil.copy2(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    images = []
    import os
    # A close-up's explicit view size applies to this call only.
    if view:
        os.environ["C3X_LAB_CITY_VIEW"] = view
    else:
        os.environ.pop("C3X_LAB_CITY_VIEW", None)
    for case, zoom, name in shots(only, zooms):
        name += suffix
        centre = focus or city_cases.centre(case)
        env = {"C3X_RENDERER_PREVIEW_OBJECTS": "", "C3X_RENDERER_PREVIEW_CITY": "",
               "C3X_LAB_TILE_CITIES": city_cases.city_spec(case, centre),
               "C3X_LAB_TILE_RESOURCES": city_cases.resources(CATEGORY, case, centre)}
        if pack:
            env["C3X_RENDERER_CITY_PACK"] = "Renderer\\packs\\" + pack
        env.update(extra_env or {})
        entry = renderer.native_render(CATEGORY, case, hour, zoom, root / name, candidate=dll,
                                       center=(16 + centre[0], 16 + centre[1]), extra_env=env)
        images.append({**entry, "name": name})
    (root / "render.json").write_text(json.dumps({"label": label, "pack": pack or "CityCompositionRuntime",
                                                  "dll_sha256": renderer.checksum(dll),
                                                  "images": images}, indent=2) + "\n")
    print(root / "render.json")


def image_of(label, case, name):
    found = sorted((OUT / label / name).glob(f"{case}-h*-z*.bmp"))
    if not found:
        found = sorted((OUT / label).glob(name + "/*.bmp"))
    return found[0] if found else OUT / label / name / "missing.bmp"


def sheet(labels, only=(), zooms=(128,), crop=None, vertical=False, suffix=""):
    from PIL import Image, ImageDraw
    out = OUT / ("compare-" + "-".join(labels))
    out.mkdir(parents=True, exist_ok=True)
    for case, _zoom, name in shots(only, zooms):
        name += suffix
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
        if vertical:
            canvas = Image.new("RGB", (w, len(images) * (h + 30)), "#1d2420")
        else:
            canvas = Image.new("RGB", (len(images) * (w + 8) - 8, h + 30), "#1d2420")
        pen = ImageDraw.Draw(canvas)
        for index, (label, image) in enumerate(zip(labels, images)):
            x, y = (0, index * (h + 30)) if vertical else (index * (w + 8), 0)
            canvas.paste(image, (x, y + 30))
            pen.text((x + 8, y + 8), f"{name} / {label}", fill="white")
        target = out / f"{name}.png"
        canvas.save(target)
        print(target)


def closeups(labels, packs, dlls, names=(), hour=12):
    """Zoom-256 single-site renders per label plus one labelled sheet per shot."""
    for label, pack, dll in zip(labels, packs, dlls):
        for case, focus, name in city_cases.CLOSE_UPS:
            if names and name not in names:
                continue
            render(label, (case,), (256,), hour, "" if pack == "-" else pack, Path(dll), None,
                   focus, "768x480", "-" + name)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    out = OUT / ("closeups-" + "-".join(labels))
    out.mkdir(parents=True, exist_ok=True)
    for case, _focus, name in city_cases.CLOSE_UPS:
        if names and name not in names:
            continue
        paths = [image_of(label, case, f"{case}-z256-{name}") for label in labels]
        if not all(p.is_file() for p in paths):
            continue
        images = [Image.open(p).convert("RGB") for p in paths]
        w, h = images[0].size
        canvas = Image.new("RGB", (len(images) * (w + 8) - 8, h + 30), "#1d2420")
        pen = ImageDraw.Draw(canvas)
        for index, (label, image) in enumerate(zip(labels, images)):
            canvas.paste(image, (index * (w + 8), 30))
            pen.text((index * (w + 8) + 8, 8), f"{name} / {label}", fill="white")
        canvas.save(out / f"{name}.png")
        print(out / f"{name}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--only", nargs="*", default=())
    run.add_argument("--zoom", nargs="*", type=int, default=[128])
    run.add_argument("--hour", type=int, default=12)
    run.add_argument("--pack", default="", help="candidate city pack folder under Renderer/packs")
    run.add_argument("--dll", type=Path, help="render with this pinned DLL")
    run.add_argument("--env", nargs="*", default=(), help="extra C3X_*=value switches")
    run.add_argument("--focus", help="view centre offset dx,dy from the scene origin (close-ups)")
    run.add_argument("--view", help="explicit view size WxH in pixels (close-ups)")
    run.add_argument("--suffix", default="", help="output name suffix for close-ups")
    close = sub.add_parser("closeups")
    close.add_argument("--labels", nargs="+", required=True)
    close.add_argument("--packs", nargs="+", required=True, help="pack per label, or - for production")
    close.add_argument("--dlls", nargs="+", required=True)
    close.add_argument("--names", nargs="*", default=())
    close.add_argument("--hour", type=int, default=12)
    compare = sub.add_parser("sheet")
    compare.add_argument("labels", nargs="+")
    compare.add_argument("--only", nargs="*", default=())
    compare.add_argument("--zoom", nargs="*", type=int, default=[128])
    compare.add_argument("--crop", nargs=4, type=float)
    compare.add_argument("--vertical", action="store_true")
    compare.add_argument("--suffix", default="")
    args = parser.parse_args()
    if args.command == "render":
        render(args.label, tuple(args.only), tuple(args.zoom), args.hour, args.pack, args.dll,
               dict(item.split("=", 1) for item in args.env),
               tuple(int(v) for v in args.focus.split(",")) if args.focus else None, args.view, args.suffix)
    elif args.command == "closeups":
        closeups(args.labels, args.packs, args.dlls, tuple(args.names), args.hour)
    else:
        sheet(args.labels, tuple(args.only), tuple(args.zoom), args.crop, args.vertical, args.suffix)


if __name__ == "__main__":
    main()


def animate(label, case, focus, zoom=256, view="512x360", frames=8, period=3.4, hour=12,
            pack="CityCompositionLab", pinned=None, name="effects"):
    """Render visual-clock stills across one smoke period and join them as a GIF."""
    paths = []
    for index in range(frames):
        seconds = period * index / frames
        frame = f"{case}-z{zoom}-{name}-t{index}"
        if not image_of(label, case, frame).is_file():  # resume after an interrupted batch
            render(label, (case,), (zoom,), hour, pack, pinned, {"C3X_LAB_EFFECT_TIME": f"{seconds:.3f}"},
                   focus, view, f"-{name}-t{index}")
        paths.append(image_of(label, case, frame))
    try:
        from PIL import Image
    except ImportError:
        return paths
    images = [Image.open(path).convert("RGB") for path in paths]
    target = OUT / label / f"{case}-{name}.gif"
    images[0].save(target, save_all=True, append_images=images[1:], duration=int(period * 1000 / frames), loop=0)
    print(target)
    return target
