#!/usr/bin/env python3
"""Before/after study of river banks, crossings, reflections and rapids.

Renders through the production renderer with a pinned copy of the current
candidate DLL (shaders still come from the checkout's generated files):

- river-reach (Renderer/lab/studies/rivers/cases.py): farmland, bridges,
  hills, forest and resources beside a meandering river, at gameplay zoom
  plus 256-pixel close-ups;
- farms-water: the farm study's river, bridge and coast;
- the rivers category's watershed (detail and gameplay views).

All shots use the shared scene surface, whose mirror pass covers rivers.
`motion` saves 48 frames of one close-up for an animated sheet. Outputs are
disposable and stay under Renderer/lab/out/river-study/.

    python3 Renderer/lab/studies/rivers/study.py render before
    python3 Renderer/lab/studies/rivers/study.py render after --hour 18
    python3 Renderer/lab/studies/rivers/study.py motion after
    $C3X_RENDERER_PYTHON Renderer/lab/studies/rivers/study.py sheet before after
    $C3X_RENDERER_PYTHON Renderer/lab/studies/rivers/study.py gif after
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
from Renderer.lab.studies.rivers import cases as river_cases

OUT = ROOT / "Renderer/lab/out/river-study"
MOTION = ("infrastructure", "river-reach", 256, (2, 0))


def shots(only=(), close=True):
    """(category, case, zoom, centre offset, name) for every requested render."""
    listed = [("infrastructure", "river-reach", 128, (0, 0), "river-reach-z128")]
    if close:
        listed += [("infrastructure", "river-reach", 256, centre, f"river-reach-z256-{centre[0]}_{centre[1]}")
                   for centre in river_cases.CLOSE_UPS["river-reach"]]
    listed += [("infrastructure", "farms-water", 128, (0, 0), "farms-water-z128")]
    if close:
        listed += [("infrastructure", "farms-water", 256, (-2, -2), "farms-water-z256--2_-2")]
    listed += [("rivers", "detail", 128, None, "rivers-detail-z128"),
               ("rivers", "gameplay", 128, None, "rivers-gameplay-z128")]
    for shot in listed:
        if not only or shot[4] in only or shot[1] in only:
            yield shot


def pin(root, pinned):
    dll = root / "C3XRenderer.dll"
    if pinned:
        if pinned.resolve() != dll.resolve():
            shutil.copy2(pinned, dll)
    else:
        renderer.prepare_sources(["infrastructure", "rivers"])
        renderer.ensure_candidate(["infrastructure", "rivers"])
        shutil.copy2(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    return dll


def render_one(category, case, zoom, centre, target, dll, hour, extra_env=None):
    env = dict(extra_env or {})
    # The preview places listed resources relative to the view centre.
    if renderer.lab_tile_cases(category, case) in (farm_cases, river_cases):
        listed = renderer.lab_tile_cases(category, case).resources(category, case, centre)
        if listed:
            env["C3X_LAB_TILE_RESOURCES"] = listed
    kwargs = {} if centre is None else {"center": (16 + centre[0], 16 + centre[1])}
    try:
        return renderer.native_render(category, case, hour, zoom, target, candidate=dll,
                                      shared_surface=True, extra_env=env, **kwargs)
    except ValueError as error:
        # The rivers category also runs the water-motion lifecycle witness after
        # writing its still. Keep the still for review, but record the failure.
        image = target / f"{case}-h{hour:02}-z{zoom}.bmp"
        if category != "rivers" or not image.is_file():
            raise
        print(f"Keeping {image.name}; witness failed: {error}", flush=True)
        return {"image": renderer.relative(image), "sha256": renderer.checksum(image),
                "case": case, "hour": hour, "zoom": zoom, "witness": "failed: " + str(error)}


def render(label, only=(), close=True, hour=12, pinned=None, extra_env=None):
    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    dll = pin(root, pinned)
    images = []
    for category, case, zoom, centre, name in shots(only, close):
        entry = render_one(category, case, zoom, centre, root / f"{name}-h{hour:02}", dll, hour, extra_env)
        images.append({**entry, "name": name})
    record = root / f"render-h{hour:02}.json"
    record.write_text(json.dumps({"label": label, "dll_sha256": renderer.checksum(dll),
                                  "hour": hour, "images": images}, indent=2) + "\n")
    print(record)


def motion(label, hour=12, pinned=None, shader_root=None):
    """48 water-clock frames (1/15 s apart) of one close-up."""
    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    dll = pin(root, pinned)
    category, case, zoom, centre = MOTION
    target = root / f"motion-h{hour:02}"
    env = {"C3X_LAB_WATER_MOTION_STUDY": "1", "C3X_LAB_WATER_FRAMES": "1"}
    if shader_root:
        env["C3X_RENDERER_SHADER_SOURCE_ROOT"] = shader_root
    try:
        render_one(category, case, zoom, centre, target, dll, hour, env)
    except ValueError as error:
        # The frames are the deliverable; the stricter playback witness that
        # follows them is reported but does not discard them.
        if not list(target.glob("*.water-047.bmp")):
            raise
        print(f"Kept motion frames; witness failed: {error}", flush=True)
    print(target)


def image_of(label, name, hour):
    found = sorted((OUT / label / f"{name}-h{hour:02}").glob("*-h*-z*.bmp"))
    found = [path for path in found if ".water-" not in path.name]
    return found[0] if found else OUT / label / "missing.bmp"


def sheet(labels, only=(), hour=12, crop=None):
    """One labelled side-by-side PNG per shot across the given labels (Pillow)."""
    from PIL import Image, ImageDraw
    out = OUT / ("compare-" + "-".join(labels) + f"-h{hour:02}")
    out.mkdir(parents=True, exist_ok=True)
    for _category, _case, _zoom, _centre, name in shots(only):
        paths = [image_of(label, name, hour) for label in labels]
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


def gif(label, hour=12, crop=None):
    """Animated GIF of the saved motion frames (Pillow)."""
    from PIL import Image
    folder = OUT / label / f"motion-h{hour:02}"
    frames = []
    for path in sorted(folder.glob("*.water-[0-9][0-9][0-9].bmp")):
        with Image.open(path) as source:
            image = source.convert("RGB")
        if crop:
            w, h = image.size
            image = image.crop((int(w * crop[0]), int(h * crop[1]), int(w * crop[2]), int(h * crop[3])))
        frames.append(image)
    if not frames:
        raise SystemExit("No motion frames in " + str(folder))
    target = OUT / label / f"motion-h{hour:02}.gif"
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=67, loop=0)
    print(target)


def shader_root_path(name):
    """A complete runtime shader tree generated outside the checkout's outputs,
    so tuning never disturbs other sessions' renders (preview-relative path)."""
    return "..\\..\\" + renderer.relative(OUT / name).replace("/", "\\")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--only", nargs="*", default=())
    run.add_argument("--no-close", action="store_true", help="overview renders only")
    run.add_argument("--hour", type=int, default=12)
    run.add_argument("--dll", type=Path, help="render with this pinned DLL instead of rebuilding")
    run.add_argument("--env", nargs="*", default=(), help="extra C3X_*=value renderer switches")
    run.add_argument("--shader-root", help="private shader root folder under the study output")
    move = sub.add_parser("motion")
    move.add_argument("label")
    move.add_argument("--hour", type=int, default=12)
    move.add_argument("--dll", type=Path)
    move.add_argument("--shader-root")
    compare = sub.add_parser("sheet")
    compare.add_argument("labels", nargs="+")
    compare.add_argument("--only", nargs="*", default=())
    compare.add_argument("--hour", type=int, default=12)
    compare.add_argument("--crop", nargs=4, type=float, help="fractions: left top right bottom")
    animate = sub.add_parser("gif")
    animate.add_argument("label")
    animate.add_argument("--hour", type=int, default=12)
    animate.add_argument("--crop", nargs=4, type=float)
    args = parser.parse_args()
    if args.command == "render":
        env = dict(item.split("=", 1) for item in args.env)
        if args.shader_root:
            env["C3X_RENDERER_SHADER_SOURCE_ROOT"] = shader_root_path(args.shader_root)
        render(args.label, args.only, not args.no_close, args.hour, args.dll, env)
    elif args.command == "motion":
        motion(args.label, args.hour, args.dll,
               shader_root_path(args.shader_root) if args.shader_root else None)
    elif args.command == "sheet":
        sheet(args.labels, args.only, args.hour, args.crop)
    else:
        gif(args.label, args.hour, args.crop)


if __name__ == "__main__":
    main()
