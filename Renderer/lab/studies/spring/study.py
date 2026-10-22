#!/usr/bin/env python3
"""Spring look study on the production renderer, kept entirely in the Lab.

Adapted from the autumn study harness (../autumn/study.py). Renders the
selected spring concept's own view (unchanged test.biq, camera
20,75, tile 128, noon) through the current candidate DLL. A spring look is a
private shader root: a copy of the flattened production shaders with the
study's patches (spring.py) applied, selected with C3X_RENDERER_SHADER_SOURCE_ROOT.
Production shaders, packs and staged binaries are never modified.

    python3 Renderer/lab/studies/spring/study.py render summer
    python3 Renderer/lab/studies/spring/study.py render v1 --look v1
    python3 Renderer/lab/studies/spring/study.py sheet summer v1 --target   # needs Pillow

Outputs are disposable and stay under Renderer/lab/out/spring/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.platform import ROOT, run_native_fixture
from Renderer.lab.studies.spring import spring

BIQ = ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"
PREVIEW = ROOT / "Renderer/lab/.cache/native_preview.exe"
CANDIDATE = ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"
OUT = ROOT / "Renderer/lab/out/spring"
TARGET = ROOT / "Renderer/lab/out/seasons/scene-concepts/spring-vibrant.png"
SAVE_1498 = ROOT / "Renderer/.cache/frame-delivery-step/input-1498AD.SAV"
# name: (center, tile width, hour, viewport[, scene]). Scenes: test.biq by default;
# "village" adds ancient cities, roads, farms and a mine to the concept view;
# "salamanca" is a 32x32 window of the 1498 AD save (cities, farms, rails, mines).
# Option "units" composes the Lab's six-unit roster (top left); it currently
# fails in the 32-bit Lab ("required unit source admission incomplete").
VIEWS = {
    "concept": ((20, 75), 128, 12, (1600, 900)),
    "close": ((22, 77), 256, 12, (1280, 800)),
    "evening": ((20, 75), 128, 18, (1600, 900)),
    "plains": ((60, 65), 128, 12, (1600, 900)),
    "village": ((18, 74), 160, 12, (1600, 900), "village", "no-units"),
    "village-night": ((18, 74), 160, 0, (1600, 900), "village", "no-units"),
    "salamanca": ((16, 16), 128, 12, (1600, 900), "salamanca", "no-units"),
    "salamanca-close": ((15, 17), 224, 12, (1600, 900), "salamanca", "no-units"),
    "salamanca-night": ((16, 16), 128, 0, (1600, 900), "salamanca", "no-units"),
}
# Lab village on test.biq (doubled lattice coordinates): ancient cities on open
# grassland, roads between them, irrigated farms beside them, one hill mine.
VILLAGE_CITIES = ((12, 76, 1, 0), (25, 79, 0, 0), (21, 71, 1, 1))  # x, y, size, capital
VILLAGE_MINE = (23, 71)


def windows(path: Path) -> str:
    return "..\\..\\" + path.relative_to(ROOT).as_posix().replace("/", "\\")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shader_root(look: str) -> Path:
    """Copy the flattened production shaders and apply one look's patches."""
    root = OUT / "shader-roots" / look
    if not root.exists():
        # Seed the runtime's per-entry .cso cache from the newest other look;
        # each entry is keyed by its source hash, so only changed files compile.
        others = sorted((OUT / "shader-roots").glob("*"), key=lambda p: p.stat().st_mtime)
        if others:
            shutil.copytree(others[-1], root)
    for source in (ROOT / "Renderer/native").rglob("*.hlsl"):
        relative = source.relative_to(ROOT)
        if "build" in relative.parts:
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text()
        patched = spring.apply(look, relative.as_posix(), text)
        # Rewrite only on change so the runtime's per-entry .cso cache survives.
        if not target.is_file() or target.read_text() != patched:
            target.write_text(patched)
    return root


def village_scene(base: Path) -> tuple[Path, list]:
    """Write the village scene CSV; return it and its cities as (x, y, size, capital)."""
    import csv
    target = OUT / "scenes/village/scene.csv"
    rows = list(csv.reader(base.open()))
    header, tiles = rows[0], {(int(r[0]), int(r[1])): r for r in rows[1:]}
    farmland = {1, 2, 4}  # plains, grassland, floodplain
    def mark(x, y, bit):
        row = tiles.get((x, y))
        if row and int(row[3]) not in (11, 12, 13):
            row[5] = str(int(row[5]) | bit)
    for x, y, _, _ in VILLAGE_CITIES:
        for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1), (2, 0), (-2, 0), (0, 2)):
            row = tiles.get((x + dx, y + dy))
            if row and int(row[3]) in farmland:
                mark(x + dx, y + dy, 8)
    # Roads: from each city, step to whichever of the eight Civ III
    # neighbours is nearest the capital (screen-space distance).
    capital = next(city for city in VILLAGE_CITIES if city[3])
    steps = ((1, 1), (-1, 1), (1, -1), (-1, -1), (2, 0), (-2, 0), (0, 2), (0, -2))
    for x, y, _, _ in VILLAGE_CITIES:
        while (x, y) != capital[:2]:
            mark(x, y, 1)
            x, y = min(((x + dx, y + dy) for dx, dy in steps),
                       key=lambda p: (p[0] - capital[0]) ** 2 + ((p[1] - capital[1]) / 2) ** 2)
    mark(*VILLAGE_MINE, 4)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows[1:])
    return target, list(VILLAGE_CITIES)


def salamanca_scene() -> tuple[Path, list]:
    folder = OUT / "scenes/salamanca"
    scene, cities = folder / "scene.csv", folder / "cities.json"
    if not scene.is_file() or not cities.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(["node", str(ROOT / "Renderer/lab/studies/roads/save_window.js"), str(SAVE_1498),
                        str(scene), str(cities), "113", "75"], cwd=ROOT, check=True)
    return scene, [(c["x"], c["y"], 0 if (c.get("pop") or 0) < 7 else 1, 0) for c in json.loads(cities.read_text())]


def vm_busy() -> bool:
    """A peer's native fixture is running (the shared VM runs one preview at a time)."""
    lines = subprocess.run(["ps", "-Ao", "args="], capture_output=True, text=True).stdout.splitlines()
    return any(line.startswith("/Applications/Parallels") and "prlctl exec" in line for line in lines)


def vm_preview_running() -> bool:
    """Ask Windows itself, which also catches previews left by stopped callers."""
    from Renderer.lab.platform import native_command_result
    result = native_command_result("Renderer/native",
                                  'tasklist /FO CSV /NH /FI "IMAGENAME eq native_preview.exe"', timeout_seconds=60)
    return "native_preview.exe" in result.get("output_tail", "")


def wait_for_vm() -> None:
    # Overlapping previews on the shared VM crash (0xC0000409) or time out.
    while True:
        while vm_busy():
            time.sleep(5)
        time.sleep(random.uniform(1, 4))
        if not vm_busy() and not vm_preview_running():
            return
        time.sleep(5)


def capture(label: str, view: str, dll: Path, scene: Path, look: str | None, season: int, retry=True,
            extra=None, trace=False) -> Path:
    center, zoom, hour, (width, height), *variant = VIEWS[view]
    context = {}
    if variant:
        scene, cities = village_scene(scene) if variant[0] == "village" else salamanca_scene()
        era = 0 if variant[0] == "village" else 2
        # "dx,dy,culture,era,size,capital,walled" from the view centre.
        options = variant[1:]
        context = {"C3X_LAB_TILE_OVERLAYS": "1", "C3X_LAB_TILE_CITIES": "" if "no-cities" in options else
                   ";".join(f"{x - center[0]},{y - center[1]},0,{era},{size},{cap},0" for x, y, size, cap in cities)}
        if "units" in options:
            # The Lab unit roster composes over the classic map surface (as the
            # units category does), not the shared resident scene surface.
            context.update({"C3X_LAB_UNIT_STUDY": "1", "C3X_LAB_ACTION_CURSOR": "7",
                            "C3X_RENDERER_SHARED_SCENE_SURFACE": "0", "C3X_RENDERER_WORLD_REGIONS": "",
                            "C3X_RENDERER_WORLD_WAVES": ""})
    folder = OUT / label / view
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / "preview.bmp"
    run_id = uuid.uuid4().hex
    settings = {
        "C3X_RENDERER_VISUAL_PROFILE": "",
        "C3X_RENDERER_TRACE": "0",
        "C3X_RENDERER_TRACE_FILE": "",
        "C3X_RENDERER_PREVIEW_OBJECTS": "",
        "C3X_RENDERER_PREVIEW_DENSE_SCENE": "",
        "C3X_RENDERER_PREVIEW_SEASON": str(season),
        "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS": windows(ROOT / "Renderer/custom.custom_rendering.txt"),
        "C3X_RENDERER_SHARED_SCENE_SURFACE": "1",
        "C3X_RENDERER_WORLD_REGIONS": "1",
        "C3X_RENDERER_WORLD_WAVES": "1",
        "C3X_RENDERER_REFLECTION_CONTROL": "0",
        "C3X_RENDERER_WATER_MOTION": "1",
        "C3X_RENDERER_WAVES": "1",
        "C3X_LAB_PID_FILE": windows(folder / "process.txt"),
        "C3X_LAB_RUN_ID": run_id,
        **context,
        **(extra or {}),
    }
    if trace:
        # Shader compile errors are reported only through the renderer trace.
        settings.update({"C3X_RENDERER_TRACE": "1", "C3X_RENDERER_TRACE_FILE": windows(folder / "trace.log")})
    if look:
        settings["C3X_RENDERER_SHADER_SOURCE_ROOT"] = windows(shader_root(look))
    command = "\n".join(f'set "{key}={value}"' for key, value in settings.items())
    command += (f'\n{windows(PREVIEW)} "{windows(dll)}" '
                f'..\\.. "..\\..\\Renderer\\default.custom_rendering.txt" '
                f'"{windows(scene)}" "{windows(image)}" '
                f'{width} {height} {center[0]} {center[1]} {zoom} {hour}')
    log = windows(folder / "native.log")
    receipt = windows(folder / "completion.txt")
    (folder / "render.bat").write_text(
        "@echo off\nsetlocal\n" + command + f' > "{log}" 2>&1\n'
        'set "C3X_LAB_EXIT=%errorlevel%"\n'
        f'> "{receipt}" echo {run_id} %C3X_LAB_EXIT%\n'
        f'type "{log}"\nexit /b %C3X_LAB_EXIT%\n')
    wait_for_vm()
    result = run_native_fixture(folder, f'call "{windows(folder / "render.bat")}"', run_id)
    if result["status"] != "pass" or "0 fallback, output=" not in result["output_tail"] or not image.is_file() or \
            ("C3X_LAB_UNIT_STUDY" in context and "PASS category unit study" not in result["output_tail"]):
        if trace and (folder / "trace.log").is_file():
            print("\n".join(line for line in (folder / "trace.log").read_text(errors="replace").splitlines()
                            if "shader-compile" in line or "error X" in line)[:4000], flush=True)
        if retry:
            print(f"Retrying {label}/{view} once after a failed preview", flush=True)
            return capture(label, view, dll, scene, look, season, retry=False, trace=trace)
        raise RuntimeError(f"Spring preview failed: {label}/{view}: {result['output_tail']}")
    png = folder / "preview.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(image), str(png)], check=True)
    image.unlink()
    return png


def render(label: str, look: str | None, season: int, views, dll: Path, trace=False) -> None:
    if not dll.is_file() or not PREVIEW.is_file():
        raise FileNotFoundError("Candidate DLL or native preview tool is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    scene = OUT / "test-biq.csv"
    if not scene.is_file():
        subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"), str(BIQ), str(scene)],
                       cwd=ROOT, check=True)
    images = {view: capture(label, view, dll, scene, look, season, trace=trace) for view in views}
    (OUT / label / "render.json").write_text(json.dumps({
        "label": label, "look": look, "season": season, "dll_sha256": digest(dll),
        "preview_sha256": digest(PREVIEW), "scene_sha256": digest(scene),
        "views": {view: digest(path) for view, path in images.items()}}, indent=2) + "\n")
    for path in images.values():
        print(path)


# Concept-view regions (1600x900) for OKLab comparison against the target.
REGIONS = {
    "grass-west": (250, 400, 430, 460), "grass-east": (960, 540, 1120, 620), "grass-mid": (820, 580, 900, 660),
    "plains-top": (620, 85, 860, 125), "hills": (890, 195, 950, 230), "trees": (190, 475, 260, 515),
    "trees2": (510, 510, 580, 545), "water": (1350, 120, 1550, 300), "tundra": (220, 720, 480, 850),
}


def measure(labels) -> None:
    """Mean OKLab L/C, median hue and lightness spread per region (needs NumPy)."""
    import numpy as np
    from PIL import Image

    def lch(path):
        with Image.open(path) as source:
            c = np.asarray(source.convert("RGB").resize(VIEWS["concept"][3], Image.LANCZOS), np.float64) / 255
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        lms = np.cbrt(np.maximum(c @ np.array([[0.4122214708, 0.2119034982, 0.0883024619],
                                               [0.5363325363, 0.6806995451, 0.2817188376],
                                               [0.0514459929, 0.1073969566, 0.6299787005]]), 1e-9))
        lab = lms @ np.array([[0.2104542553, 1.9779984951, 0.0259040371], [0.7936177850, -2.4285922050, 0.7827717662],
                              [-0.0040720468, 0.4505937099, -0.8086757660]])
        return np.stack([lab[..., 0], np.hypot(lab[..., 1], lab[..., 2]),
                         np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360], -1)

    images = {"target": lch(TARGET), **{label: lch(OUT / label / "concept" / "preview.png") for label in labels}}
    for region, (left, top, right, bottom) in REGIONS.items():
        cells = []
        for name, image in images.items():
            v = image[top:bottom, left:right].reshape(-1, 3)
            cells.append(f"{name} L{v[:, 0].mean():.3f}~{v[:, 0].std():.3f} C{v[:, 1].mean():.3f} h{np.median(v[:, 2]):5.1f}")
        print(f"{region:11s} " + " | ".join(cells))


def sheet(labels, view: str, crop=None, target=False) -> Path:
    """Labelled side-by-side PNG (Pillow); the concept target can lead the row."""
    from PIL import Image, ImageDraw
    panels = []
    if target:
        with Image.open(TARGET) as source:
            panels.append(("target concept", source.convert("RGB").resize(VIEWS[view][3], Image.LANCZOS)))
    for label in labels:
        with Image.open(OUT / label / view / "preview.png") as source:
            panels.append((label, source.convert("RGB")))
    if crop:
        panels = [(name, image.crop((int(image.width * crop[0]), int(image.height * crop[1]),
                                     int(image.width * crop[2]), int(image.height * crop[3]))))
                  for name, image in panels]
    w, h = panels[0][1].size
    canvas = Image.new("RGB", (len(panels) * (w + 8) - 8, h + 28), "#1d2420")
    pen = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(panels):
        canvas.paste(image, (index * (w + 8), 28))
        pen.text((index * (w + 8) + 8, 8), f"{view} / {name}", fill="white")
    suffix = "-crop" if crop else ""
    target_path = OUT / "sheets" / f"{view}-{'-'.join(labels)}{suffix}.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target_path)
    print(target_path)
    return target_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--look", choices=spring.LOOKS)
    run.add_argument("--season", type=int, help="frame season (default: 3 = Spring with a look, else 0)")
    run.add_argument("--views", nargs="*", default=["concept"], choices=list(VIEWS))
    run.add_argument("--dll", type=Path, default=CANDIDATE)
    run.add_argument("--trace", action="store_true", help="renderer trace (reports shader compile errors)")
    compare = sub.add_parser("sheet")
    compare.add_argument("labels", nargs="+")
    compare.add_argument("--view", default="concept", choices=list(VIEWS))
    compare.add_argument("--crop", nargs=4, type=float, help="fractions: left top right bottom")
    compare.add_argument("--target", action="store_true", help="lead with the concept image")
    roots = sub.add_parser("shaders")
    roots.add_argument("look", choices=spring.LOOKS)
    scores = sub.add_parser("measure")
    scores.add_argument("labels", nargs="+")
    args = parser.parse_args()
    if args.command == "render":
        season = args.season if args.season is not None else (3 if args.look else 0)
        render(args.label, args.look, season, args.views, args.dll.resolve(), args.trace)
    elif args.command == "sheet":
        sheet(args.labels, args.view, args.crop, args.target)
    elif args.command == "measure":
        measure(args.labels)
    else:
        print(shader_root(args.look))


if __name__ == "__main__":
    main()
