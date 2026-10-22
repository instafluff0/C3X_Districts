#!/usr/bin/env python3
"""Winter look study on the production renderer, kept entirely in the Lab.

Renders the selected winter concept's own view (unchanged test.biq, camera
20,75, tile 128, noon) and city/farm context views through the current
candidate DLL with frame season 2 (Winter). A winter look is a private shader
root: a copy of the flattened production shaders with this study's patches
(winter.py) applied, selected with C3X_RENDERER_SHADER_SOURCE_ROOT. Production
shaders, packs and staged binaries are never modified.

    python3 Renderer/lab/studies/winter/study.py render base --views concept village
    python3 Renderer/lab/studies/winter/study.py render v1 --look v1 --views concept
    python3 Renderer/lab/studies/winter/study.py sheet summer base v1 --target

Outputs are disposable and stay under Renderer/lab/out/winter/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.platform import ROOT, NativeFixturePending, run_native_fixture
from Renderer.lab.studies.winter import winter

BIQ = ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"
PREVIEW = ROOT / "Renderer/lab/.cache/native_preview.exe"
CANDIDATE = ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"
OUT = ROOT / "Renderer/lab/out/winter"
TARGET = ROOT / "Renderer/lab/out/seasons/scene-concepts/winter-icy-blue.png"
SAVE_1498 = ROOT / "Renderer/.cache/frame-delivery-step/input-1498AD.SAV"
# name: (center, tile width, hour, viewport[, scene, options...]). Scenes:
# test.biq by default (the concept's own render: no tile overlays, so every
# forest draws as broadleaf); "biq" is test.biq through the Lab tile overlays,
# which honour the BIQ pine bit as the game does; "village" adds ancient
# cities, roads, farms and a mine; "salamanca" is a 32x32 window of the
# 1498 AD save. Option "snow-pines" previews pine forests drawn with their
# authored snowy variety (production picks it only for pines on tundra; the
# Lab marks those tiles' base terrain tundra).
VIEWS = {
    "concept": ((20, 75), 128, 12, (1600, 900)),
    "close": ((22, 77), 256, 12, (1280, 800)),
    "evening": ((20, 75), 128, 18, (1600, 900)),
    "night": ((20, 75), 128, 0, (1600, 900)),
    "pines": ((20, 75), 128, 12, (1600, 900), "biq"),
    "snow-pines": ((20, 75), 128, 12, (1600, 900), "biq", "snow-pines"),
    "pines-night": ((20, 75), 128, 0, (1600, 900), "biq"),
    "village": ((18, 74), 160, 12, (1600, 900), "village"),
    "village-close": ((21, 73), 256, 12, (1600, 900), "village"),
    "village-night": ((18, 74), 160, 0, (1600, 900), "village"),
    "village-sp": ((18, 74), 160, 12, (1600, 900), "village", "snow-pines"),
    "village-close-sp": ((21, 73), 256, 12, (1600, 900), "village", "snow-pines"),
    "village-night-sp": ((18, 74), 160, 0, (1600, 900), "village", "snow-pines"),
    # Wide Coast, Sea and Ocean bands (readability of the three water types).
    "seas": ((60, 26), 96, 12, (1600, 900)),
    "salamanca": ((16, 16), 128, 12, (1600, 900), "salamanca"),
    "salamanca-close": ((15, 17), 224, 12, (1600, 900), "salamanca"),
    "salamanca-night": ((16, 16), 128, 0, (1600, 900), "salamanca"),
    "salamanca-close-sp": ((15, 17), 224, 12, (1600, 900), "salamanca", "snow-pines"),
    "evening-sp": ((20, 75), 128, 18, (1600, 900), "biq", "snow-pines"),
    "pines-evening": ((20, 75), 128, 18, (1600, 900), "biq"),
    # All four open land types with interior tiles (desert band at the top, tundra lower left).
    "lands": ((20, 70), 96, 12, (1600, 900), "biq"),
}
# Lab village on test.biq (doubled lattice coordinates): ancient cities on open
# grassland, roads between them, irrigated farms beside them, one hill mine.
VILLAGE_CITIES = ((12, 76, 1, 0), (25, 79, 0, 0), (21, 71, 1, 1))  # x, y, size, capital
VILLAGE_MINE = (23, 71)


def windows(path: Path) -> str:
    return "..\\..\\" + path.relative_to(ROOT).as_posix().replace("/", "\\")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shader_root(look: str, source: Path = ROOT, tag: str = "") -> Path:
    """Copy the flattened production shaders (or a private tree's) and apply one look's patches."""
    root = OUT / "shader-roots" / (look + tag)
    if not root.exists():
        # Seed the runtime's per-entry .cso cache from the newest other look;
        # each entry is keyed by its source hash, so only changed files compile.
        others = sorted((OUT / "shader-roots").glob("*"), key=lambda p: p.stat().st_mtime)
        if others:
            shutil.copytree(others[-1], root)
    for path in (source / "Renderer/native").rglob("*.hlsl"):
        relative = path.relative_to(source)
        if "build" in relative.parts:
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        patched = winter.apply(look, relative.as_posix(), path.read_text())
        # Rewrite only on change so the runtime's per-entry .cso cache survives.
        if not target.is_file() or target.read_text() != patched:
            target.write_text(patched)
    return root


def village_scene(base: Path) -> tuple[Path, list]:
    """Write the village scene CSV; return it and its cities as (x, y, size, capital)."""
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


def snow_pines(scene: Path) -> Path:
    """Copy a scene with every pine forest tile (BIQ bonus 0x20) on tundra base."""
    target = scene.with_name(scene.stem + "-snow-pines.csv")
    rows = list(csv.reader(scene.open()))
    for row in rows[1:]:
        if int(row[3]) == 7 and int(row[4]) & 0x20:
            row[2] = "3"
    with target.open("w", newline="") as file:
        csv.writer(file, lineterminator="\n").writerows(rows)
    return target


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
    """Ask Windows itself, which also catches previews left by stopped callers
    (the autumn study's check; overlapping previews crash or time out)."""
    from Renderer.lab.platform import native_command_result
    result = native_command_result("Renderer/native",
                                   'tasklist /FO CSV /NH /FI "IMAGENAME eq native_preview.exe"', timeout_seconds=60)
    return "native_preview.exe" in result.get("output_tail", "")


def wait_for_vm() -> None:
    """One preview at a time on the shared VM: no peer `prlctl exec` on the
    Mac and no native_preview.exe in the VM's own task list. Run this study's
    renders from one process (one `render` call or one queue) at a time."""
    while True:
        while vm_busy():
            time.sleep(5)
        time.sleep(random.uniform(1, 4))
        if not vm_busy() and not vm_preview_running():
            return
        time.sleep(5)


def capture(label: str, view: str, dll: Path, scene: Path, look: str | None, season: int, retry=True,
            tree: str | None = None, flurry=False, frames=0, effect_time=None, winter_pines=False) -> Path:
    center, zoom, hour, (width, height), *variant = VIEWS[view]
    context = {}
    if variant:
        kind, options = variant[0], variant[1:]
        scene, cities = (village_scene(scene) if kind == "village" else
                         salamanca_scene() if kind == "salamanca" else (scene, []))
        if "snow-pines" in options:
            scene = snow_pines(scene)
        era = 0 if kind == "village" else 2
        # "dx,dy,culture,era,size,capital,walled" from the view centre.
        context = {"C3X_LAB_TILE_OVERLAYS": "1", "C3X_LAB_TILE_CITIES":
                   ";".join(f"{x - center[0]},{y - center[1]},0,{era},{size},{cap},0" for x, y, size, cap in cities)}
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
    }
    preview = PREVIEW
    if tree:
        # A private Lab tree (flurry_tree.py): its DLL, preview tool and shaders.
        private = OUT / f"tree-{tree}"
        dll, preview = private / "Renderer/native/build/candidate/C3XRenderer.dll", private / "Renderer/lab/.cache/native_preview.exe"
    if flurry:
        settings["C3X_LAB_MOUNTAIN_FLURRY"] = "1"
    if winter_pines:
        settings["C3X_LAB_WINTER_PINES"] = "1"
    if frames:
        settings["C3X_LAB_EFFECT_FRAMES"] = str(frames)
    if effect_time is not None:
        settings["C3X_LAB_EFFECT_TIME"] = f"{effect_time:.6f}"
    if look or tree:
        settings["C3X_RENDERER_SHADER_SOURCE_ROOT"] = windows(
            shader_root(look or "identity", OUT / f"tree-{tree}" if tree else ROOT, f"@{tree}" if tree else ""))
    command = "\n".join(f'set "{key}={value}"' for key, value in settings.items())
    command += (f'\n{windows(preview)} "{windows(dll)}" '
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
    try:
        result = run_native_fixture(folder, f'call "{windows(folder / "render.bat")}"', run_id)
    except ValueError:  # NativeFixturePending or an unconfirmed completion
        if not frames:
            raise
        # Effect clips render every frame cold (~30 s each on the shared VM),
        # far past the fixture timeout: wait for render.bat's own receipt.
        receipt = folder / "completion.txt"
        while not (receipt.is_file() and receipt.read_text().split()[:1] == [run_id]):
            time.sleep(10)
        status = receipt.read_text().split()[1]
        result = {"status": "pass" if status == "0" else "fail", "output_tail": (folder / "native.log").read_text()}
    done = "EFFECT frames: saved" in result["output_tail"] if frames else "0 fallback, output=" in result["output_tail"]
    if result["status"] != "pass" or not done or not image.is_file():
        if retry:
            print(f"Retrying {label}/{view} once after a failed preview", flush=True)
            return capture(label, view, dll, scene, look, season, False, tree, flurry, frames, effect_time, winter_pines)
        raise RuntimeError(f"Winter preview failed: {label}/{view}: {result['output_tail']}")
    if frames:
        # 15 fps effect frames -> MP4 (the still stays as preview.png).
        clip = folder / "effect.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "15", "-i",
                        str(folder / "preview.bmp.effect-%03d.bmp"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", "-movflags", "+faststart", str(clip)], check=True)
        for bmp in folder.glob("preview.bmp.effect-*.bmp"):
            bmp.unlink()
    png = folder / "preview.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(image), str(png)], check=True)
    image.unlink()
    return png


def render(label: str, look: str | None, season: int, views, dll: Path, tree=None, flurry=False, frames=0,
           effect_time=None, winter_pines=False) -> None:
    if not dll.is_file() or not PREVIEW.is_file():
        raise FileNotFoundError("Candidate DLL or native preview tool is missing")
    if tree:
        dll = OUT / f"tree-{tree}" / "Renderer/native/build/candidate/C3XRenderer.dll"
    OUT.mkdir(parents=True, exist_ok=True)
    scene = OUT / "test-biq.csv"
    if not scene.is_file():
        subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"), str(BIQ), str(scene)],
                       cwd=ROOT, check=True)
    images = {view: capture(label, view, dll, scene, look, season, tree=tree, flurry=flurry, frames=frames,
                            effect_time=effect_time, winter_pines=winter_pines)
              for view in views}
    receipt = OUT / label / "render.json"
    previous = json.loads(receipt.read_text())["views"] if receipt.is_file() else {}
    receipt.write_text(json.dumps({
        "label": label, "look": look, "season": season, "dll_sha256": digest(dll),
        "preview_sha256": digest(PREVIEW), "scene_sha256": digest(scene),
        "views": {**previous, **{view: digest(path) for view, path in images.items()}}}, indent=2) + "\n")
    for path in images.values():
        print(path)


def load(path: Path, size=None):
    """RGB uint8 array through ffmpeg (this Mac has numpy but no Pillow)."""
    import numpy as np
    if size is None:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True, check=True).stdout.strip()
        size = tuple(int(v) for v in probe.split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf",
                          f"scale={size[0]}:{size[1]}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3)


def save(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                    f"{image.shape[1]}x{image.shape[0]}", "-i", "-", str(path)],
                   input=image.tobytes(), check=True)


def sheet(labels, view: str, crop=None, target=False) -> Path:
    """Side-by-side PNG (concept first when requested, then labels in order).
    Panels are unlabelled; the review page names them."""
    import numpy as np
    size = VIEWS[view][3]
    panels = ([load(TARGET, size)] if target else []) + [load(OUT / label / view / "preview.png") for label in labels]
    if crop:
        panels = [p[int(p.shape[0] * crop[1]):int(p.shape[0] * crop[3]), int(p.shape[1] * crop[0]):int(p.shape[1] * crop[2])]
                  for p in panels]
    h, w = panels[0].shape[:2]
    columns = 2 if len(panels) == 4 else len(panels)
    rows = (len(panels) + columns - 1) // columns
    canvas = np.full((rows * (h + 8) - 8, columns * (w + 8) - 8, 3), 29, np.uint8)
    for index, panel in enumerate(panels):
        x, y = (index % columns) * (w + 8), (index // columns) * (h + 8)
        canvas[y:y + h, x:x + w] = panel
    suffix = "-crop" if crop else ""
    target_path = OUT / "sheets" / f"{view}-{'target-' if target else ''}{'-'.join(labels)}{suffix}.png"
    save(canvas, target_path)
    print(target_path)
    return target_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--look", choices=winter.LOOKS)
    run.add_argument("--season", type=int, default=2, help="frame season (default 2, Winter; 0 = Summer)")
    run.add_argument("--views", nargs="*", default=["concept"], choices=list(VIEWS))
    run.add_argument("--dll", type=Path, default=CANDIDATE)
    run.add_argument("--tree", help="private Lab tree under out/winter (e.g. flurry)")
    run.add_argument("--flurry", action="store_true", help="Lab switch: snow flurry over mountain tiles (tree flurry)")
    run.add_argument("--frames", type=int, default=0, help="attached-effect frames at 15 fps -> effect.mp4")
    run.add_argument("--effect-time", type=float, help="visual clock of the first frame, seconds (preview default 1)")
    run.add_argument("--winter-pines", action="store_true", help="Lab switch: snowy-pine variety for every pine forest (tree flurry)")
    compare = sub.add_parser("sheet")
    compare.add_argument("labels", nargs="+")
    compare.add_argument("--view", default="concept", choices=list(VIEWS))
    compare.add_argument("--crop", nargs=4, type=float, help="fractions: left top right bottom")
    compare.add_argument("--target", action="store_true", help="lead with the concept image")
    roots = sub.add_parser("shaders")
    roots.add_argument("look", choices=winter.LOOKS)
    args = parser.parse_args()
    if args.command == "render":
        render(args.label, args.look, args.season, args.views, args.dll.resolve(), args.tree, args.flurry, args.frames,
               args.effect_time, args.winter_pines)
    elif args.command == "sheet":
        sheet(args.labels, args.view, args.crop, args.target)
    else:
        print(shader_root(args.look))


if __name__ == "__main__":
    main()
