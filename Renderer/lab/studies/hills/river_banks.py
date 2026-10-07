#!/usr/bin/env python3
"""Hills beside rivers: before/after through the production renderer.

Civ III rivers run on tile edges, but the authored hill bodies reach across
them. One Lab case lays a meandering river past the hill arrangements that
matter: a hill on the far bank with the river wrapping its foot (the in-game
report), a hill on the near bank, a valley squeezed between four hills with a
bridged road, a mined chain on the far bank and a forested hill facing it. The
case is supplied to the dispatcher in-process; renderer.py is unchanged.
Outputs are disposable, under Renderer/lab/out/hills/river-banks/.

    python3 Renderer/lab/studies/hills/river_banks.py render before --tree hill-banks --env C3X_LAB_HILL_BANKS=0
    python3 Renderer/lab/studies/hills/river_banks.py render after --tree hill-banks
    $C3X_RENDERER_PYTHON Renderer/lab/studies/hills/river_banks.py sheet before after

`--tree NAME` renders with a private source tree built by
Renderer/lab/studies/mountains/private_tree.py (DLL, preview and shaders), so
an unaccepted look never reaches other sessions' builds. `--env KEY=VALUE`
passes study switches to the renderer.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer

OUT = ROOT / "Renderer/lab/out/hills/river-banks"
TREES = ROOT / "Renderer/lab/out/mountains"
# The infrastructure fixture renders routes, mines and farms from BIQ bits; its
# object witness needs a route or improvement in every view.
CATEGORY = "infrastructure"
CASE = "hill-banks"
ROAD, MINE, IRRIGATION = 0x1, 0x4, 0x8
GRASSLAND, HILLS, FOREST, COAST = 2, 5, 7, 11
# River path of tile-corner nodes in (column, row); "c" steps one column (down
# right on screen), "r" one row (up right). The river crosses the view left to
# right to a coast (a drainage-less river renders differently); tile (c, r) is
# raw (c + r, c - r).
START = (-7, -7)
MOVES = "crccrrccrrrcccrrcrrcccrrccrr"
HILL_TILES = {
    # Far bank: the river wraps the hill's lower edges (the in-game report).
    (-5, -6),
    # Near bank: the river follows the hill's upper edges.
    (-2, -2),
    # Valley: two hills on each bank, crossed by a bridged road.
    (0, 0), (1, 1), (1, 0), (2, 1),
    # Far-bank chain, its middle hill mined.
    (3, 3), (4, 3), (4, 4),
    # Near-bank hill facing the chain; forest on its other diagonals makes
    # Civ III draw it as a forested hill.
    (5, 4),
}
MINED = {(4, 3)}
FOREST_TILES = {(5, 5), (5, 3), (6, 4)}
ROAD_TILES = {(-1, 0), (0, 0), (1, 0), (2, 0), (3, 0)}
VIEWS = (
    ("overview", 128, (0, 0), (1600, 640)),
    ("far-bank", 256, (-11, 1), (1024, 768)),
    ("near-bank", 256, (-4, 0), (1024, 768)),
    ("valley", 256, (1, 1), (1024, 768)),
    ("chain", 256, (8, 1), (1024, 768)),
)


def nodes():
    c, r = START
    result = [(c, r)]
    for move in MOVES:
        c, r = (c + 1, r) if move == "c" else (c, r + 1)
        result.append((c, r))
    return result


def river_bits() -> dict[tuple[int, int], int]:
    # Civ III records each edge on both incident tiles (raw coordinates).
    bits: dict[tuple[int, int], int] = {}

    def add(c, r, bit):
        raw = (c + r, c - r)
        bits[raw] = bits.get(raw, 0) | bit

    path = nodes()
    for a, b in zip(path, path[1:]):
        c, r = min(a, b)
        if a[1] == b[1]:
            add(c, r, 32)
            add(c, r - 1, 2)
        else:
            add(c, r, 128)
            add(c - 1, r, 8)
    return bits


RIVER = river_bits()


class Cases:
    """The tile-case interface the dispatcher calls for BIQ terrain bits."""

    viewport_size = (1024, 768)

    @staticmethod
    def applies(category: str, case: str) -> bool:
        return category == CATEGORY and case == CASE

    @staticmethod
    def terrain(category, case, dx, dy, base, real):
        tile = ((dx + dy) // 2, (dx - dy) // 2)
        if dx >= 14:
            return COAST, COAST, 0, 0, 0
        base = real = GRASSLAND
        overlays = 0
        if tile in HILL_TILES:
            real = HILLS
        elif tile in FOREST_TILES:
            real = FOREST
        elif abs(dy) in (5, 6):
            # A farm band at the close-ups' edges keeps every view's object
            # witness satisfied without dressing the banks under study.
            overlays |= IRRIGATION
        if tile in MINED:
            overlays |= MINE
        if tile in ROAD_TILES:
            overlays |= ROAD
        return base, real, RIVER.get((dx, dy), 0), 0, overlays

    @staticmethod
    def resources(category, case, centre=(0, 0)) -> str:
        return ""

    @staticmethod
    def viewport_for(case: str, zoom: int) -> tuple[int, int]:
        return Cases.viewport_size


def install() -> None:
    previous = renderer.lab_tile_cases
    renderer.lab_tile_cases = lambda category, case: Cases if Cases.applies(category, case) else \
        previous(category, case)


def windows_path(path: Path) -> str:
    return "..\\..\\" + renderer.relative(path).replace("/", "\\")


def render(label: str, tree: str, views=None, hours=(12,), extra=()) -> None:
    install()
    source = TREES / f"tree-{tree}"
    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    dll, preview = root / "C3XRenderer.dll", root / "native_preview.exe"
    shutil.copy2(source / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    shutil.copy2(source / "Renderer/lab/.cache/native_preview.exe", preview)
    env = {"C3X_RENDERER_SHADER_SOURCE_ROOT": windows_path(source)}
    env.update(item.split("=", 1) for item in extra)
    images = []
    for hour in hours:
        for name, zoom, (dx, dy), size in VIEWS:
            if views and name not in views:
                continue
            Cases.viewport_size = size
            output = root / f"{name}-h{hour:02}"
            for attempt in range(3):
                try:
                    renderer.native_render(CATEGORY, CASE, hour, zoom, output, candidate=dll, preview=preview,
                                           center=(16 + dx, 16 + dy), shared_surface=True, extra_env=env)
                    break
                except ValueError:
                    # Parallels occasionally drops a transport job before the
                    # batch starts; retry only if no preview process started.
                    if attempt == 2 or (output / "process.txt").exists():
                        raise
                    shutil.rmtree(output)
                    time.sleep(10)
            images.append(f"{name}-h{hour:02}")
    receipt = root / "render.json"
    previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
    receipt.write_text(json.dumps(
        {"label": label, "tree": tree, "env": env, "dll_sha256": renderer.checksum(dll),
         "images": sorted(set(previous.get("images", [])) | set(images))}, indent=2) + "\n")
    print(receipt.relative_to(ROOT))


def image_path(label: str, view: str, hour: int) -> Path:
    zoom = next(z for n, z, _, _ in VIEWS if n == view)
    return OUT / label / f"{view}-h{hour:02}" / f"{CASE}-h{hour:02}-z{zoom}.bmp"


def sheet(labels: list[str], hour: int = 12, views=None, crop: float = 1.0) -> None:
    """One PNG per view: the labels side by side (optionally centre-cropped)."""
    from PIL import Image, ImageDraw
    for view, *_ in VIEWS:
        if views and view not in views:
            continue
        paths = [(label, image_path(label, view, hour)) for label in labels]
        paths = [(label, path) for label, path in paths if path.is_file()]
        if not paths:
            continue
        frames = [Image.open(path).convert("RGB") for _, path in paths]
        width, height = frames[0].size
        cw, ch = int(width * crop), int(height * crop)
        box = ((width - cw) // 2, (height - ch) // 2, (width + cw) // 2, (height + ch) // 2)
        canvas = Image.new("RGB", (len(frames) * (cw + 8) + 8, ch + 38), "#1c211f")
        pen = ImageDraw.Draw(canvas)
        for i, ((label, _), frame) in enumerate(zip(paths, frames)):
            canvas.paste(frame.crop(box), (8 + i * (cw + 8), 30))
            pen.text((12 + i * (cw + 8), 8), f"{label} / {view} / {hour:02}:00", fill="white")
        destination = OUT / f"compare-{'-'.join(labels)}-{view}-h{hour:02}.png"
        canvas.save(destination)
        print(destination.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--tree", required=True, help="private source tree NAME (DLL, preview and shaders)")
    run.add_argument("--views", help="comma-separated subset of " + ",".join(n for n, *_ in VIEWS))
    run.add_argument("--hours", default="12")
    run.add_argument("--env", action="append", default=[], help="KEY=VALUE study switch for the renderer")
    compare = sub.add_parser("sheet")
    compare.add_argument("labels", nargs="+")
    compare.add_argument("--views")
    compare.add_argument("--hour", type=int, default=12)
    compare.add_argument("--crop", type=float, default=1.0)
    args = parser.parse_args()
    views = set(args.views.split(",")) if args.views else None
    if args.command == "render":
        render(args.label, args.tree, views, [int(h) for h in args.hours.split(",")], args.env)
    else:
        sheet(args.labels, args.hour, views, args.crop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
