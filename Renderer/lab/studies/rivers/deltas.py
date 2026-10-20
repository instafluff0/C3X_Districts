#!/usr/bin/env python3
"""River deltas meeting different coasts: before/after through the production renderer.

Four rivers cross farmland to one coast: a bay, a straight stretch, a
headland corner and a one-tile inlet. Each mouth's distributaries should
enter the sea; on the old fixed-angle delta many stopped on the beach (user,
October 19). The case is supplied to the dispatcher in-process; renderer.py
is unchanged. Outputs are disposable, under Renderer/lab/out/river-deltas/.

    python3 Renderer/lab/studies/rivers/deltas.py render before --tree river-delta-base
    python3 Renderer/lab/studies/rivers/deltas.py render after --tree river-delta
    $C3X_RENDERER_PYTHON Renderer/lab/studies/rivers/deltas.py sheet before after

`--tree NAME` renders with a private source tree built by
Renderer/lab/studies/mountains/private_tree.py (DLL, preview and shaders).
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

OUT = ROOT / "Renderer/lab/out/river-deltas"
TREES = ROOT / "Renderer/lab/out/mountains"
CATEGORY = "infrastructure"
CASE = "river-deltas"
IRRIGATION = 0x8
GRASSLAND, COAST, SEA = 2, 11, 12
# Raw source nodes (x, y) on the left; each river alternates column and row
# steps (down-right, up-right) until a node touches water: its mouth.
SOURCES = ((-9, -9), (-9, -3), (-9, 1), (-9, 7))
MOVES = "crcrccrrcrcr" * 3


def coast_edge(dy: int) -> int:
    """The first water column (raw x) on raw row dy."""
    if -11 <= dy <= -6:
        return 1      # a bay
    if -1 <= dy <= 2:
        return 5      # a headland
    if dy == 7:
        return 2      # a one-tile inlet
    return 3          # straight coast


def water(dx: int, dy: int) -> bool:
    return dx >= coast_edge(dy)


def tile_water(c: int, r: int) -> bool:
    return water(c + r, c - r)


def rivers():
    """(river bits by raw tile, mouths as raw node positions)."""
    bits: dict[tuple[int, int], int] = {}
    mouths = []

    def add(c, r, bit):
        raw = (c + r, c - r)
        bits[raw] = bits.get(raw, 0) | bit

    for sx, sy in SOURCES:
        c, r = (sx + sy) // 2, (sx - sy) // 2
        path = [(c, r)]
        for move in MOVES:
            c, r = (c + 1, r) if move == "c" else (c, r + 1)
            path.append((c, r))
            if any(tile_water(c - i, r - j) for i in (0, 1) for j in (0, 1)):
                mouths.append((c + r, c - r))
                break
        for a, b in zip(path, path[1:]):
            c, r = min(a, b)
            if a[1] == b[1]:
                add(c, r, 32)
                add(c, r - 1, 2)
            else:
                add(c, r, 128)
                add(c - 1, r, 8)
    return bits, mouths


RIVER, MOUTHS = rivers()
# The overview at gameplay zoom, then a close-up on each mouth.
VIEWS = (("overview", 128, (0, 0)),) + tuple(
    (f"mouth-{index}", 256, (x - 1, y)) for index, (x, y) in enumerate(MOUTHS))


class Cases:
    """The tile-case interface the dispatcher calls for BIQ terrain bits."""

    @staticmethod
    def applies(category: str, case: str) -> bool:
        return category == CATEGORY and case == CASE

    @staticmethod
    def terrain(category, case, dx, dy, base, real):
        if water(dx, dy):
            kind = COAST if dx < coast_edge(dy) + 3 else SEA
            return kind, kind, 0, 0, 0
        river = RIVER.get((dx, dy), 0)
        # Farmland keeps the infrastructure object witness satisfied in every
        # view; the strip along the coast stays clear for the deltas.
        overlays = IRRIGATION if not river and dx <= coast_edge(dy) - 3 else 0
        return GRASSLAND, GRASSLAND, river, 0, overlays

    @staticmethod
    def resources(category, case, centre=(0, 0)) -> str:
        return ""

    @staticmethod
    def viewport_for(case: str, zoom: int) -> tuple[int, int]:
        return 1024, 768


def install() -> None:
    previous = renderer.lab_tile_cases
    renderer.lab_tile_cases = lambda category, case: Cases if Cases.applies(category, case) else \
        previous(category, case)


def windows_path(path: Path) -> str:
    return "..\\..\\" + renderer.relative(path).replace("/", "\\")


def render(label: str, tree: str, hour: int = 12) -> None:
    install()
    source = TREES / f"tree-{tree}"
    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    dll, preview = root / "C3XRenderer.dll", root / "native_preview.exe"
    shutil.copy2(source / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    shutil.copy2(source / "Renderer/lab/.cache/native_preview.exe", preview)
    env = {"C3X_RENDERER_SHADER_SOURCE_ROOT": windows_path(source)}
    for name, zoom, (dx, dy) in VIEWS:
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
    (root / "render.json").write_text(json.dumps(
        {"label": label, "tree": tree, "dll_sha256": renderer.checksum(dll), "hour": hour,
         "views": [name for name, _, _ in VIEWS]}, indent=2) + "\n")
    print((root / "render.json").relative_to(ROOT))


def sheet(labels: list[str], hour: int = 12) -> None:
    """One PNG per view with the labels side by side, plus all close-ups in one."""
    from PIL import Image, ImageDraw
    out = OUT / ("compare-" + "-".join(labels))
    out.mkdir(parents=True, exist_ok=True)
    for name, zoom, _ in VIEWS:
        images = []
        for label in labels:
            path = OUT / label / f"{name}-h{hour:02}" / f"{CASE}-h{hour:02}-z{zoom}.bmp"
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        w, h = images[0].size
        canvas = Image.new("RGB", (len(images) * (w + 8) - 8, h + 30), "#1d2420")
        pen = ImageDraw.Draw(canvas)
        for index, (label, image) in enumerate(zip(labels, images)):
            canvas.paste(image, (index * (w + 8), 30))
            pen.text((index * (w + 8) + 8, 8), f"{name} / {label}", fill="white")
        canvas.save(out / f"{name}.png")
        print(out / f"{name}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--tree", required=True, help="private source tree NAME (DLL, preview and shaders)")
    run.add_argument("--hour", type=int, default=12)
    compare = sub.add_parser("sheet")
    compare.add_argument("labels", nargs="+")
    compare.add_argument("--hour", type=int, default=12)
    args = parser.parse_args()
    if args.command == "render":
        render(args.label, args.tree, args.hour)
    else:
        sheet(args.labels, args.hour)


if __name__ == "__main__":
    main()
