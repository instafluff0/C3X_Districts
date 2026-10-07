#!/usr/bin/env python3
"""Before/after study of the production mine on grassland, hills and mountains.

Renders two infrastructure Lab cases through the production renderer with a
pinned copy of the candidate DLL and whatever ImprovementsNormalized/
mine_runtime.bin is current: a gameplay-zoom overview and 256-pixel close-ups.
The cases are supplied to the dispatcher in-process; renderer.py is unchanged.
Outputs are disposable, under Renderer/lab/out/mines/in-game/.

    python3 Renderer/lab/studies/mines/in_game.py render before --dll Renderer/native/build/candidate/C3XRenderer.dll
    python3 Renderer/lab/studies/mines/in_game.py render after

Cases (raw Civ III offsets from the view centre; 0x1 road, 0x2 railroad,
0x4 mine, 0x8 irrigation, as in the farm cases). Each keeps a road, a railroad,
a mine and a farm in view for the infrastructure ownership witness.
- mines-terrain: a mine alone on grassland, hills and a mountain, and the same
  three on a road;
- mines-range: mines among a small mountain range, hills, a wood and roads.
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

OUT = ROOT / "Renderer/lab/out/mines/in-game"
CATEGORY = "infrastructure"
CASES = ("mines-terrain", "mines-range")
ROAD, RAIL, MINE, IRRIGATION = 0x1, 0x2, 0x4, 0x8
GRASSLAND, HILLS, MOUNTAINS, FOREST = 2, 5, 6, 7
TERRAIN_MINES = {(-4, -4): GRASSLAND, (0, -4): HILLS, (4, -4): MOUNTAINS,
                 (-4, 2): GRASSLAND, (0, 2): HILLS, (4, 2): MOUNTAINS}
RANGE = {(2, -2): MOUNTAINS, (3, -1): MOUNTAINS, (4, 0): MOUNTAINS, (5, 1): MOUNTAINS, (6, 2): MOUNTAINS,
         (0, -2): HILLS, (1, -1): HILLS, (2, 0): HILLS, (3, 1): HILLS, (4, 2): HILLS,
         (-1, 1): FOREST, (0, 2): FOREST, (-2, 0): FOREST}
RANGE_MINES = {(3, -1), (5, 1), (1, -1), (2, 0), (4, 2), (-4, -2)}
CLOSE_UPS = {"mines-terrain": ((-4, -4), (0, -4), (4, -4), (0, 2), (4, 2)),
             "mines-range": ((2, -1), (4, 1))}


class Cases:
    """The farm-case interface the dispatcher calls for BIQ object bits."""

    @staticmethod
    def applies(category: str, case: str) -> bool:
        return category == CATEGORY and case in CASES

    @staticmethod
    def terrain(category, case, dx, dy, base, real):
        base = real = GRASSLAND
        overlays = 0
        if (dx, dy) == (6, -8):
            overlays = IRRIGATION
        if dy == 8:
            overlays = ROAD | RAIL
        if case == "mines-terrain":
            if (dx, dy) in TERRAIN_MINES:
                real, overlays = TERRAIN_MINES[(dx, dy)], MINE
            if dy == 2 and -8 <= dx <= 8:
                overlays |= ROAD
        else:
            real = RANGE.get((dx, dy), GRASSLAND)
            if (dx, dy) in RANGE_MINES:
                overlays |= MINE
            if dx + dy == 2 and -6 <= dx <= 2 or (dx, dy) in ((3, -1), (2, 0)):
                overlays |= ROAD
        return base, real, 0, 0, overlays

    @staticmethod
    def resources(category, case, centre=(0, 0)) -> str:
        return ""

    @staticmethod
    def viewport(zoom: int) -> tuple[int, int]:
        return min(8 * zoom, 1024), min(6 * zoom, 768)


def install() -> None:
    previous = renderer.lab_tile_cases
    renderer.lab_tile_cases = lambda category, case: Cases if Cases.applies(category, case) else \
        previous(category, case)


def render(label: str, pinned: Path | None = None) -> None:
    install()
    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    dll = root / "C3XRenderer.dll"
    if pinned:
        shutil.copy2(pinned, dll)
    else:
        renderer.prepare_sources([CATEGORY])
        renderer.ensure_candidate([CATEGORY])
        shutil.copy2(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    runtime = ROOT / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    images = []
    for case in CASES:
        for zoom, (dx, dy) in [(128, (0, 0))] + [(256, centre) for centre in CLOSE_UPS[case]]:
            name = f"{case}-z{zoom}" + (f"-{dx}_{dy}" if zoom == 256 else "")
            renderer.native_render(CATEGORY, case, 12, zoom, root / name, candidate=dll, center=(16 + dx, 16 + dy))
            images.append(name)
    (root / "render.json").write_text(json.dumps(
        {"label": label, "dll_sha256": renderer.checksum(dll), "mine_runtime_sha256": renderer.checksum(runtime),
         "images": images}, indent=2) + "\n")
    print(root / "render.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--dll", type=Path, help="render with this pinned DLL instead of the rebuilt candidate")
    args = parser.parse_args()
    render(args.label, args.dll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
