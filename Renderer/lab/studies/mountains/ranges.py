#!/usr/bin/env python3
"""Mountain shape and range study: before/after through the production renderer.

One Lab case lays out the mountain configurations that matter on a Civ III map:
an isolated peak, straight ranges along both tile diagonals, a range running
horizontally on screen (corner-adjacent tiles), a zigzag of mixed joins, a 3x3
massif and foothills. The case is supplied to the dispatcher in-process;
renderer.py is unchanged. Outputs are disposable, under
Renderer/lab/out/mountains/ranges/.

    python3 Renderer/lab/studies/mountains/ranges.py render before --dll PATH
    python3 Renderer/lab/studies/mountains/ranges.py render after
    python3 Renderer/lab/studies/mountains/ranges.py sheet before after

Run with a Python that has Pillow for `sheet`.
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

OUT = ROOT / "Renderer/lab/out/mountains/ranges"
CATEGORY = "mountains"
CASE = "ranges"
PLAINS, GRASSLAND, HILLS, MOUNTAINS, FOREST = 1, 2, 5, 6, 7
# Raw Civ III offsets from the view centre (x+y even). (x±1, y±1) share an
# edge; (x±2, y) and (x, y±2) share only a corner.
LAYOUT = {
    # Isolated peak.
    (-8, -6): MOUNTAINS,
    # Edge-joined range running down-right on screen.
    **{(dx, dy): MOUNTAINS for dx, dy in ((-5, -11), (-4, -10), (-3, -9), (-2, -8), (-1, -7), (0, -6))},
    # Corner-joined range running horizontally on screen.
    **{(dx, 0): MOUNTAINS for dx in (-12, -10, -8, -6)},
    # Edge-joined range running up-right on screen.
    **{(dx, dy): MOUNTAINS for dx, dy in ((3, 3), (4, 2), (5, 1), (6, 0), (7, -1), (8, -2))},
    # 3x3 massif (a square block of natural tiles).
    **{(8 + a + b, 8 + a - b): MOUNTAINS for a in range(3) for b in range(3)},
    # Zigzag with mixed edge and corner joins.
    **{(dx, dy): MOUNTAINS for dx, dy in ((-8, 8), (-7, 9), (-5, 9), (-4, 10), (-2, 10))},
    # A second isolated peak, snow-capped.
    (-4, -2): MOUNTAINS,
    # Foothills and a wood beside the ranges.
    (-3, -7): HILLS, (-1, -5): HILLS, (1, -7): HILLS, (2, 4): HILLS, (4, 4): HILLS,
    (-6, 10): HILLS, (-7, 1): HILLS, (2, -6): FOREST, (3, -5): FOREST,
}
# Civ III's snow-capped flag (BIQ bonus 0x10, game Tile field_30 0x100000):
# the down-right range, the massif, the second isolated peak and the upper
# half of the up-right range. Older candidates ignore it and snow every peak.
SNOW = 0x10
SNOWY = {(-5, -11), (-4, -10), (-3, -9), (-2, -8), (-1, -7), (0, -6), (-4, -2), (6, 0), (7, -1), (8, -2),
         *((8 + a + b, 8 + a - b) for a in range(3) for b in range(3))}
VIEWS = (
    ("overview", 128, (0, 0), (1600, 1000)),
    ("isolated", 256, (-8, -6), (1024, 768)),
    ("isolated-snow", 256, (-4, -2), (1024, 768)),
    ("down-right", 256, (-3, -9), (1024, 768)),
    ("horizontal", 256, (-9, 1), (1024, 768)),
    ("up-right", 256, (5, 1), (1024, 768)),
    ("massif", 256, (10, 8), (1024, 768)),
    ("zigzag", 256, (-5, 9), (1024, 768)),
)


VOLCANO, COAST, SEA = 10, 11, 12
# Volcano layout (`--layout volcanoes`): an isolated volcano, one inside an
# edge-joined range, one ringed by forest, a coastal one, an adjacent pair and
# one among hills. Active renders mark every volcano active.
VOLCANO_LAYOUT = {
    (-8, -6): VOLCANO,
    **{(dx, dy): MOUNTAINS for dx, dy in ((-5, -11), (-4, -10), (-2, -8), (-1, -7))}, (-3, -9): VOLCANO,
    (4, -6): VOLCANO, **{(dx, dy): FOREST for dx, dy in ((3, -7), (5, -7), (3, -5), (5, -5), (2, -6), (6, -6))},
    (9, 1): VOLCANO, **{(dx, dy): COAST for dx in range(11, 16) for dy in range(-16, 16) if (dx + dy) % 2 == 0},
    (-7, 7): VOLCANO, (-6, 8): VOLCANO,
    (2, 8): VOLCANO, **{(dx, dy): HILLS for dx, dy in ((1, 7), (3, 7), (1, 9), (3, 9))},
}
VOLCANO_VIEWS = (
    ("overview", 128, (0, 0), (1600, 1000)),
    ("isolated", 256, (-8, -6), (1024, 768)),
    ("in-range", 256, (-3, -9), (1024, 768)),
    ("forest", 256, (4, -6), (1024, 768)),
    ("coastal", 256, (9, 1), (1024, 768)),
    ("pair", 256, (-7, 7), (1024, 768)),
    ("hills", 256, (2, 8), (1024, 768)),
)


def use_layout(name: str) -> None:
    """Select the study layout; volcanoes render under their own output root."""
    global LAYOUT, SNOWY, VIEWS, OUT
    if name == "volcanoes":
        LAYOUT, SNOWY, VIEWS = VOLCANO_LAYOUT, set(), VOLCANO_VIEWS
        OUT = ROOT / "Renderer/lab/out/mountains/volcanoes"


# Real map context: the unchanged test.biq terrain (exported by
# Renderer/sandbox/export_biq.js) at its two densest mountain groups. Centres
# are absolute raw tiles. Its 72 mountains carry no snow-capped bit.
BIQ = ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"
BIQ_VIEWS = (
    ("biq-range", 128, (64, 56), (1600, 1000)),
    ("biq-ridge", 128, (57, 35), (1600, 1000)),
    ("biq-range-close", 256, (64, 56), (1024, 768)),
)


class Cases:
    """The tile-case interface the dispatcher calls for BIQ terrain bits."""

    viewport_size = (1024, 768)

    @staticmethod
    def applies(category: str, case: str) -> bool:
        return category == CATEGORY and case in (CASE, "biq")

    @staticmethod
    def terrain(category, case, dx, dy, base, real):
        base = PLAINS if dx + dy > 10 else GRASSLAND
        real = LAYOUT.get((dx, dy), base)
        if real >= COAST:
            base = real
        return base, real, 0, SNOW if (dx, dy) in SNOWY and real == MOUNTAINS else 0, 0

    @staticmethod
    def resources(category, case, centre=(0, 0)) -> str:
        return ""

    @staticmethod
    def viewport_for(case: str, zoom: int) -> tuple[int, int]:
        return Cases.viewport_size


def install_biq() -> None:
    scene = OUT / "biq/test-biq.csv"
    if not scene.is_file():
        import subprocess
        scene.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"), str(BIQ), str(scene)], cwd=ROOT, check=True)
    original = renderer.scene

    def biq_scene(category, case, destination, *, world_size=32):
        if case != "biq":
            return original(category, case, destination, world_size=world_size)
        shutil.copyfile(scene, destination)
    renderer.scene = biq_scene


def install() -> None:
    previous = renderer.lab_tile_cases
    renderer.lab_tile_cases = lambda category, case: Cases if Cases.applies(category, case) else \
        previous(category, case)


def shader_root(name: str) -> str:
    """A complete private runtime shader tree under the study output, so
    tuning never changes the generated shaders other sessions render with."""
    return "..\\..\\" + renderer.relative(OUT / name).replace("/", "\\")


# Active states pin the effect clock so plume frames are repeatable.
STATES = {"dormant": {}, "smoldering": {"C3X_RENDERER_PREVIEW_ACTIVE_VOLCANO": "1", "C3X_LAB_EFFECT_TIME": "1.7"},
          "erupting": {"C3X_RENDERER_PREVIEW_ERUPTING_VOLCANO": "1", "C3X_LAB_EFFECT_TIME": "1.7"}}


def render(label: str, pinned: Path | None = None, views=None, hour: int = 12, root_name: str = "",
           tree: str = "", biq: bool = False, state: str = "dormant") -> None:
    """tree: a private source tree (private_tree.py) supplying the DLL,
    preview tool and shaders together."""
    install()
    env = {"C3X_RENDERER_SHADER_SOURCE_ROOT": shader_root(root_name)} if root_name else None
    preview = None
    if tree:
        source = OUT.parent / f"tree-{tree}"
        pinned = source / "Renderer/native/build/candidate/C3XRenderer.dll"
        preview = root_preview = OUT / label / "native_preview.exe"
        root_preview.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "Renderer/lab/.cache/native_preview.exe", root_preview)
        env = {"C3X_RENDERER_SHADER_SOURCE_ROOT": shader_root(root_name) if root_name else
               "..\\..\\" + renderer.relative(source).replace("/", "\\")}
    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    dll = root / "C3XRenderer.dll"
    if pinned:
        if pinned.resolve() != dll.resolve():
            shutil.copy2(pinned, dll)
    else:
        renderer.prepare_sources([CATEGORY])
        renderer.ensure_candidate([CATEGORY])
        shutil.copy2(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    images = []
    if biq:
        install_biq()
    for name, zoom, (dx, dy), size in (BIQ_VIEWS if biq else VIEWS):
        if views and name not in views:
            continue
        case, centre = ("biq", (dx, dy)) if biq else (CASE, (16 + dx, 16 + dy))
        view_env = {**(env or {}), **STATES[state]} or None
        Cases.viewport_size = size
        output = root / f"{name}-h{hour:02}"
        for attempt in range(3):
            try:
                renderer.native_render(CATEGORY, case, hour, zoom, output, candidate=dll,
                                       center=centre, extra_env=view_env, preview=preview)
                break
            except ValueError:
                # Parallels occasionally drops a transport job (PrlJob_GetResult)
                # before the batch starts. Retry only when no preview process
                # was ever recorded for this invocation.
                if attempt == 2 or (output / "process.txt").exists():
                    raise
                shutil.rmtree(output)
                time.sleep(10)
        images.append(f"{name}-h{hour:02}")
    receipt = root / "render.json"
    previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
    (receipt).write_text(json.dumps(
        {"label": label, "dll_sha256": renderer.checksum(dll), "shader_root": root_name, "tree": tree,
         "images": sorted(set(previous.get("images", [])) | set(images))}, indent=2) + "\n")
    print(receipt.relative_to(ROOT))


def image_path(label: str, view: str, hour: int) -> Path:
    zoom, case = next((z, "biq" if n.startswith("biq") else CASE) for n, z, _, _ in VIEWS + BIQ_VIEWS if n == view)
    return OUT / label / f"{view}-h{hour:02}" / f"{case}-h{hour:02}-z{zoom}.bmp"


def sheet(labels: list[str], hour: int = 12, views=None) -> None:
    from PIL import Image, ImageDraw
    names = [n for n, *_ in VIEWS if not views or n in views]
    for view in names:
        paths = [image_path(label, view, hour) for label in labels]
        paths = [(label, p) for label, p in zip(labels, paths) if p.is_file()]
        if not paths:
            continue
        frames = [Image.open(p).convert("RGB") for _, p in paths]
        width, height = frames[0].size
        canvas = Image.new("RGB", (len(frames) * (width + 8) + 8, height + 38), "#1c211f")
        pen = ImageDraw.Draw(canvas)
        for i, ((label, _), frame) in enumerate(zip(paths, frames)):
            canvas.paste(frame, (8 + i * (width + 8), 30))
            pen.text((12 + i * (width + 8), 8), f"{label} / {view} / {hour:02}:00", fill="white")
        destination = OUT / f"compare-{'-'.join(labels)}-{view}-h{hour:02}.png"
        canvas.save(destination)
        print(destination.relative_to(ROOT))


def review(labels: list[str], hours: list[int], views=None, crop: float = 1.0) -> None:
    """One JPEG per view: a row per hour, a column per label (centre crop)."""
    from PIL import Image, ImageDraw
    names = [n for n, *_ in VIEWS + BIQ_VIEWS if not views or n in views]
    for view in names:
        cells = [[image_path(label, view, hour) for label in labels] for hour in hours]
        if not all(p.is_file() for row in cells for p in row):
            print("skipping", view, "(missing renders)")
            continue
        frames = [[Image.open(p).convert("RGB") for p in row] for row in cells]
        width, height = frames[0][0].size
        cw, ch = int(width * crop), int(height * crop)
        box = ((width - cw) // 2, (height - ch) // 2, (width + cw) // 2, (height + ch) // 2)
        canvas = Image.new("RGB", (len(labels) * (cw + 6) + 6, len(hours) * (ch + 28) + 6), "#1c211f")
        pen = ImageDraw.Draw(canvas)
        for r, (hour, row) in enumerate(zip(hours, frames)):
            for c, (label, frame) in enumerate(zip(labels, row)):
                x, y = 6 + c * (cw + 6), 6 + r * (ch + 28)
                canvas.paste(frame.crop(box), (x, y + 22))
                pen.text((x + 4, y + 4), f"{label} / {view} / {hour:02}:00", fill="white")
        destination = OUT / f"review-{'-'.join(labels)}-{view}.jpg"
        canvas.save(destination, quality=90)
        print(destination.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--dll", type=Path, help="render with this pinned DLL instead of the rebuilt candidate")
    run.add_argument("--views", help="comma-separated subset of " + ",".join(n for n, *_ in VIEWS))
    run.add_argument("--hour", type=int, default=12)
    run.add_argument("--shader-root", default="", help="private shader tree folder under the study output")
    run.add_argument("--tree", default="", help="private source tree NAME (DLL, preview and shaders)")
    run.add_argument("--biq", action="store_true", help="render the test.biq context views instead")
    run.add_argument("--state", choices=tuple(STATES), default="dormant", help="volcano activity for every volcano")
    looks = sub.add_parser("review")
    looks.add_argument("labels", nargs="+")
    looks.add_argument("--hours", default="9,12,15")
    looks.add_argument("--views")
    looks.add_argument("--crop", type=float, default=1.0)
    compare = sub.add_parser("sheet")
    compare.add_argument("labels", nargs="+")
    compare.add_argument("--views")
    compare.add_argument("--hour", type=int, default=12)
    parser.add_argument("--layout", choices=("ranges", "volcanoes"), default="ranges")
    args = parser.parse_args()
    use_layout(args.layout)
    views = set(args.views.split(",")) if args.views else None
    if args.command == "review":
        review(args.labels, [int(h) for h in args.hours.split(",")], views, args.crop)
        return 0
    if args.command == "render":
        render(args.label, args.dll, views, args.hour, args.shader_root, args.tree, args.biq, args.state)
    else:
        sheet(args.labels, args.hour, views)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
