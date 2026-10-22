#!/usr/bin/env python3
"""Lab prototype: a gentle flurry of small snowflakes over mountain tiles.

    python3 Renderer/lab/studies/winter/flurry.py LABEL VIEW [--seconds 6] [--crop l t r b]

Composites animated flakes over a still Winter render (Renderer/lab/out/winter/
LABEL/VIEW/preview.png) and writes a looping MP4 under out/winter/flurry/.

The flakes are computed the way a GPU effect would: each mountain tile owns a
column standing on its diamond (the preview's own tile anchors and 2:1
projection). Three parallax layers of cells fall through the column; every
cell holds at most one flake at a hashed position, with a hashed sway phase.
Everything is a pure function of (tile, pixel, time), so the same equations
port to one pixel shader over one quad per mountain tile. The column fades in
at its top and out near the ground; terrain occlusion is not modelled.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.studies.winter.study import OUT, VIEWS, load, salamanca_scene, village_scene

FPS = 30
# Per layer at tile width 128 (scaled with zoom): cell size px, fall px/s,
# flake radius px, opacity, share of cells holding a flake, sway px.
LAYERS = [(22.0, 20.0, 1.25, 0.90, 0.55, 3.0),
          (15.0, 14.0, 0.95, 0.70, 0.55, 2.2),
          (10.0, 9.0, 0.70, 0.50, 0.50, 1.5)]
COLUMN = 2.1          # column height above the diamond, in tile heights
SWAY_PERIOD = 2.6     # seconds
FLAKE = np.array([0.97, 0.98, 1.0])


def hash2(ix, iy, seed):
    """Deterministic [0,1) hashes of integer cells (GPU-friendly arithmetic)."""
    h = np.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453
    return h - np.floor(h)


def mountain_tiles(view):
    center, tw, _, (w, h), *variant = VIEWS[view]
    kind = variant[0] if variant else "biq"
    base = OUT / "test-biq.csv"
    scene = (village_scene(base)[0] if kind == "village" else salamanca_scene()[0] if kind == "salamanca" else base)
    rows = list(csv.reader(scene.open()))[1:]
    th = tw // 2
    shift_x, shift_y = w // 2 - tw // 2 - center[0] * tw // 2, h // 2 - th // 2 - center[1] * th // 2
    tiles = []
    for r in rows:
        x, y, real = int(r[0]), int(r[1]), int(r[3])
        if real != 6:
            continue
        cx, cy = x * tw // 2 + shift_x + tw / 2, y * th // 2 + shift_y + th / 2
        if -tw < cx < w + tw and -th < cy < h + th * (1 + COLUMN):
            tiles.append((cx, cy, x * 7.0 + y * 13.0))
    return tiles, tw


def flurry(shape, tiles, tw, t):
    """Flake opacity over the frame at time t (seconds)."""
    h, w = shape
    th = tw / 2
    zoom = tw / 128.0
    alpha = np.zeros((h, w))
    top = th * COLUMN
    for cx, cy, seed in tiles:
        x0, x1 = int(max(cx - tw / 2, 0)), int(min(cx + tw / 2, w))
        y0, y1 = int(max(cy - th / 2 - top, 0)), int(min(cy + th / 2, h))
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(float)
        dx = (xs - cx) / (tw / 2)
        edge = th / 2 * (1 - np.abs(dx))               # diamond half-height at this column
        bottom, upper = cy + edge, cy - edge - top      # column floor (front edge) and ceiling
        inside = (np.abs(dx) < 1) & (ys < bottom) & (ys > upper)
        # Fade in under the ceiling, out over the last part of the fall.
        fade = np.clip((ys - upper) / (0.35 * top), 0, 1) * np.clip((bottom - ys) / (0.6 * th), 0, 1)
        fade *= np.clip((1 - np.abs(dx)) / 0.25, 0, 1)
        column = np.zeros_like(xs)
        for layer, (cell, speed, radius, opacity, share, sway) in enumerate(LAYERS):
            cell, speed, radius, sway = cell * zoom, speed * zoom, radius * zoom, sway * zoom
            lx = xs - cx
            ly = ys - (cy + edge.max()) - speed * t     # cells move down the screen with time
            col = np.floor(lx / cell)
            phase = hash2(col, layer, seed + 3.1)
            lx = lx + sway * np.sin(2 * np.pi * (t / SWAY_PERIOD + phase))
            ix, iy = np.floor(lx / cell), np.floor(ly / cell)
            fx, fy = lx / cell - ix, ly / cell - iy
            px = 0.2 + 0.6 * hash2(ix, iy, seed + layer)
            py = 0.2 + 0.6 * hash2(iy, ix, seed + layer + 9.3)
            present = hash2(ix + 17.0, iy, seed + layer * 5.0) < share
            d = np.hypot((fx - px) * cell, (fy - py) * cell)
            column = np.maximum(column, present * opacity * np.clip((radius + 0.6 - d) / 1.0, 0, 1))
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], column * fade * inside)
    return alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label")
    parser.add_argument("view")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--crop", nargs=4, type=float, help="fractions l t r b")
    parser.add_argument("--tint", nargs=3, type=float, help="flake colour (night: moonlit blue)")
    args = parser.parse_args()
    background = load(OUT / args.label / args.view / "preview.png").astype(float) / 255
    tiles, tw = mountain_tiles(args.view)
    colour = np.array(args.tint) if args.tint else FLAKE
    h, w = background.shape[:2]
    l, t_, r, b = args.crop or (0, 0, 1, 1)
    x0, x1, y0, y1 = int(w * l) // 2 * 2, int(w * r) // 2 * 2, int(h * t_) // 2 * 2, int(h * b) // 2 * 2
    target = OUT / "flurry" / f"{args.label}-{args.view}{'-crop' if args.crop else ''}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{x1 - x0}x{y1 - y0}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(target)],
                               stdin=subprocess.PIPE)
    for frame in range(int(args.seconds * FPS)):
        a = flurry((h, w), tiles, tw, frame / FPS)[..., None]
        image = background * (1 - a) + colour * a
        encoder.stdin.write((np.clip(image[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8).tobytes())
    encoder.stdin.close()
    encoder.wait()
    (target.with_suffix(".json")).write_text(json.dumps({"tiles": len(tiles), "tile_width": tw,
                                                         "layers": LAYERS, "column": COLUMN}) + "\n")
    print(target, len(tiles), "mountain tiles")


if __name__ == "__main__":
    main()
