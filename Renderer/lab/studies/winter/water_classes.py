#!/usr/bin/env python3
"""Can players tell Coast, Sea and Ocean apart? Per-class water colour in a render.

    python3 Renderer/lab/studies/winter/water_classes.py summer v9 [--view concept]

Samples the central part of every Coast (11), Sea (12) and Ocean (13) tile's
screen diamond (the preview's own tile anchors), skipping tiles that touch
land (Coast: tile centre only) so beaches do not blur the means. Prints per-class OKLab means and the separation between classes
(OKLab distance x100; ~2 is a just-noticeable difference, ~5+ is clear).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.studies.winter.study import OUT, VIEWS, load

NAMES = {11: "coast", 12: "sea", 13: "ocean"}
# --land: the four open land types (forests, hills etc. are other real types).
LAND = {0: "desert", 1: "plains", 2: "grassland", 3: "tundra"}


def oklab(rgb):
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lms = np.cbrt(np.stack([c @ [0.4122214708, 0.5363325363, 0.0514459929],
                            c @ [0.2119034982, 0.6806995451, 0.1073969566],
                            c @ [0.0883024619, 0.2817188376, 0.6299787005]], -1))
    return np.stack([lms @ [0.2104542553, 0.7936177850, -0.0040720468],
                     lms @ [1.9779984951, -2.4285922050, 0.4505937099],
                     lms @ [0.0259040371, 0.7827717662, -0.8086757660]], -1)


def class_means(image, view, names=NAMES):
    (cx, cy), tw, _, (w, h), *_ = VIEWS[view]
    th = tw // 2
    rows = list(csv.reader((OUT / "test-biq.csv").open()))[1:]
    real = {(int(r[0]), int(r[1])): int(r[3]) for r in rows}
    shift_x, shift_y = w // 2 - tw // 2 - cx * tw // 2, h // 2 - th // 2 - cy * th // 2
    with np.errstate(all="ignore"):  # numpy 2.0 + Accelerate warns spuriously in matmul
        lab = oklab(image.astype(float))
    samples = {k: [] for k in names}
    ys, xs = np.mgrid[0:h, 0:w]
    for (x, y), kind in real.items():
        if kind not in names:
            continue
        # Sea and Ocean tiles away from land; Coast always borders land, so
        # only its centre is sampled (beaches lie near the tile edges).
        neighbours = [real.get((x + dx, y + dy)) for dx, dy in
                      ((1, 1), (-1, 1), (1, -1), (-1, -1), (2, 0), (-2, 0), (0, 2), (0, -2))]
        if names is NAMES and kind != 11 and any(n not in NAMES for n in neighbours):
            continue
        # Land: tiles mostly surrounded by their own type (6 of 8), centre
        # only, so beaches and neighbouring types barely blur the means.
        if names is not NAMES and sum(n == kind for n in neighbours) < 6:
            continue
        ax, ay = x * tw // 2 + shift_x + tw / 2, y * th // 2 + shift_y + th / 2
        if not (tw < ax < w - tw and th < ay < h - th):
            continue
        x0, x1, y0, y1 = int(ax - tw / 2), int(ax + tw / 2), int(ay - th / 2), int(ay + th / 2)
        sub_x, sub_y = xs[y0:y1, x0:x1], ys[y0:y1, x0:x1]
        inside = np.abs(sub_x - ax) / (tw / 2) + np.abs(sub_y - ay) / (th / 2) < (0.35 if kind == 11 or names is not NAMES else 0.6)
        samples[kind].append(lab[y0:y1, x0:x1][inside].mean(0))
    return {k: np.array(v) for k, v in samples.items() if v}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--view", default="concept")
    parser.add_argument("--land", action="store_true", help="desert, plains, grassland and tundra instead of water")
    args = parser.parse_args()
    names = LAND if args.land else NAMES
    for label in args.labels:
        means = class_means(load(OUT / label / args.view / "preview.png"), args.view, names)
        print(f"{label}/{args.view}")
        centre = {}
        for kind, values in means.items():
            m = values.mean(0)
            centre[kind] = m
            spread = np.linalg.norm(values - m, axis=1).mean() * 100
            print(f"  {names[kind]:9s} n={len(values):3d}  L {m[0]:.3f}  C {np.hypot(m[1], m[2]):.3f}  "
                  f"h {np.degrees(np.arctan2(m[2], m[1])) % 360:5.1f}  tile spread {spread:.1f}")
        keys = sorted(centre)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                print(f"  {names[a]}-{names[b]} separation {np.linalg.norm(centre[a] - centre[b]) * 100:.1f}")


if __name__ == "__main__":
    main()
