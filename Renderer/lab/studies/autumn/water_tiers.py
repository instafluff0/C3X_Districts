#!/usr/bin/env python3
"""Can players tell coast, sea and ocean apart? Measures rendered study views.

Samples each water tile's centre (from the view's scene CSV and the preview's
exact tile anchors), groups by Civ III terrain (11 coast, 12 sea, 13 ocean)
and reports OKLab class means, spread and the separation between tiers.
Needs Pillow and NumPy (C3X_RENDERER_PYTHON).

    $C3X_RENDERER_PYTHON Renderer/lab/studies/autumn/water_tiers.py summer v10 --views concept plains
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.studies.autumn import study

TIERS = {11: "coast", 12: "sea", 13: "ocean"}


def oklab(rgb):
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return np.cbrt(np.maximum(c @ m1.T, 1e-9)) @ m2.T


def scene_for(view):
    variant = study.VIEWS[view][4:]
    if not variant:
        return study.OUT / "test-biq.csv"
    return study.OUT / ("scenes/village/scene.csv" if variant[0] == "village" else "scenes/salamanca/scene.csv")


def samples(label, view):
    (cx, cy), w, _, (width, height), *_ = study.VIEWS[view]
    rows = list(csv.reader(scene_for(view).open()))
    map_width = int(rows[0][1])
    image = np.asarray(Image.open(study.OUT / label / view / "preview.png").convert("RGB")).astype(float)
    h = w // 2
    found = {tier: [] for tier in TIERS}
    for row in rows[1:]:
        x, y, real = int(row[0]), int(row[1]), int(row[3])
        if real not in TIERS:
            continue
        for wrap in (-1, 0, 1):
            px = (x + wrap * map_width - cx) * w // 2 + width // 2
            py = (y - cy) * h // 2 + height // 2
            rx, ry = max(2, w // 8), max(2, h // 8)
            if rx <= px < width - rx and ry <= py < height - ry:
                patch = image[py - ry:py + ry, px - rx:px + rx].reshape(-1, 3)
                found[real].append(oklab(np.median(patch, axis=0)))
    return {tier: np.array(values) for tier, values in found.items() if values}


def report(label, view):
    tiers = samples(label, view)
    means = {tier: values.mean(0) for tier, values in tiers.items()}
    line = [f"{label:8s} {view:16s}"]
    for tier, values in tiers.items():
        mean, chroma = means[tier], np.hypot(*means[tier][1:])
        hue = np.degrees(np.arctan2(means[tier][2], means[tier][1])) % 360
        line.append(f"{TIERS[tier]} n{len(values)} L{mean[0]:.3f} C{chroma:.3f} h{hue:.0f} sd{values[:, 0].std():.3f}")
    print(" | ".join(line))
    for a, b in ((11, 12), (12, 13), (11, 13)):
        if a in tiers and b in tiers:
            delta = np.linalg.norm(means[a] - means[b]) * 100
            spread = np.sqrt((tiers[a].var(0).sum() + tiers[b].var(0).sum()) / 2) * 100
            dl = (means[a][0] - means[b][0]) * 100
            print(f"   {TIERS[a]}-{TIERS[b]}: dE {delta:5.1f}  (lightness {dl:+5.1f})  within-tier spread {spread:4.1f}  ratio {delta / max(spread, 0.1):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--views", nargs="+", default=["concept", "plains", "salamanca"])
    args = parser.parse_args()
    for view in args.views:
        for label in args.labels:
            if (study.OUT / label / view / "preview.png").is_file():
                report(label, view)
        print()


if __name__ == "__main__":
    main()
