#!/usr/bin/env python3
"""Can players spot farms? Rendered field vs open turf contrast per view.

Samples the centres of irrigated, city-free land tiles (farms) and of plain
grassland/plains tiles with no overlay or forest (turf), from the view's scene
CSV and the preview's exact tile anchors. Needs Pillow and NumPy.

    $C3X_RENDERER_PYTHON Renderer/lab/studies/autumn/field_contrast.py summer v10 v17 --views village
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
from Renderer.lab.studies.autumn.water_tiers import oklab, scene_for


def classes(label, view):
    (cx, cy), w, _, (width, height), *variant = study.VIEWS[view]
    rows = list(csv.reader(scene_for(view).open()))
    map_width = int(rows[0][1])
    cities = set()
    if variant and variant[0] == "village":
        cities = {(x, y) for x, y, _, _ in study.VILLAGE_CITIES}
    image = np.asarray(Image.open(study.OUT / label / view / "preview.png").convert("RGB")).astype(float)
    h = w // 2
    found = {"farm": [], "turf": []}
    for row in rows[1:]:
        x, y, real, overlays = int(row[0]), int(row[1]), int(row[3]), int(row[5])
        if real not in (1, 2) or (x, y) in cities:
            continue
        kind = "farm" if overlays & 8 else "turf" if overlays == 0 else None
        if kind is None:
            continue
        for wrap in (-1, 0, 1):
            px = (x + wrap * map_width - cx) * w // 2 + width // 2
            py = (y - cy) * h // 2 + height // 2
            rx, ry = max(2, w // 6), max(2, h // 6)
            if rx <= px < width - rx and ry <= py < height - ry:
                found[kind].append(oklab(np.median(image[py - ry:py + ry, px - rx:px + rx].reshape(-1, 3), axis=0)))
    return {kind: np.array(values) for kind, values in found.items() if values}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--views", nargs="+", default=["village", "salamanca-close"])
    args = parser.parse_args()
    for view in args.views:
        for label in args.labels:
            if not (study.OUT / label / view / "preview.png").is_file():
                continue
            found = classes(label, view)
            if len(found) < 2:
                print(f"{label:8s} {view:16s} too few samples")
                continue
            farm, turf = found["farm"].mean(0), found["turf"].mean(0)
            hue = lambda c: np.degrees(np.arctan2(c[2], c[1])) % 360
            print(f"{label:8s} {view:16s} farm n{len(found['farm'])} L{farm[0]:.3f} h{hue(farm):.0f} | "
                  f"turf n{len(found['turf'])} L{turf[0]:.3f} h{hue(turf):.0f} | "
                  f"dE {np.linalg.norm(farm - turf) * 100:5.1f}  dL {(farm[0] - turf[0]) * 100:+5.1f}")
        print()


if __name__ == "__main__":
    main()
