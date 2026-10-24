#!/usr/bin/env python3
"""Top-down footprints of each recommended wonder on its Civ III tile (Lab check).

Places every finished piece exactly as kits.bake does and measures it against
the square Civ III tile (u, v in 0..1): how much raised geometry and flat ground
falls outside, and how hexagonal the raised footprint is (its convex hull's share
of its bounding box: a square footprint is near 1.0, a Civ VI hex near 0.75).
Writes Renderer/lab/out/wonders/footprints.png and footprints.json.

    python3 Renderer/lab/studies/wonders/footprints.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.wonders import kits, plan

SIZE = 180


def placed(part: dict, facing: float):
    """(u, v, z, raised) of every vertex of a finished part, as bake places it."""
    finished, cx, cy, scale, _low, _high = kits.layout(part)
    pieces = kits.kept(finished)
    c, s = math.cos(facing), math.sin(facing)
    du, dv = part.get("at", (0.0, 0.0))
    for _path, _material, vertices, indices in pieces:
        for i in range(0, len(indices) - 2, 3):
            tri = []
            for j in indices[i:i + 3]:
                (x, y, z), _n, _uv = vertices[j]
                x, y = x - cx, y - cy
                tri.append((.5 + du + (x * c - y * s) * scale, .5 + dv + (x * s + y * c) * scale, z * scale))
            yield tri


def hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return points
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - \
                    (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out
    return half(points)[:-1] + half(points[::-1])[:-1]


def area(poly):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))) / 2


def main() -> int:
    import numpy as np
    from Renderer.lab.studies.resources import audit
    report, tiles = {}, []
    for _group, _title, wonders in plan.GROUPS:
        for wonder in wonders:
            pick = plan.PICKS[wonder][0]
            facing = plan.FACING.get(wonder, plan.FACE_SW)
            tris = [t for part in pick["parts"] for t in placed(part, facing)]
            raised = [p for t in tris for p in t if p[2] > .02]
            flat = [p for t in tris for p in t if p[2] <= .02]
            out = lambda pts: sum(not (0 <= u <= 1 and 0 <= v <= 1) for u, v, _ in pts) / max(len(pts), 1)
            over = max([max(-u, u - 1, -v, v - 1, 0) for u, v, _ in raised + flat] or [0])
            h = hull([(round(u, 3), round(v, 3)) for u, v, _ in raised])
            box = (max(p[0] for p in h) - min(p[0] for p in h)) * (max(p[1] for p in h) - min(p[1] for p in h)) if h else 1
            report[wonder] = {"pick": pick["label"], "raised_outside": round(out(raised), 3),
                              "flat_outside": round(out(flat), 3), "max_overhang_tiles": round(over, 3),
                              "hull_box_ratio": round(area(h) / box, 3) if h and box else None}
            # Top-down tile: square tile outline, flat ground light, raised dark (by height).
            img = np.full((SIZE, SIZE, 3), 236, np.uint8)
            m = SIZE * .25                                   # margin: the tile spans the middle half
            def px(u, v):
                return int(m + u * (SIZE - 2 * m)), int(m + v * (SIZE - 2 * m))
            for t in sorted(tris, key=lambda t: max(p[2] for p in t)):
                zmax = max(p[2] for p in t)
                colour = (200, 190, 160) if zmax <= .02 else tuple(int(v) for v in np.clip(
                    np.array((120, 70, 40)) - zmax * 60, 0, 255))
                xs, ys = zip(*(px(u, v) for u, v, _ in t))
                for a in np.linspace(0, 1, 4):
                    for b in np.linspace(0, 1 - a, 4):
                        x = int(xs[0] + a * (xs[1] - xs[0]) + b * (xs[2] - xs[0]))
                        y = int(ys[0] + a * (ys[1] - ys[0]) + b * (ys[2] - ys[0]))
                        if 0 <= x < SIZE and 0 <= y < SIZE:
                            img[y, x] = colour
            lo, hi = int(m), int(SIZE - m)
            img[lo:hi + 1, [lo, hi]] = (220, 40, 40)
            img[[lo, hi], lo:hi + 1] = (220, 40, 40)
            tiles.append(img)
            print(wonder, report[wonder], flush=True)
    cols = 6
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * SIZE, cols * SIZE, 3), 255, np.uint8)
    for k, img in enumerate(tiles):
        sheet[(k // cols) * SIZE:(k // cols + 1) * SIZE, (k % cols) * SIZE:(k % cols + 1) * SIZE] = img
    audit.write_png(kits.OUT / "footprints.png", sheet.shape[1], sheet.shape[0], [r.tobytes() for r in sheet])
    (kits.OUT / "footprints.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
