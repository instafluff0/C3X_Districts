#!/usr/bin/env python3
"""Measure Civ III unit sprites against the production 3D unit pack.

Both sides are measured at normal zoom (128-pixel tiles), idle pose, all eight
directions (the median is reported). Civ III: the DEFAULT FLC's first frame per
direction; body = palette indices below 224 (shadow, smoke and transparency
excluded); civ colour = indices 0..63, which ntpNN.pcx replaces. Ours: the
bound idle payload's first palette, skinned on the CPU and projected exactly as
the live vertex shader does (`sandbox/direct_units.h`), with the pack's scale
and ground offset. Nothing is rendered or written outside lab/out.

    python3 Renderer/lab/studies/unit_readability/audit.py [--pack UnitAnimationFidelity]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import civ3_flc  # noqa: E402
sys.path.insert(0, str(HERE.parents[3]))
from Renderer.native.environment_refresh import prepare_units as units  # noqa: E402
from Renderer.tools.asset_compiler import civ3_unit_sprites as sprites  # noqa: E402

ROOT = HERE.parents[3]                      # C3X checkout
GAME = ROOT.parents[1]                      # Civilization III Complete
OUT = ROOT / "Renderer/lab/out/unit-readability"
PACKS = ROOT / "Renderer/packs"


def pedia_names() -> dict[str, str]:
    return sprites.pedia_names(sprites.game_roots(GAME))


def default_flc(name: str) -> Path | None:
    return sprites.default_flc(sprites.game_roots(GAME), name)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114


def civ3_measure(path: Path, civ: np.ndarray) -> dict:
    flc = civ3_flc.read(path)
    rows = []
    for direction in range(min(8, flc.directions)):
        frame = flc.direction_frame(direction)
        body = frame < 224
        if not body.any():
            continue
        ys, xs = np.nonzero(body)
        table = flc.palette.copy()
        table[:64] = civ[:64]
        rgb = table[frame[body]].astype(float)
        rows.append({"height": int(np.ptp(ys)) + 1, "width": int(np.ptp(xs)) + 1,
                     "area": int(body.sum()),
                     "foot": int(ys.max()) + flc.y_offset - flc.full_height // 2,
                     "civ": float((frame[body] < 64).mean()),
                     "luma": float(luminance(rgb).mean()),
                     "luma90": float(np.percentile(luminance(rgb), 90))})
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def ours_measure(pack: Path, unit: dict) -> dict:
    idle = unit["idle"]
    parts = [units.idle_geometry((pack / idle[f"part{i}"]["mesh"]).read_bytes()) for i in range(idle["part_count"])]
    base, triangles = 0, []
    for points, tris in parts:
        triangles.append(tris + base); base += len(points)
    return units.silhouette(np.concatenate([p for p, _ in parts]), np.concatenate(triangles),
                            unit["scale"], unit["offset_z"], unit.get("yaw_offset", 225.0))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", default="UnitAnimationFidelity")
    parser.add_argument("--civ", type=int, default=1, help="ntpNN civ palette for luminance")
    args = parser.parse_args(argv)
    pack = PACKS / args.pack
    bindings = json.loads((pack / "bindings.json").read_text())
    names = pedia_names()
    civ = civ3_flc.read_pcx_palette(GAME / "Art/Units/Palettes" / f"ntp{args.civ:02d}.pcx")
    results = []
    for key, unit in sorted(bindings.items()):
        if not isinstance(unit, dict) or "idle" not in unit:
            continue
        prtos = [unit[f"key{i}"] for i in range(unit["key_count"])]
        ours = ours_measure(pack, unit)
        civ3 = None
        for prto in prtos:
            flc = default_flc(names.get(prto, prto[5:]))
            if flc:
                civ3 = civ3_measure(flc, civ)
                break
        results.append({"binding": key, "prto": prtos, "scale": unit["scale"],
                        "fit": unit.get("fit_policy"), "ours": ours, "civ3": civ3})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "size_audit.json").write_text(json.dumps(results, indent=1))
    print(f"{'unit':24} {'c3 h':>5} {'our h':>6} {'ratio':>6} {'c3 w':>5} {'our w':>6} {'c3 a':>6} {'our a':>6} {'lin':>5}  {'civ%':>5} {'luma':>5}")
    for r in sorted(results, key=lambda r: (r["civ3"] or {}).get("height", 0) / max(1, r["ours"]["height"])):
        c = r["civ3"] or {}
        ratio = r["ours"]["height"] / c["height"] if c else float("nan")
        print(f"{r['prto'][0][5:]:24} {c.get('height', 0):5.0f} {r['ours']['height']:6.1f} {ratio:6.2f} "
              f"{c.get('width', 0):5.0f} {r['ours']['width']:6.1f} {c.get('area', 0):6.0f} {r['ours']['area']:6.0f} "
              f"{math.sqrt(r['ours']['area'] / c['area']) if c else float('nan'):5.2f}  {100 * c.get('civ', 0):5.1f} {c.get('luma', 0):5.0f}")
    paired = [r for r in results if r["civ3"]]
    for label, metric in (("height", lambda r: r["ours"]["height"] / r["civ3"]["height"]),
                          ("linear area", lambda r: math.sqrt(r["ours"]["area"] / r["civ3"]["area"]))):
        ratios = np.array([metric(r) for r in paired])
        print(f"{len(paired)} paired units: {label} ratio ours/Civ III median {np.median(ratios):.2f}, "
              f"10th {np.percentile(ratios, 10):.2f}, 90th {np.percentile(ratios, 90):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
