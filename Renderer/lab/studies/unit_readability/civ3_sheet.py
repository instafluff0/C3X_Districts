#!/usr/bin/env python3
"""Civ III's own unit sprites on a 128x64 tile grid, for side-by-side review.

Each unit's DEFAULT FLC, first frame, is placed with its 240x240 frame centre on
the tile centre (Civ III's unit anchor), facing SW, recoloured with a civ
palette. Output: lab/out/unit-readability/civ3-<palette>.png (1x and 2x).

    python3 Renderer/lab/studies/unit_readability/civ3_sheet.py [--civ 6] [--units Warrior,Tank]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit  # noqa: E402
import civ3_flc  # noqa: E402
from image_io import over, scale_nearest, write_png  # noqa: E402

ROSTER = ("Warrior", "Spearman", "Archer", "Horseman", "Settler", "Worker",
          "Swordsman", "Knight", "Catapult", "Musketman", "Cavalry", "Cannon",
          "Rifleman", "Infantry", "Artillery", "Tank", "Modern_Armor", "Mech_Infantry")
COLUMNS, TILE_W, TILE_H, CELL_W, CELL_H = 6, 128, 64, 160, 150
GRASS = np.array([112, 136, 52], np.uint8)


def tile(canvas, cx, cy, colour):
    ys, xs = np.mgrid[0:canvas.shape[0], 0:canvas.shape[1]]
    inside = np.abs(xs + .5 - cx) / (TILE_W / 2) + np.abs(ys + .5 - cy) / (TILE_H / 2) <= 1
    canvas[inside] = colour


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--civ", type=int, default=6)
    parser.add_argument("--units", default=",".join(ROSTER))
    args = parser.parse_args(argv)
    names = audit.pedia_names()
    civ = civ3_flc.read_pcx_palette(audit.GAME / "Art/Units/Palettes" / f"ntp{args.civ:02d}.pcx")
    units = args.units.split(",")
    rows = (len(units) + COLUMNS - 1) // COLUMNS
    canvas = np.empty((rows * CELL_H + 20, COLUMNS * CELL_W, 3), np.uint8)
    canvas[:] = (88, 104, 44)
    for index, unit in enumerate(units):
        cx = (index % COLUMNS) * CELL_W + CELL_W // 2
        cy = (index // COLUMNS) * CELL_H + CELL_H // 2 + 25
        tile(canvas, cx, cy, GRASS)
        flc = civ3_flc.read(audit.default_flc(names["PRTO_" + unit]))
        sprite = civ3_flc.rgba(flc.direction_frame(0), flc.palette, civ)
        over(canvas, sprite, cx - flc.full_width // 2 + flc.x_offset, cy - flc.full_height // 2 + flc.y_offset)
    audit.OUT.mkdir(parents=True, exist_ok=True)
    write_png(audit.OUT / f"civ3-ntp{args.civ:02d}.png", canvas)
    write_png(audit.OUT / f"civ3-ntp{args.civ:02d}-2x.png", scale_nearest(canvas, 2))
    print(audit.OUT / f"civ3-ntp{args.civ:02d}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
