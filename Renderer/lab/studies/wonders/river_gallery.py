#!/usr/bin/env python3
"""A gallery of river-wonder sites rendered through the production path (prod_parity.py).

C3X builds a wonder whose config needs a river (buildable_on_rivers) on any tile of its
buildable terrain that holds a river: the river may run along any edge, bend, fork,
start or end there, and the neighbours are arbitrary. Each cell here is one such site
for Hoover Dam on a mountain, with its own river and neighbours; the renderer must
place the dam across the river and cut the valley from facts alone.

    python3 Renderer/lab/studies/wonders/river_gallery.py [--zoom 128] [--zoom 256]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.wonders import prod_parity

# Civ III river edge bits (direction_to_neighbor_bit) and each edge's neighbour in world
# (column, row) offsets; a river edge sets its bit on both tiles.
NE, SE, SW, NW = 2, 8, 32, 128
ACROSS = {NE: ((0, 1), SW), SE: ((1, 0), NW), SW: ((0, -1), NE), NW: ((-1, 0), SE)}
WONDER = "hoover_dam"


def straight(bit: int, start: int = -2, stop: int = 2) -> list:
    """A straight river along `bit` edges, from `start` to `stop` tiles along it."""
    along = (1, 0) if bit in (NE, SW) else (0, 1)
    return [(along[0] * k, along[1] * k, bit) for k in range(start, stop + 1)]


# (label, river edges as (column, row, bit) from the dam tile, neighbours as
# (column, row, terrain), options). Rivers run along tile edges; corners join them.
SITES = (
    ("south-west edge, lone mountain", straight(SW), (), {}),
    ("north-east edge", straight(NE), (), {}),
    ("south-east edge", straight(SE), (), {}),
    ("north-west edge", straight(NW), (), {}),
    ("bend at the south corner", straight(SW, -2, 0) + straight(SE, 0, 2), (), {}),
    ("bend at the north corner", straight(NE, 0, 2) + straight(NW, -2, 0), (), {}),
    ("fork: a tributary joins", straight(SW) + straight(SE, 0, 2), (), {}),
    ("source: the river starts here", straight(SW, 0, 2), (), {}),
    ("mountain range on both banks", straight(SW),
     tuple((c, r, "Mountains") for c in (-1, 1) for r in (-1, 0, 1)) + ((0, 1, "Mountains"), (0, -1, "Mountains")), {}),
    ("hills and forest beside", straight(SE),
     ((-1, 0, "Hills"), (1, 0, "Hills"), (0, 1, "Forest"), (0, -1, "Forest"), (1, 1, "Hills")), {}),
    ("alternate facing", straight(SW), ((1, 0, "Mountains"), (-1, 0, "Mountains")), {"alternate": 1}),
    ("snow-capped, under construction (half built)", straight(NW), ((0, 1, "Mountains"), (0, -1, "Mountains")),
     {"built": .5, "snow": True}),
)
# Raw Civ III offsets of the cells: four columns, three rows, far enough apart that
# rivers two tiles long never meet.
PLACES = [(dx, dy) for dy in (-8, 0, 8) for dx in (-12, -4, 4, 12)]


def raw(base: tuple[int, int], column: int, row: int) -> tuple[int, int]:
    """Raw offset of the tile (column, row) world tiles from a raw base tile."""
    return base[0] + column + row, base[1] + column - row


def catalog() -> dict:
    cells, rivers = [], {}
    for number, ((label, edges, neighbours, options), base) in enumerate(zip(SITES, PLACES)):
        cell = {"name": f"River wonder {number + 1}", "label": label, "wonder": WONDER, "dx": base[0], "dy": base[1],
                "terrain": "Mountains", "parts": [{"built": options.get("built", 1.0)}]}
        if "alternate" in options:
            cell["alternate"] = options["alternate"]
        if options.get("snow"):
            cell["snow"] = True   # Civ III's snow-capped mountain bit
        cells.append(cell)
        for column, row, terrain in neighbours:
            x, y = raw(base, column, row)
            cells.append({"name": f"River site {number + 1} {column},{row}", "ghost": True, "dx": x, "dy": y,
                          "terrain": terrain})
        for column, row, bit in edges:
            (oc, orow), back = ACROSS[bit]
            here, there = raw(base, column, row), raw(base, column + oc, row + orow)
            rivers[here] = rivers.get(here, 0) | bit
            rivers[there] = rivers.get(there, 0) | back
    return {"id": "wonder-river-sites", "cells": cells, "rivers": rivers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zoom", action="append", type=int)
    args = parser.parse_args()
    zooms = tuple(args.zoom or (128,))
    spec = {**catalog(), "shots": tuple({"zoom": zoom, "hour": 12} for zoom in zooms)}
    chosen = {spec["id"]: spec}
    prod_parity.rivers(chosen)
    for stale in (prod_parity.PARITY / "renders").glob(f"{spec['id']}-*"):
        shutil.rmtree(stale)
    prod_parity.render(chosen, zooms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
