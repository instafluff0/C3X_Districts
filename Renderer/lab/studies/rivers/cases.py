"""River Lab cases: a meandering river through farmland, crossings and forest.

Offsets are raw Civ III coordinates relative to the Lab view centre; raw x
and y share parity on a tile. The river is a path of tile-corner nodes in
(column, row) = ((x + y) / 2, (x - y) / 2); Civ III records each edge on both
incident tiles, as in the rivers category's watershed. Object markers use the
C3C BIQ overlay bits decoded by the Lab preview: 0x1 road, 0x2 railroad,
0x4 mine, 0x8 irrigation.

Cases (category "infrastructure"; the object witness needs all four flags):
- river-reach: the river meanders across farmland to a coast on the right.
  A road and a railroad cross it on diagonal edges, so both get bridges.
  Hills (one mined) line the far bank. A forest stands on the near bank in
  front of the water and another on the far bank behind it, which the
  water reflects; resources sit beside the river.
"""
from __future__ import annotations

CASES = ("river-reach",)
ROAD, RAIL, MINE, IRRIGATION = 0x1, 0x2, 0x4, 0x8
PLAINS, GRASSLAND, HILLS, FOREST, COAST, SEA = 1, 2, 5, 7, 11, 12
# Raw view extent at the gameplay zoom (as the farm cases): 8 x 6 tiles.
VIEW = (8, 12)
# Moves from the source node: "c" steps one column (down-right on screen),
# "r" one row (up-right). Runs of either bend the course into meanders.
START = (-5, -5)
MOVES = "ccrrrrcccccrrrrc"
# Raw tiles: banks and context.
# Tiles with river bit 2 or 128 lie on the near (lower) bank, so their trees
# stand in front of the water; bits 8 and 32 mark the far bank.
HILL_RUN = ((1, 1), (2, 0), (3, -1))
MINED_HILL = (2, 0)
FOREST_CELLS = ((-3, 1), (-2, 2), (-1, 3), (-2, 4), (-3, 3),
                (-4, -2), (-3, -1), (-5, -3), (-4, -4))
RESOURCE_CELLS = ((0, -2, "Cattle"), (-6, 0, "Wheat"), (0, 2, "Wines"))
CLOSE_UPS = {"river-reach": ((-6, -2), (-2, 2), (2, 0), (5, 0))}


def applies(category: str, case: str) -> bool:
    return category == "infrastructure" and case in CASES


def _nodes():
    c, r = START
    nodes = [(c, r)]
    for move in MOVES:
        c, r = (c + 1, r) if move == "c" else (c, r + 1)
        nodes.append((c, r))
    return nodes


def _river_bits() -> dict[tuple[int, int], int]:
    bits: dict[tuple[int, int], int] = {}

    def add(c, r, bit):
        raw = (c + r, c - r)
        bits[raw] = bits.get(raw, 0) | bit

    nodes = _nodes()
    for a, b in zip(nodes, nodes[1:]):
        c, r = min(a, b)
        if a[1] == b[1]:
            add(c, r, 32)
            add(c, r - 1, 2)
        else:
            add(c, r, 128)
            add(c - 1, r, 8)
    return bits


RIVER = _river_bits()


def _coast(dx: int) -> int:
    return COAST if dx in (6, 7) else SEA if dx >= 8 else 0


def _routes(dx: int, dy: int) -> int:
    # A road on a down-right diagonal and a railroad on an up-right one.
    road = dy == dx + 4 and -10 <= dx <= -4
    rail = dy == -dx + 4 and -1 <= dx <= 5
    return (ROAD if road or rail else 0) | (RAIL if rail else 0)


def terrain(category: str, case: str, dx: int, dy: int, base: int, real: int) -> tuple[int, int, int, int, int]:
    """(base, real, river, bonus, overlays) for one raw Lab tile."""
    bonus = 0
    water = _coast(dx)
    if water:
        return water, water, 0, 0, 0
    base = real = GRASSLAND if dy < 4 else PLAINS
    river = RIVER.get((dx, dy), 0)
    overlays = _routes(dx, dy)
    inside = abs(dx) <= VIEW[0] + 1 and abs(dy) <= VIEW[1] + 1
    if inside and (dx, dy) not in FOREST_CELLS and (dx, dy) != MINED_HILL:
        overlays |= IRRIGATION
    if (dx, dy) in FOREST_CELLS:
        real = FOREST
    if (dx, dy) in HILL_RUN:
        real = HILLS
    if (dx, dy) == MINED_HILL:
        real, overlays = HILLS, MINE
    return base, real, river, bonus, overlays


def resources(category: str, case: str, centre: tuple[int, int] = (0, 0)) -> str:
    """'dx,dy,Name;...' relative to the view centre."""
    return ";".join(f"{dx - centre[0]},{dy - centre[1]},{name}" for dx, dy, name in RESOURCE_CELLS)


def viewport(zoom: int) -> tuple[int, int]:
    return min(8 * zoom, 1024), min(6 * zoom, 768)
