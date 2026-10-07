"""Forest and jungle Lab cases: canopy beside routes, resources and sites.

Offsets are raw Civ III coordinates relative to the Lab view centre; raw
x and y share parity on a tile. Edge neighbours are (dx+-1, dy+-1); (dx+-2, dy)
and (dx, dy+-2) share only a corner. Object markers use the C3C BIQ bits, carried
in the scene CSV's bonus/overlay columns and decoded by the Lab preview only:

- bonus 0x20: pine forest (Civ III's per-tile flag);
- overlays 0x1 road, 0x2 railroad, 0x4 mine, 0x20 goody hut, 0x80 barbarian camp.

Resources use the BIQ display names the game passes to the renderer.

Cases:
- roads: a road grid, a crossing railroad, a branch and a dead-end spur;
- resources: each native resource in the canopy, one beside a road;
- rivers-sites: a river, a goody hut, a camp, mined hill and mountain canopies;
- pines (forests only): pine forest on grassland, broadleaf control, and
  pine forest on tundra (snow), each with a road and a resource.
"""
from __future__ import annotations

CASES = ("roads", "resources", "rivers-sites", "pines")
ROAD, RAIL, MINE, HUT, CAMP = 0x1, 0x2, 0x4, 0x20, 0x80
PINE = 0x20
GRASSLAND, TUNDRA, HILLS, MOUNTAINS = 2, 3, 5, 6
NATIVE = {
    "forests": ("Furs", "Game", "Silks", "Dyes", "Spices", "Ivory", "Rubber", "Uranium"),
    "jungles": ("Gems", "Silks", "Dyes", "Spices", "Rubber", "Coal", "Tropical Fruit"),
}
RESOURCE_CELLS = ((-4, -6), (0, -6), (4, -6), (-4, 0), (0, 0), (4, 0), (-4, 6), (0, 6), (4, 6))
# The extent of the canopy patch; the viewport shows about 7 x 10 raw units.
PATCH = (8, 11)


def cases(category: str) -> tuple[str, ...]:
    return CASES if category == "forests" else CASES[:3]


def applies(category: str, case: str) -> bool:
    return category in NATIVE and case in cases(category)


def _roads(dx: int, dy: int) -> int:
    road = dy == -6 or (dy == dx + 2 and -7 <= dx <= 5) or (dx == -4 and -2 <= dy <= 10)
    road = road or (dx, dy) == (3, -5)
    rail = dy == -dx + 4 and -4 <= dx <= 8
    return (ROAD if road or rail else 0) | (RAIL if rail else 0)


def _resource_cells(category: str) -> list[tuple[int, int, str]]:
    names = NATIVE[category]
    cells = [(dx, dy, name) for (dx, dy), name in zip(RESOURCE_CELLS, names)]
    # The last cell repeats the first resource with a road through the tile.
    return cells + [(RESOURCE_CELLS[-1][0], RESOURCE_CELLS[-1][1], names[0])] if len(cells) < len(RESOURCE_CELLS) else cells


def terrain(category: str, case: str, dx: int, dy: int, base: int, real: int) -> tuple[int, int, int, int, int]:
    """(base, real, river, bonus, overlays) for one raw Lab tile."""
    vegetation = 7 if category == "forests" else 8
    river = bonus = overlays = 0
    inside = abs(dx) <= PATCH[0] and abs(dy) <= PATCH[1]
    base, real = GRASSLAND, vegetation if inside else GRASSLAND
    if case == "roads":
        overlays = _roads(dx, dy)
    elif case == "resources":
        if dy == 6 and dx >= 2:
            overlays = ROAD
    elif case == "rivers-sites":
        river = 2 if dx == dy else 32 if dx == dy + 2 else 0
        overlays = HUT if (dx, dy) == (-4, 2) else CAMP if (dx, dy) == (4, -2) else 0
        if (dx, dy) in ((-5, -3), (1, -7)):
            real = HILLS
        if (dx, dy) == (5, 7):
            real = MOUNTAINS
        if (dx, dy) in ((-5, -3), (5, 7)):
            overlays = MINE
    elif case == "pines":
        # Pine forest on grassland, a broadleaf control band, and pine
        # forest on tundra, which Civ III draws with its snow sheet.
        if dx >= 3:
            base = TUNDRA
            real = vegetation if inside else TUNDRA
        if real == vegetation and abs(dx) >= 3:
            bonus = PINE
        if (dx, dy) in ((-5, -7), (5, -7)):
            real = HILLS
        if dy == -4:
            overlays = ROAD
    return base, real, river, bonus, overlays


def resources(category: str, case: str) -> str:
    """'dx,dy,Name;...' for the Lab preview."""
    if case == "resources":
        cells = _resource_cells(category)
    elif case == "pines":
        cells = [(-5, 3, "Game"), (5, 3, "Furs"), (0, 5, "Silks")]
    else:
        cells = []
    return ";".join(f"{dx},{dy},{name}" for dx, dy, name in cells)


def viewport(zoom: int) -> tuple[int, int]:
    return 7 * zoom, 5 * zoom
