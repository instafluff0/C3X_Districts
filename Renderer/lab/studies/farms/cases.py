"""Farm Lab cases: irrigated farmland beside terrain, routes, resources and water.

Offsets are raw Civ III coordinates relative to the Lab view centre; raw x and
y share parity on a tile. Edge neighbours are (dx+-1, dy+-1); (dx+-2, dy) and
(dx, dy+-2) share only a corner. Object markers use the C3C BIQ overlay bits
carried in the scene CSV and decoded by the Lab preview only:
0x1 road, 0x2 railroad, 0x4 mine, 0x8 irrigation.

Every case keeps a railroad and a mine in view: the infrastructure ownership
witness requires all four custom route/improvement flags.

Cases (category "infrastructure"):
- farms-terrain: 3x3 farm blocks on desert, plains, grassland, tundra, flood
  plain and grassland hills, with a road and railroad between the rows, and a
  mountain ridge and a hill beside the plains block;
- farms-routes: farmland crossed by a corner (E-W) road, an edge (NW-SE) road,
  a corner (N-S) road, an edge (NE-SW) railroad, junctions and a spur;
- farms-resources: farmed resource tiles inside farmland, one beside a road;
- farms-water: farmland along a straight river and a coast, with a bridged road;
- farms-network: late-game farmland where every tile has a road to all its
  neighbours and three railroad corridors cross it (as on the 1498 AD save).
"""
from __future__ import annotations

CASES = ("farms-terrain", "farms-routes", "farms-resources", "farms-water", "farms-network")
ROAD, RAIL, MINE, IRRIGATION = 0x1, 0x2, 0x4, 0x8
DESERT, PLAINS, GRASSLAND, TUNDRA, FLOODPLAIN, HILLS, MOUNTAIN, COAST = 0, 1, 2, 3, 4, 5, 6, 11
# Raw view extent: 8 x 6 tiles at the gameplay zoom.
VIEW = (8, 12)
# farms-terrain patches: centre -> (base, real, label).
PATCHES = {
    (-6, -6): (DESERT, DESERT, "desert"),
    (0, -6): (PLAINS, PLAINS, "plains"),
    (6, -6): (GRASSLAND, GRASSLAND, "grassland"),
    (-6, 6): (TUNDRA, TUNDRA, "tundra"),
    (0, 6): (FLOODPLAIN, FLOODPLAIN, "flood plain"),
    (6, 6): (GRASSLAND, HILLS, "hills"),
}
RESOURCE_CELLS = (
    ((-4, -6), "Wheat", GRASSLAND), ((0, -6), "Cattle", GRASSLAND), ((4, -6), "Horses", PLAINS),
    ((-4, 0), "Wines", GRASSLAND), ((0, 0), "Sugar", FLOODPLAIN), ((4, 0), "Tobacco", GRASSLAND),
    ((-4, 6), "Incense", PLAINS), ((0, 6), "Oasis", DESERT), ((4, 6), "Wheat", PLAINS),
)


NETWORK_RESOURCES = (((-3, -3), "Wheat"), ((3, -1), "Cattle"))


def applies(category: str, case: str) -> bool:
    return category == "infrastructure" and case in CASES


def _block(center: tuple[int, int], dx: int, dy: int) -> bool:
    """Inside the 3x3 lattice block around a raw centre."""
    return abs(dx - center[0]) + abs(dy - center[1]) <= 2


def _nearest(dx: int, dy: int, centers) -> tuple[int, int]:
    return min(centers, key=lambda c: (abs(dx - c[0]) + abs(dy - c[1]), c))


def _routes(dx: int, dy: int) -> int:
    road = dy == -6 or (dy == dx + 2 and -7 <= dx <= 5) or (dx == -4 and -2 <= dy <= 12)
    road = road or (dx, dy) == (3, -5)
    rail = dy == -dx + 4 and -4 <= dx <= 8
    return (ROAD if road or rail else 0) | (RAIL if rail else 0)


def terrain(category: str, case: str, dx: int, dy: int, base: int, real: int) -> tuple[int, int, int, int, int]:
    """(base, real, river, bonus, overlays) for one raw Lab tile."""
    river = bonus = overlays = 0
    base = real = GRASSLAND
    inside = abs(dx) <= VIEW[0] + 1 and abs(dy) <= VIEW[1] + 1
    if case == "farms-terrain":
        center = _nearest(dx, dy, PATCHES)
        base, real, _label = PATCHES[center]
        if _block(center, dx, dy):
            overlays = IRRIGATION
        if dy == 0:
            overlays = ROAD | (RAIL if dx >= 2 else 0)
        if (dx, dy) == (4, 10):
            overlays = MINE
        # A mountain ridge and a hill beside the plains block: farms stop at
        # their foot.
        if (dx, dy) in ((2, -4), (3, -5), (3, -7), (2, -8)):
            real, overlays = MOUNTAIN, 0
        if (dx, dy) == (-2, -4):
            real, overlays = HILLS, 0
    elif case == "farms-routes":
        overlays = _routes(dx, dy) | (IRRIGATION if inside else 0)
        if (dx, dy) == (6, 10):
            real, overlays = HILLS, MINE
    elif case == "farms-resources":
        for (cx, cy), _name, ground in RESOURCE_CELLS:
            if abs(dx - cx) + abs(dy - cy) <= 2:
                base = real = ground
                overlays = IRRIGATION
        if dy == 6 and dx >= 2:
            overlays |= ROAD
        if dy == 10:
            overlays = ROAD | RAIL
        if (dx, dy) == (-7, -9):
            real, overlays = HILLS, MINE
    elif case == "farms-network":
        if (dx * 5 + dy * 3) % 11 < 4:
            base = real = PLAINS
        rail = dy == 2 or dx - dy == 6 or dx + dy == -6
        overlays = ROAD | (RAIL if rail else 0) | (IRRIGATION if inside else 0)
        if (dx, dy) == (5, 9):
            real, overlays = HILLS, ROAD | MINE
    elif case == "farms-water":
        if dx + dy >= 12:
            return COAST, COAST, 0, 0, 0
        river = 2 if dx == dy else 32 if dx == dy + 2 else 0
        overlays = IRRIGATION if inside else 0
        if dx + dy == -4 and -6 <= dx <= 2:
            overlays |= ROAD | RAIL
        if (dx, dy) == (-6, 8):
            real, overlays = HILLS, MINE
    return base, real, river, bonus, overlays


def resources(category: str, case: str, centre: tuple[int, int] = (0, 0)) -> str:
    """'dx,dy,Name;...' for the Lab preview, relative to the view centre
    (the preview places resources around it; the terrain CSV is fixed)."""
    cells = RESOURCE_CELLS if case == "farms-resources" else NETWORK_RESOURCES if case == "farms-network" else ()
    return ";".join(f"{dx - centre[0]},{dy - centre[1]},{name}" for (dx, dy), name, *_ground in cells)


def viewport(zoom: int) -> tuple[int, int]:
    return min(8 * zoom, 1024), min(6 * zoom, 768)


# Close-up centres (raw offsets) per case for the 256 zoom review.
CLOSE_UPS = {
    "farms-terrain": ((-6, -6), (0, -6), (6, -6), (-6, 6), (0, 6), (6, 6)),
    "farms-routes": ((-4, -6), (0, 2), (2, 2), (-4, 4)),
    "farms-resources": tuple(cell for cell, _name, _ground in RESOURCE_CELLS),
    "farms-water": ((-2, -2), (4, 4)),
    "farms-network": ((0, 0), (-3, -3), (2, 2)),
}
