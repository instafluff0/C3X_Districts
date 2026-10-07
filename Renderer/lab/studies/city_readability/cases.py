"""City Lab cases: era/size ladders, difficult sites and a busy gameplay map.

Offsets are raw Civ III coordinates relative to the scene origin (16, 16);
raw x and y share parity on a tile. Natural coordinates are
(column, row) = ((x + y) / 2, (x - y) / 2): a column step is down-right on
screen and a row step is up-right. Object markers use the C3C BIQ overlay bits
decoded by the Lab preview (0x1 road, 0x2 railroad, 0x4 mine, 0x8
irrigation). Cities travel separately as "dx,dy,culture,era,size,capital,
walled;..." in C3X_LAB_TILE_CITIES, relative to the view centre.

Cases (category "cities"):
- city-ladder-CULTURE: four era rows by Town / walled Town / City /
  Metropolis columns on flat grassland joined by roads.
- city-sites-ERA: six Metropolis or City sites: a river on the city's own
  edges with a bridged road, a river through the outskirts, a coast, hills
  under a mountain range, forest and jungle all round, and a road/railroad
  junction among farms.
- city-gameplay-ERA: a dense late-game map in the spirit of the 1498 AD save:
  roads on most tiles, two railroads, farms, mined hills, a mountain range,
  a river, forests and five cities of every size.
"""
from __future__ import annotations

import os

CULTURES = ("american", "european", "roman", "middle_eastern", "asian")
ERAS = ("ancient", "medieval", "industrial", "modern")
CASES = tuple([f"city-ladder-{c}" for c in CULTURES] +
              [f"city-sites-{e}" for e in ERAS] +
              [f"city-gameplay-{e}" for e in ERAS])
ROAD, RAIL, MINE, IRRIGATION = 0x1, 0x2, 0x4, 0x8
DESERT, PLAINS, GRASSLAND, TUNDRA, FLOODPLAIN, HILLS, MOUNTAIN, FOREST, JUNGLE = range(9)
COAST, SEA, OCEAN = 11, 12, 13


def applies(category: str, case: str) -> bool:
    return category == "cities" and case in CASES


def raw(c: int, r: int) -> tuple[int, int]:
    return c + r, c - r


def natural(dx: int, dy: int) -> tuple[int, int]:
    return (dx + dy) // 2, (dx - dy) // 2


def _river(nodes) -> dict[tuple[int, int], int]:
    """Civ III edge bits on both incident raw tiles for a path of corner nodes."""
    bits: dict[tuple[int, int], int] = {}

    def add(c, r, bit):
        key = raw(c, r)
        bits[key] = bits.get(key, 0) | bit

    for a, b in zip(nodes, nodes[1:]):
        c, r = min(a, b)
        if a[1] == b[1]:
            add(c, r, 32)
            add(c, r - 1, 2)
        else:
            add(c, r, 128)
            add(c - 1, r, 8)
    return bits


def _walk(start, moves):
    c, r = start
    nodes = [(c, r)]
    for move in moves:
        c, r = (c + 1, r) if move == "c" else (c - 1, r) if move == "C" else \
            (c, r + 1) if move == "r" else (c, r - 1)
        nodes.append((c, r))
    return nodes


def _parse(case: str) -> tuple[str, str]:
    kind, _, value = case.removeprefix("city-").partition("-")
    return kind, value


def culture_index(case: str) -> int:
    kind, value = _parse(case)
    if kind == "ladder":
        return CULTURES.index(value)
    override = os.environ.get("C3X_LAB_CITY_CULTURE", "")
    return CULTURES.index(override) if override in CULTURES else CULTURES.index("european")


# --- Ladder -----------------------------------------------------------------
LADDER_COLUMNS = (-9, -3, 3, 9)          # Town, walled Town, City, Metropolis
LADDER_ROWS = (-9, -3, 3, 9)             # ancient .. modern
LADDER_STATES = ((0, 0), (0, 1), (1, 0), (2, 0))   # (size, walled)


def _ladder_cities(culture: int):
    for era, dy in enumerate(LADDER_ROWS):
        for (size, walled), dx in zip(LADDER_STATES, LADDER_COLUMNS):
            yield dx, dy, culture, era, size, 0, walled


def _ladder_terrain(dx, dy):
    base = real = GRASSLAND
    overlays = 0
    if dy in LADDER_ROWS and -11 <= dx <= 11:
        overlays = ROAD
    # Ordinary context the eye needs to judge scale: a farm, a mined hill.
    if (dx, dy) in ((-6, 0), (6, 0), (0, -6), (0, 6)):
        overlays |= IRRIGATION
    if (dx, dy) in ((-6, 6), (6, -6)):
        real, overlays = HILLS, MINE
    if (dx, dy) in ((0, 0), (1, 1), (-1, 1)):
        real = FOREST
    return base, real, 0, 0, overlays


# --- Sites ------------------------------------------------------------------
# Six sites in natural coordinates around the scene origin.
SITE_CENTRES = {
    "river-edge": (-3, 3), "river-outskirts": (2, 6), "coast": (6, 0),
    "hills": (-6, -2), "forest": (-1, -4), "junction": (3, -6),
}


def _site_size(era: int, name: str) -> int:
    return 1 if name in ("hills", "forest") else 2


def _sites_cities(era: int, culture: int):
    for name, (c, r) in SITE_CENTRES.items():
        dx, dy = raw(c, r)
        capital = 1 if name == "junction" else 0
        yield dx, dy, culture, era, _site_size(era, name), capital, 0


_SITE_RIVER = {}
# River along the river-edge city's own up-right and down-right edges, then
# away to the coast; a road crosses it on a diagonal edge (a bridge).
_SITE_RIVER.update(_river(_walk((-4, 2), "rrcccccc")))
# A second river through the outskirts of the next city, ending at the sea.
_SITE_RIVER.update({k: _SITE_RIVER.get(k, 0) | v for k, v in _river(_walk((0, 4), "crcrcc")).items()})


def _sites_terrain(dx, dy):
    c, r = natural(dx, dy)
    base = real = GRASSLAND
    overlays = 0
    river = _SITE_RIVER.get((dx, dy), 0)
    # Coast to the right of the coastal city.
    if c >= 7 and r <= 2:
        water = COAST if c == 7 else SEA
        return water, water, 0, 0, 0
    hc, hr = SITE_CENTRES["hills"]
    if abs(c - hc) <= 1 and abs(r - hr) <= 1:
        real = HILLS
    if (c - hc, r - hr) in ((-2, 0), (-2, -1), (-2, 1), (-1, -2), (0, -2), (1, -2)):
        real = MOUNTAIN
    fc, fr = SITE_CENTRES["forest"]
    if abs(c - fc) <= 1 and abs(r - fr) <= 1 and (c, r) != (fc, fr):
        real = FOREST if c - fc <= 0 else JUNGLE
    jc, jr = SITE_CENTRES["junction"]
    if abs(c - jc) <= 2 and abs(r - jr) <= 2:
        overlays = ROAD | (IRRIGATION if (c + r) % 2 else 0)
        if c == jc or r == jr:
            overlays |= RAIL
    # Roads join the sites; one crosses the river-edge city's river.
    for name, (sc, sr) in SITE_CENTRES.items():
        if (c, r) == (sc, sr):
            overlays |= ROAD
    rc, rr = SITE_CENTRES["river-edge"]
    if (c == rc and rr - 3 <= r <= rr + 2) or (r == rr and rc - 2 <= c <= rc + 3):
        overlays |= ROAD
    oc, orr = SITE_CENTRES["river-outskirts"]
    if r == orr and oc - 3 <= c <= oc + 1:
        overlays |= ROAD
    if real in (MOUNTAIN, FOREST, JUNGLE) and overlays & IRRIGATION:
        overlays &= ~IRRIGATION
    return base, real, river, 0, overlays


# --- Gameplay ---------------------------------------------------------------
GAMEPLAY_CITIES = (
    # (c, r, size, capital, walled)
    (0, 0, 2, 1, 0), (-4, 4, 1, 0, 0), (4, -3, 0, 0, 1), (5, 5, 1, 0, 0), (-5, -3, 0, 0, 0),
)
_GAMEPLAY_RIVER = _river(_walk((-7, 1), "crrcrcccrc"))
_GAMEPLAY_RANGE = {(2, 3), (3, 2), (3, 3), (4, 2), (5, 1), (6, 0), (6, 1), (7, -1)}
_GAMEPLAY_HILLS = {(1, 4), (2, 4), (4, 3), (5, 2), (7, 0), (-2, -2), (-3, -1), (6, 2)}
_GAMEPLAY_FOREST = {(-6, 0), (-6, 1), (-7, 0), (1, -5), (2, -5), (1, -6)}


def _gameplay_cities(era: int, culture: int):
    for c, r, size, capital, walled in GAMEPLAY_CITIES:
        dx, dy = raw(c, r)
        yield dx, dy, culture, era, size, capital, walled if size == 0 else 0


def _hash(c, r):
    value = (c * 73856093) ^ (r * 19349663)
    value = ((value ^ (value >> 13)) * 0x5BD1E995) & 0xFFFFFFFF
    return value ^ (value >> 15)


def _gameplay_terrain(dx, dy):
    c, r = natural(dx, dy)
    base = real = PLAINS if (c * 5 + r * 3) % 11 < 4 else GRASSLAND
    river = _GAMEPLAY_RIVER.get((dx, dy), 0)
    if (c, r) in _GAMEPLAY_RANGE:
        real = MOUNTAIN
    elif (c, r) in _GAMEPLAY_HILLS:
        real = HILLS
    elif (c, r) in _GAMEPLAY_FOREST:
        real = FOREST
    seed = _hash(c, r)
    overlays = 0
    if real != MOUNTAIN and seed % 100 < 82:
        overlays = ROAD
    rail = r == 0 or c - r == 4
    if rail and real != MOUNTAIN:
        overlays |= ROAD | RAIL
    if real in (PLAINS, GRASSLAND) and seed % 7 < 3:
        overlays |= IRRIGATION
    if real == HILLS and seed % 3 == 0:
        overlays |= MINE
    for cc, cr, *_ in GAMEPLAY_CITIES:
        if (c, r) == (cc, cr):
            overlays = ROAD | (overlays & RAIL)
    return base, real, river, 0, overlays


# --- Interface used by Renderer/renderer.py ----------------------------------
def terrain(category: str, case: str, dx: int, dy: int, base: int, real: int):
    """(base, real, river, bonus, overlays) for one raw Lab tile."""
    kind, _value = _parse(case)
    if kind == "ladder":
        return _ladder_terrain(dx, dy)
    if kind == "sites":
        return _sites_terrain(dx, dy)
    return _gameplay_terrain(dx, dy)


def cities(case: str):
    kind, value = _parse(case)
    culture = culture_index(case)
    if kind == "ladder":
        return list(_ladder_cities(culture))
    era = ERAS.index(value)
    return list((_sites_cities if kind == "sites" else _gameplay_cities)(era, culture))


def city_spec(case: str, centre: tuple[int, int] = (0, 0)) -> str:
    """C3X_LAB_TILE_CITIES relative to the view centre (scene origin offsets)."""
    return ";".join(f"{dx - centre[0]},{dy - centre[1]},{c},{e},{s},{cap},{w}"
                    for dx, dy, c, e, s, cap, w in cities(case))


def resources(category: str, case: str, centre: tuple[int, int] = (0, 0)) -> str:
    kind, _value = _parse(case)
    cells = {"ladder": ((-6, -6, "Wheat"), (6, 6, "Cattle")),
             "sites": ((-2, 6, "Wheat"), (8, 2, "Horses")),
             "gameplay": ((-4, 0, "Wheat"), (4, 6, "Cattle"), (-8, 2, "Horses"))}[kind]
    return ";".join(f"{dx - centre[0]},{dy - centre[1]},{name}" for dx, dy, name in cells)


# Close-up review shots: (case, view-centre offset, name suffix) at zoom 256.
CLOSE_UPS = (
    ("city-sites-industrial", (0, -6), "river-edge"),
    ("city-sites-industrial", (8, -4), "river-outskirts"),
    ("city-sites-industrial", (6, 6), "coast"),
    ("city-sites-industrial", (-8, -4), "hills"),
    ("city-sites-industrial", (-5, 3), "forest"),
    ("city-sites-industrial", (-3, 9), "junction"),
    ("city-gameplay-industrial", (0, 0), "metropolis"),
    ("city-gameplay-industrial", (0, -8), "river-city"),
    ("city-ladder-european", (9, 3), "industrial-metropolis"),
    ("city-ladder-european", (9, 9), "modern-metropolis"),
    ("city-ladder-european", (9, -9), "ancient-metropolis"),
    ("city-ladder-european", (9, -3), "medieval-metropolis"),
)


def centre(case: str) -> tuple[int, int]:
    """View centre offset from the scene origin."""
    return (0, 0)


def viewport_for(case: str, zoom: int) -> tuple[int, int]:
    # Close-up studies set an explicit view size (pixels) for a single site.
    view = os.environ.get("C3X_LAB_CITY_VIEW", "")
    if view:
        width, height = (int(v) for v in view.split("x"))
        return width, height
    kind, _value = _parse(case)
    if kind == "ladder":
        # Four 3-tile columns by four 3-tile rows at the gameplay zoom.
        return 1472 * zoom // 128, 832 * zoom // 128
    return 1024 * zoom // 128, 768 * zoom // 128


def viewport(zoom: int) -> tuple[int, int]:
    return 1024 * zoom // 128, 768 * zoom // 128
