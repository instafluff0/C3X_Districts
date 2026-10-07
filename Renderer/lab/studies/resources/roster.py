"""Vanilla Conquests map resources at fixed Lab positions.

Names are the BIQ display names the game passes to the renderer, so the census
exercises the same name-based selection as production. Offsets are raw Civ III
coordinates relative to the Lab view center; even/even or odd/odd pairs keep
tile parity.

Cases:
- roster: all 26 resources on grassland (Fish/Whales offshore);
- relief: the same layout with every land resource on a hill;
- native-BATCH: each resource of a review batch on every terrain it can occupy;
  'Name~label' rows are Lab-only alternate compositions of Name; batches
  longer than BLOCK_ROWS wrap into further column blocks;
- oasis-roads: oases on desert crossed by a straight road, an L junction, a
  straight railroad and a crossroads, beside one without routes. Roster names
  ending in "|road" or "|rail" give that tile a route ("|road" alone: no resource).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MAPPING = ROOT / "Renderer/inventory/vanilla_conquests_to_civ6_resources.json"
NATIVE = Path(__file__).with_name("native_terrains.json")
WATER = ("Fish", "Whales")
COLUMNS = (-5, -3, -1, 1, 3, 5)
ROWS = (-7, -3, 1, 5)
WATER_ROW = 9
# Raw rows from this offset are coast for four rows, then sea.
WATER_EDGE = 7
# Civ III (base, real) terrain indices for the Lab scene.
TERRAIN = {"Desert": (0, 0), "Plains": (1, 1), "Grassland": (2, 2), "Tundra": (3, 3), "Flood Plain": (4, 4),
           "Hills": (2, 5), "Mountains": (2, 6), "Forest": (2, 7), "Jungle": (2, 8), "Marsh": (2, 9),
           "Coast": (11, 11), "Sea": (12, 12)}


@lru_cache(maxsize=None)
def mappings() -> tuple[dict, ...]:
    return tuple(json.loads(MAPPING.read_text())["mappings"])


@lru_cache(maxsize=None)
def native() -> dict:
    return json.loads(NATIVE.read_text())


def cases() -> tuple[str, ...]:
    return ("roster", "relief", "oasis-roads") + tuple("native-" + name for name in native()["batches"])


# Raw Civ III neighbour offsets: east-west (+-2, 0), north-south (0, +-2).
OASIS_ROADS = (
    ((-6, -4), "road", ((2, 0), (-2, 0))),                    # straight road
    ((6, -4), "road", ((2, 0), (0, 2))),                      # L junction
    ((-6, 4), "rail", ((2, 0), (-2, 0))),                     # straight railroad
    ((6, 4), "road", ((2, 0), (-2, 0), (0, 2), (0, -2))),     # crossroads
    ((0, 0), None, ()),                                       # no route
)


def oasis_roads() -> list[tuple[int, int, str]]:
    """(dx, dy, roster name) for every tile of the oasis-roads case."""
    entries = []
    for (x, y), route, arms in OASIS_ROADS:
        entries.append((x, y, "Oasis" + (f"|{route}" if route else "")))
        entries += [(x + ax, y + ay, f"|{route}") for ax, ay in arms]
    return entries


CASES = cases()


def layout() -> list[tuple[int, int, dict]]:
    land = [m for m in mappings() if m["civ3_name"] not in WATER]
    water = [m for m in mappings() if m["civ3_name"] in WATER]
    if len(land) != len(COLUMNS) * len(ROWS):
        raise ValueError("Resource roster no longer fits the Lab grid")
    cells = [(COLUMNS[i % len(COLUMNS)], ROWS[i // len(COLUMNS)], m) for i, m in enumerate(land)]
    cells += [(-1 + 4 * i, WATER_ROW, m) for i, m in enumerate(water)]
    return cells


# Rows beyond this wrap into another block of columns, keeping the batch on the 32-tile Lab map.
BLOCK_ROWS = 8


@lru_cache(maxsize=None)
def native_layout(case: str) -> tuple[tuple[int, int, str, str], ...]:
    """(dx, dy, resource, terrain): one row per resource, one column per native
    terrain, two tiles apart so mountains and hills stay separate bodies."""
    batch = case.removeprefix("native-")
    names = native()["batches"][batch]
    extra = native().get("batch_terrains", {}).get(batch, [])
    terrains = {name: extra + [terrain for terrain in native()["terrains"].get(name.split("~")[0], [])
                               if terrain not in extra] for name in names}
    if len(names) <= BLOCK_ROWS:
        return tuple((4 * column - 4, 4 * row - 2 * (len(names) - 1), name, terrain)
                     for row, name in enumerate(names) for column, terrain in enumerate(terrains[name]))
    width = max(len(values) for values in terrains.values()) + 1
    blocks = -(-len(names) // BLOCK_ROWS)
    cells = []
    for index, name in enumerate(names):
        block, row = divmod(index, BLOCK_ROWS)
        for column, terrain in enumerate(terrains[name]):
            cells.append((4 * (block * width + column) - 2 * (blocks * width - 2), 4 * row - 2 * (BLOCK_ROWS - 1),
                          name, terrain))
    return tuple(cells)


def placements(case: str) -> list[tuple[int, int, str, str]]:
    """Resource placements as (dx, dy, resource, terrain label)."""
    if case.startswith("native-"):
        return list(native_layout(case))
    if case == "oasis-roads":
        return [(x, y, "Oasis", "Desert") for x, y, name in oasis_roads() if name.startswith("Oasis")]
    return [(x, y, m["civ3_name"], "Hills" if case == "relief" and y < WATER_EDGE else
             "Coast" if y >= WATER_EDGE else "Grassland") for x, y, m in layout()]


def terrain(case: str, dx: int, dy: int, base: int, real: int) -> tuple[int, int]:
    if case == "oasis-roads":
        near = any(abs(dx - x) + abs(dy - y) <= 4 for (x, y), _, _ in OASIS_ROADS)
        return TERRAIN["Desert"] if near else TERRAIN["Grassland"]
    if case.startswith("native-"):
        for x, y, _, label in native_layout(case):
            if (x, y) == (dx, dy):
                return TERRAIN[label]
        return 2, 2
    if dy >= WATER_EDGE:
        return (11, 11) if dy < WATER_EDGE + 4 else (12, 12)
    if case == "relief" and any((dx, dy) == (x, y) for x, y, m in layout() if m["civ3_name"] not in WATER):
        return 2, 5
    return 2, 2


def spec(case: str) -> str:
    """Compact environment value consumed by the native Lab preview."""
    if case == "oasis-roads":
        return ";".join(f"{x},{y},{name}" for x, y, name in oasis_roads())
    return ";".join(f"{x},{y},{name}" for x, y, name, _ in placements(case))


def viewport(case: str, zoom: int) -> tuple[int, int]:
    if case.startswith("native-"):
        rows = len(native()["batches"][case.removeprefix("native-")])
        if rows > BLOCK_ROWS:
            return (max(abs(x) for x, _, _, _ in native_layout(case)) + 3) * zoom, (BLOCK_ROWS + 2) * zoom
        columns = max(x for x, _, _, _ in native_layout(case)) // 4 + 2
        return max(6, 2 * columns + 1) * zoom, (rows + 2) * zoom
    if case == "oasis-roads":
        return 11 * zoom, 6 * zoom
    return 8 * zoom, 6 * zoom
