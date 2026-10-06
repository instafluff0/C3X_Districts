"""Vanilla Conquests map resources at fixed Lab positions.

Names are the BIQ display names the game passes to the renderer, so the census
exercises the same name-based selection as production. Offsets are raw Civ III
coordinates relative to the Lab view center; even/even or odd/odd pairs keep
tile parity.

Cases:
- roster: all 26 resources on grassland (Fish/Whales offshore);
- relief: the same layout with every land resource on a hill;
- native-BATCH: each resource of a review batch on every terrain it can occupy.
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
    return ("roster", "relief") + tuple("native-" + name for name in native()["batches"])


CASES = cases()


def layout() -> list[tuple[int, int, dict]]:
    land = [m for m in mappings() if m["civ3_name"] not in WATER]
    water = [m for m in mappings() if m["civ3_name"] in WATER]
    if len(land) != len(COLUMNS) * len(ROWS):
        raise ValueError("Resource roster no longer fits the Lab grid")
    cells = [(COLUMNS[i % len(COLUMNS)], ROWS[i // len(COLUMNS)], m) for i, m in enumerate(land)]
    cells += [(-1 + 4 * i, WATER_ROW, m) for i, m in enumerate(water)]
    return cells


@lru_cache(maxsize=None)
def native_layout(case: str) -> tuple[tuple[int, int, str, str], ...]:
    """(dx, dy, resource, terrain): one row per resource, one column per native
    terrain, two tiles apart so mountains and hills stay separate bodies."""
    names = native()["batches"][case.removeprefix("native-")]
    cells = []
    for row, name in enumerate(names):
        dy = 4 * row - 2 * (len(names) - 1)
        for column, terrain in enumerate(native()["terrains"][name]):
            cells.append((4 * column - 4, dy, name, terrain))
    return tuple(cells)


def placements(case: str) -> list[tuple[int, int, str, str]]:
    """Resource placements as (dx, dy, resource, terrain label)."""
    if case.startswith("native-"):
        return list(native_layout(case))
    return [(x, y, m["civ3_name"], "Hills" if case == "relief" and y < WATER_EDGE else
             "Coast" if y >= WATER_EDGE else "Grassland") for x, y, m in layout()]


def terrain(case: str, dx: int, dy: int, base: int, real: int) -> tuple[int, int]:
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
    return ";".join(f"{x},{y},{name}" for x, y, name, _ in placements(case))


def viewport(case: str, zoom: int) -> tuple[int, int]:
    if case.startswith("native-"):
        rows = len(native()["batches"][case.removeprefix("native-")])
        return 6 * zoom, (rows + 2) * zoom
    return 8 * zoom, 6 * zoom
