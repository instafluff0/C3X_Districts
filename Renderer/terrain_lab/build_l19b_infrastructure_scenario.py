#!/usr/bin/env python3
"""Build the deterministic 192-tile L19B infrastructure Lab augmentation."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


MAGIC = "C3X_LAB_INFRASTRUCTURE_SCENARIO_V0"


def rows(path: Path) -> list[list[str]]:
    return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))


def occupied(path: Path) -> set[tuple[int, int]]:
    return {(int(row[0]), int(row[1])) for row in rows(path)[1:]}


def build(viewport: Path, roads: Path, resources: Path, cities: Path, mines: Path,
          farms: Path, tile_objects: Path, output: Path) -> None:
    source = rows(viewport)
    columns, height = int(source[0][1]), int(source[0][2])
    tiles = {
        (int(row[0]), int(row[1])): tuple(map(int, row[:9]))
        for row in source[1:1 + columns * height]
    }
    excluded = (occupied(resources) | occupied(cities) | occupied(mines) |
                occupied(farms) | occupied(tile_objects))
    road_cells = set()
    for row in rows(roads)[1:]:
        road_cells.add((int(row[0]), int(row[1])))
        road_cells.add((int(row[2]), int(row[3])))
    land = [point for point, tile in tiles.items() if tile[4] < 11 and point not in excluded]
    # Raised compounds need a full-tile visual margin. Persistent damage may
    # still exercise edge/wrap behavior independently below.
    route_land = [point for point in land if point in road_cells and
                  0 < point[0] < columns - 1 and 0 < point[1] < height - 1]
    free_land = [point for point in land if point not in road_cells]
    route_land.sort(key=lambda p: ((p[0] * 83 + p[1] * 149) % 1009, p[1], p[0]))
    free_land.sort(key=lambda p: ((p[0] * 127 + p[1] * 61) % 1013, p[1], p[0]))
    if len(route_land) < 18 or len(free_land) < 10:
        raise ValueError("L19B needs eighteen routed and ten unrouted land witnesses")
    records = []
    used = set()

    def take(pool, kind, count, road_connected):
        chosen = 0
        for point in pool:
            if point in used:
                continue
            index = len(records)
            era = chosen % 4
            owner = (chosen * 3 + kind) % 4
            variant = (point[0] * 7 + point[1] * 11 + kind) % 4
            records.append((point[0], point[1], kind, variant, era, owner, 1,
                            int(road_connected)))
            used.add(point)
            chosen += 1
            if chosen == count:
                return
        raise ValueError(f"not enough placement cells for infrastructure kind {kind}")

    # Raised strategic infrastructure. Fortifications and airfields deliberately
    # touch the connected road network; outposts/radar/victory witnesses remain
    # spread across both route and open-terrain contexts.
    take(route_land, 0, 4, True)  # Fortress
    take(route_land, 1, 4, True)  # Barricade
    take(route_land, 2, 4, True)  # Airfield
    take(route_land, 3, 4, True)  # Outpost
    take(route_land, 4, 3, True)  # Radar tower
    take(free_land, 7, 4, False)  # Victory location
    # Persistent damage prefers open ground but may intersect a road, as it can
    # in authoritative Civ III state. It never shares a raised-object cell.
    damage_land = free_land + route_land
    take(damage_land, 5, 6, False)  # Pollution
    take(damage_land, 6, 6, False)  # Crater

    # One hidden record proves that authoritative visibility suppresses a body.
    hidden = free_land[0]
    records.append((hidden[0], hidden[1], 4, 0, 3, 2, 0, 0))
    records.sort(key=lambda row: (row[1], row[0], row[2]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([MAGIC, columns, height, len(records),
                         hashlib.sha256(viewport.read_bytes()).hexdigest(),
                         "lab_augmentation"])
        writer.writerows(records)
    print("L19B infrastructure scenario: fortress=4 barricade=4 airfield=4 "
          "outpost=4 radar=4/3-visible victory=4 pollution=6 crater=6 eras=4")


def main() -> int:
    if len(sys.argv) != 9:
        print("usage: build_l19b_infrastructure_scenario.py <viewport> <roads> "
              "<resources> <cities> <mines> <farms> <tile-objects> <output>",
              file=sys.stderr)
        return 2
    build(*(Path(value) for value in sys.argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
