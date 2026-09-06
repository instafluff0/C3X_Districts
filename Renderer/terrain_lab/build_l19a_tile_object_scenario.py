#!/usr/bin/env python3
"""Build the deterministic L19A goody-hut and colony Lab scenario."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


MAGIC = "C3X_LAB_TILE_OBJECT_SCENARIO_V0"


def read_rows(path: Path) -> list[list[str]]:
    return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))


def occupied(path: Path) -> set[tuple[int, int]]:
    return {(int(row[0]), int(row[1])) for row in read_rows(path)[1:]}


def build(viewport: Path, resources: Path, cities: Path, mines: Path, farms: Path,
          output: Path) -> None:
    source = read_rows(viewport)
    columns, height = int(source[0][1]), int(source[0][2])
    tiles = {
        (int(row[0]), int(row[1])): tuple(map(int, row[:9]))
        for row in source[1 : 1 + columns * height]
    }
    resource_rows = [tuple(map(int, row)) for row in read_rows(resources)[1:]]
    visible_land_resources = [
        row for row in resource_rows
        if row[3] == 1 and row[2] != 7 and tiles[(row[0], row[1])][4] < 11
    ]
    visible_land_resources.sort(
        key=lambda row: ((row[0] * 89 + row[1] * 137 + row[2] * 31) % 997,
                         row[1], row[0])
    )
    if len(visible_land_resources) < 13:
        raise ValueError("L19A requires at least thirteen visible land resource witnesses")
    records = []
    colony_cells = set()
    for index, row in enumerate(visible_land_resources[:13]):
        column, y, resource, _visible, variant = row
        era = index % 4
        owner = (index * 3) % 4
        territory_owner = (owner + 1 + index % 2) % 4
        visible = 0 if index == 12 else 1
        records.append((column, y, 1, variant % 3, era, owner,
                        territory_owner, visible, resource))
        colony_cells.add((column, y))
    excluded = occupied(cities) | occupied(mines) | occupied(farms) | colony_cells
    hut_candidates = [
        point for point, tile in tiles.items()
        if tile[4] < 11 and point not in excluded
    ]
    hut_candidates.sort(
        key=lambda point: ((point[0] * 71 + point[1] * 149) % 1009,
                           point[1], point[0])
    )
    if len(hut_candidates) < 9:
        raise ValueError("L19A viewport lacks nine free land cells for goody huts")
    for bucket, (column, y) in enumerate(hut_candidates[:9]):
        records.append((column, y, 0, bucket % 8, 0, 0, 0,
                        0 if bucket == 8 else 1, 255))
    records.sort(key=lambda row: (row[1], row[0], row[2]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            MAGIC, columns, height, len(records),
            hashlib.sha256(viewport.read_bytes()).hexdigest(), "lab_augmentation",
        ])
        writer.writerows(records)
    print("L19A tile-object scenario: huts=9 visible_huts=8 colonies=13 "
          "visible_colonies=12 eras=4 extraterritorial=13")


def main() -> int:
    if len(sys.argv) != 7:
        print("usage: build_l19a_tile_object_scenario.py <viewport> <resources> "
              "<cities> <mines> <farms> <output>", file=sys.stderr)
        return 2
    build(*(Path(value) for value in sys.argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
