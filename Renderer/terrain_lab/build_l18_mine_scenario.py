#!/usr/bin/env python3
"""Build deterministic terrain-, route-, resource-, and city-aware L18 mines."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


MAGIC = "C3X_LAB_MINE_SCENARIO_V0"


def read_objects(path: Path) -> list[list[str]]:
    return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))


def write_scenario(viewport: Path, roads: Path, resources: Path, cities: Path, output: Path) -> None:
    view_rows = read_objects(viewport)
    columns, rows = int(view_rows[0][1]), int(view_rows[0][2])
    tiles = {(int(row[0]), int(row[1])): tuple(map(int, row[:9]))
             for row in view_rows[1:1 + columns * rows]}
    route_rows = read_objects(roads)[1:]
    route_nodes = {(int(row[0]), int(row[1])) for row in route_rows}
    route_nodes |= {(int(row[2]), int(row[3])) for row in route_rows}
    resource_rows = read_objects(resources)[1:]
    mineral_cells = {(int(row[0]), int(row[1])) for row in resource_rows
                     if int(row[2]) in (1, 2, 3) and int(row[3]) == 1}
    city_cells = {(int(row[0]), int(row[1])) for row in read_objects(cities)[1:]}
    candidates = [point for point, tile in tiles.items()
                  if tile[4] < 11 and point not in city_cells]
    candidates.sort(key=lambda point: (
        point not in mineral_cells,
        point not in route_nodes,
        (point[0] * 97 + point[1] * 151 + tiles[point][5] * 13) % 389,
        point[1], point[0]))
    selected = []
    for point in candidates:
        if all(abs(point[0] - other[0]) + abs(point[1] - other[1]) >= 2
               for other in selected):
            selected.append(point)
        if len(selected) == 20:
            break
    if len(selected) < 20:
        raise ValueError("viewport lacks twenty separated mine cells")
    records = []
    for index, (column, row) in enumerate(selected):
        era = index % 4
        variant = index % 3
        resource_owned = int((column, row) in mineral_cells)
        records.append((column, row, era, variant, 1, resource_owned))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([MAGIC, columns, rows, len(records),
                         hashlib.sha256(viewport.read_bytes()).hexdigest(), "lab_augmentation"])
        writer.writerows(records)
    relief = sum(tiles[(record[0], record[1])][5] in (6, 9) for record in records)
    print(f"L18 mine scenario: mines={len(records)} mineral={sum(r[5] for r in records)} relief={relief}")


def main() -> int:
    if len(sys.argv) != 6:
        print("usage: build_l18_mine_scenario.py <viewport> <roads> <resources> <cities> <output>", file=sys.stderr)
        return 2
    write_scenario(*(Path(value) for value in sys.argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
