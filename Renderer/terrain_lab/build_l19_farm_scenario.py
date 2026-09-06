#!/usr/bin/env python3
"""Build deterministic L19 connected farms and a tundra material witness viewport."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


MAGIC = "C3X_LAB_FARM_SCENARIO_V0"


def rows(path: Path) -> list[list[str]]:
    return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))


def write_scenario(viewport: Path, cities: Path, output: Path) -> None:
    source = rows(viewport)
    columns, height = int(source[0][1]), int(source[0][2])
    tiles = {
        (int(row[0]), int(row[1])): tuple(map(int, row[:9]))
        for row in source[1 : 1 + columns * height]
    }
    city_cells = {(int(row[0]), int(row[1])) for row in rows(cities)[1:]}
    land = {
        point
        for point, tile in tiles.items()
        if tile[4] < 11 and tile[5] in (0, 1, 2, 3) and point not in city_cells
    }
    # Select the largest flat-land components so the beauty scene contains
    # broad connected acreages rather than scattered per-mask stamps.
    remaining = set(land)
    components = []
    while remaining:
        seed = min(remaining, key=lambda point: (point[1], point[0]))
        remaining.remove(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            column, row = frontier.pop()
            for point in ((column, row - 1), (column + 1, row),
                          (column, row + 1), (column - 1, row)):
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    frontier.append(point)
        components.append(component)
    components.sort(key=lambda points: (-len(points), min((p[1], p[0]) for p in points)))
    selected = set()
    for component in components:
        for point in sorted(component, key=lambda p: ((p[0] * 97 + p[1] * 151) % 389, p[1], p[0])):
            if len(selected) >= 36:
                break
            selected.add(point)
        if len(selected) >= 36:
            break
    if len(selected) < 32:
        raise ValueError("viewport lacks enough land for the L19 farm matrix")
    directions = ((0, -1, 1), (1, 0, 2), (0, 1, 4), (-1, 0, 8))
    records = []
    for index, (column, row) in enumerate(sorted(selected, key=lambda p: (p[1], p[0]))):
        mask = sum(bit for dx, dy, bit in directions if (column + dx, row + dy) in selected)
        era = (index // 7) % 4
        terrain_family = index % 4
        visible = 0 if index == len(selected) - 1 else 1
        records.append((column, row, era, mask, terrain_family, visible, index % 3))
    # Ensure all 16 topology masks are represented as explicit Lab witnesses.
    represented = {record[3] for record in records}
    used = {(record[0], record[1]) for record in records}
    for mask in range(16):
        if mask in represented:
            continue
        frontier = [
            point
            for point in land - used
            if any(
                (point[0] + dx, point[1] + dy) in used
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
            )
        ]
        frontier.sort(
            key=lambda point: (
                (point[0] * 67 + point[1] * 103 + mask * 29) % 421,
                point[1],
                point[0],
            )
        )
        column, row = frontier[0]
        used.add((column, row))
        records.append((column, row, mask % 4, mask, mask % 4, 1, mask % 3))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                MAGIC,
                columns,
                height,
                len(records),
                hashlib.sha256(viewport.read_bytes()).hexdigest(),
                "lab_augmentation",
            ]
        )
        writer.writerows(records)
    print(f"L19 farm scenario: farms={len(records)} masks=16 hidden=1")


def write_tundra_viewport(viewport: Path, farms: Path, output: Path) -> None:
    source = rows(viewport)
    farm_rows = rows(farms)[1:]
    farm_cells = {(int(row[0]), int(row[1])) for row in farm_rows if int(row[5]) == 1}
    columns, height = int(source[0][1]), int(source[0][2])
    changed = 0
    body = []
    for row in source[1:]:
        next_row = list(row)
        column, y, base = int(row[0]), int(row[1]), int(row[4])
        in_view = 0 <= column < columns and 0 <= y < height
        witness = in_view and base < 11 and (
            (column, y) in farm_cells or (3 <= column <= 12 and 1 <= y <= 9)
        )
        if witness and (column + 2 * y) % 3 != 0:
            next_row[4] = "3"
            changed += 1
        body.append(next_row)
    # This fixture is explicitly Lab-only. The authoritative header,
    # dimensions, coordinates and all non-base fields remain unchanged.
    header = list(source[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(body)
    print(f"L19 tundra witness: changed={changed}")


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: build_l19_farm_scenario.py <viewport> <cities> <farms-output> <tundra-output>",
            file=sys.stderr,
        )
        return 2
    viewport, cities, farms, tundra = (Path(value) for value in sys.argv[1:])
    write_scenario(viewport, cities, farms)
    write_tundra_viewport(viewport, farms, tundra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
