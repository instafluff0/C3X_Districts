#!/usr/bin/env python3
"""Build a deterministic, route-connected L17 city matrix."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


MAGIC = "C3X_LAB_CITY_SCENARIO_V0"


def write_scenario(viewport: Path, roads: Path, output: Path) -> None:
    view_rows = list(csv.reader(viewport.read_text(encoding="utf-8").splitlines()))
    road_rows = list(csv.reader(roads.read_text(encoding="utf-8").splitlines()))
    columns, rows = int(view_rows[0][1]), int(view_rows[0][2])
    tiles = { (int(r[0]), int(r[1])): tuple(map(int, r[:9])) for r in view_rows[1:1 + columns * rows] }
    road_nodes = {(int(r[0]), int(r[1])) for r in road_rows[1:]} | {(int(r[2]), int(r[3])) for r in road_rows[1:]}
    candidates = [p for p in road_nodes if p in tiles and tiles[p][4] < 11 and tiles[p][5] not in (6, 9)]
    candidates.sort(key=lambda p: ((p[0] * 73 + p[1] * 131) % 257, p[1], p[0]))
    selected = []
    for point in candidates:
        if all(abs(point[0] - other[0]) + abs(point[1] - other[1]) >= 2 for other in selected):
            selected.append(point)
        if len(selected) == 12:
            break
    if len(selected) < 12:
        raise ValueError("viewport lacks twelve separated route-connected city cells")
    records = []
    for index, (column, row) in enumerate(selected):
        records.append((column, row, index % 4, index % 3, index % 5,
                        index % 4, int(index % 2 == 0), int(index in (3, 8)), 1, index))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([MAGIC, columns, rows, len(records), hashlib.sha256(viewport.read_bytes()).hexdigest(), "lab_augmentation"])
        writer.writerows(records)
    print(f"L17 city scenario: cities={len(records)} walls={sum(r[6] for r in records)} capitals={sum(r[7] for r in records)}")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: build_l17_city_scenario.py <viewport.csv> <roads.csv> <output.csv>", file=sys.stderr)
        return 2
    write_scenario(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
