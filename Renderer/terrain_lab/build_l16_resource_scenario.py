#!/usr/bin/env python3
"""Build deterministic L16 resource placements over the accepted 192-tile viewport."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


MAGIC = "C3X_LAB_RESOURCE_SCENARIO_V0"
RESOURCE_NAMES = ("horses", "iron", "uranium", "gold", "dye", "wheat", "cattle", "fish")


def write_scenario(source: Path, output: Path) -> None:
    rows = list(csv.reader(source.read_text(encoding="utf-8").splitlines()))
    if not rows or rows[0][0] not in ("C3X_BIQ_TERRAIN_WINDOW_V1", "C3X_BIQ_TERRAIN_WINDOW_V2"):
        raise ValueError("not a decoded Terrain Lab viewport")
    columns, row_count = int(rows[0][1]), int(rows[0][2])
    tiles = [tuple(map(int, row[:9])) for row in rows[1:1 + columns * row_count]]
    records = []
    land_index = water_index = 0
    for column, row, _sx, _sy, base, real, _bonus, overlays, _river in tiles:
        witness = (column * 37 + row * 53 + base * 11 + real * 7) % 17
        if base >= 11:
            if witness in (0, 1, 2):
                records.append((column, row, 7, 1, water_index))
                water_index += 1
        elif witness in (0, 1, 2, 3):
            resource = (column * 5 + row * 3 + land_index) % 7
            # Minerals also get a few elevated/hill witnesses; animals and crops
            # remain on compatible lowland cells so every silhouette is legible.
            if real in (6, 9) and resource not in (1, 2, 3):
                resource = 1 + resource % 3
            records.append((column, row, resource, 1, land_index))
            land_index += 1
    # One authoritative hidden witness verifies suppression without changing the
    # visible resource roster.
    hidden_tile = next(tile for tile in tiles if tile[4] < 11)
    records.append((hidden_tile[0], hidden_tile[1], 3, 0, 999))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            MAGIC, columns, row_count, len(records), hashlib.sha256(source.read_bytes()).hexdigest(),
            "lab_augmentation",
        ])
        writer.writerows(records)
    counts = {name: sum(record[2] == i and record[3] for record in records)
              for i, name in enumerate(RESOURCE_NAMES)}
    print(f"L16 resource scenario: visible={sum(record[3] for record in records)} hidden=1 {counts}")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: build_l16_resource_scenario.py <decoded-viewport.csv> <resource-scenario.csv>", file=sys.stderr)
        return 2
    write_scenario(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
