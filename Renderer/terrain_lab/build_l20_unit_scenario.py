#!/usr/bin/env python3
"""Build the deterministic 192-tile L20 unit/action Lab augmentation."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


MAGIC = "C3X_LAB_UNIT_SCENARIO_V0"


def rows(path: Path) -> list[list[str]]:
    return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))


def occupied(path: Path) -> set[tuple[int, int]]:
    return {(int(row[0]), int(row[1])) for row in rows(path)[1:] if int(row[6])}


def build(viewport: Path, cities: Path, tile_objects: Path,
          infrastructure: Path, output: Path) -> None:
    source = rows(viewport)
    columns, height = int(source[0][1]), int(source[0][2])
    tiles = {
        (int(row[0]), int(row[1])): tuple(map(int, row[:9]))
        for row in source[1:1 + columns * height]
    }
    blocked = occupied(cities) | occupied(tile_objects) | occupied(infrastructure)
    land = [point for point, tile in tiles.items()
            if tile[4] < 11 and point not in blocked and
            0 < point[0] < columns - 1 and 0 < point[1] < height - 1]
    water = [point for point, tile in tiles.items()
             if tile[4] >= 11 and point not in blocked and
             0 < point[0] < columns - 1 and 0 < point[1] < height - 1]
    land.sort(key=lambda p: ((p[0] * 179 + p[1] * 83) % 1019, p[1], p[0]))
    water.sort(key=lambda p: ((p[0] * 97 + p[1] * 157) % 1021, p[1], p[0]))
    if len(land) < 48 or len(water) < 4:
        raise ValueError("L20 needs forty-eight clear land and four clear water witnesses")

    records = []
    land_index = 0

    def add_land(kind, owner, facing, action, phase=0, stack=0,
                 move=(0, 0, 0), visible=1, point=None):
        nonlocal land_index
        if point is None:
            point = land[land_index]
            land_index += 1
        records.append((point[0], point[1], kind, owner, facing, action, phase,
                        visible, stack, move[0], move[1], move[2]))
        return point

    # Eight-direction source-art turntable distributed across the real scene.
    for facing in range(8):
        add_land(0, facing % 4, facing, 0)

    # Every basic action and four actual source-pose phases for the key motions.
    actions = [(1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0),
               (4, 0), (4, 1), (4, 2), (4, 3), (5, 0), (6, 0),
               (7, 0), (7, 1), (7, 2), (7, 3)]
    for index, (action, phase) in enumerate(actions):
        move = (1, 0, (phase + 1) * 200) if action == 2 else (0, 0, 0)
        add_land(1 if index < 8 else 2, (index + 1) % 4,
                 (index * 3) % 8, action, phase, move=move)

    # One moving stack: the second body is deliberately offset and uses a
    # different source family/owner rather than becoming an invented roster.
    stack_point = add_land(2, 2, 5, 3, stack=0)
    add_land(0, 2, 5, 0, stack=1, point=stack_point)

    # Air and naval silhouettes use clear domain-correct witnesses.
    for index in range(4):
        add_land(3, index, index * 2, (0, 2, 4, 7)[index],
                 0 if index != 1 else 2)

    # Source-authored compound bodies: two mounted facings, crewed siege,
    # vehicle-and-gunner, then Army commander plus exactly one ordinary member.
    compound_actions = (0, 2, 4)
    used = {(record[0], record[1]) for record in records}
    deep_land = [point for point in land if point not in used and all(
        tiles.get((point[0] + dx, point[1] + dy), (0, 0, 0, 0, 11))[4] < 11
        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        if dx or dy
    )]
    if len(deep_land) < 16:
        raise ValueError("L20 needs sixteen interior-land unit witnesses")
    compound_index = 0
    for kind in range(5, 9):
        for index, action in enumerate(compound_actions):
            phase = index if action in (2, 4) else 0
            point = deep_land[compound_index]
            compound_index += 1
            move = (0, 1, 600) if action == 2 else (0, 0, 0)
            if action == 2 and tiles.get((point[0], point[1] + 1),
                                         (0, 0, 0, 0, 11))[4] >= 11:
                move = (1, 0, 600)
            add_land(kind, (kind + index) % 4, (kind * 2 + index * 3) % 8,
                     action, phase, move=move, point=point)

    # Builder-only source clips: light work, heavy work, cutting, and capture.
    for index, action in enumerate(range(8, 12)):
        point = deep_land[compound_index]
        compound_index += 1
        add_land(9, index, index * 2, action, index, point=point)
    for index, point in enumerate(water[:4]):
        records.append((point[0], point[1], 4, index, index * 2,
                        (0, 2, 4, 7)[index], 0 if index != 1 else 2,
                        1, 0, 0, 0, 0))

    # Viewer-hidden unit is present in the fixture but must emit no body.
    add_land(0, 3, 7, 0, visible=0)
    records.sort(key=lambda row: (row[1], row[0], row[8], row[2]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([MAGIC, columns, height, len(records),
                         hashlib.sha256(viewport.read_bytes()).hexdigest(),
                         "lab_augmentation_absolute_time_samples"])
        writer.writerows(records)
    print(f"L20 unit scenario: records={len(records)} facings=8 actions=12 "
          "move_phases=4 attack_phases=4 death_phases=4 compounds=12 "
          "workers=4 army_commander_plus_member=1 hidden=1 stack=2")


def main() -> int:
    if len(sys.argv) != 6:
        print("usage: build_l20_unit_scenario.py <viewport> <cities> "
              "<tile-objects> <infrastructure> <output>", file=sys.stderr)
        return 2
    build(*(Path(value) for value in sys.argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
