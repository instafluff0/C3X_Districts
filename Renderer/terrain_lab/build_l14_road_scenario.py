#!/usr/bin/env python3
"""Build the deterministic Lab-only L14 road graph over an unchanged BIQ viewport."""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import deque
from pathlib import Path


MAGIC = "C3X_LAB_ROAD_SCENARIO_V0"


def load_land(path: Path) -> tuple[int, int, dict[tuple[int, int], dict[str, int]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0][0] not in {
        "C3X_BIQ_TERRAIN_WINDOW_V1",
        "C3X_BIQ_TERRAIN_WINDOW_V2",
    }:
        raise ValueError("not a decoded C3X BIQ terrain viewport")
    columns, row_count = int(rows[0][1]), int(rows[0][2])
    tiles: dict[tuple[int, int], dict[str, int]] = {}
    for row in rows[1 : 1 + columns * row_count]:
        x, y = int(row[0]), int(row[1])
        tiles[x, y] = {
            "base": int(row[4]),
            "real": int(row[5]),
            "river": int(row[8]) if len(row) > 8 else 0,
        }
    if len(tiles) != columns * row_count:
        raise ValueError("incomplete BIQ viewport")
    return columns, row_count, tiles


def neighbors(cell: tuple[int, int], columns: int, rows: int):
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < columns and 0 <= ny < rows:
            yield nx, ny


def shortest_path(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    land: set[tuple[int, int]],
    columns: int,
    rows: int,
) -> list[tuple[int, int]]:
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    found = None
    while queue and found is None:
        cell = queue.popleft()
        if cell in goals:
            found = cell
            break
        for nxt in neighbors(cell, columns, rows):
            if nxt in land and nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    if found is None:
        raise ValueError(f"road target {start} is disconnected from the main landmass")
    path = []
    while found is not None:
        path.append(found)
        found = parent[found]
    return path


def edge_key(a: tuple[int, int], b: tuple[int, int]):
    return (a, b) if a < b else (b, a)


def crosses_river(
    a: tuple[int, int],
    b: tuple[int, int],
    tiles: dict[tuple[int, int], dict[str, int]],
    wraps: bool = False,
) -> bool:
    """Match the reciprocal BIQ river bits on the road's shared tile edge."""
    if wraps or a[1] == b[1]:
        left, right = (a, b) if (wraps or a[0] < b[0]) else (b, a)
        return bool((tiles[left]["river"] & 8) or (tiles[right]["river"] & 128))
    top, bottom = (a, b) if a[1] < b[1] else (b, a)
    return bool((tiles[top]["river"] & 2) or (tiles[bottom]["river"] & 32))


def build_graph(columns: int, rows: int, tiles: dict[tuple[int, int], dict[str, int]]):
    land = {cell for cell, tile in tiles.items() if tile["base"] < 11}
    root = min(land, key=lambda cell: (abs(cell[0] - columns // 2) + abs(cell[1] - rows // 2), cell))
    targets = {
        cell
        for cell in land
        if ((cell[0] * 17 + cell[1] * 31) % 11 < 2)
        or tiles[cell]["river"]
        or tiles[cell]["real"] in (5, 6, 10)
    }
    for y in range(rows):
        if (0, y) in land and (columns - 1, y) in land:
            targets.update({(0, y), (columns - 1, y)})
    network = {root}
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for target in sorted(targets, key=lambda cell: (abs(cell[0] - root[0]) + abs(cell[1] - root[1]), cell)):
        path = shortest_path(target, network, land, columns, rows)
        network.update(path)
        edges.update(edge_key(a, b) for a, b in zip(path, path[1:]))

    # Add deterministic chords so the fixture contains real loops and alternate routes.
    for cell in sorted(network):
        for nxt in neighbors(cell, columns, rows):
            if nxt in network and cell < nxt and ((cell[0] * 13 + cell[1] * 7 + nxt[0] * 5 + nxt[1]) % 5 == 0):
                edges.add(edge_key(cell, nxt))

    # Explicit wrap continuations are recorded as their own edges, never folded into BIQ state.
    wrap_edges = []
    for y in range(rows):
        a, b = (0, y), (columns - 1, y)
        if a in network and b in network:
            wrap_edges.append((b, a))
    return edges, wrap_edges, network


def write_scenario(source: Path, output: Path) -> None:
    columns, rows, tiles = load_land(source)
    edges, wrap_edges, network = build_graph(columns, rows, tiles)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    records = []
    for a, b in sorted(edges):
        bridge = int(crosses_river(a, b, tiles))
        style = (a[0] * 3 + a[1] + b[0] * 5 + b[1]) % 4
        pillaged = int((a[0] * 19 + a[1] * 11 + b[0] * 7 + b[1]) % 23 == 0)
        records.append((*a, *b, 0, style, pillaged, bridge))
    for a, b in wrap_edges:
        bridge = int(crosses_river(a, b, tiles, True))
        style = (a[1] * 3) % 4
        records.append((*a, *b, 1, style, 0, bridge))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([MAGIC, columns, rows, len(records), source_hash, "lab_augmentation"])
        writer.writerows(records)
    degrees = {cell: 0 for cell in network}
    for x0, y0, x1, y1, *_ in records:
        degrees[x0, y0] = degrees.get((x0, y0), 0) + 1
        degrees[x1, y1] = degrees.get((x1, y1), 0) + 1
    print(
        f"L14 road scenario: nodes={len(network)} edges={len(records)} "
        f"junctions={sum(value >= 3 for value in degrees.values())} "
        f"ends={sum(value == 1 for value in degrees.values())} "
        f"bridges={sum(record[-1] for record in records)} wraps={len(wrap_edges)}"
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: build_l14_road_scenario.py <biq-window.csv> <road-scenario.csv>", file=sys.stderr)
        return 2
    write_scenario(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
