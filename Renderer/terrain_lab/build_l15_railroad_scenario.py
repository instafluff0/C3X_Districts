#!/usr/bin/env python3
"""Build a deterministic connected L15 railroad subset of the accepted Lab road graph."""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import deque
from pathlib import Path


MAGIC = "C3X_LAB_RAILROAD_SCENARIO_V0"


def load_roads(path: Path):
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    if not rows or rows[0][0] != "C3X_LAB_ROAD_SCENARIO_V0":
        raise ValueError("not an L14 Lab road scenario")
    columns, row_count = int(rows[0][1]), int(rows[0][2])
    edges = []
    for row in rows[1:]:
        a = int(row[0]), int(row[1])
        b = int(row[2]), int(row[3])
        edges.append((a, b, int(row[4]), int(row[7])))
    return columns, row_count, edges


def build_subset(columns: int, rows: int, edges):
    adjacency = {}
    edge_data = {}
    for a, b, wraps, bridge in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        edge_data[frozenset((a, b))] = wraps, bridge
    root = min(adjacency, key=lambda n: (abs(n[0] - columns // 2) + abs(n[1] - rows // 2), n))
    targets = {
        node for node in adjacency
        if (node[0] * 23 + node[1] * 37) % 13 < 2
        or len(adjacency[node]) >= 4
    }
    targets.update(node for node in adjacency if node[0] in (0, columns - 1))
    selected = set()
    selected_nodes = {root}
    for target in sorted(targets, key=lambda n: (abs(n[0] - root[0]) + abs(n[1] - root[1]), n)):
        queue = deque([target])
        parent = {target: None}
        found = None
        while queue and found is None:
            node = queue.popleft()
            if node in selected_nodes:
                found = node
                break
            for nxt in sorted(adjacency[node]):
                if nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        if found is None:
            continue
        path = [found]
        while path[-1] != target:
            path.append(parent[path[-1]])
        selected_nodes.update(path)
        selected.update(frozenset((a, b)) for a, b in zip(path, path[1:]))
    # Preserve deterministic alternate routes and every reachable wrap witness.
    for key, (wraps, _bridge) in edge_data.items():
        a, b = tuple(key)
        if a in selected_nodes and b in selected_nodes and (wraps or (a[0] * 11 + a[1] * 7 + b[0] * 5 + b[1]) % 7 == 0):
            selected.add(key)
    records = []
    for key in sorted(selected, key=lambda k: sorted(k)):
        a, b = sorted(key)
        wraps, bridge = edge_data[key]
        if wraps and a[0] == 0:
            a, b = b, a
        pillaged = int((a[0] * 29 + a[1] * 17 + b[0] * 7 + b[1]) % 31 == 0)
        records.append((*a, *b, wraps, 4, pillaged, bridge))
    return records


def write_scenario(source: Path, output: Path) -> None:
    columns, rows, edges = load_roads(source)
    records = build_subset(columns, rows, edges)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([MAGIC, columns, rows, len(records), source_hash, "lab_augmentation"])
        writer.writerows(records)
    nodes = {(r[0], r[1]) for r in records} | {(r[2], r[3]) for r in records}
    degree = {node: 0 for node in nodes}
    for x0, y0, x1, y1, *_ in records:
        degree[x0, y0] += 1
        degree[x1, y1] += 1
    print(
        f"L15 railroad scenario: nodes={len(nodes)} edges={len(records)} "
        f"junctions={sum(v >= 3 for v in degree.values())} "
        f"ends={sum(v == 1 for v in degree.values())} "
        f"bridges={sum(r[-1] for r in records)} wraps={sum(r[4] for r in records)}"
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: build_l15_railroad_scenario.py <road-scenario.csv> <railroad-scenario.csv>", file=sys.stderr)
        return 2
    write_scenario(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
