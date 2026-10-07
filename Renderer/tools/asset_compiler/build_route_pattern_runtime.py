"""Compile connection-mask route patterns into a generic centerline pack.

The importer reads a Civ III style route sheet: a 16-column grid of 128x64
isometric cells, one per 8-neighbor connection mask (bit k is neighbor k+1:
NE, E, SE, S, SW, W, NW, N), with two transparent key colors. Cells past the
first 256 are further variants of the fully connected mask (the railroad
sheet has 16, which Civ III picks at random). Each cell's painted path is
thinned to a centerline graph, joined exactly to the shared edge midpoint or
corner of every connected neighbor, lightly smoothed and written in
tile-local (u, v) coordinates. Runtime receives only these generic polylines;
it never reads the source sheet or its palette.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Renderer/packs/RoutePatternSources"
OUTPUT = ROOT / "Renderer/packs/RoutePatternsRuntime"
# Sheet, the closing radius (cell pixels) that fills a painted ladder of rails
# and sleepers into one band before thinning, and the longest dead-end branch
# off a junction that is thinning noise (a closed band's blunt ends leave
# slightly longer ones).
SHEETS = {"road": ("roads.pcx", 0, 4.0), "railroad": ("railroads.pcx", 1, 7.0)}
MAGIC = b"C3XRPAT1"
CELL_W, CELL_H, GRID = 128, 64, 16
KEY_COLORS = (254, 255)  # palette indices of the two transparent key colors
# Exact shared join point of each neighbor direction, in cell pixels.
CONNECT = ((96.0, 16.0), (128.0, 32.0), (96.0, 48.0), (64.0, 64.0),
           (32.0, 48.0), (0.0, 32.0), (32.0, 16.0), (64.0, 0.0))
SPUR = 4.0        # unconnected skeleton branches shorter than this are thinning noise
BUBBLE = 24.0     # rings or duplicate paths shorter than this are thinning noise
ATTACH = 9.0      # a connected direction must have painted path this close to its join
CLUSTER = 3.0     # cell pixels; junction pixels this close are one junction
SMOOTHING = 1.1   # cell pixels; removes the paint staircase but keeps painted bends
SPACING = 2.0     # resampling spacing in cell pixels, before simplification
SIMPLIFY = 0.3    # cell pixels a dropped point may deviate (under a screen pixel at 3x)
MAX_SEGMENT = 5.0 # cell pixels; keeps strips sampling the terrain they drape over
SCALE = 3         # thin an upsampled mask; 2px diagonal strokes otherwise erode away


def read_pcx(data: bytes):
    if len(data) < 128 + 769 or data[0] != 10 or data[3] != 8 or data[65] != 1:
        raise ValueError("Expected an 8-bit single-plane PCX route sheet")
    xmin, ymin, xmax, ymax = struct.unpack_from("<4H", data, 4)
    stride = struct.unpack_from("<H", data, 66)[0]
    width, height = xmax - xmin + 1, ymax - ymin + 1
    out = bytearray()
    index, need = 128, stride * height
    while len(out) < need and index < len(data) - 769:
        value = data[index]
        index += 1
        if value >= 0xC0:
            out += bytes((data[index],)) * (value & 0x3F)
            index += 1
        else:
            out.append(value)
    if len(out) < need:
        raise ValueError("Truncated PCX route sheet")
    return np.frombuffer(bytes(out[:need]), np.uint8).reshape(height, stride)[:, :width]


def thin(image):
    """Zhang-Suen thinning of a boolean image."""
    im = np.pad(image.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = im
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9, center = p[1:-1, :-2], p[:-2, :-2], p[1:-1, 1:-1]
            ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            count = sum(x.astype(np.int32) for x in ring[:8])
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                remove = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                remove = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove &= (center == 1) & (count >= 2) & (count <= 6) & (transitions == 1)
            if remove.any():
                im[1:-1, 1:-1][remove] = 0
                changed = True
    return im[1:-1, 1:-1].astype(bool)


def pixel_graph(skeleton):
    """8-connected skeleton graph without redundant diagonal shortcuts."""
    pixels = {(int(x), int(y)) for y, x in zip(*np.nonzero(skeleton))}
    links = {p: set() for p in pixels}
    for x, y in pixels:
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            q = (x + dx, y + dy)
            if q not in pixels:
                continue
            if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                continue
            links[(x, y)].add(q)
            links[q].add((x, y))
    return links


def walk(links, start, first):
    path = [start, first]
    while len(links[path[-1]]) == 2:
        nxt = next(iter(links[path[-1]] - {path[-2]}))
        path.append(nxt)
        if nxt == start:
            break
    return path


def prune(links, keep, spur=SPUR):
    while True:
        removed = False
        for leaf in [p for p, n in links.items() if len(n) == 1 and p not in keep]:
            if leaf not in links or len(links[leaf]) != 1:
                continue
            path = walk(links, leaf, next(iter(links[leaf])))
            end = path[-1]
            if len(links[end]) < 3:
                continue  # an isolated stroke or a stroke to a join stays
            length = sum(math.dist(a, b) for a, b in zip(path, path[1:])) / SCALE
            if length >= spur:
                continue
            for a, b in zip(path, path[1:]):
                links[a].discard(b)
                links[b].discard(a)
            for p in path[:-1]:
                if not links[p]:
                    del links[p]
            removed = True
        if not removed:
            return


def chains(links):
    nodes = {p for p, n in links.items() if len(n) != 2}
    seen, result = set(), []
    for node in sorted(nodes, key=lambda p: (p[1], p[0])):
        for first in sorted(links[node], key=lambda p: (p[1], p[0])):
            if (node, first) in seen:
                continue
            path = walk(links, node, first)
            for a, b in zip(path, path[1:]):
                seen.add((a, b))
                seen.add((b, a))
            result.append(path)
    remaining = {p for p in links if p not in nodes and not any((p, q) in seen for q in links[p])}
    while remaining:
        start = min(remaining, key=lambda p: (p[1], p[0]))
        path = walk(links, start, min(links[start], key=lambda p: (p[1], p[0])))
        remaining.difference_update(path)
        for a, b in zip(path, path[1:]):
            seen.add((a, b))
            seen.add((b, a))
        result.append(path)
    return result


def resample(points, spacing):
    steps = np.hypot(*np.diff(points, axis=0).T)
    total = float(steps.sum())
    if total < 1e-6:
        return None
    count = max(1, int(round(total / spacing)))
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    targets = np.linspace(0.0, total, count + 1)
    out = np.stack([np.interp(targets, distance, points[:, 0]),
                    np.interp(targets, distance, points[:, 1])], axis=1)
    out[0], out[-1] = points[0], points[-1]  # joins stay exact across tiles
    return out


def simplify(points, tolerance):
    """Douglas-Peucker; the first and last points are kept exactly."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    length = float(np.hypot(*ab))
    offsets = points - a
    distance = (np.abs(ab[0] * offsets[:, 1] - ab[1] * offsets[:, 0]) / length if length > 1e-9
                else np.hypot(offsets[:, 0], offsets[:, 1]))
    index = int(np.argmax(distance))
    if distance[index] <= tolerance:
        return np.array([a, b])
    return np.vstack([simplify(points[:index + 1], tolerance)[:-1], simplify(points[index:], tolerance)])


def bound_segments(points, longest):
    out = [points[0]]
    for p, q in zip(points, points[1:]):
        count = max(1, int(np.ceil(np.hypot(*(q - p)) / longest)))
        out.extend(p + (q - p) * k / count for k in range(1, count))
        out.append(q)  # exact, so joins stay on their shared points
    return np.array(out)


def smooth_and_resample(points):
    # Laplacian smoothing at a fixed fine spacing approximates a Gaussian of
    # SMOOTHING cell pixels; endpoints stay on their exact joins/junctions.
    # Simplification then drops points on near-straight stretches, which keeps
    # dense late-game road networks within the renderer's geometry budget.
    points = resample(np.asarray(points, np.float64), 0.5)
    if points is None:
        return None
    for _ in range(int(round(2.0 * SMOOTHING * SMOOTHING / 0.25))):
        points[1:-1] = 0.5 * points[1:-1] + 0.25 * (points[:-2] + points[2:])
    points = resample(points, SPACING)
    return bound_segments(simplify(points, SIMPLIFY), MAX_SEGMENT)


def cell_point(pixel):
    return ((pixel[0] + 0.5) / SCALE, (pixel[1] + 0.5) / SCALE)


def close(painted, radius):
    """Morphological closing: fills gaps up to about twice the radius wide."""
    def grow(mask):
        padded = np.pad(mask, 1)
        out = np.zeros_like(mask)
        for dy in range(3):
            for dx in range(3):
                out |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
        return out
    filled = painted
    for _ in range(radius):
        filled = grow(filled)
    empty = ~filled
    for _ in range(radius):
        empty = grow(empty)
    return ~empty


def extract(cell, mask, closing=0, spur=SPUR):
    """Centerline polylines for one connection mask, in cell pixel units."""
    painted = close(~np.isin(cell, KEY_COLORS), closing)
    links = pixel_graph(thin(painted.repeat(SCALE, 0).repeat(SCALE, 1)))
    if not links:
        raise ValueError(f"Route mask {mask} has no painted path")
    # Snap each connected direction to its exact shared join point.
    joins = {}
    for direction in range(8):
        if not (mask >> direction) & 1:
            continue
        target = CONNECT[direction]
        nearest = min(links, key=lambda p: math.dist(cell_point(p), target))
        distance = math.dist(cell_point(nearest), target)
        if distance > ATTACH:
            raise ValueError(f"Route mask {mask} has no path near direction {direction} ({distance:.1f} px)")
        joins[direction] = nearest
    prune(links, set(joins.values()), spur)
    junctions = {p for p, n in links.items() if len(n) >= 3}
    lines = []
    paths = chains(links)
    # Thick painted bends can thin into a tiny ring or a two-path bubble.
    # Keep only the shortest path between the same pair of nodes when the
    # alternatives are that small, and drop tiny rings entirely.
    def length(path):
        return sum(math.dist(a, b) for a, b in zip(path, path[1:])) / SCALE
    shortest = {}
    for path in paths:
        key = frozenset((path[0], path[-1]))
        if key not in shortest or length(path) < length(shortest[key]):
            shortest[key] = path
    paths = [path for path in paths if not (
        (path[0] == path[-1] and length(path) < BUBBLE) or
        (shortest[frozenset((path[0], path[-1]))] is not path and length(path) < BUBBLE))]
    # Pieces carry their end keys: a join direction, or the pixel node.
    pieces = []
    for path in paths:
        ends = [-1, -1]
        for side, pixel in ((0, path[0]), (1, path[-1])):
            for direction, join in joins.items():
                if join == pixel and pixel not in junctions:
                    ends[side] = direction
        # A short stub between two junction pixels is the same crossing.
        if len(path) <= SCALE + 1 and path[0] in junctions and path[-1] in junctions:
            continue
        points = [cell_point(p) for p in path]
        if ends[0] >= 0:
            points.insert(0, CONNECT[ends[0]])
        if ends[1] >= 0:
            points.append(CONNECT[ends[1]])
        pieces.append([("join", ends[0]) if ends[0] >= 0 else ("node", path[0]),
                       ("join", ends[1]) if ends[1] >= 0 else ("node", path[-1]), points])
    # A join on a through-path pixel (degree two) or junction still needs its
    # own short connector to the exact shared point.
    for direction, join in joins.items():
        if any(("join", direction) in (a, b) for a, b, _ in pieces):
            continue
        pieces.append([("node", join), ("join", direction), [cell_point(join), CONNECT[direction]]])
    # Thinning a wide band leaves several junction pixels a pixel or two
    # apart. Each such cluster is one junction: its pieces end on the
    # cluster's center and the tiny pieces inside it are dropped, so no short
    # crosswise stub remains there.
    node_pixels = sorted({key[1] for piece in pieces for key in piece[:2] if key[0] == "node"})
    parent = {pixel: pixel for pixel in node_pixels}
    def root(pixel):
        while parent[pixel] != pixel:
            parent[pixel] = parent[parent[pixel]]
            pixel = parent[pixel]
        return pixel
    for index, a in enumerate(node_pixels):
        for b in node_pixels[index + 1:]:
            if math.dist(cell_point(a), cell_point(b)) <= CLUSTER:
                parent[root(a)] = root(b)
    members = {}
    for pixel in node_pixels:
        members.setdefault(root(pixel), []).append(cell_point(pixel))
    center = {key: tuple(np.mean(points, axis=0)) for key, points in members.items()}
    clustered = []
    for a, b, points in pieces:
        a = ("node", root(a[1])) if a[0] == "node" else a
        b = ("node", root(b[1])) if b[0] == "node" else b
        points = list(points)
        if a[0] == "node":
            points[0] = center[a[1]]
        if b[0] == "node":
            points[-1] = center[b[1]]
        if a == b and sum(math.dist(p, q) for p, q in zip(points, points[1:])) < 2 * CLUSTER:
            continue
        clustered.append([a, b, points])
    pieces = clustered
    # Pieces meeting at a node with no other branch are one path: join them
    # before smoothing so the path stays continuous through that node.
    looped = set()
    while True:
        degree = {}
        for a, b, _ in pieces:
            for key in (a, b):
                if key[0] == "node":
                    degree[key] = degree.get(key, 0) + 1
        node = next((key for key, count in degree.items() if count == 2 and key not in looped), None)
        if node is None:
            break
        meeting = [piece for piece in pieces if node in (piece[0], piece[1])]
        if len(meeting) != 2:
            looped.add(node)  # one closed piece returns to its own start
            continue
        first, second = meeting
        if first[0] == node:
            first = [first[1], first[0], first[2][::-1]]
        if second[1] == node:
            second = [second[1], second[0], second[2][::-1]]
        pieces = [piece for piece in pieces if piece not in meeting]
        pieces.append([first[0], second[1], first[2] + second[2][1:]])
    lines = []
    for a, b, points in pieces:
        resampled = smooth_and_resample(points)
        if resampled is not None:
            lines.append((a[1] if a[0] == "join" else -1, b[1] if b[0] == "join" else -1, resampled))
    return lines


def to_tile(points):
    """Cell pixels to tile-local (u, v); u runs toward NE/E, v toward NW/W."""
    x, y = points[:, 0] / (CELL_W * 0.5), points[:, 1] / (CELL_H * 0.5)
    return np.stack([(x - 1.0 + y) * 0.5, (y - x + 1.0) * 0.5], axis=1)


def compile_sheet(data, closing=0, spur=SPUR):
    image = read_pcx(data)
    rows = image.shape[0] // CELL_H
    if image.shape[1] != CELL_W * GRID or image.shape[0] % CELL_H or rows < GRID or rows > GRID + 1:
        raise ValueError("Expected a 2048-pixel-wide route sheet of 16 or 17 cell rows")
    cells = image.reshape(rows, CELL_H, GRID, CELL_W).transpose(0, 2, 1, 3).reshape(rows * GRID, CELL_H, CELL_W)
    # Cells past 255 are variants of the fully connected mask.
    return [extract(cells[index], min(index, 255), closing, spur) for index in range(rows * GRID)]


def encode(patterns):
    offsets, records, points = [0], [], []
    for lines in patterns:
        for start, end, line in lines:
            uv = to_tile(line)
            if len(uv) > 0xFFFF:
                raise ValueError("Route pattern line is too long")
            records.append(struct.pack("<IHbb", len(points), len(uv), start, end))
            points.extend(map(tuple, uv))
        offsets.append(len(records))
    header = MAGIC + struct.pack("<III", len(patterns), len(records), len(points))
    body = struct.pack(f"<{len(offsets)}I", *offsets) + b"".join(records)
    body += struct.pack(f"<{2 * len(points)}f", *(c for p in points for c in p))
    return header + body


def sources():
    return [(SOURCE / name).relative_to(ROOT).as_posix() for name, *_ in SHEETS.values()]


def build(output=OUTPUT, source=SOURCE):
    consumed, manifest = {}, {"schema": "c3x.route_patterns.v1", "sets": {}}
    output.mkdir(parents=True, exist_ok=True)
    for kind, (name, closing, spur) in SHEETS.items():
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.relative_to(ROOT)}; run "
                                    "Renderer/tools/asset_compiler/build_route_pattern_runtime.py import")
        data = path.read_bytes()
        consumed[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
        patterns = compile_sheet(data, closing, spur)
        payload = encode(patterns)
        (output / f"{kind}_patterns.bin").write_bytes(payload)
        manifest["sets"][kind] = {"file": f"{kind}_patterns.bin", "masks": len(patterns),
                                  "lines": sum(len(p) for p in patterns),
                                  "sha256": hashlib.sha256(payload).hexdigest()}
    manifest["source_sha256"] = consumed
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return consumed


def civ3_root():
    configured = os.environ.get("C3X_CIV3_ROOT")
    # The default C3X layout is <Civ III>/Conquests/<C3X checkout>.
    return Path(configured).expanduser() if configured else ROOT.parents[1]


def import_sheets(root=None):
    root = Path(root) if root else civ3_root()
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name, *_ in SHEETS.values():
        found = root / "Art/Terrain" / name
        if not found.is_file():
            raise FileNotFoundError(f"Route sheet not found under the Civ III root: Art/Terrain/{name}")
        shutil.copyfile(found, SOURCE / name)
    (SOURCE / "README.md").write_text(
        "Local copies of the installed game's route sheets. Ignored by Git; derived\n"
        "patterns are compiled into RoutePatternsRuntime and never redistributed.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("import", "build"))
    parser.add_argument("--civ3-root", help="Civ III Complete folder (default: C3X_CIV3_ROOT or ../.. of C3X)")
    args = parser.parse_args(argv)
    if args.command == "import":
        import_sheets(args.civ3_root)
    else:
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
