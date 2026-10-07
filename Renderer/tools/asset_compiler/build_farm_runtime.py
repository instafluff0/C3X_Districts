#!/usr/bin/env python3
"""Build the compact generic farm bundle for Renderer Lab."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import struct
import sys
from collections import deque
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Renderer.tools.asset_compiler.build_mine_runtime import (
    MAGIC,
    bundle_string,
    collect_parts,
    group_payload,
    merged_asset,
)


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Counter-clockwise hull (y down: clockwise on screen)."""
    points = sorted(set(points))
    if len(points) < 3:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def clip_convex(polygon: list[tuple[float, float]], hull: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman: the part of polygon inside a counter-clockwise hull."""
    for index in range(len(hull)):
        a, b = hull[index], hull[(index + 1) % len(hull)]

        def inside(point):
            return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0
        result = []
        for k in range(len(polygon)):
            p, q = polygon[k], polygon[(k + 1) % len(polygon)]
            if inside(p):
                result.append(p)
            if inside(p) != inside(q):
                dp = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                dq = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
                t = dp / (dp - dq)
                result.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
        polygon = result
        if len(polygon) < 3:
            return []
    return polygon


def patchwork(pack: Path, atlas: str, size: float, smallest: int = 40, cover: bool = False) -> list[dict]:
    """The planted crop atlas as one patchwork of separate field pieces.

    The atlas is Civ VI's complete farm patchwork: about thirty green fields
    of mixed shapes and row directions with soil fringes, and transparent
    paths between them. Each field (its opaque blocks, grown 2.5 blocks to
    keep the fringe) becomes one flat convex piece, gridded finer than a road
    verge so it clips cleanly and drapes closely. The pieces share one
    placement; the side of the whole atlas is `size` tiles. Separate pieces
    let the runtime drop a field that clipping cuts down to a sliver.

    With `cover`, every path block joins its nearest field, so the pieces
    tile the whole atlas (for its opaque copy): the gap-free patchwork that
    dense route networks bound instead of the atlas paths.
    """
    labels = field_labels((pack / atlas).read_bytes())
    blocks = len(labels)
    sizes: dict[int, int] = defaultdict(int)
    for row in labels:
        for label in row:
            sizes[label] += 1
    if cover:
        owner = [[label if label and sizes[label] >= smallest else 0 for label in row] for row in labels]
        queue = deque((x, y) for y in range(blocks) for x in range(blocks) if owner[y][x])
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < blocks and 0 <= ny < blocks and not owner[ny][nx]:
                    owner[ny][nx] = owner[y][x]
                    queue.append((nx, ny))
        labels = owner
    cells: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for y in range(blocks):
        for x in range(blocks):
            if labels[y][x]:
                cells[labels[y][x]].append((x, y))
    grow, step = (1.0 if cover else 2.5), .078 / size * blocks
    pieces = []
    for label in sorted(cells):
        if len(cells[label]) < smallest:
            continue
        hull = convex_hull([(min(blocks, max(0, x + dx)), min(blocks, max(0, y + dy)))
                            for x, y in cells[label] for dx in (-grow, 1 + grow) for dy in (-grow, 1 + grow)])
        x0, x1 = min(p[0] for p in hull), max(p[0] for p in hull)
        y0, y1 = min(p[1] for p in hull), max(p[1] for p in hull)
        lookup, vertices, indices = {}, [], []

        def vertex(point):
            key = (round(point[0], 4), round(point[1], 4))
            if key not in lookup:
                lookup[key] = len(vertices)
                u, v = point[0] / blocks, point[1] / blocks
                # Same orientation as the source decal: u follows +x, v follows +y.
                vertices.append({"position": [(u - .5) * size, (v - .5) * size, .002],
                                 "normal": [0.0, 0.0, 1.0], "uv0": [u, v]})
            return lookup[key]
        columns, rows = max(1, math.ceil((x1 - x0) / step)), max(1, math.ceil((y1 - y0) / step))
        for row in range(rows):
            for column in range(columns):
                cx0, cy0 = x0 + (x1 - x0) * column / columns, y0 + (y1 - y0) * row / rows
                cx1, cy1 = x0 + (x1 - x0) * (column + 1) / columns, y0 + (y1 - y0) * (row + 1) / rows
                polygon = clip_convex([(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)], hull)
                ring = [vertex(point) for point in polygon]
                for k in range(1, len(ring) - 1):
                    indices.extend((ring[0], ring[k], ring[k + 1]))
        pieces.append({"vertices": vertices, "topology": {"indices": indices}})
    return pieces


def solid_texture(pack: Path, source: str) -> str:
    """An opaque copy of a BC3 atlas: only its alpha blocks change (no colour
    recompression), so the colour filled under the source's transparent paths
    shows as grassy lanes between the fields on every terrain."""
    data = bytearray((pack / source).read_bytes())
    if data[:4] != b"DDS " or data[84:88] != b"DX10" or struct.unpack_from("<I", data, 128)[0] not in (77, 78):
        raise ValueError(f"Solid patchwork needs a BC3 DX10 atlas: {source}")
    for at in range(148, len(data) - 15, 16):
        data[at:at + 8] = b"\xff\xff\x00\x00\x00\x00\x00\x00"
    target = "textures/farm/patchwork_solid.dds"
    (pack / target).parent.mkdir(parents=True, exist_ok=True)
    (pack / target).write_bytes(data)
    return target


def field_labels(data: bytes) -> list[list[int]]:
    """Each planted field of a BC3 atlas (its opaque alpha) as a numbered
    region on the top mip's 4x4 block grid; 0 is a path between fields."""
    blocks = struct.unpack_from("<I", data, 16)[0] // 4
    solid = []
    for index in range(blocks * blocks):
        at = 148 + index * 16
        a0, a1 = data[at], data[at + 1]
        palette = [a0, a1] + ([((7 - k) * a0 + k * a1) // 7 for k in range(1, 7)] if a0 > a1 else
                              [((5 - k) * a0 + k * a1) // 5 for k in range(1, 5)] + [0, 255])
        bits = int.from_bytes(data[at + 2:at + 8], "little")
        solid.append(sum(palette[(bits >> (3 * t)) & 7] for t in range(16)) > 16 * 128)
    labels = [[0] * blocks for _ in range(blocks)]
    count = 0
    for start in range(blocks * blocks):
        if not solid[start] or labels[start // blocks][start % blocks]:
            continue
        count += 1
        queue = deque([start])
        labels[start // blocks][start % blocks] = count
        while queue:
            at = queue.popleft()
            y, x = divmod(at, blocks)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < blocks and 0 <= nx < blocks and solid[ny * blocks + nx] and not labels[ny][nx]:
                    labels[ny][nx] = count
                    queue.append(ny * blocks + nx)
    # Soil flecks inside a field are small transparent holes, not paths: they
    # join the field around them.
    seen = [[False] * blocks for _ in range(blocks)]
    for start in range(blocks * blocks):
        y0, x0 = divmod(start, blocks)
        if labels[y0][x0] or seen[y0][x0]:
            continue
        hole, around, queue = [], set(), deque([(y0, x0)])
        seen[y0][x0] = True
        while queue:
            y, x = queue.popleft()
            hole.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if not (0 <= ny < blocks and 0 <= nx < blocks):
                    around.add(0)
                elif labels[ny][nx]:
                    around.add(labels[ny][nx])
                elif not seen[ny][nx]:
                    seen[ny][nx] = True
                    queue.append((ny, nx))
        if len(hole) < 64 and len(around) == 1 and 0 not in around:
            for y, x in hole:
                labels[y][x] = next(iter(around))
    return labels


def ripe_rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Green crop to pale ripe straw; soil fringes keep their colour."""
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    degrees = h * 360
    if s < .05 or degrees < 48 or degrees > 190:
        return r, g, b
    weight = min(1.0, (degrees - 48) / 14)
    target = 48 - 4 * min(1.0, (degrees - 60) / 60)
    degrees += (target - degrees) * weight
    s = min(1.0, s * (1 + .10 * weight))
    v = min(1.0, v * (1 + .70 * weight))
    r, g, b = colorsys.hsv_to_rgb(degrees / 360, s, v)
    return round(r * 255), round(g * 255), round(b * 255)


def ripe_texture(pack: Path, source: str, fields: str, target: str = "textures/farm/patchwork_ripe.dds",
                 share: float = .8) -> str:
    """A copy of the patchwork texture with about `share` of its fields ripe.
    Only BC1 colour endpoints change (alpha and indices are kept), and whole
    fields of the planted atlas `fields` turn together."""
    data = bytearray((pack / source).read_bytes())
    labels = field_labels((pack / fields).read_bytes())
    width = struct.unpack_from("<I", data, 16)[0]
    mips = max(1, struct.unpack_from("<I", data, 28)[0])
    at = 148
    for level in range(mips):
        blocks = max(1, (width >> level) // 4)
        for by in range(blocks):
            for bx in range(blocks):
                label = labels[min(len(labels) - 1, by << level)][min(len(labels) - 1, bx << level)]
                if label and (label * 2654435761) % 1000 < share * 1000:
                    for k in (8, 10):
                        c = struct.unpack_from("<H", data, at + k)[0]
                        r, g, b = ripe_rgb((c >> 11) * 255 // 31, (c >> 5 & 63) * 255 // 63, (c & 31) * 255 // 31)
                        struct.pack_into("<H", data, at + k,
                                         (r * 31 + 127) // 255 << 11 | (g * 63 + 127) // 255 << 5 | (b * 31 + 127) // 255)
                at += 16
    (pack / target).parent.mkdir(parents=True, exist_ok=True)
    (pack / target).write_bytes(data)
    return target


def build(pack: Path, runtime_name: str = "farm_runtime.bin",
          field_size: float = 3.3, ripe: tuple[str, ...] = ("Wheat",)) -> Path:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads(
        (pack / manifest["improvement_catalog"]).read_text(encoding="utf-8")
    )
    crop = catalog["farm"]["crop_styles"][0]["pieces"][1]
    roots: list[tuple[str, str]] = []
    for era_index, era in enumerate(catalog["farm"]["eras"]):
        roots.extend(
            [
                (f"farm_{era_index}:building", era["building_pieces"][1]),
                (f"farm_{era_index}:crop", crop),
                (f"farm_{era_index}:tree", era["tile_bases"][0]),
            ]
        )
    all_parts = {
        role: collect_parts(pack, manifest, asset_id)
        for role, asset_id in roots
    }
    crop_counts: dict[str, int] = defaultdict(int)
    for role, parts in all_parts.items():
        if role.endswith(":crop"):
            for _mesh, base, _emissive in parts:
                crop_counts[base] += 1
    if len(crop_counts) != 5:
        raise ValueError("Farm crop source should contain five authored materials")
    ranked_crops = sorted(crop_counts, key=lambda item: (-crop_counts[item], item))
    # Six bound color slots serve farms. Four field palettes leave room for
    # actual source tree canopies and buildings; the fifth crop palette is a
    # soil/path duplicate and cannot justify stripping all raised geometry.
    base_textures = [ranked_crops[i] for i in (0, 2, 3, 4)]
    for role_name in (":tree", ":building"):
        counts: dict[str, int] = defaultdict(int)
        for role, parts in all_parts.items():
            if role.endswith(role_name):
                for mesh, base, _emissive in parts:
                    if max(vertex["position"][2] for vertex in mesh["vertices"]) > 0.02:
                        counts[base] += len(mesh["vertices"])
        choice = max(counts, key=counts.get)
        if choice in base_textures:
            raise ValueError("Farm raised material duplicates a field palette")
        base_textures.append(choice)
    emissive_textures = sorted(
        {
            emissive
            for parts in all_parts.values()
            for _mesh, _base, emissive in parts
            if emissive
        }
    )
    if len(emissive_textures) != 2:
        raise ValueError("Compact farm bundle expects two confirmed emissive channels")
    textures = base_textures + emissive_textures
    # The kit uses no tan, muddy or soil palettes. Slot 2 carries the opaque
    # patchwork for dense route networks; slots 1 and 3 the ripe copies of
    # both patchworks for resource kits.
    planted = textures[0]
    textures[2] = solid_texture(pack, planted)
    if ripe:
        textures[1] = ripe_texture(pack, planted, planted)
        textures[3] = ripe_texture(pack, textures[2], planted, "textures/farm/patchwork_ripe_solid.dds")
    fields = patchwork(pack, planted, field_size)
    dense_fields = patchwork(pack, planted, field_size, cover=True)
    assets: list[bytes] = []
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for role, _asset_id in roots:
        era = int(role.split(":", 1)[0].rsplit("_", 1)[1])
        if role.endswith(":tree"):
            # A centered source tree can be scattered independently along the
            # plots without importing its large prearranged tile cluster.
            candidates = [mesh for mesh, base, _emissive in all_parts[role]
                          if base == base_textures[4] and
                          max(vertex["position"][2] for vertex in mesh["vertices"]) > .025]
            if not candidates:
                raise ValueError(f"No raised source tree in {role}")
            tree = max(candidates, key=lambda mesh: max(
                vertex["position"][2] for vertex in mesh["vertices"]))
            cx = (min(vertex["position"][0] for vertex in tree["vertices"]) +
                  max(vertex["position"][0] for vertex in tree["vertices"])) * .5
            cy = (min(vertex["position"][1] for vertex in tree["vertices"]) +
                  max(vertex["position"][1] for vertex in tree["vertices"])) * .5
            centered = {**tree, "vertices": [
                {**vertex, "position": [vertex["position"][0] - cx,
                                        vertex["position"][1] - cy,
                                        vertex["position"][2]]}
                for vertex in tree["vertices"]]}
            asset_index = len(assets)
            assets.append(merged_asset(f"{role}:source", 4, 0, [centered]))
            grouped[era].append((asset_index, .04))
            continue
        if role.endswith(":building"):
            candidates = [mesh for mesh, base, _emissive in all_parts[role]
                          if base == base_textures[5] and
                          max(vertex["position"][2] for vertex in mesh["vertices"]) > .03]
            if not candidates:
                raise ValueError(f"No raised source building in {role}")
            building = max(candidates, key=lambda mesh: len(mesh["vertices"]))
            cx = (min(vertex["position"][0] for vertex in building["vertices"]) +
                  max(vertex["position"][0] for vertex in building["vertices"])) * .5
            cy = (min(vertex["position"][1] for vertex in building["vertices"]) +
                  max(vertex["position"][1] for vertex in building["vertices"])) * .5
            centered = {**building, "vertices": [
                {**vertex, "position": [vertex["position"][0] - cx,
                                        vertex["position"][1] - cy,
                                        vertex["position"][2]]}
                for vertex in building["vertices"]]}
            asset_index = len(assets)
            assets.append(merged_asset(f"{role}:source", 5, 0, [centered]))
            grouped[era].append((asset_index, .08))
            continue
        # The kit's patchwork pieces, on the planted (first) crop texture,
        # shared by every era.
        if era == 0:
            first_field = len(assets)
            for index, mesh in enumerate(fields):
                assets.append(merged_asset(f"farm_kit:crop:field{index}", 0, 0, [mesh]))
        grouped[era].extend((first_field + index, .71) for index in range(len(fields)))
    groups = [group_payload(f"farm_{era}", grouped[era]) for era in range(3)]
    # Marks this pack as a farm kit: the runtime lays its patchwork out alike
    # on every terrain and keeps routes, resources and water open.
    groups.append(group_payload("farm_kit", grouped[0][:1]))

    # Dense route networks bound the gap-free patchwork instead.
    first_dense = len(assets)
    for index, mesh in enumerate(dense_fields):
        assets.append(merged_asset(f"farm_kit:crop:dense{index}", 2, 0, [mesh]))
    groups.append(group_payload("farm_kit:dense", [(first_dense + index, .71) for index in range(len(dense_fields))]))
    if ripe:
        # A resource's own kit: its farm grows ripe fields ("NAME"), or is the
        # resource's own planting and keeps no yard around it ("NAME:crop").
        first_ripe = len(assets)
        for index, mesh in enumerate(fields):
            assets.append(merged_asset(f"farm_kit:crop:ripe{index}", 1, 0, [mesh]))
        first_ripe_dense = len(assets)
        for index, mesh in enumerate(dense_fields):
            assets.append(merged_asset(f"farm_kit:crop:ripedense{index}", 3, 0, [mesh]))
        for name in ripe:
            base, crop = name.lower().split(":")[0], name.lower().endswith(":crop")
            groups.append(group_payload(f"farm_kit:{name.lower()}",
                                        [(first_ripe + index, .71) for index in range(len(fields))]))
            groups.append(group_payload(f"farm_kit:{base}:dense" + (":crop" if crop else ""),
                                        [(first_ripe_dense + index, .71) for index in range(len(dense_fields))]))
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target = pack / runtime_name
    target.write_bytes(output)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack", type=Path, default=Path("Renderer/packs/ImprovementsNormalized")
    )
    parser.add_argument("--output", default="farm_runtime.bin",
                        help="runtime file name in the pack; Lab candidates use e.g. farm_runtime~kit.bin")
    parser.add_argument("--field-size", type=float, default=3.3,
                        help="patchwork side in tiles (larger means fewer, bigger fields per tile)")
    parser.add_argument("--ripe", nargs="*", default=["Wheat"],
                        help="resources whose farms grow ripe fields, e.g. Wheat (yard kept) or Wheat:crop (no yard)")
    args = parser.parse_args()
    if "/" in args.output or "\\" in args.output or not args.output.endswith(".bin"):
        parser.error("--output is a .bin file name inside the pack")
    target = build(args.pack.resolve(), args.output, args.field_size, tuple(args.ripe))
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
