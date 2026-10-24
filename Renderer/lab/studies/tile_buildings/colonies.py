"""Resource-aware colonies (Lab): the colony stands around its resource.

A Civ III colony is built on a resource tile. Its pieces are laid out around
the resource's footprint, measured as the renderer measures it
(c3x_renderer.cpp resource_footprint: model bounds, animals by a box, decals
left out), with buildings at the back of the tile and props at its sides.
`layout` is the prototype of the runtime placer.

The Lab draws one composition per tile, so `merge` writes the production
resource pack's assets and the chosen resource variant into this study's
composition pack beside the colony pieces: production resource atlases keep
their slots (0-3, 7) and the colony's atlases take the free ones (4-6).
"""
from __future__ import annotations

import json
import math
import shutil
import struct
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PRODUCTION = ROOT / "Renderer/packs/ResourceCompositions"
MAGIC = b"C3XVEG1\0"
GRASSLAND = 2
ANIMAL_HALF = .06          # an animal's half extent (tile units), for want of its bone bounds here
MARGIN = .015              # kept between the colony and the resource, and between pieces
EDGE = .03                 # pieces stay this far inside the tile


def read_pack(path: Path) -> dict:
    data = path.read_bytes()
    at = len(MAGIC)

    def take(form):
        nonlocal at
        values = struct.unpack_from(form, data, at)
        at += struct.calcsize(form)
        return values

    def text():
        nonlocal at
        (length,) = take("<I")
        at += length
        return data[at - length:at].decode()

    version, slots, count, groups = take("<IIII")
    textures = [text() for _ in range(slots)]
    assets = []
    for _ in range(count):
        name = text()
        texture, vertices, indices = take("<III")
        payload = data[at:at + vertices * 32 + indices * 4]
        points = [struct.unpack_from("<3f", payload, k * 32) for k in range(vertices)]
        at += vertices * 32 + indices * 4
        assets.append({"name": name, "texture": texture, "vertices": vertices, "indices": indices,
                       "payload": payload, "points": points})
    group_list = []
    for _ in range(groups):
        name = text()
        (count,) = take("<I")
        # Placements kept as their raw records (<IffIIIIff): asset, scale, variation, ...
        group_list.append((name, [data[at + k * 36:at + k * 36 + 36] for k in range(count)]))
        at += count * 36
    if version == 1:
        return {"version": version, "textures": textures, "assets": assets, "groups": group_list,
                "compositions": [], "aliases": []}
    (count,) = take("<I")
    compositions = []
    for _ in range(count):
        name = text()
        (variants,) = take("<I")
        items = []
        for _ in range(variants):
            mask, instances = take("<II")
            items.append((mask, [list(take("<I6f")) for _ in range(instances)]))
        compositions.append((name, items))
    (count,) = take("<I")
    aliases = [(text(), take("<I")[0]) for _ in range(count)]
    return {"version": version, "textures": textures, "assets": assets, "groups": group_list,
            "compositions": compositions, "aliases": aliases}


def write_pack(pack: dict, path: Path) -> None:
    def text(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<I", len(encoded)) + encoded
    blob = bytearray(MAGIC)
    blob += struct.pack("<IIII", pack["version"], len(pack["textures"]), len(pack["assets"]), 0)
    for texture in pack["textures"]:
        blob += text(texture)
    for asset in pack["assets"]:
        blob += text(asset["name"]) + struct.pack("<III", asset["texture"], asset["vertices"], asset["indices"])
        blob += asset["payload"]
    blob += struct.pack("<I", len(pack["compositions"]))
    for name, variants in pack["compositions"]:
        blob += text(name) + struct.pack("<I", len(variants))
        for mask, instances in variants:
            blob += struct.pack("<II", mask, len(instances))
            for item in instances:
                blob += struct.pack("<I6f", *item)
    blob += struct.pack("<I", len(pack["aliases"]))
    for name, index in pack["aliases"]:
        blob += text(name) + struct.pack("<I", index)
    path.write_bytes(bytes(blob))


@lru_cache(maxsize=None)
def production() -> dict:
    return read_pack(PRODUCTION / "resource_runtime.bin")


def variant(resource: str, terrain: int = GRASSLAND) -> list:
    """The resource's first variant authored for the terrain (else its first)."""
    pack = production()
    found = next(variants for name, variants in pack["compositions"] if name == resource)
    return next((instances for mask, instances in found if mask >> terrain & 1), found[0][1])


def resource_boxes(resource: str, terrain: int = GRASSLAND) -> list:
    """Footprint boxes (u0, v0, u1, v1) in tile UV, as resource_footprint keeps them."""
    pack, boxes, decals = production(), [], []
    for index, u, v, rotation, scale, _lift, _fit in variant(resource, terrain):
        asset = pack["assets"][index]
        if asset["name"].startswith("animated/"):
            half = ANIMAL_HALF * scale
            boxes.append((u - half, v - half, u + half, v + half))
            continue
        if asset["name"].startswith("clearance/"):
            continue
        c, s = math.cos(rotation), math.sin(rotation)
        us = [u + (x * c - y * s) * scale for x, y, _ in asset["points"]]
        vs = [v + (x * s + y * c) * scale for x, y, _ in asset["points"]]
        if asset["name"].startswith("decal/"):
            # A ground decal fades toward its edge: its core (60%), as resource_footprint keeps it.
            cu, cv, hu, hv = (min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2, (max(us) - min(us)) * .3, (max(vs) - min(vs)) * .3
            decals.append((cu - hu, cv - hv, cu + hu, cv + hv))
            continue
        boxes.append((min(us), min(vs), max(us), max(vs)))
    # Ground decals under rocks and plants may span the tile; a resource drawn
    # only as decals (oil's slick) keeps their cores clear.
    return boxes or decals


GRID = 100                 # occupancy cells per tile side (0.01 tile)


def resource_grid(resource: str, terrain: int = GRASSLAND):
    """The tile ground the resource's bodies cover, triangle by triangle, on a
    GRID x GRID occupancy grid indexed [u][v]: tighter than whole-model boxes,
    so a colony can stand in the gaps of a rock cluster. Animals keep a box; a
    resource drawn only as decals keeps their cores."""
    import numpy as np
    pack, grid = production(), np.zeros((GRID, GRID), bool)
    decals = []

    def mark(u0, v0, u1, v1, into):
        a, b = max(0, int(u0 * GRID)), min(GRID - 1, int(u1 * GRID))
        c, d = max(0, int(v0 * GRID)), min(GRID - 1, int(v1 * GRID))
        if a <= b and c <= d:
            into[a:b + 1, c:d + 1] = True

    for index, u, v, rotation, scale, _lift, _fit in variant(resource, terrain):
        asset = pack["assets"][index]
        if asset["name"].startswith("animated/"):
            half = ANIMAL_HALF * scale
            mark(u - half, v - half, u + half, v + half, grid)
            continue
        if asset["name"].startswith("clearance/"):
            continue
        c, s = math.cos(rotation), math.sin(rotation)
        points = [(u + (x * c - y * s) * scale, v + (x * s + y * c) * scale, z) for x, y, z in asset["points"]]
        if asset["name"].startswith("decal/"):
            us, vs = [p[0] for p in points], [p[1] for p in points]
            cu, cv, hu, hv = (min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2, (max(us) - min(us)) * .3, (max(vs) - min(vs)) * .3
            decals.append((cu - hu, cv - hv, cu + hu, cv + hv))
            continue
        payload = asset["payload"]
        indices = struct.unpack_from(f"<{asset['indices']}I", payload, asset["vertices"] * 32)
        for k in range(0, len(indices) - 2, 3):
            triangle = [points[i] for i in indices[k:k + 3]]
            if max(p[2] for p in triangle) * scale < .002:
                continue                  # ground-hugging skirts
            mark(min(p[0] for p in triangle), min(p[1] for p in triangle),
                 max(p[0] for p in triangle), max(p[1] for p in triangle), grid)
    if not grid.any():
        for box in decals:
            mark(*box, grid)
    return grid


def layout(pieces: list, occupied) -> list:
    """Place pieces (name, role, half_u, half_v) on the free ground of an
    occupancy grid (resource_grid).

    Anchors (the pieces that name the colony) first, then other buildings,
    largest first, as far toward the back of the tile (small u + v, the top
    of the screen diamond) as fits; props next to them on the sides. A piece
    that cannot fit shrinks to 85% and 70% before it is left out; an anchor
    shrinks down to half rather than be left out. Returns (name, u, v, factor)
    for each placed piece."""
    import numpy as np
    taken = occupied.copy()
    pad, edge = int(round(MARGIN * GRID)), int(round(EDGE * GRID))
    if pad:
        # The resource keeps its margin: dilate its cells.
        grown = taken.copy()
        for du in range(-pad, pad + 1):
            for dv in range(-pad, pad + 1):
                grown |= np.roll(np.roll(taken, du, 0), dv, 1)
        taken = grown
    centre = (np.arange(GRID) + .5) / GRID
    gu, gv = np.meshgrid(centre, centre, indexing="ij")
    rank = {"anchor": 0, "building": 1, "prop": 2}
    order = sorted(pieces, key=lambda piece: (rank[piece[1]], -piece[2] * piece[3]))
    placed, buildings = [], []
    for name, role, hu, hv in order:
        for factor in (1.0, .85, .7, .6, .5) if role == "anchor" else (1.0, .85, .7):
            ru, rv = max(1, int(math.ceil(hu * factor * GRID))), max(1, int(math.ceil(hv * factor * GRID)))
            table = np.pad(taken.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            rows, cols = np.arange(GRID)[:, None], np.arange(GRID)[None, :]
            u0, u1 = np.clip(rows - ru, 0, GRID), np.clip(rows + ru, 0, GRID)
            v0, v1 = np.clip(cols - rv, 0, GRID), np.clip(cols + rv, 0, GRID)
            free = (table[u1, v1] - table[u0, v1] - table[u1, v0] + table[u0, v0]) == 0
            free &= (rows - ru >= edge) & (rows + ru <= GRID - edge) & (cols - rv >= edge) & (cols + rv <= GRID - edge)
            if not free.any():
                continue
            if role != "prop":
                cost = (gu + gv) + .35 * np.abs(gu - gv)
            else:
                near = np.full(gu.shape, 0.0)
                if buildings:
                    near = np.min([np.hypot(gu - bu, gv - bv) for bu, bv in buildings], axis=0)
                cost = near + .5 * (gu + gv)
            index = np.unravel_index(np.argmin(np.where(free, cost, np.inf)), cost.shape)
            i, j = int(index[0]), int(index[1])
            spread = max(1, pad // 2)
            taken[max(0, i - ru - spread):i + ru + spread, max(0, j - rv - spread):j + rv + spread] = True
            placed.append((name, (i + .5) / GRID, (j + .5) / GRID, factor))
            if role != "prop":
                buildings.append(((i + .5) / GRID, (j + .5) / GRID))
            break
    return placed


def merge(catalog_pack: Path, cells: list) -> dict:
    """Add the production resource assets and each cell's resource variant to
    the study's composition pack (written by kits.bake)."""
    mine, base = read_pack(catalog_pack / "resource_runtime.bin"), production()
    used = sorted({asset["texture"] for asset in base["assets"]})
    free = [slot for slot in range(len(base["textures"])) if slot not in used]
    own = sorted({asset["texture"] for asset in mine["assets"]})
    if len(own) > len(free):
        raise ValueError(f"colony atlases ({len(own)}) exceed the free resource slots {free}")
    slot_of = dict(zip(own, free))
    textures = list(base["textures"])
    folder = catalog_pack / "textures"
    staged = {}
    for old, new in slot_of.items():
        staged[new] = (folder / f"atlas_{old}.dds").read_bytes()
        textures[new] = f"textures/atlas_{new}.dds"
    for stale in folder.glob("*.dds"):
        stale.unlink()
    for slot, data in staged.items():
        (folder / f"atlas_{slot}.dds").write_bytes(data)
    for slot in used:
        shutil.copy2(PRODUCTION / base["textures"][slot], catalog_pack / base["textures"][slot])
    shutil.copy2(PRODUCTION / "textures/unused.dds", folder / "unused.dds")
    assets = [{**asset, "texture": slot_of[asset["texture"]]} for asset in mine["assets"]] + base["assets"]
    offset = len(mine["assets"])
    resource_of = {cell["name"]: cell["resource"] for cell in cells if cell.get("resource")}
    compositions = []
    for name, variants in mine["compositions"]:
        if name in resource_of:
            extra = [[index + offset, *rest] for index, *rest in variant(resource_of[name])]
            # Bodies before veils (Civ VI's silk webbing blends last).
            veil = lambda item: assets[item[0]]["name"].startswith("veil/")
            variants = [(mask, sorted(extra + instances, key=veil)) for mask, instances in variants]
        compositions.append((name, variants))
    write_pack({**mine, "textures": textures, "assets": assets, "compositions": compositions},
               catalog_pack / "resource_runtime.bin")
    return {"colony_slots": slot_of, "resource_assets": len(base["assets"])}


# Largest footprint side (tile units) a piece keeps: Civ VI's big pieces (the
# Quonset hangar, a windmill yard) are enlarged less than its small ones, as
# the 2026-10-07 draft did (2.6-3.5x small pieces, 1.2-2.0x towers and hangars).
CAP = {"anchor": .40, "building": .28, "prop": .14}


def piece_extent(compiler, record: dict, size: float) -> tuple[float, float, float, float]:
    """A piece's kit-space footprint centre and its half extents in tile units,
    measured on its lower part (a windmill's sails overhang, they do not stand)."""
    from Renderer.lab.studies.districts import kits
    points = []
    for _component, landmark, transform in kits.components(compiler.manifest, compiler.root(record), kits.IDENTITY, []):
        for binding in landmark["draw_bindings"]:
            if "worked" not in binding["states"]:
                continue
            mesh = json.loads((kits.SOURCES / landmark["components"]["geometry"][binding["geometry"]]).read_text())
            for vertex in mesh["vertices"]:
                x, y, z = vertex["position"]
                points.append((x * transform[0] + y * transform[4] + transform[12],
                               x * transform[1] + y * transform[5] + transform[13], z))
    top = max((z for _, _, z in points), default=0.0)
    # Foundation skirts (z <= .003) widen a building; a flat prop has nothing else.
    low = [(x, y) for x, y, z in points if .003 < z <= max(.01, top * .4)] or [(x, y) for x, y, _ in points]
    if not low:
        raise ValueError(f"{record['entry']}: no geometry")
    xs, ys = [x for x, _ in low], [y for _, y in low]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
            (max(xs) - min(xs)) / 2 * size, (max(ys) - min(ys)) / 2 * size)


def fit(role: str, half_u: float, half_v: float) -> float:
    """The scale that keeps a piece within its role's cap."""
    return min(1.0, CAP[role] / (2 * max(half_u, half_v)))
