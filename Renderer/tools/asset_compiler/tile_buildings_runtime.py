"""Civ III's tile buildings in the site pack (user picks 2026-10-23).

Renderer/lab/studies/tile_buildings bakes the picks (fortress, barricade,
airfield, radar, outpost and victory point by era, and colony pieces) into the
frozen pack Renderer/packs/TileBuildingsFrozen: a version-2 feature bundle of
compositions over its own atlases. Its first BC1 atlas also holds the hut and
camp textures, keyed by their bytes, so the site pack stays within its eight
texture slots: build_site_runtime moves the hut and camp meshes into those
cells and appends the compositions after its own assets.
"""
from pathlib import Path
import hashlib
import json
import struct

from Renderer.tools.asset_compiler.build_resource_compositions import within_texture

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "Renderer/packs/TileBuildingsFrozen"


def load(read, frozen=FROZEN):
    """The frozen pack through `read` (None without one): its manifest, the
    bundle's assets as bytes, and its compositions and aliases."""
    if not (frozen / "manifest.json").is_file():
        return None
    manifest = json.loads(read(frozen / "manifest.json"))
    data = read(frozen / manifest["bundle"])
    for atlas in manifest["atlases"]:
        read(frozen / atlas)
    at = 8
    def take(layout):
        nonlocal at
        values = struct.unpack_from(layout, data, at)
        at += struct.calcsize(layout)
        return values
    def text():
        length, = take("<I")
        return take(f"<{length}s")[0].decode()
    version, slots, count, groups = take("<IIII")
    if data[:8] != b"C3XVEG1\0" or version != 2 or groups:
        raise ValueError("The frozen tile buildings must be a version-2 bundle of compositions")
    for _ in range(slots):
        text()
    assets = []
    for _ in range(count):
        start = at
        text()
        texture, vertices, indices = take("<III")
        if texture >= len(manifest["atlases"]):
            raise ValueError("A frozen tile building samples a missing atlas")
        at += vertices * 32 + indices * 4
        assets.append(data[start:at])
    compositions = []
    for _ in range(take("<I")[0]):
        name, variants = text(), []
        for _ in range(take("<I")[0]):
            mask, instances = take("<II")
            variants.append((mask, [list(take("<I6f")) for _ in range(instances)]))
        compositions.append((name, variants))
    aliases = [(text(), take("<I")[0]) for _ in range(take("<I")[0])]
    return {"manifest": manifest, "assets": assets, "compositions": compositions, "aliases": aliases}


def cell(frozen, texture: bytes):
    """The atlas cell holding a hut or camp texture: (atlas, u0, v0, extent, width)."""
    found = frozen["manifest"]["site_cells"].get(hashlib.sha256(texture).hexdigest())
    if found is None:
        raise ValueError("A hut or camp texture changed since the tile buildings were frozen; "
                         "re-run Renderer/lab/studies/tile_buildings/sitepack.py")
    return found


def atlas_mesh(mesh: dict, place) -> dict:
    """A hut or camp mesh moved into its atlas cell; tiling UVs are first clipped
    at whole-texture boundaries (an atlas cell cannot repeat)."""
    _, u0, v0, extent, width = place
    vertices = [(vertex["position"], vertex["normal"], vertex["uv0"]) for vertex in mesh["vertices"]]
    indices = list(mesh["topology"]["indices"])
    if any(not 0 <= value <= 1 for _, _, uv in vertices for value in uv):
        vertices, indices = within_texture(vertices, indices)
    edge = .5 / width
    return {"vertices": [{"position": list(p), "normal": list(n),
                          "uv0": [u0 + min(1 - edge, max(edge, uv[0])) * extent,
                                  v0 + min(1 - edge, max(edge, uv[1])) * extent]} for p, n, uv in vertices],
            "topology": {"indices": indices}}


def tail(frozen, offset: int) -> bytes:
    """The compositions (their assets after the site pack's first `offset`) and aliases."""
    data = bytearray(struct.pack("<I", len(frozen["compositions"])))
    for name, variants in frozen["compositions"]:
        data += struct.pack("<I", len(name.encode())) + name.encode() + struct.pack("<I", len(variants))
        for mask, instances in variants:
            data += struct.pack("<II", mask, len(instances))
            for asset, *rest in instances:
                data += struct.pack("<I6f", asset + offset, *rest)
    data += struct.pack("<I", len(frozen["aliases"]))
    for name, index in frozen["aliases"]:
        data += struct.pack("<I", len(name.encode())) + name.encode() + struct.pack("<I", index)
    return bytes(data)
