#!/usr/bin/env python3
"""Bake resource tile compositions (models + ground decals) into a generic pack.

Art direction lives in Renderer/inventory/resource_composition_profiles.json.
This builder turns those choices plus the normalized source placements into
deterministic per-variant layouts: each instance's tile position, rotation,
final scale and sink. Ground decals become flat "decal/" mesh assets mapped to
one atlas cell. Model textures are block-copied (no recompression) into BC1
atlases so the whole pack fits the runtime's eight resource texture slots.
The runtime only instantiates the baked records; it never sizes or scatters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Renderer.tools.asset_compiler.build_resource_runtime import SELECTIONS, bundle_string, material_texture

PROFILES = ROOT / "Renderer/inventory/resource_composition_profiles.json"
SOURCE = ROOT / "Renderer/packs/ResourceNormalized"
DECALS = ROOT / "Renderer/packs/DecalsNormalized"
OUTPUT = ROOT / "Renderer/packs/ResourceCompositionLab"
MAGIC = b"C3XVEG1\0"
ANCILLARY = ("decal", "boulder", "snow_boulder", "tree_pine", "jungle_clump", "shrub")
TEXTURE_SLOTS = 8
CELL = 512
MODEL_ATLAS = (2048, 7, 8, 72, (71, 72))      # size, mips, block bytes, DXGI, accepted sources
DECAL_ATLAS = (1024, 8, 16, 78, (77, 78))
DECAL_GRID = 16


def read(path: Path):
    return json.loads(path.read_text())


def base_placements(record: dict) -> list[dict]:
    """First occurrence of each authored placement; terrain-variant ancillaries are
    omitted until their source conditions are recovered."""
    seen, result = set(), []
    for placement in record["placements"]:
        key = (placement["asset"], placement.get("pack"), placement["count"], placement["scale"],
               placement["scale_variation"])
        if key not in seen:
            seen.add(key)
            result.append(placement)
    return result


def dds(path: Path) -> tuple[int, int, int, int, bytes]:
    data = path.read_bytes()
    if data[:4] != b"DDS " or data[84:88] != b"DX10":
        raise ValueError(f"Unsupported DDS container: {path.name}")
    height, width = struct.unpack_from("<II", data, 12)
    mips = max(1, struct.unpack_from("<I", data, 28)[0])
    return width, height, mips, struct.unpack_from("<I", data, 128)[0], data[148:]


def dds_header(width: int, height: int, mips: int, dxgi: int, top_bytes: int) -> bytes:
    pixel_format = struct.pack("<II4s5I", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
    header = struct.pack("<7I44s32s5I", 124, 0xA1007, height, width, top_bytes, 0, mips, bytes(44),
                         pixel_format, 0x401008, 0, 0, 0, 0)
    return b"DDS " + header + struct.pack("<5I", dxgi, 3, 0, 1, 0)


def mip_offsets(width: int, height: int, mips: int, block: int) -> list[tuple[int, int, int]]:
    result, offset = [], 0
    for level in range(mips):
        bw, bh = max(1, (max(1, width >> level) + 3) // 4), max(1, (max(1, height >> level) + 3) // 4)
        result.append((offset, bw, bh))
        offset += bw * bh * block
    return result


def pack_atlases(textures: list[Path], kind: tuple, first: int = 0):
    """Block-copy compressed textures (no recompression) into atlases of 512
    cells; returns atlas DDS bytes and, per texture, (slot, u0, v0, uv extent)."""
    size, levels, block, dxgi_out, accepted = kind
    per_atlas = (size // CELL) ** 2
    atlases, placement = [], {}
    for start in range(0, len(textures), per_atlas):
        layout = mip_offsets(size, size, levels, block)
        blob = bytearray(layout[-1][0] + layout[-1][1] * layout[-1][2] * block)
        for slot, path in enumerate(textures[start:start + per_atlas]):
            width, height, mips, dxgi, payload = dds(path)
            if dxgi not in accepted or width != height or width not in (256, 512) or \
                    (width >> (levels - 1)) < 4 or mips < levels:
                raise ValueError(f"Atlas source must be square 256/512 with enough mips: {path.name}")
            cx, cy = slot % (size // CELL), slot // (size // CELL)
            source = mip_offsets(width, height, mips, block)
            for level in range(levels):
                origin, atlas_bw, _ = layout[level]
                offset, bw, bh = source[level]
                base_x, base_y = cx * (CELL >> level) // 4, cy * (CELL >> level) // 4
                for row in range(bh):
                    target = origin + ((base_y + row) * atlas_bw + base_x) * block
                    blob[target:target + bw * block] = payload[offset + row * bw * block: offset + (row + 1) * bw * block]
            placement[path] = (first + len(atlases), cx * CELL / size, cy * CELL / size, width / size)
        atlases.append(dds_header(size, size, levels, dxgi_out, layout[0][1] * layout[0][2] * block) + bytes(blob))
    return atlases, placement


def decal_cells(path: Path) -> list[tuple[float, float, float, float]]:
    """Split a BC3 decal atlas at its transparent gutters (nearest the centre);
    a texture without gutters is one cell."""
    width, height, _, dxgi, payload = dds(path)
    if dxgi not in (77, 78):
        raise ValueError(f"Decal must be BC3: {path.name}")
    blocks_x, blocks_y = (width + 3) // 4, (height + 3) // 4
    column, row = [0] * width, [0] * height
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = payload[(by * blocks_x + bx) * 16:(by * blocks_x + bx) * 16 + 8]
            a0, a1 = block[0], block[1]
            palette = [a0, a1] + ([((7 - k) * a0 + k * a1) // 7 for k in range(1, 7)] if a0 > a1 else
                                  [((5 - k) * a0 + k * a1) // 5 for k in range(1, 5)] + [0, 255])
            bits = int.from_bytes(block[2:8], "little")
            for texel in range(16):
                value = palette[(bits >> (3 * texel)) & 7]
                x, y = bx * 4 + texel % 4, by * 4 + texel // 4
                column[x] = max(column[x], value)
                row[y] = max(row[y], value)

    def gutter(values: list[int]) -> float | None:
        size = len(values)
        clear = [i for i in range(size * 3 // 8, size * 5 // 8) if values[i] < 8]
        return (min(clear, key=lambda i: abs(i - size / 2)) + .5) / size if clear else None
    gx, gy = gutter(column), gutter(row)
    if gx is None or gy is None:
        return [(0.0, 0.0, 1.0, 1.0)]
    return [(0.0, 0.0, gx, gy), (gx, 0.0, 1.0, gy), (0.0, gy, gx, 1.0), (gx, gy, 1.0, 1.0)]


def mesh_payload(asset_id: str, texture: int, vertices: list, indices: list) -> bytes:
    payload = bytearray(bundle_string(asset_id))
    payload.extend(struct.pack("<III", texture, len(vertices), len(indices)))
    for position, normal, uv in vertices:
        payload.extend(struct.pack("<8f", *position, *normal, *uv))
    payload.extend(struct.pack(f"<{len(indices)}I", *indices))
    return bytes(payload)


TERRAIN_INDEX = {"Desert": 0, "Plains": 1, "Grassland": 2, "Tundra": 3, "Flood Plain": 4, "Hills": 5,
                 "Mountains": 6, "Forest": 7, "Jungle": 8, "Marsh": 9, "Volcano": 10}


def bake_variant(key: str, setting: dict, pieces: list, decals: list, *, model, decal_cell_ids) -> list[dict]:
    """One deterministic layout: decal records first, then sunk model bodies."""
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    inside = setting["keep_inside"]
    clamp = lambda value: min(1 - inside, max(inside, value))
    centre = setting["centre"]
    first = (clamp(centre[0] + rng.uniform(-1, 1) * setting["spread"]),
             clamp(centre[1] + rng.uniform(-1, 1) * setting["spread"]))
    centers = [first]
    for _ in range(setting["clusters"] - 1):
        if setting["arrangement"] == "flank":
            # Side by side across the view (tile u-v axis), so neither hides the other.
            sign = rng.choice((-1, 1)) * setting["cluster_separation"] / math.sqrt(2)
            centers.append((clamp(first[0] + sign), clamp(first[1] - sign)))
        else:
            angle = rng.uniform(0, 2 * math.pi)
            centers.append((clamp(first[0] + math.cos(angle) * setting["cluster_separation"]),
                            clamp(first[1] + math.sin(angle) * setting["cluster_separation"])))
    scale_factor = setting.get("scale", 1.0)
    # Without a decal, a cluster spans the default decal footprint.
    radii = [setting["decal_world_scale"] * scale_factor] * len(centers)
    placed = []
    for index, placement in enumerate(decals):
        cells = decal_cell_ids(placement["asset"])
        cluster = index % len(centers)
        size = setting["decal_world_scale"] * scale_factor * placement["scale"] * \
            (1 + placement["scale_variation"] * rng.uniform(-1, 1))
        radii[cluster] = size if index < len(centers) else max(radii[cluster], size)
        size *= setting.get("decal_scale", 1.0)
        cx, cy = centers[cluster]
        offset = setting.get("decal_offset", (0.0, 0.0))
        placed.append({"decal": cells[rng.randrange(len(cells))],
                       "u": cx + offset[0] + rng.uniform(-.03, .03), "v": cy + offset[1] + rng.uniform(-.03, .03),
                       "rotation": rng.uniform(0, 2 * math.pi), "scale": size,
                       "lift": setting["decal_lift"], "radius": 0.0, "ground_fit": 0.0})
    # The terrain layout may densify (scree) or thin the resource's own pieces.
    target = max(1, int(round(len(pieces) * setting.get("layout_count_scale", 1.0))))
    order = (list(pieces) * (target // max(1, len(pieces)) + 1))[:target]
    rng.shuffle(order)
    bodies = []
    for index, placement in enumerate(order):
        meta = model(placement["asset"])
        scale = setting["world_scale"] * scale_factor * placement["scale"] * \
            (1 + placement["scale_variation"] * rng.uniform(-1, 1))
        radius = meta["radius"] * scale
        cluster = index % len(centers)
        cx, cy = centers[cluster]
        best = None
        for _ in range(40):
            distance = math.sqrt(rng.random()) * radii[cluster] * setting["cluster_radius"]
            angle = rng.uniform(0, 2 * math.pi)
            u, v = clamp(cx + math.cos(angle) * distance), clamp(cy + math.sin(angle) * distance)
            clearance = min((math.hypot(u - p["u"], v - p["v"]) - setting["spacing"] * (radius + p["radius"])
                             for p in bodies), default=1.0)
            if best is None or clearance > best[0]:
                best = (clearance, u, v)
            if clearance >= 0:
                break
        bodies.append({"model": placement["asset"], "u": best[1], "v": best[2],
                       "rotation": rng.uniform(0, 2 * math.pi), "scale": scale,
                       "lift": -setting["sink"] * meta["height"] * scale, "radius": radius,
                       "ground_fit": radius * setting["ground_fit"]})
    return placed + bodies


def build(output: Path = OUTPUT) -> dict:
    profiles = read(PROFILES)
    manifest = read(SOURCE / "manifest.json")
    decal_manifest = read(DECALS / "manifest.json")
    defaults = profiles["defaults"]
    models: dict[str, dict] = {}      # asset id -> mesh/meta, in first-use order
    decal_assets: dict[tuple[str, int], dict] = {}
    decal_textures: list[Path] = []
    compositions, report = [], {"schema": "c3x.resource_composition_report.v0", "resources": {}}

    def model(asset_id: str, z_offset: float = 0.0) -> dict:
        if asset_id not in models:
            record = manifest["assets"][asset_id]
            mesh = read(SOURCE / record["mesh"])
            vertices = [([v["position"][0], v["position"][1], v["position"][2] + z_offset], v["normal"], v["uv0"])
                        for v in mesh["vertices"]]
            low = [min(v[0][a] for v in vertices) for a in range(3)]
            high = [max(v[0][a] for v in vertices) for a in range(3)]
            models[asset_id] = {"texture": SOURCE / material_texture(read(SOURCE / record["material"])),
                                "vertices": vertices, "indices": mesh["topology"]["indices"],
                                "radius": max(high[0] - low[0], high[1] - low[1]) * .5, "height": high[2] - low[2]}
        return models[asset_id]

    def decal_cell_ids(asset_id: str) -> list[tuple[str, int]]:
        definition = read(DECALS / decal_manifest["assets"][asset_id]["decal"])
        texture = DECALS / definition["channels"]["base_color"]["texture"]
        if texture not in decal_textures:
            decal_textures.append(texture)
        keys = []
        for cell, rect in enumerate(decal_cells(texture)):
            key = (texture.name, cell)
            decal_assets.setdefault(key, {"texture": texture, "rect": rect})
            keys.append(key)
        return keys

    for name, setting in profiles["resources"].items():
        profile = {**defaults, **profiles["families"].get(setting["family"], {}), **setting}
        record = manifest["resources"][profile["source"]]
        pieces, decals = [], []
        for placement in base_placements(record):
            short = placement["asset"].split("/")[-1]
            if placement.get("pack") == "DecalsNormalized":
                count = profile.get("decal_counts", {}).get(short, placement["count"])
                decals += [placement] * max(0, int(round(count * profile.get("decal_count_scale", 1.0))))
            elif not short.startswith(ANCILLARY) and placement["count"] > 0:
                pieces += [placement] * max(1, int(round(placement["count"] * profile["count_scale"])))
        variants = []
        for layout_name, layout in profile["terrain_layouts"].items():
            setting = {**profile, **layout}
            mask = sum(1 << TERRAIN_INDEX[terrain] for terrain in layout["terrains"])
            for variant in range(profile["variants"]):
                variants.append((mask, bake_variant(f"{name}:{layout_name}:{variant}", setting, pieces, decals,
                                                    model=model, decal_cell_ids=decal_cell_ids)))
        compositions.append((name, variants))
        report["resources"][name] = {"models": len(pieces), "decals": len(decals), "variants": len(variants)}

    # Production's legacy static groups stay available for resources without a composition.
    legacy = []
    for name, asset_id, scale, count in SELECTIONS:
        legacy.append((name, asset_id, scale, count))
        model(asset_id, .060 if name == "fish" else 0.0)

    # Textures: BC1 model atlases first, then each BC3 decal atlas, padded to the slot count.
    sources = list(dict.fromkeys(meta["texture"] for meta in models.values()))
    atlases, atlas_place = pack_atlases(sources, MODEL_ATLAS)
    decal_atlases, decal_place = pack_atlases(decal_textures, DECAL_ATLAS, len(atlases))
    atlases += decal_atlases
    atlas_place.update(decal_place)
    slots = [f"textures/atlas_{i}.dds" for i in range(len(atlases))]
    if len(slots) > TEXTURE_SLOTS:
        raise ValueError(f"Composition pack needs {len(slots)} texture slots; the runtime binds {TEXTURE_SLOTS}")
    padded = slots + [slots[-1]] * (TEXTURE_SLOTS - len(slots))

    assets, asset_index = [], {}
    for asset_id, meta in models.items():
        atlas, u0, v0, extent = atlas_place[meta["texture"]]
        edge = .5 / (extent * MODEL_ATLAS[0])
        vertices = [(p, n, (u0 + min(1 - edge, max(edge, uv[0])) * extent, v0 + min(1 - edge, max(edge, uv[1])) * extent))
                    for p, n, uv in meta["vertices"]]
        asset_index[asset_id] = len(assets)
        assets.append(mesh_payload(asset_id, atlas, vertices, meta["indices"]))
    step = 2.0 / DECAL_GRID
    grid_indices = []
    for j in range(DECAL_GRID):
        for i in range(DECAL_GRID):
            a = j * (DECAL_GRID + 1) + i
            grid_indices += [a, a + 1, a + DECAL_GRID + 2, a, a + DECAL_GRID + 2, a + DECAL_GRID + 1]
    for key, decal in decal_assets.items():
        atlas, a0, b0, extent = atlas_place[decal["texture"]]
        u0, v0, u1, v1 = (a0 + decal["rect"][0] * extent, b0 + decal["rect"][1] * extent,
                          a0 + decal["rect"][2] * extent, b0 + decal["rect"][3] * extent)
        inset = .01
        vertices = [((-1 + i * step, -1 + j * step, 0.0), (0.0, 0.0, 1.0),
                     (u0 + (u1 - u0) * (inset + (1 - 2 * inset) * i / DECAL_GRID),
                      v0 + (v1 - v0) * (inset + (1 - 2 * inset) * j / DECAL_GRID)))
                    for j in range(DECAL_GRID + 1) for i in range(DECAL_GRID + 1)]
        identifier = f"decal/{key[0]}/{key[1]}"
        asset_index[key] = len(assets)
        assets.append(mesh_payload(identifier, atlas, vertices, grid_indices))
    if len(assets) > 256:
        raise ValueError("Composition pack exceeds the runtime asset limit")

    blob = bytearray(MAGIC)
    blob.extend(struct.pack("<IIII", 2, TEXTURE_SLOTS, len(assets), len(legacy)))
    for texture in padded:
        blob.extend(bundle_string(texture))
    for payload in assets:
        blob.extend(payload)
    for name, asset_id, scale, count in legacy:
        blob.extend(bundle_string(name))
        blob.extend(struct.pack("<I", 1))
        blob.extend(struct.pack("<IffIIIIff", asset_index[asset_id], scale, 0.10 if count > 1 else 0.03,
                                count, 1, 5, 0, 0.0, 0.0))
    blob.extend(struct.pack("<I", len(compositions)))
    for name, variants in compositions:
        blob.extend(bundle_string(name))
        blob.extend(struct.pack("<I", len(variants)))
        for mask, instances in variants:
            blob.extend(struct.pack("<II", mask, len(instances)))
            for item in instances:
                identifier = asset_index[item.get("decal") or item["model"]]
                blob.extend(struct.pack("<I6f", identifier, item["u"], item["v"], item["rotation"],
                                        item["scale"], item["lift"], item["ground_fit"]))
    names = [name for name, _ in compositions]
    aliases = [(alias, names.index(name)) for name, values in profiles["bindings"].items() for alias in values]
    blob.extend(struct.pack("<I", len(aliases)))
    for alias, index in aliases:
        blob.extend(bundle_string(alias))
        blob.extend(struct.pack("<I", index))

    if output.exists():
        shutil.rmtree(output)
    (output / "textures").mkdir(parents=True)
    for index, data in enumerate(atlases):
        (output / f"textures/atlas_{index}.dds").write_bytes(data)
    (output / "resource_runtime.bin").write_bytes(blob)
    report["textures"] = padded
    report["layouts"] = {name: [{"terrain_mask": mask, "instances": [{k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in item.items()} for item in instances]} for mask, instances in variants]
                         for name, variants in compositions}
    (output / "composition_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps({"resources": report["resources"], "textures": report["textures"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
