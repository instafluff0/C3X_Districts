#!/usr/bin/env python3
"""Bake resource tile compositions (models + ground decals) into a generic pack.

Art direction lives in Renderer/inventory/resource_composition_profiles.json.
This builder turns those choices plus the imported source placement sets
(ResourceCompositionSources, which keeps each model's authored burial) into
deterministic per-variant layouts: each instance's tile position, rotation,
final scale and extra sink. Source terrain variants become their own terrain
masks when their placements differ. Ground decals become flat "decal/" mesh
assets mapped to one atlas cell. Model textures are block-copied (no
recompression) into BC1 atlases so the whole pack fits the runtime's eight
resource texture slots. The runtime only instantiates the baked records; it
never sizes or scatters.
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
from Renderer.tools.asset_compiler import resource_composition_sources

PROFILES = ROOT / "Renderer/inventory/resource_composition_profiles.json"
LEGACY = ROOT / "Renderer/packs/ResourceNormalized"
SOURCES = resource_composition_sources.OUTPUT
OUTPUT = ROOT / "Renderer/packs/ResourceCompositionLab"
CATALOG = ROOT / "Renderer/packs/ResourceCatalogLab"
MAGIC = b"C3XVEG1\0"
TEXTURE_SLOTS = 8
CELL = 512
MODEL_ATLAS = (2048, 7, 8, 72, (71, 72))      # size, mips, block bytes, DXGI, accepted sources
MASKED_ATLAS = (2048, 7, 16, 78, (77, 78))    # alpha-masked bodies (BC3: opacity + colour)
DECAL_ATLAS = (1024, 8, 16, 78, (77, 78))
DECAL_GRID = 16


def read(path: Path):
    return json.loads(path.read_text())


def effective(placements: list[dict]) -> tuple:
    """The placements a composition uses: ancillaries (trees, clumps) are omitted."""
    return tuple(sorted((p["asset"], p["kind"], p["count"], p["scale"], p["scale_variation"], p["center"])
                        for p in placements if p["kind"] != "ancillary"))


# Generic source conditions mapped onto Civ III terrains.
VARIANT_TERRAINS = {("forest", None): "Forest", ("jungle", None): "Jungle", ("marsh", None): "Marsh",
                    ("flood_plain", None): "Flood Plain", (None, "tundra"): "Tundra", (None, "desert"): "Desert",
                    (None, "plains"): "Plains", (None, "grass"): "Grassland"}


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


def write_dds(path: Path, width: int, mips: int, dxgi: int, block: int, levels: list[bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dds_header(width, width, mips, dxgi, len(levels[0])) + b"".join(levels))
    return path


def levels_from(path: Path, block: int, size: int = 512) -> tuple[int, int, list[bytes]]:
    """Square DDS mip levels starting at no larger than size (larger top mips are dropped)."""
    width, height, mips, dxgi, payload = dds(path)
    if width != height:
        raise ValueError(f"Square texture required: {path.name}")
    layout = mip_offsets(width, height, mips, block)
    levels = [payload[offset:offset + bw * bh * block] for offset, bw, bh in layout]
    skip = 0
    while (width >> skip) > size:
        skip += 1
    return width >> skip, dxgi, levels[skip:]


def fitted(path: Path, cache: Path) -> Path:
    """A BC1 texture no larger than an atlas cell (top mips dropped, no recompression)."""
    width = dds(path)[0]
    if width <= CELL:
        return path
    size, dxgi, levels = levels_from(path, 8)
    return write_dds(cache / f"fit_{path.parent.name}_{path.name}", size, len(levels), dxgi, 8, levels)


def fully_opaque(opacity: Path, cutoff: int = 128) -> bool:
    """True when every top-mip BC4 texel is at or above the cutoff (nothing to cut out)."""
    width, height, _, _, payload = dds(opacity)
    for offset in range(0, ((width + 3) // 4) * ((height + 3) // 4) * 8, 8):
        a0, a1 = payload[offset], payload[offset + 1]
        palette = [a0, a1] + ([((7 - k) * a0 + k * a1) // 7 for k in range(1, 7)] if a0 > a1 else
                              [((5 - k) * a0 + k * a1) // 5 for k in range(1, 5)] + [0, 255])
        bits = int.from_bytes(payload[offset + 2:offset + 8], "little")
        if any(palette[(bits >> (3 * t)) & 7] < cutoff for t in range(16)):
            return False
    return True


def masked(colour: Path, opacity: Path, cache: Path) -> Path:
    """BC3 = the BC4 opacity block + the BC1 colour block, block by block (no
    recompression). BC1 blocks in three-colour mode read as four-colour in BC3,
    a slight shift on their interpolated texel; source cards use none transparent."""
    size, dxgi, colour_levels = levels_from(colour, 8)
    alpha_size, alpha_dxgi, alpha_levels = levels_from(opacity, 8)
    if alpha_dxgi not in (80, 81) or dxgi not in (71, 72):
        raise ValueError(f"Masked merge needs BC1 colour and BC4 opacity: {colour.name}")
    while alpha_size > size:
        alpha_size, alpha_levels = alpha_size >> 1, alpha_levels[1:]
    while size > alpha_size:
        size, colour_levels = size >> 1, colour_levels[1:]
    levels = []
    for colour_level, alpha_level in zip(colour_levels, alpha_levels):
        if len(colour_level) != len(alpha_level):
            raise ValueError(f"Opacity and colour mips differ: {colour.name}")
        levels.append(b"".join(alpha_level[i:i + 8] + colour_level[i:i + 8] for i in range(0, len(colour_level), 8)))
    return write_dds(cache / f"masked_{colour.name}", size, len(levels), 78 if dxgi == 72 else 77, 16, levels)


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
            full_chain = (width // 4).bit_length()   # mips down to one 4x4 block
            if dxgi not in accepted or width != height or width not in (128, 256, 512) or \
                    mips < min(levels, full_chain):
                raise ValueError(f"Atlas source must be square 128-512 with a full mip chain: {path.name}")
            cx, cy = slot % (size // CELL), slot // (size // CELL)
            source = mip_offsets(width, height, mips, block)
            for level in range(levels):
                origin, atlas_bw, _ = layout[level]
                # A small source repeats its last 4x4 mip for the atlas's tiniest levels.
                offset, bw, bh = source[min(level, len(source) - 1, full_chain - 1)]
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


def cluster_centres(setting: dict, rng: random.Random) -> list[tuple[float, float]]:
    """Cluster centres around the layout centre. flank: side by side across the view
    (the tile's u-v axis), centred, with outer clusters raised toward the tile centre
    by arc (to hug a mountain's curved foot); random: scattered, then re-centred so
    the group stays centred."""
    centre, count, separation = setting["centre"], setting["clusters"], setting["cluster_separation"]
    jitter = (rng.uniform(-1, 1) * setting["spread"], rng.uniform(-1, 1) * setting["spread"])
    step = separation / math.sqrt(2)
    if setting["arrangement"] == "flank":
        rise = setting.get("arc", 0.0) * step
        offsets = [((k - (count - 1) / 2) * step - abs(k - (count - 1) / 2) * rise,
                    -(k - (count - 1) / 2) * step - abs(k - (count - 1) / 2) * rise) for k in range(count)]
    else:
        angles = [rng.uniform(0, 2 * math.pi) for _ in range(count - 1)]
        offsets = [(0.0, 0.0)] + [(math.cos(a) * separation, math.sin(a) * separation) for a in angles]
        mean = (sum(u for u, _ in offsets) / count, sum(v for _, v in offsets) / count)
        offsets = [(u - mean[0], v - mean[1]) for u, v in offsets]
    inside = setting["keep_inside"]
    clamp = lambda value: min(1 - inside, max(inside, value))
    return [(clamp(centre[0] + jitter[0] + du), clamp(centre[1] + jitter[1] + dv)) for du, dv in offsets]


def bake_variant(key: str, setting: dict, pieces: list, decals: list, *, model, decal_cell_ids,
                 accent: dict | None = None) -> list[dict]:
    """One deterministic layout: decal records first, then an optional accent, then
    the bodies at their authored burial. Each cluster's first decal lies directly
    under it and its rocks spread over that decal's footprint."""
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    inside = setting["keep_inside"]
    clamp = lambda value: min(1 - inside, max(inside, value))
    centers = cluster_centres(setting, rng)
    footprints = setting.get("cluster_scales") or [1.0] * len(centers)
    weights = setting.get("cluster_weights") or [1.0] * len(centers)
    bounds = [sum(weights[:k + 1]) / sum(weights) for k in range(len(centers))]
    scale_factor = setting.get("scale", 1.0)
    decal_scale = setting.get("decal_scale", 1.0)
    # Without a decal, a cluster spans the default decal footprint.
    radii = [setting["decal_world_scale"] * scale_factor * decal_scale * f for f in footprints]
    if decals and len(decals) < len(centers):
        decals = (list(decals) * len(centers))[:len(centers)]   # every cluster gets a decal
    placed = []
    for index, placement in enumerate(decals):
        cells = decal_cell_ids(placement["asset"])
        cluster = index % len(centers)
        size = setting["decal_world_scale"] * scale_factor * decal_scale * footprints[cluster] * \
            placement["scale"] * (1 + placement["scale_variation"] * rng.uniform(-1, 1))
        radii[cluster] = size if index < len(centers) else max(radii[cluster], size)
        cx, cy = centers[cluster]
        nudge = .03 * footprints[cluster] if index >= len(centers) else 0.0
        placed.append({"decal": cells[rng.randrange(len(cells))],
                       "u": cx + rng.uniform(-1, 1) * nudge, "v": cy + rng.uniform(-1, 1) * nudge,
                       "rotation": rng.uniform(0, 2 * math.pi), "scale": size,
                       "lift": setting["decal_lift"], "radius": 0.0, "ground_fit": 0.0})

    def body(placement: dict, scale: float, u: float, v: float) -> dict:
        meta = model(placement["asset"])
        return {"model": placement["asset"], "u": u, "v": v, "rotation": rng.uniform(0, 2 * math.pi),
                "scale": scale, "lift": -setting["sink"] * meta["height"] * scale,
                "radius": meta["radius"] * scale, "ground_fit": meta["radius"] * scale * setting["ground_fit"]}
    bodies = []
    if accent is not None:
        # The accent sits at the group's centre.
        at = (sum(u for u, _ in centers) / len(centers), sum(v for _, v in centers) / len(centers))
        bodies.append(body(accent, setting["world_scale"] * scale_factor * setting["accent_scale"] * accent["scale"],
                           *at))
    if setting.get("subject") or setting.get("subjects"):
        # A small herd of the animated subject ("animated/<binding>"), grouped at
        # the layout centre; the runtime stands each on the highest ground under
        # its footprint (ground_fit) so no part of the body is buried.
        herd, radius = setting["herd"], setting["subject_radius"]
        subjects = setting.get("subjects") or [setting["subject"]]   # herd members cycle through these
        for subject in subjects:
            model("animated/" + subject)   # registers the placeholder asset
        cx, cy = (sum(u for u, _ in centers) / len(centers), sum(v for _, v in centers) / len(centers))
        start = rng.uniform(0, 2 * math.pi)
        for index in range(herd):
            angle = start + 2 * math.pi * index / herd + rng.uniform(-.3, .3)
            distance = 0.0 if herd == 1 else radius * setting.get("herd_spread", 1.3)
            scale = setting.get("subject_scale", 1.0) * (1 + rng.uniform(-.08, .08))
            subject = subjects[index % len(subjects)]
            bodies.append({"model": "animated/" + subject,
                           "u": clamp(cx + math.cos(angle) * distance), "v": clamp(cy + math.sin(angle) * distance),
                           "rotation": rng.uniform(-1, 1) * setting.get("yaw_jitter", 0.0), "scale": scale,
                           "lift": 0.0, "radius": radius * scale,
                           "ground_fit": radius * scale * setting.get("subject_fit", .6)})
    # The terrain layout may densify (scree) or thin the resource's own pieces.
    target = max(1, int(round(len(pieces) * setting.get("layout_count_scale", 1.0))))
    if not pieces:
        return placed + bodies
    order = (list(pieces) * (target // max(1, len(pieces)) + 1))[:target]
    rng.shuffle(order)
    if setting.get("planting") == "rows":
        # Planted rows (vineyards, plantations): parallel rows along the tile's u
        # axis, which runs diagonally down-right on screen, centred on the layout.
        rows, spacing, length = setting["rows"], setting["row_spacing"], setting["row_length"]
        per_row = -(-len(order) // rows)
        cx, cy = (sum(u for u, _ in centers) / len(centers), sum(v for _, v in centers) / len(centers))
        for index, placement in enumerate(order):
            row, slot = divmod(index, per_row)
            along = ((slot + .5) / per_row - .5) * length + rng.uniform(-.15, .15) * length / per_row
            across = (row - (rows - 1) / 2) * spacing + rng.uniform(-.1, .1) * spacing
            scale = setting["world_scale"] * scale_factor * placement["scale"] * \
                (1 + placement["scale_variation"] * rng.uniform(-1, 1))
            bodies.append(body(placement, scale, clamp(cx + along), clamp(cy + across)))
        return placed + bodies
    for index, placement in enumerate(order):
        scale = setting["world_scale"] * scale_factor * placement["scale"] * \
            (1 + placement["scale_variation"] * rng.uniform(-1, 1))
        radius = model(placement["asset"])["radius"] * scale
        cluster = next(k for k, bound in enumerate(bounds) if (index + .5) / len(order) <= bound)
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
        bodies.append(body(placement, scale, best[1], best[2]))
    return placed + bodies


def build(output: Path = OUTPUT, alternates: bool = True, catalog: str | None = None) -> dict:
    """alternates=False bakes only the recommended compositions (promotion);
    catalog names one of the profile's Lab reference catalogs to bake alone, without legacy groups."""
    profiles = read(PROFILES)
    sources = resource_composition_sources.ensure(SOURCES, PROFILES)
    legacy_manifest = read(LEGACY / "manifest.json")
    defaults = profiles["defaults"]
    models: dict[str, dict] = {}      # asset id -> mesh/meta, in first-use order
    decal_assets: dict[tuple[str, int], dict] = {}
    decal_textures: list[Path] = []
    compositions, report = [], {"schema": "c3x.resource_composition_report.v0", "resources": {}}
    cache = SOURCES.parent / (SOURCES.name + "_merged")

    def model(asset_id: str, z_offset: float = 0.0) -> dict:
        if asset_id not in models and asset_id.startswith("animated/"):
            # Placeholder naming an animated subject; the runtime places the skinned body.
            models[asset_id] = {"texture": None, "masked": False, "vertices": [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                                (0.0, 0.0))] * 3, "indices": [0, 1, 2], "radius": 0.0, "height": 0.0}
        if asset_id not in models:
            pack, records = (SOURCES, sources["assets"]) if asset_id.startswith("source/") else \
                (LEGACY, legacy_manifest["assets"])
            record = records[asset_id]
            mesh = read(pack / record["mesh"])
            vertices = [([v["position"][0], v["position"][1], v["position"][2] + z_offset], v["normal"], v["uv0"])
                        for v in mesh["vertices"]]
            low = [min(v[0][a] for v in vertices) for a in range(3)]
            high = [max(v[0][a] for v in vertices) for a in range(3)]
            material = read(pack / record["material"])
            colour = pack / material_texture(material)
            opacity = material.get("opacity")
            opacity = opacity.get("texture") if isinstance(opacity, dict) else opacity
            if opacity and fully_opaque(pack / opacity):
                opacity = None   # an all-opaque mask stays a BC1 body
            texture = masked(colour, pack / opacity, cache) if opacity else fitted(colour, cache)
            models[asset_id] = {"texture": texture, "masked": bool(opacity),
                                "vertices": vertices, "indices": mesh["topology"]["indices"],
                                "radius": max(high[0] - low[0], high[1] - low[1]) * .5, "height": high[2] - low[2]}
        return models[asset_id]

    def decal_cell_ids(asset_id: str) -> list[tuple[str, int]]:
        definition = read(SOURCES / sources["assets"][asset_id]["decal"])
        texture = SOURCES / definition["channels"]["base_color"]["texture"]
        if texture not in decal_textures:
            decal_textures.append(texture)
        keys = []
        for cell, rect in enumerate(decal_cells(texture)):
            key = (texture.name, cell)
            decal_assets.setdefault(key, {"texture": texture, "rect": rect})
            keys.append(key)
        return keys

    def selection(profile: dict, placements: list[dict], decal_placements: list[dict]):
        pieces, decals = [], []
        for placement in placements if profile.get("static_pieces", True) else ():
            count = profile.get("piece_counts", {}).get((placement["asset"] or "").split("/")[-1], placement["count"])
            imported = sources["assets"].get(placement["asset"] or "", {}).get("type") == "feature"
            kinds = ("model", "accessory") if profile.get("accessories", True) else ("model",)
            if placement["kind"] in kinds and imported and count > 0 and not placement["center"]:
                pieces += [placement] * max(1, int(round(count * profile["count_scale"])))
        for placement in decal_placements:
            if placement["kind"] != "decal" or sources["assets"].get(placement["asset"], {}).get("type") != "decal":
                continue
            count = profile.get("decal_counts", {}).get(placement["asset"].split("/")[-1], placement["count"])
            decals += [placement] * max(0, int(round(count * profile.get("decal_count_scale", 1.0))))
        return pieces, decals

    entries = [(name, setting) for name, setting in profiles["resources"].items()]
    if catalog:
        spec = profiles["catalogs"][catalog]
        entries = [(name, {"family": spec["family"], **entry}) for name, entry in spec["entries"].items()]
    elif alternates:
        entries += [(f"{name}~{label}", {**profiles["resources"][name], **override})
                    for name, labels in profiles.get("alternates", {}).items() if isinstance(labels, dict)
                    for label, override in labels.items()]
    for name, setting in entries:
        profile = {**defaults, **profiles["families"].get(setting["family"], {}), **setting}
        source = sources["sources"][profile["source"]]
        decal_source = sources["sources"][profile.get("decal_source", profile["source"])]
        # The accent is a centre model (Civ VI's authored pile) heading the outcrop.
        accent = None
        accent_from = profile.get("accent_source") or (profile["source"] if profile.get("accent_scale", 0) > 0 else None)
        if accent_from:
            options = sources["sources"][accent_from]
            accent = next((p for p in options["sets"][options["base"]]["placements"]
                           if p["kind"] == "model" and p["center"] and p["asset"] in sources["assets"]), None)
        base_set = source["sets"][source["base"]]["placements"]
        decal_set = decal_source["sets"][decal_source["base"]]["placements"]
        # Terrains whose source variant changes the effective placements get their own sets.
        overrides: dict[str, list[dict]] = {}
        for variant in source["variants"]:
            terrain = VARIANT_TERRAINS.get((variant["when"]["feature"], variant["when"]["terrain"]))
            placements = source["sets"][variant["set"]]["placements"]
            if terrain and not variant["when"]["hills"] and effective(placements) != effective(base_set):
                overrides.setdefault(terrain, placements)
        groups = []
        for layout_name, layout in profile["terrain_layouts"].items():
            groups.append((layout_name, layout, [t for t in layout["terrains"] if t not in overrides], base_set))
            groups += [(f"{layout_name}/{t}", layout, [t], overrides[t]) for t in layout["terrains"] if t in overrides]
        groups = [group for group in groups if group[2]]
        # The runtime holds 16 variants per composition; terrain overrides share that budget.
        per_group = min(profile["variants"], 16 // len(groups))
        if per_group < 1:
            raise ValueError(f"{name}: {len(groups)} terrain groups exceed the runtime limit of 16 variants")
        variants = []
        for group_name, layout, terrains, placements in groups:
            own_decals = placements if decal_source is source else decal_set
            pieces, decals = selection(profile, placements, own_decals)
            # A resource may override any layout ("*" for all) after the layout's own values.
            overrides = profile.get("layout_overrides", {})
            setting_for = {**profile, **layout, **overrides.get("*", {}),
                           **overrides.get(group_name.split("/")[0], {})}
            mask = sum(1 << TERRAIN_INDEX[terrain] for terrain in terrains)
            for variant in range(per_group):
                variants.append((mask, bake_variant(f"{name}:{group_name}:{variant}", setting_for, pieces, decals,
                                                    model=model, decal_cell_ids=decal_cell_ids, accent=accent)))
        compositions.append((name, variants))
        pieces, decals = selection(profile, base_set, base_set if decal_source is source else decal_set)
        report["resources"][name] = {"source": profile["source"], "models": len(pieces), "decals": len(decals),
                                     "variants": len(variants), "terrain_overrides": sorted(overrides)}

    # Production's legacy static groups stay available for resources without a
    # composition; a group whose name a composition alias covers is unreachable.
    covered = [alias.lower() for values in profiles["bindings"].values() for alias in values]
    legacy = []
    for name, asset_id, scale, count in () if catalog else SELECTIONS:
        if any(name in alias for alias in covered):
            continue
        legacy.append((name, asset_id, scale, count))
        model(asset_id, .060 if name == "fish" else 0.0)

    # Textures: BC1 model atlases, BC3 masked-model atlases, then BC3 decal
    # atlases, padded to the slot count.
    opaque = list(dict.fromkeys(meta["texture"] for meta in models.values() if not meta["masked"] and meta["texture"]))
    cutout = list(dict.fromkeys(meta["texture"] for meta in models.values() if meta["masked"]))
    atlases, atlas_place = pack_atlases(opaque, MODEL_ATLAS)
    for textures, kind in ((cutout, MASKED_ATLAS), (decal_textures, DECAL_ATLAS)):
        more, place = pack_atlases(textures, kind, len(atlases))
        atlases += more
        atlas_place.update(place)
    if not atlases:   # only animated subjects: the bundle still needs one (unused) texture
        atlases.append(dds_header(4, 4, 1, MODEL_ATLAS[3], 8) + bytes(8))
    slots = [f"textures/atlas_{i}.dds" for i in range(len(atlases))]
    if len(slots) > TEXTURE_SLOTS:
        raise ValueError(f"Composition pack needs {len(slots)} texture slots; the runtime binds {TEXTURE_SLOTS}")
    padded = slots + [slots[-1]] * (TEXTURE_SLOTS - len(slots))

    assets, asset_index = [], {}
    for asset_id, meta in models.items():
        atlas, u0, v0, extent = atlas_place[meta["texture"]] if meta["texture"] else (0, 0.0, 0.0, 1.0 / CELL)
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
    aliases = [(name, index) for index, name in enumerate(names)] if catalog else \
        [(alias + name[len(base):], names.index(name)) for name in names
         for base in [name.split("~")[0]] for alias in profiles["bindings"].get(base, ())]
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
    parser.add_argument("--no-alternates", action="store_true", help="bake only the recommended compositions")
    parser.add_argument("--catalog", help="bake this Lab reference catalog (e.g. minerals) into ResourceCatalogLab")
    args = parser.parse_args()
    report = build(CATALOG if args.catalog and args.output == OUTPUT else args.output, not args.no_alternates, args.catalog)
    print(json.dumps({"resources": report["resources"], "textures": report["textures"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
