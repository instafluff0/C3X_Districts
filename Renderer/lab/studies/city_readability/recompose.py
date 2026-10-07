#!/usr/bin/env python3
"""Recompose the compiled city library toward Civ III map readability.

Civ III cities read as one dense, high-contrast block on a light ground plate.
The selected Civ VI-derived recipes are sparse at map scale. This Lab step
keeps every selected source mesh, material and facade light, and changes only
placement data in a separate candidate pack:

- grows each building about its own anchor until it nearly meets its
  neighbours (uniform scale, never a proportion change);
- fills the remaining gaps with more of the same composition's buildings;
- adds era accents (industrial smokestacks, modern towers) at the back of
  Industrial/Modern settlements, replacing the plots they cover;
- adds a crisp ground plate under buildings and lanes, from an era paving
  texture made tileable offline; and
- marks bodies that may yield to a river, water or steep site (version 5).

Input is the frozen pre-readability library, Renderer/packs/CityCompositionFrozen
(the last pack the retired recipe builder made; its intermediates no longer
exist, so it is never rebuilt or overwritten). The `cities` asset job runs this
builder into Renderer/packs/CityCompositionRuntime; run directly, it writes the
Lab candidate Renderer/packs/CityCompositionLab (ignored local data). Era
ground textures come from a local cache made once from the installed source
decals with --ground.

    python3 Renderer/lab/studies/city_readability/recompose.py [--ground]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.city_readability import runtime_pack as rp

SOURCE = ROOT / "Renderer/packs/CityCompositionFrozen"
ACCENTS = Path("Renderer/packs/CityAccentsLab")
GROUND = ACCENTS / "ground"
OUT = ROOT / "Renderer/packs/CityCompositionLab"
STYLE = Path(__file__).with_name("style.json")
ERAS = ("ancient", "medieval", "industrial", "modern")
CULTURES = ("american", "european", "roman", "middle_eastern", "asian")  # Civ III culture groups
TEXTURE_PREFIX = "Renderer/packs/CityCompositionRuntime/textures/"
SOURCE_Z = 0.648266978876
SHARED_DATA = (Path.home() / "Library/Application Support/Steam/steamapps/common"
               / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets/Base/Platforms/Windows/BLPs/SHARED_DATA")


# --- Footprint geometry (offset space: x = column, y = -row, as Instance.offset) --
def transform(hull, scale, yaw, offset):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(offset[0] + scale * (x * c - y * s), offset[1] + scale * (x * s + y * c)) for x, y in hull]


def box(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def overlap(a, b, gap):
    """Separating-axis test between convex polygons, with a required gap."""
    ba, bb = box(a), box(b)
    if ba[2] + gap <= bb[0] or bb[2] + gap <= ba[0] or ba[3] + gap <= bb[1] or bb[3] + gap <= ba[1]:
        return False
    for poly in (a, b):
        for i in range(len(poly)):
            p, q = poly[i], poly[(i + 1) % len(poly)]
            nx, ny = q[1] - p[1], p[0] - q[0]
            length = math.hypot(nx, ny)
            if length < 1e-12:
                continue
            nx, ny = nx / length, ny / length
            pa = [x * nx + y * ny for x, y in a]
            pb = [x * nx + y * ny for x, y in b]
            if max(pa) + gap <= min(pb) or max(pb) + gap <= min(pa):
                return False
    return True


def reach(poly):
    """Superellipse (L6) radius, matching the runtime's metropolis wall metric."""
    return max((abs(x) ** 6 + abs(y) ** 6) ** (1 / 6) for x, y in poly)


def area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2


def polygon_distance(x, y, polygon):
    inside = True
    distance = math.inf
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        inside = inside and dx * (y - a[1]) - dy * (x - a[0]) >= 0
        length = dx * dx + dy * dy
        t = max(0., min(1., ((x - a[0]) * dx + (y - a[1]) * dy) / length)) if length else 0.
        distance = min(distance, math.hypot(x - a[0] - t * dx, y - a[1] - t * dy))
    return -distance if inside else distance


def segment_distance(x, y, segment):
    x0, y0, x1, y1 = segment
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    t = max(0., min(1., ((x - x0) * dx + (y - y0) * dy) / length)) if length else 0.
    return math.hypot(x - x0 - t * dx, y - y0 - t * dy)


def polygon_field(gx, gy, polygon):
    """Signed distance from a grid to a counterclockwise convex polygon."""
    import numpy as np
    inside = np.ones(gx.shape, bool)
    distance = np.full(gx.shape, np.inf)
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        inside &= dx * (gy - a[1]) - dy * (gx - a[0]) >= 0
        length = dx * dx + dy * dy
        t = np.clip(((gx - a[0]) * dx + (gy - a[1]) * dy) / length, 0, 1) if length else 0
        distance = np.minimum(distance, np.hypot(gx - a[0] - t * dx, gy - a[1] - t * dy))
    return np.where(inside, -distance, distance)


def segment_field(gx, gy, segment):
    import numpy as np
    x0, y0, x1, y1 = segment
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    t = np.clip(((gx - x0) * dx + (gy - y0) * dy) / length, 0, 1) if length else 0
    return np.hypot(gx - x0 - t * dx, gy - y0 - t * dy)


def convex_hull(points):
    points = sorted(set((round(x, 7), round(y, 7)) for x, y in points))
    if len(points) < 3:
        raise ValueError("footprint needs three points")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(sequence):
        result = []
        for p in sequence:
            while len(result) > 1 and cross(result[-2], result[-1], p) <= 0:
                result.pop()
            result.append(p)
        return result

    return half(points)[:-1] + half(list(reversed(points)))[:-1]


def stable_rng(*values):
    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# --- Texture closure --------------------------------------------------------------
class Textures:
    """Content-addressed DDS closure of the candidate pack."""

    def __init__(self, out: Path):
        self.directory = out / "textures"
        self.directory.mkdir(parents=True, exist_ok=True)

    def add_bytes(self, data: bytes) -> str:
        name = hashlib.sha256(data).hexdigest() + ".dds"
        target = self.directory / name
        if not target.exists():
            target.write_bytes(data)
        return TEXTURE_PREFIX + name

    def add_file(self, path: Path) -> str:
        return self.add_bytes(path.read_bytes())


def dds_header(width, height, mips, dxgi, top_bytes):
    pixel_format = struct.pack("<II4s5I", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
    header = struct.pack("<7I44s32s5I", 124, 0xA1007, height, width, top_bytes, 0, mips, bytes(44),
                         pixel_format, 0x401008, 0, 0, 0, 0)
    return b"DDS " + header + struct.pack("<5I", dxgi, 3, 0, 1, 0)


def bc3_block(texels):
    """BC3 block from 16 (r,g,b,a) sRGB texels: least-squares-free endpoint fit."""
    alphas = [int(round(t[3])) for t in texels]
    a0, a1 = max(alphas), min(alphas)
    if a0 == a1:
        abits = 0
    else:
        palette = [a0, a1] + [((7 - k) * a0 + k * a1) // 7 for k in range(1, 7)]
        abits = 0
        for index, alpha in enumerate(alphas):
            abits |= min(range(8), key=lambda k: abs(palette[k] - alpha)) << (3 * index)
    # Endpoints along the principal luminance range, inset to reduce bias.
    luma = [t[0] * .3 + t[1] * .59 + t[2] * .11 for t in texels]
    hi, lo = texels[luma.index(max(luma))], texels[luma.index(min(luma))]
    inset = [(h - l) / 16 for h, l in zip(hi[:3], lo[:3])]
    hi = [min(255, max(0, h - i)) for h, i in zip(hi[:3], inset)]
    lo = [min(255, max(0, l + i)) for l, i in zip(lo[:3], inset)]

    def pack565(c):
        return (int(c[0] + 4) // 8 if c[0] < 252 else 31) << 11 | (int(c[1] + 2) // 4 if c[1] < 254 else 63) << 5 | \
            (int(c[2] + 4) // 8 if c[2] < 252 else 31)

    c0, c1 = pack565(hi), pack565(lo)
    if c0 < c1:
        c0, c1 = c1, c0
    if c0 == c1:
        indices = 0
    else:
        rgb = [((v >> 11) * 255 / 31, (v >> 5 & 63) * 255 / 63, (v & 31) * 255 / 31) for v in (c0, c1)]
        colours = rgb + [tuple((2 * a + b) / 3 for a, b in zip(rgb[0], rgb[1])),
                         tuple((a + 2 * b) / 3 for a, b in zip(rgb[0], rgb[1]))]
        indices = 0
        for index, texel in enumerate(texels):
            nearest = min(range(4), key=lambda k: sum((colours[k][c] - texel[c]) ** 2 for c in range(3)))
            indices |= nearest << (2 * index)
    return struct.pack("<BB", a0, a1) + abits.to_bytes(6, "little") + struct.pack("<HHI", c0, c1, indices)


def ground_texture(recipe: dict, size: int = 256) -> bytes:
    """A tileable, era-toned BC3 sRGB paving texture with a complete mip chain."""
    from PIL import Image
    import numpy as np
    from Renderer.tools.asset_compiler.clutter_blp_extractor import extract_civbig_texture
    import io
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "source.dds"
        extract_civbig_texture(SHARED_DATA / recipe["source"], target)
        data = bytearray(target.read_bytes())
    fmt = struct.unpack_from("<I", data, 128)[0]
    if fmt in (72, 78):  # decode sRGB blocks through the matching UNORM path
        struct.pack_into("<I", data, 128, fmt - 1)
    image = Image.open(io.BytesIO(bytes(data)))
    image.load()
    patch = image.convert("RGBA").crop(tuple(recipe["crop"])).resize((size, size), Image.LANCZOS)
    pixels = np.asarray(patch).astype(np.float64)
    # Tileable: blend with the half-offset copy across a seam mask.
    shifted = np.roll(np.roll(pixels, size // 2, 0), size // 2, 1)
    ramp = np.minimum(np.arange(size), size - 1 - np.arange(size)) / (size / 2)
    weight = np.clip(np.minimum.outer(ramp, ramp) * 3.0, 0, 1)[..., None]
    pixels = pixels * weight + shifted * (1 - weight)
    rgb = pixels[..., :3]
    luma = (rgb @ np.array([.2126, .7152, .0722]))[..., None]
    rgb = luma + (rgb - luma) * recipe.get("saturation", 1.0)
    rgb = np.clip(rgb * recipe.get("gain", 1.0), 0, 255)
    alpha = np.full((size, size), 255.0) if recipe.get("opaque") else np.where(pixels[..., 3] > 127, 255.0, 0.0)
    return bc3_mips(np.dstack([rgb, alpha]))


def bc3_mips(level) -> bytes:
    """A square RGBA float image as BC3 sRGB DDS bytes with a complete mip chain."""
    import numpy as np
    size = level.shape[0]
    levels = []
    while True:
        h, w = level.shape[:2]
        blocks = bytearray()
        for by in range(0, h, 4):
            for bx in range(0, w, 4):
                tile = level[by:by + 4, bx:bx + 4]
                if tile.shape[0] < 4 or tile.shape[1] < 4:
                    tile = np.pad(tile, ((0, 4 - tile.shape[0]), (0, 4 - tile.shape[1]), (0, 0)), mode="edge")
                blocks.extend(bc3_block([tuple(v) for v in tile.reshape(16, 4)]))
        levels.append(bytes(blocks))
        if h <= 4:
            break
        level = level.reshape(h // 2, 2, w // 2, 2, 4).mean(axis=(1, 3))
    return dds_header(size, size, len(levels), 78, len(levels[0])) + b"".join(levels)


def decode_dds(path: Path):
    """RGBA float pixels of a DDS top level; sRGB blocks decode through UNORM."""
    from PIL import Image
    import numpy as np
    import io
    data = bytearray(path.read_bytes())
    if data[84:88] == b"DX10":
        fmt = struct.unpack_from("<I", data, 128)[0]
        if fmt in (72, 78, 99):
            struct.pack_into("<I", data, 128, fmt - 1)
    image = Image.open(io.BytesIO(bytes(data)))
    image.load()
    return np.asarray(image.convert("RGBA")).astype(np.float64)


def recolor_texture(path: Path, recipe: dict) -> bytes:
    """Pull one hue family (e.g. terracotta trim) to a target colour, keeping the
    texture's own light and dark variation."""
    import numpy as np
    pixels = decode_dds(path)
    rgb = pixels[..., :3] / 255
    high, low = rgb.max(-1), rgb.min(-1)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0)
    delta = np.maximum(high - low, 1e-6)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.where(high == r, ((g - b) / delta) % 6, np.where(high == g, (b - r) / delta + 2, (r - g) / delta + 4)) * 60
    first, last = recipe["hue"]
    window = np.clip(np.minimum(hue - first, last - hue) / 8 + 1, 0, 1)
    s0 = recipe["saturation"]
    weight = (window * np.clip((saturation - s0) / .12, 0, 1))[..., None]
    luma = rgb @ np.array([.2126, .7152, .0722])
    reference = np.median(luma[weight[..., 0] > .5]) if (weight > .5).any() else 1.0
    target = np.array(recipe["target"]) / 255
    replaced = target * np.clip(luma / reference, .6, 1.12)[..., None]
    rgb = rgb * (1 - weight) + replaced * weight
    return bc3_mips(np.dstack([np.clip(rgb, 0, 1) * 255, pixels[..., 3]]))


# --- New models from the accent pack -----------------------------------------------
def scaled_direction(vector, z_scale):
    """A frame direction after a vertical scale, as prepare_pack's model()."""
    v = [vector[0], vector[1], vector[2] * z_scale]
    length = math.sqrt(sum(c * c for c in v))
    return [c / length for c in v] if length else v


def plinth(mesh) -> bool:
    """Source palace plinths and ground planes, as flat_palace.py removes them:
    a slab sunk below the ground, or a flat upward plane at ground level."""
    levels = [v["position"][2] for v in mesh["vertices"]]
    if min(levels) < -.007 and max(levels) < .005:
        return True
    return (0 <= min(levels) <= .002 and max(levels) - min(levels) < .002 and
            all(v["normal"][2] > .95 for v in mesh["vertices"]))


def accent_model(asset: str, textures: Textures, materials: list, material_ids: dict,
                 pack: Path = ACCENTS, flatten: bool = False) -> dict:
    from Renderer.lab.shared.cities.assets import component
    from Renderer.tools.prepare_city_recipes import normalized_frame
    body = component(asset, pack)
    if flatten:
        body["parts"] = [(mesh, material) for mesh, material in body["parts"] if not plinth(mesh)]
        points = [v["position"] for mesh, material in body["parts"] if material["alpha_mode"] != "blend"
                  for v in mesh["vertices"]]
        body["lo"] = [min(p[j] for p in points) for j in range(3)]
        body["hi"] = [max(p[j] for p in points) for j in range(3)]
    cx, cy = [(body["lo"][j] + body["hi"][j]) / 2 for j in (0, 1)]
    parts = []
    hull_points = []
    for mesh, material in body["parts"]:
        if material["alpha_mode"] == "blend":
            continue
        frame = normalized_frame(mesh)
        channels = material["channels"]
        names = ["base_color", "emissive", "ambient_occlusion", "normal_0", "gloss", "metalness", "opacity"]
        paths = []
        for name in names:
            texture = channels.get(name, {}).get("texture", "")
            paths.append(textures.add_file(ROOT / texture) if texture else "")
        if paths[2] and not all("uv1" in v for v in mesh["vertices"]):
            paths[2] = ""
        if paths[1] and not all("uv2" in v for v in mesh["vertices"]):
            paths[1] = ""
        mode = 3 if channels["base_color"].get("address_u") == "clamp" else 0
        bits = (bool(paths[2]) * 1 + bool(paths[3]) * 2 + (mode == 0) * 4 + bool(paths[4]) * 8 +
                bool(paths[5]) * 16 + bool(paths[6]) * 32)
        key = tuple(paths) + (mode, bits, 0)
        if key not in material_ids:
            material_ids[key] = len(materials)
            materials.append({"address": mode, "bits": bits, "ground": 0, "textures": paths})
        vertices = []
        for i, v in enumerate(mesh["vertices"]):
            # Lab city designs keep the source's vertical proportion: the
            # runtime divides z by this metric, as for every selected building.
            position = [v["position"][0] - cx, v["position"][1] - cy, v["position"][2] * SOURCE_Z]
            hull_points.append(position[:2])
            vertices.append(position + list(v["uv0"]) + scaled_direction(frame["normals"][i], 1 / SOURCE_Z) +
                            list(v.get("uv1", [0, 0])) + scaled_direction(frame["tangents"][i], SOURCE_Z) +
                            scaled_direction(frame["bitangents"][i], SOURCE_Z) + list(v.get("uv2", [0, 0])))
        parts.append({"material": material_ids[key], "vertices": vertices,
                      "indices": mesh["topology"]["indices"]})
    hull = [list(p) for p in convex_hull(hull_points)]
    low = [body["lo"][0], body["lo"][1], body["lo"][2] * SOURCE_Z]
    high = [body["hi"][0], body["hi"][1], body["hi"][2] * SOURCE_Z]
    return {"low": low, "high": high, "hull": hull, "materials": [p["material"] for p in parts],
            "vertex_count": sum(len(p["vertices"]) for p in parts),
            "wire": rp.model_wire(parts, low, high, hull), "asset": asset, "centre": [cx, cy]}


def model_sockets(pack: Path, asset: str, centre) -> list:
    """Operational flame/smoke/light points in an imported model's frame."""
    from Renderer.lab.studies.city_readability.extract_sockets import (
        bind_position, from_pack, operational, semantic)
    sockets = []
    for point in from_pack(ROOT / pack, asset):
        kind = semantic(point)
        if kind and operational(point):
            x, y, z = bind_position(point["skeleton_data"], point["bone"])
            sockets.append({"kind": kind, "position": [x - centre[0], y - centre[1], z * SOURCE_Z]})
    return sockets


def pack_closure(pack: Path, asset: str) -> set:
    """Every file an imported model's landmark names: what component() and the
    socket reader consume, for the asset job's input record."""
    manifest = pack / "manifest.json"
    entry = json.loads((ROOT / manifest).read_text())["assets"][asset]
    landmark = json.loads((ROOT / pack / entry["landmark"]).read_text())
    files = {manifest, pack / entry["landmark"]}
    parts = landmark["components"]
    files.update(pack / name for name in parts.get("geometry", []) + parts.get("skeletons", []))
    for name in parts.get("materials", []):
        files.add(pack / name)
        for channel in json.loads((ROOT / pack / name).read_text())["channels"].values():
            if isinstance(channel, dict) and channel.get("texture"):
                files.add(pack / channel["texture"])
    return files


def ground_recipes(style: dict) -> dict:
    return {era: style["ground"][era] for era in ERAS}


def write_ground(style: dict):
    """Make the era ground textures from the installed source decals (local)."""
    (ROOT / GROUND).mkdir(parents=True, exist_ok=True)
    for era in ERAS:
        (ROOT / GROUND / f"{era}.dds").write_bytes(ground_texture(style["ground"][era]))
    (ROOT / GROUND / "ground.json").write_text(json.dumps({"recipes": ground_recipes(style)}, indent=1) + "\n")


def ground_files(style: dict) -> dict:
    record = ROOT / GROUND / "ground.json"
    if not record.exists() or json.loads(record.read_text())["recipes"] != ground_recipes(style):
        raise ValueError("Era ground textures are stale; run recompose.py --ground")
    return {era: GROUND / f"{era}.dds" for era in ERAS}


# --- Composition --------------------------------------------------------------------
class Item:
    __slots__ = ("model", "scale", "yaw", "offset", "lights", "capital", "kind", "flags", "f")

    def __init__(self, model, scale, yaw, offset, lights, capital, kind, flags=0):
        self.model, self.scale, self.yaw, self.offset = model, scale, yaw, list(offset)
        self.lights, self.capital, self.kind, self.flags = lights, capital, kind, flags
        self.f = 1.0


def instance_kind(instance, model_records):
    if instance["capital"]:
        return "palace"
    source = model_records[instance["model"]]["pack"]
    if "farm-tree" in source:
        return "tree"
    if "Adjuncts" in source:
        return "wall"
    return "building"


def scaled_lights(lights, factor, yaw_delta=0.0):
    c, s = math.cos(yaw_delta), math.sin(yaw_delta)
    result = []
    for light in lights:
        light = list(light)
        x, y = light[0] * factor, light[1] * factor
        light[0], light[1] = x * c - y * s, x * s + y * c
        light[2] *= factor
        light[3] = min(1.0, light[3] * factor)
        dx, dy = light[8], light[9]
        light[8], light[9] = dx * c - dy * s, dx * s + dy * c
        result.append(light)
    return result


class Composer:
    def __init__(self, library, model_records, style, accent_ids, sockets=None, palaces=None, siblings=None):
        self.library = library
        self.records = model_records
        self.style = style
        self.accent_ids = accent_ids
        self.gap = style["gap"]
        self.sockets = sockets or {}
        # (culture, era) -> (generic palace model, replacement model)
        self.palaces = palaces or {}
        # (culture, era, size, walled, variant) -> the source instances of the
        # same city without a palace
        self.siblings = siblings or {}

    def swap_palace(self, items, culture, era):
        """Give a capital its culture's palace in place of the shared generic
        one, at the same footprint centre and width."""
        swap = self.palaces.get((culture, era))
        if not swap:
            return
        generic, chosen = swap
        for item in items:
            if item.kind != "palace" or item.model != generic:
                continue
            old, new = box(self.hull(generic)), box(self.hull(chosen))
            centre = transform([((old[0] + old[2]) / 2, (old[1] + old[3]) / 2)], item.scale, item.yaw, item.offset)[0]
            shift = (centre[0] - item.offset[0], centre[1] - item.offset[1])
            item.scale *= max(old[2] - old[0], old[3] - old[1]) / max(new[2] - new[0], new[3] - new[1])
            item.model, item.offset = chosen, list(centre)
            item.lights = [[l[0] - shift[0], l[1] - shift[1], *l[2:]] for l in item.lights]

    def effects(self, instances, seed):
        """Attached flame, smoke and night-light records per instance, capped
        per city by priority: accent smoke and furnaces, palace fires, then
        ordinary fires, lamps and chimney smoke."""
        settings = self.style.get("effects")
        if not settings:
            return
        kinds = {"flame": rp.FLAME, "smoke": rp.SMOKE, "night_light": rp.NIGHT_LIGHT}
        candidates = []
        authored = settings.get("authored", {})
        for index, instance in enumerate(instances):
            asset = self.records[instance["model"]]["asset"]
            source = asset.split("/")[1]  # component, palace or accent
            sockets = list(self.sockets.get(asset, []))
            for point in authored.get(asset, []):
                # An authored point at the model's top centre (e.g. a stack's
                # mouth whose source smoke belongs to the district, not it).
                top = self.library["models"][instance["model"]]["high"][2]
                sockets.append({"kind": point["kind"], "position": [0.0, 0.0, top * point.get("height", 1.0)]})
            for ordinal, socket in enumerate(sockets):
                profile = settings.get(f"{source}/{socket['kind']}") or settings.get(socket["kind"])
                if not profile or profile["strength"] <= 0:  # zero strength drops the kind
                    continue
                rng = stable_rng("effect", seed, index, ordinal)
                record = [*socket["position"], kinds[socket["kind"]], profile["width"], profile["height"],
                          rng.random() * 64, profile["strength"]]
                candidates.append((profile["priority"], index, record))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for instance in instances:
            instance["effects"] = []
        for priority, index, record in candidates[:settings.get("per_city", 24)]:
            if len(instances[index]["effects"]) < 16:
                instances[index]["effects"].append(record)

    def hull(self, model):
        return self.library["models"][model]["hull"]

    def poly(self, item, scale=None):
        return transform(self.hull(item.model), item.scale if scale is None else scale, item.yaw, item.offset)

    def items(self, template):
        return [Item(i["model"], i["scale"], i["rotation"], i["offset"], i["lights"], i["capital"],
                     instance_kind(i, self.records)) for i in template["instances"]]

    def civic(self, items):
        buildings = [i for i in items if i.kind == "building"]
        if not buildings:
            return
        core = min(buildings, key=lambda i: math.hypot(*i.offset))
        if math.hypot(*core.offset) < .14:
            core.kind = "civic"

    def place_accents(self, items, era, size, limit, seed):
        roles = self.style["accents"].get(ERAS[era], [[], [], []])[size]
        rng = stable_rng("accent", seed)
        removed_total = 0
        for index, role in enumerate(roles):
            model = self.accent_ids[role]
            hull = self.hull(model)
            width = max(box(hull)[2] - box(hull)[0], box(hull)[3] - box(hull)[1])
            scale = self.style["accent_width"][role] / width
            # Smoke from an accent straight behind the palace reads as the
            # palace's own chimney; keep smoking accents off that line.
            asset = self.records[model]["asset"]
            authored = self.style.get("effects", {}).get("authored", {}).get(asset, [])
            smoky = any(p["kind"] == "smoke" for p in list(self.sockets.get(asset, [])) + authored)
            best = self.site(items, model, scale, 0.0, limit, rng, index, smoky)
            if best is None:
                continue
            for other in best[2]:
                items.remove(other)
                removed_total += other.kind == "building"
            items.append(Item(model, scale, 0.0, best[1], [], 0, "accent", rp.ACCENT))
        return removed_total

    def site(self, items, model, scale, yaw, limit, rng, index=0, smoky=False):
        """The cheapest ring position for a new body: it may replace a few
        ordinary buildings but never a palace, civic, wall or accent."""
        hull = self.hull(model)
        width = max(box(hull)[2] - box(hull)[0], box(hull)[3] - box(hull)[1])
        palace = any(o.kind == "palace" for o in items)
        best = None
        for ring in (.18, .26, .34, .42, .5, .58):
            for step in range(36):
                angle = 2 * math.pi * (step + rng.random() * .5) / 36
                offset = [ring * math.cos(angle), ring * math.sin(angle)]
                poly = transform(hull, scale, yaw, offset)
                if reach(poly) > limit:
                    continue
                blocked = False
                covered = []
                for other in items:
                    if not overlap(poly, self.poly(other), self.gap):
                        continue
                    if other.kind in ("palace", "civic", "wall", "accent"):
                        blocked = True
                        break
                    covered.append(other)
                if blocked or sum(o.kind == "building" for o in covered) > self.style["accent_removal_limit"]:
                    continue
                # Prefer the back of the settlement (screen up: x + y < 0)
                # so stacks and towers rise behind the street front.
                back = (offset[0] + offset[1]) / max(ring, 1e-6)
                # Thin accents (stacks, water towers) show best in profile at
                # the back corners rather than straight behind the centre.
                side = abs(offset[0] - offset[1]) / max(ring, 1e-6) if width * scale < .2 else 0.0
                behind = 0.0
                if smoky and palace and back < 0:
                    behind = -back / math.sqrt(2) * (1 - abs(offset[0] - offset[1]) / (ring * math.sqrt(2)))
                cost = sum(area(self.poly(o)) for o in covered if o.kind == "building") + .05 * back + \
                    .02 * index * ring - .04 * side + .12 * behind
                if best is None or cost < best[0]:
                    best = (cost, offset, covered)
        return best

    def capital_towers(self, items, template, limit, seed):
        """A large late-era capital keeps its peers' skyline: the tallest
        towers of the same city without a palace stand beside the palace."""
        count = self.style.get("capital_towers", {}).get(ERAS[template["era"]], [0, 0, 0])[template["size"]]
        sibling = self.siblings.get((template["culture"], template["era"], template["size"],
                                     template["walled"], template["variant"]))
        if not template["capital"] or not count or not sibling:
            return
        def height(i):
            m = self.library["models"][i["model"]]
            return (m["high"][2] - m["low"][2]) * i["scale"]
        def width(i):
            m = self.library["models"][i["model"]]
            return max(m["high"][0] - m["low"][0], m["high"][1] - m["low"][1]) * i["scale"]
        towers = sorted((i for i in sibling if not i["capital"] and height(i) >= 1.4 * width(i)),
                        key=height, reverse=True)[:count]
        rng = stable_rng("towers", seed)
        for index, tower in enumerate(towers):
            best = self.site(items, tower["model"], tower["scale"], tower["rotation"], limit, rng, index)
            if best is None:
                continue
            for other in best[2]:
                items.remove(other)
            items.append(Item(tower["model"], tower["scale"], tower["rotation"], best[1],
                              [list(l) for l in tower["lights"]], 0, "civic", rp.SITE_OPTIONAL))

    def grow(self, items, factor, limit):
        movable = [i for i in items if i.kind in ("building", "civic", "palace", "accent")]
        fixed = [i for i in items if i.kind == "wall"]
        for item in movable:
            item.f = 1.0 if item.kind == "accent" else factor
        base_reach = {id(i): reach(self.poly(i)) for i in movable}
        for _ in range(60):
            polys = {id(i): self.poly(i, i.scale * i.f) for i in movable}
            walls = [self.poly(w) for w in fixed]
            bad = set()
            for a_index, a in enumerate(movable):
                pa = polys[id(a)]
                if reach(pa) > max(limit, base_reach[id(a)]) and a.f > 1.0:
                    bad.add(id(a))
                for w in walls:
                    if a.f > 1.0 and overlap(pa, w, self.gap):
                        bad.add(id(a))
                for b in movable[a_index + 1:]:
                    if (a.f > 1.0 or b.f > 1.0) and overlap(pa, polys[id(b)], self.gap):
                        if a.f > 1.0:
                            bad.add(id(a))
                        if b.f > 1.0:
                            bad.add(id(b))
            if not bad:
                break
            for item in movable:
                if id(item) in bad:
                    item.f = max(1.0, item.f * .965)
        for item in movable:
            if item.f != 1.0:
                item.lights = scaled_lights(item.lights, item.f)
                item.scale *= item.f
        # Trees yield to grown architecture.
        bodies = [self.poly(i) for i in movable]
        items[:] = [i for i in items if i.kind != "tree" or not any(overlap(self.poly(i), b, 0.0) for b in bodies)]

    def grow_priority(self, items, factor, limit):
        """Larger bodies claim room first; a smaller neighbour shrinks and, if
        it still collides at its authored size, gives way to infill later."""
        fixed = [self.poly(i) for i in items if i.kind == "wall"]
        movable = [i for i in items if i.kind in ("building", "civic", "palace", "accent")]
        movable.sort(key=lambda i: (i.kind not in ("palace", "civic", "accent"), -area(self.poly(i))))
        placed = []
        removed = []
        for item in movable:
            base = reach(self.poly(item))
            f = 1.0 if item.kind == "accent" else factor
            while True:
                poly = self.poly(item, item.scale * f)
                clash = (reach(poly) > max(limit, base) or any(overlap(poly, w, self.gap) for w in fixed) or
                         any(overlap(poly, q, self.gap) for q in placed))
                if not clash or f <= 1.0:
                    break
                f = max(1.0, f * .965)
            if clash and item.kind == "building":
                removed.append(item)
                continue
            if f != 1.0:
                item.lights = scaled_lights(item.lights, f)
                item.scale *= f
            placed.append(self.poly(item))
        for item in removed:
            items.remove(item)
        bodies = placed
        items[:] = [i for i in items if i.kind != "tree" or not any(overlap(self.poly(i), b, 0.0) for b in bodies)]
        return len(removed)

    def infill(self, items, limit, seed):
        rng = stable_rng("infill", seed)
        settings = self.style["infill"]
        buildings = [i for i in items if i.kind == "building"]
        if not buildings:
            return 0
        pool = {}
        for item in buildings:
            pool.setdefault(item.model, item)
        pool = sorted(pool.values(), key=lambda i: -area(self.poly(i)))
        budget = int(len(buildings) * settings["max_fraction"] + .5)
        light_budget = 128 - sum(len(i.lights) for i in items)
        occupied = [self.poly(i) for i in items if i.kind != "tree"]
        trees = [i for i in items if i.kind == "tree"]
        anchors = [i.offset for i in items if i.kind in ("building", "civic", "palace", "accent")]
        step = settings["step"]
        span = int(limit / step) + 1
        points = [(x * step + (rng.random() - .5) * step * .4, y * step + (rng.random() - .5) * step * .4)
                  for x in range(-span, span + 1) for y in range(-span, span + 1)]
        points.sort(key=lambda p: math.hypot(*p) + rng.random() * .02)
        added = 0
        for x, y in points:
            if added >= budget or len(items) >= 126:
                break
            if math.hypot(x, y) < settings["clear_center"]:
                continue
            if min(math.hypot(x - a[0], y - a[1]) for a in anchors) > settings["reach"]:
                continue
            for template in pool if rng.random() < .5 else pool[::-1]:
                candidate = Item(template.model, template.scale, template.yaw, (x, y), [], 0, "building",
                                 rp.SITE_OPTIONAL)
                poly = self.poly(candidate)
                if reach(poly) > limit or any(overlap(poly, o, self.gap) for o in occupied):
                    continue
                if len(template.lights) <= light_budget:
                    candidate.lights = [list(light) for light in template.lights]
                    light_budget -= len(template.lights)
                items.append(candidate)
                occupied.append(poly)
                anchors.append((x, y))
                # A planted tree under new architecture gives way.
                for tree in trees:
                    if tree in items and overlap(self.poly(tree), poly, 0.0):
                        items.remove(tree)
                added += 1
                break
        return added

    def paving(self, items, era, size, seed, material):
        settings = self.style["paving"]
        rng = stable_rng("paving", seed)
        bodies = [i for i in items if i.kind in ("building", "civic", "palace", "accent")]
        polys = [self.poly(i) for i in bodies]
        walls = [i for i in items if i.kind == "wall"]
        # Lanes: a minimum spanning tree over body centres, plus spokes from
        # the plaza, gives the plate connected streets rather than islands.
        centres = [tuple(i.offset) for i in bodies] + [(0.0, 0.0)]
        connected = {len(centres) - 1}
        lanes = []
        while len(connected) < len(centres):
            _, a, b = min(((centres[i][0] - centres[j][0]) ** 2 + (centres[i][1] - centres[j][1]) ** 2, i, j)
                          for i in connected for j in range(len(centres)) if j not in connected)
            lanes.append((*centres[a], *centres[b]))
            connected.add(b)
        margin, feather, lane, plaza = settings["margin"], settings["feather"], settings["lane"], settings["plaza"]
        boxes = [box(p) for p in polys]
        low = [min(b[0] for b in boxes) - margin - feather, min(b[1] for b in boxes) - margin - feather]
        high = [max(b[2] for b in boxes) + margin + feather, max(b[3] for b in boxes) + margin + feather]
        step = settings["step"]
        width = int(math.ceil((high[0] - low[0]) / step)) + 1
        height = int(math.ceil((high[1] - low[1]) / step)) + 1
        import numpy as np
        xs = low[0] + np.arange(width) * step
        ys = low[1] + np.arange(height) * step
        gx, gy = np.meshgrid(xs, ys)
        d = np.hypot(gx, gy) - plaza
        for poly in polys:
            d = np.minimum(d, polygon_field(gx, gy, poly) - margin)
        for segment in lanes:
            d = np.minimum(d, segment_field(gx, gy, segment) - lane)
        # Coarse blue-noise-like jitter keeps the plate edge organic but crisp.
        cells = (int(height / 3) + 2, int(width / 3) + 2)
        field = np.array([[(rng.random() - .5) * 2 * settings["noise"] for _ in range(cells[1])]
                          for _ in range(cells[0])])
        d = d + field[np.arange(height)[:, None] // 3, np.arange(width)[None, :] // 3]
        if walls:
            # Stay inside a town wall ring when one is present.
            ring = .9 * min(math.hypot(*w.offset) for w in walls)
            d = np.maximum(d, np.hypot(gx, gy) - ring)
        value = np.clip(-d / feather, 0, 1)
        alpha = (value * value * (3 - 2 * value)).ravel().tolist()
        used = {}
        vertices = []
        indices = []
        for iy in range(height - 1):
            for ix in range(width - 1):
                i = iy * width + ix
                quad = (i, i + 1, i + width, i + width + 1)
                if max(alpha[q] for q in quad) <= 0:
                    continue
                for tri in ((i, i + 1, i + width), (i + 1, i + width + 1, i + width)):
                    for q in tri:
                        if q not in used:
                            used[q] = len(vertices)
                            qx, qy = q % width, q // width
                            # Paving vertices use world rows (y down): flip.
                            vertices.append([low[0] + qx * step, -(low[1] + qy * step), alpha[q]])
                        indices.append(used[q])
        if len(vertices) > 30000:
            raise ValueError("paving exceeds the runtime vertex bound")
        return {"material": material, "period": [settings["period"], settings["period"]],
                "atlas": [0.0, 0.0, 1.0, 1.0], "vertices": vertices, "indices": indices}

    def compose(self, template, ground_material, seed):
        era, size = template["era"], template["size"]
        limit = self.style["extent"][size]
        items = self.items(template)
        self.swap_palace(items, template["culture"], era)
        self.capital_towers(items, template, limit, seed)
        self.civic(items)
        removed = self.place_accents(items, era, size, limit, seed)
        yielded = 0
        if self.style.get("growth_mode", {}).get(ERAS[era]) == "priority":
            yielded = self.grow_priority(items, self.style["growth"][ERAS[era]], limit)
        else:
            self.grow(items, self.style["growth"][ERAS[era]], limit)
        added = self.infill(items, limit, seed)
        paving = self.paving(items, era, size, seed, ground_material)
        instances = []
        # Rigid draw order is irrelevant; keep architecture before plantings.
        order = {"palace": 0, "civic": 1, "accent": 2, "building": 3, "wall": 4, "tree": 5}
        for item in sorted(items, key=lambda i: order[i.kind]):
            poly = self.poly(item)
            b = box(poly)
            bounds = [b[0] - item.offset[0], b[1] - item.offset[1], b[2] - item.offset[0], b[3] - item.offset[1]]
            flags = item.flags
            if item.kind in ("building", "accent"):
                flags |= rp.SITE_OPTIONAL
            if item.kind == "tree":
                flags |= rp.SITE_OPTIONAL | rp.TREE
            lights = [list(light) for light in item.lights]
            for light in lights:
                light[7] *= self.style.get("light_intensity", 1.0)
            instances.append({"model": item.model, "capital": 1 if item.kind == "palace" else 0,
                              "scale": item.scale, "rotation": item.yaw, "offset": item.offset,
                              "bounds": bounds, "flags": flags, "lights": lights})
        self.effects(instances, seed)
        total = sum(len(i["lights"]) for i in instances)
        while total > 128:
            for instance in reversed(instances):
                if instance["lights"]:
                    instance["lights"].pop()
                    total -= 1
                    break
        return instances, paving, {"accent_removed": removed, "infill": added, "yielded": yielded,
                                   "instances": len(instances)}


def build(out: Path = OUT, version: int = 5, only=None, return_inputs=False):
    style = json.loads(STYLE.read_text())
    here = Path(__file__).resolve()
    read = {here, here.with_name("runtime_pack.py"), here.with_name("extract_sockets.py"), STYLE.resolve(),
            SOURCE / "city.bin", SOURCE / "manifest.json"}
    library = rp.decode(SOURCE / "city.bin")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    records = manifest["models"]
    if len(records) != len(library["models"]):
        raise ValueError("runtime manifest and library disagree")
    out = Path(out).resolve()
    if out == SOURCE.resolve():
        raise ValueError("never overwrite the frozen source city library")
    if (out / "textures").exists():
        shutil.rmtree(out / "textures")
    textures = Textures(out)
    for path in sorted((SOURCE / "textures").glob("*.dds")):
        read.add(path)
        # Content-addressed and never rewritten in place: link, don't copy.
        try:
            os.link(path, textures.directory / path.name)
        except OSError:
            shutil.copyfile(path, textures.directory / path.name)
    materials = library["materials"]
    # Material recolours (e.g. stone trim on a wall set) apply to every
    # material of the matching models: their base-colour texture is replaced.
    for prefix, recipe in style.get("recolor", {}).items():
        targets = {m for model, record in zip(library["models"], records)
                   if record["asset"].startswith(prefix) for m in model["materials"]}
        for index in sorted(targets):
            source = materials[index]["textures"][0]
            name = source.split("/")[-1]
            materials[index]["textures"][0] = textures.add_bytes(recolor_texture(SOURCE / "textures" / name, recipe))
    material_ids = {tuple(m["textures"]) + (m["address"], m["bits"], m["ground"]): i
                    for i, m in enumerate(materials)}
    palace_sockets = {}
    # Era accents.
    accent_ids = {}
    for role in sorted({role for roles in style["accents"].values() for tier in roles for role in tier}):
        model = accent_model("city/accent/" + role.replace("_", "-"), textures, materials, material_ids)
        read.update(ROOT / f for f in pack_closure(ACCENTS, model["asset"]))
        accent_ids[role] = len(library["models"])
        library["models"].append(model)
        records.append({"asset": model["asset"], "pack": str(ACCENTS)})
    # Culture palaces for capitals that share the generic one.
    palaces = {}
    if style.get("palaces"):
        settings = style["palaces"]
        pack = Path(settings["pack"])
        generic = next(i for i, r in enumerate(records) if r["asset"] == settings["replace"])
        chosen_ids = {}
        for key, tag in settings["choices"].items():
            asset = "city/palace/root/" + tag
            if asset not in chosen_ids:
                model = accent_model(asset, textures, materials, material_ids, pack, flatten=True)
                read.update(ROOT / f for f in pack_closure(pack, asset))
                chosen_ids[asset] = len(library["models"])
                library["models"].append(model)
                records.append({"asset": asset, "pack": str(pack)})
                palace_sockets[asset] = model_sockets(pack, asset, model["centre"])
            culture, era = key.split("/")
            for e in (range(len(ERAS)) if era == "*" else (ERAS.index(era),)):
                palaces.setdefault((CULTURES.index(culture), e), (generic, chosen_ids[asset]))
    # Era ground plates.
    ground_materials = {}
    grounds = ground_files(style)
    read.add(ROOT / GROUND / "ground.json")
    for era in ERAS:
        read.add(ROOT / grounds[era])
        texture = textures.add_file(ROOT / grounds[era])
        ground_materials[era] = len(materials)
        materials.append({"address": 0, "bits": 4, "ground": 1, "textures": [texture] + [""] * 6})
    sockets_path = ROOT / ACCENTS / "sockets.json"
    sockets = json.loads(sockets_path.read_text())["models"] if sockets_path.exists() else {}
    if sockets_path.exists():
        read.add(sockets_path)
    sockets.update(palace_sockets)
    if sockets and style.get("effects"):
        # Attached effects draw with a ground-flagged material: read-only depth,
        # no shadow casting. Its texture is bound but not sampled.
        library["effect_material"] = len(materials)
        materials.append({"address": 0, "bits": 4, "ground": 1,
                          "textures": [materials[ground_materials["ancient"]]["textures"][0]] + [""] * 6})
    siblings = {(t["culture"], t["era"], t["size"], t["walled"], t["variant"]): t["instances"]
                for t in library["templates"] if not t["capital"]}
    composer = Composer(library, records, style, accent_ids, sockets, palaces, siblings)
    report = []
    for index, template in enumerate(library["templates"]):
        if only and (ERAS[template["era"]], template["size"]) not in only:
            continue
        seed = (template["culture"], template["era"], template["size"], template["capital"],
                template["walled"], template["variant"])
        instances, paving, stats = composer.compose(template, ground_materials[ERAS[template["era"]]], seed)
        template["instances"] = instances
        template["paving"] = paving
        template["authority"] = template["authority"].replace("lab-fixed-", "lab-readable-")
        report.append({"template": index, "seed": seed, **stats, "paving_vertices": len(paving["vertices"])})
    library["look"] = style.get("look", [0.0] * rp.LOOK_FIELDS)
    data = rp.encode(library, version)
    (out / "city.bin").write_bytes(data)
    (out / "recompose-report.json").write_text(json.dumps({
        "source_city_bin_sha256": hashlib.sha256((SOURCE / "city.bin").read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(STYLE.read_bytes()).hexdigest(),
        "version": version, "bytes": len(data), "templates": report}, indent=1) + "\n")
    inputs = {p.resolve().relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(read)}
    (out / "manifest.json").write_text(json.dumps({
        "schema": "c3x.city_composition.v1",
        "builder": here.relative_to(ROOT).as_posix(), "version": version,
        "texture_prefix": TEXTURE_PREFIX,
        "models": [{"asset": r["asset"], "pack": r["pack"]} for r in records],
        "source_sha256": inputs}, indent=1) + "\n")
    print("PASS", len(library["models"]), "models", len(materials), "materials",
          len(library["templates"]), "templates; bytes", len(data), flush=True)
    summary = {"bytes": len(data), "templates": report}
    return (summary, inputs) if return_inputs else summary


def declared_sources() -> dict:
    """The builder's fixed inputs, before a build has recorded its full read set."""
    here = Path(__file__).resolve()
    paths = [here, here.with_name("runtime_pack.py"), here.with_name("extract_sockets.py"), STYLE.resolve(),
             SOURCE / "city.bin", SOURCE / "manifest.json", ROOT / ACCENTS / "sockets.json",
             ROOT / GROUND / "ground.json"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
            for p in paths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--version", type=int, default=5, choices=(4, 5))
    parser.add_argument("--ground", action="store_true", help="remake the era ground texture cache first")
    args = parser.parse_args()
    if args.ground:
        write_ground(json.loads(STYLE.read_text()))
    build(args.output, args.version)
