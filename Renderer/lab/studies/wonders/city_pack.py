"""Lab wonders as Renderer64 city bodies (the production city library format).

Civ VI maps a wonder's night lights (emissive) and ambient occlusion with its
second and third UV sets, as it does for city buildings. Resource bodies carry
one UV set, so the Lab draws wonders as city bodies instead: Civ VI's full
material (colour, emissive by uv2, AO by uv1, normal, gloss, metalness,
opacity), the city shader's night response, attached effects (fires, smoke,
lamps) and point lights.

The Lab library is the production CityCompositionRuntime library (the runtime
requires its complete city set) plus one composition per wonder cell, whose
authority is "wonder:<cell name>"; effects_tree.py teaches the private tree to
select it on the tile whose Lab resource composition has that name. Production
textures are hard-linked into the Lab pack, where the loader looks for them.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

from Renderer.lab.studies.city_readability import runtime_pack as rp
from Renderer.lab.studies.city_readability.recompose import convex_hull, scaled_direction
from Renderer.tools.prepare_city_recipes import normalized_frame

ROOT = Path(__file__).resolve().parents[4]
PRODUCTION = ROOT / "Renderer/packs/CityCompositionRuntime"
PACK = ROOT / "Renderer/packs/WonderCityLab"
TEXTURES = "Renderer/packs/WonderCityLab/textures/"
SOURCE_Z = 0.648266978876
# Resource bodies stand local_z * 150 / .82 height units tall; city bodies
# scale * z / SOURCE_Z * 112. Wonders keep the height the resource path gave them.
LIFT = 150 / .82 / 112
# sRGB scale of a light map's colours drawn as night emission (about .27 linear).
LIGHT_MAP = .55
CHANNELS = ("base_color", "emissive", "ambient_occlusion", "normal_0", "gloss", "metalness", "opacity")


class Library:
    """The production library with wonder models, materials and compositions appended."""

    def __init__(self) -> None:
        self.library = rp.decode(PRODUCTION / "city.bin")
        self.library["templates"] = [t for t in self.library["templates"] if not t["authority"].startswith("wonder:")]
        self.material_ids: dict[tuple, int] = {}
        (PACK / "textures").mkdir(parents=True, exist_ok=True)

    def texture(self, path: Path, role: str = "") -> str:
        return self.add(path.read_bytes(), role)

    def add(self, data: bytes, role: str = "") -> str:
        name = hashlib.sha256(data).hexdigest() + ".dds"
        target = PACK / "textures" / name
        if not target.exists():
            target.write_bytes(data)
        return TEXTURES + name

    def material(self, material: dict, sources: Path, has_uv1: bool, has_uv2: bool,
                 colour: Path | None = None) -> tuple[int, bool]:
        """The material's id, and whether its night emission is Civ VI's baked light
        map (laid out by uv1) rather than an emissive map (by uv2)."""
        channels = dict(material.get("channels", {}))
        light_map = (not isinstance(channels.get("emissive"), dict) and isinstance(channels.get("light_map"), dict)
                     and has_uv1)
        if light_map:
            channels["emissive"] = channels["light_map"]
            has_uv2 = True
        paths = []
        for name in CHANNELS:
            channel = channels.get(name)
            texture = channel.get("texture") if isinstance(channel, dict) else None
            if name == "base_color" and colour is not None:
                paths.append(self.texture(colour, name))
            elif name == "emissive" and light_map:
                # A baked light map holds all the building's light, not just its
                # lamps: drawn at LIGHT_MAP of an emissive map's strength.
                paths.append(self.add(scaled_bc1((sources / texture).read_bytes(), LIGHT_MAP), name))
            else:
                paths.append(self.texture(sources / texture, name) if texture else "")
        # AO is mapped by uv1 and emissive by uv2, as the production city pack reads them.
        if not has_uv1:
            paths[2] = ""
        if not has_uv2:
            paths[1] = ""
        clamp = isinstance(channels.get("base_color"), dict) and channels["base_color"].get("address_u") == "clamp"
        bits = (bool(paths[2]) * 1 + bool(paths[3]) * 2 + (not clamp) * 4 + bool(paths[4]) * 8 +
                bool(paths[5]) * 16 + bool(paths[6]) * 32)
        key = tuple(paths) + (bits,)
        if key not in self.material_ids:
            self.material_ids[key] = len(self.library["materials"])
            self.library["materials"].append({"address": 3 if clamp else 0, "bits": bits, "ground": 0,
                                              "textures": paths})
        return self.material_ids[key], light_map

    def model(self, pieces: list) -> tuple[int, list]:
        """A model from pieces in tile space: (material id, vertices of
        (x, y, z, normal, uv0, uv1, uv2) with x, y from the tile centre and z the
        rendered height in tiles, indices). Returns the model index and bounds."""
        parts, points, low, high = [], [], [9.0] * 3, [-9.0] * 3
        for material, vertices, indices in pieces:
            mesh = {"vertices": [{"position": [x, y, z], "normal": n, "uv0": uv0}
                                 for x, y, z, n, uv0, _uv1, _uv2 in vertices],
                    "topology": {"indices": indices}}
            frame = normalized_frame(mesh)
            wire = []
            for i, (x, y, z, _n, uv0, uv1, uv2) in enumerate(vertices):
                position = [x, y, z * SOURCE_Z]
                points.append(position[:2])
                for axis in range(3):
                    low[axis], high[axis] = min(low[axis], position[axis]), max(high[axis], position[axis])
                wire.append(position + list(uv0) + scaled_direction(frame["normals"][i], 1 / SOURCE_Z) +
                            list(uv1) + scaled_direction(frame["tangents"][i], SOURCE_Z) +
                            scaled_direction(frame["bitangents"][i], SOURCE_Z) + list(uv2))
            parts.append({"material": material, "vertices": wire, "indices": indices})
        hull = [list(p) for p in convex_hull(points)]
        self.library["models"].append({"low": low, "high": high, "hull": hull,
                                       "materials": [p["material"] for p in parts],
                                       "vertex_count": sum(len(p["vertices"]) for p in parts),
                                       "wire": rp.model_wire(parts, low, high, hull)})
        return len(self.library["models"]) - 1, [low[0], low[1], high[0], high[1]]

    def composition(self, name: str, bodies: list) -> None:
        """One wonder cell: bodies of (model index, bounds, effects, lights)."""
        instances = [{"model": model, "capital": 0, "scale": 1.0, "rotation": 0.0, "offset": [0.0, 0.0],
                      "bounds": bounds, "flags": 0, "lights": lights[:16], "effects": effects[:16]}
                     for model, bounds, effects, lights in bodies]
        self.library["templates"].append({
            "culture": 0, "era": 0, "size": 0, "capital": 0, "environment": 0, "variant": 0, "walled": 0,
            "owns_walls": 0, "anchor_layout": 1, "authority": "wonder:" + name, "clearance": [0.0, 0.0, 0.0, 0.0],
            "instances": instances, "paving": None, "foundation": None})

    def save(self) -> Path:
        # The loader reads production texture paths from this pack's folder.
        for source in (PRODUCTION / "textures").iterdir():
            target = PACK / "textures" / source.name
            if not target.exists():
                os.link(source, target)
        data = rp.encode(self.library, self.library["version"])
        (PACK / "city.bin").write_bytes(data)
        return PACK / "city.bin"


def effect_record(du: float, dv: float, lift: float, kind: int, width: float, height: float, seed: float,
                  strength: float) -> list:
    """A city effect record from the resource path's tile offsets (effect_lines)."""
    return [du - .5, .5 - dv, lift * LIFT * SOURCE_Z, float(kind), width, height, seed, strength]


def scaled_bc1(data: bytes, factor: float) -> bytes:
    """A BC1 DDS with every block's two endpoint colours scaled (all its texels
    scale with them); the endpoint order, which selects the block mode, is kept."""
    from Renderer.tools.asset_compiler import build_resource_compositions as rc
    import struct
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".dds", delete=False) as handle:
        handle.write(data)
    try:
        _w, _h, _mips, dxgi, payload = rc.dds(Path(handle.name))
    finally:
        os.unlink(handle.name)
    if dxgi not in (71, 72):
        return data
    start = len(data) - len(payload)
    out = bytearray(data)

    def scale(colour: int) -> int:
        r, g, b = colour >> 11, (colour >> 5) & 63, colour & 31
        return (int(r * factor + .5) << 11) | (int(g * factor + .5) << 5) | int(b * factor + .5)
    for at in range(start, len(data) - 7, 8):
        c0, c1 = struct.unpack_from("<HH", data, at)
        s0, s1 = scale(c0), scale(c1)
        if c0 > c1 and s0 <= s1:
            s0 = min(0xFFFF, s1 + 1)
        elif c0 <= c1 and s0 > s1:
            s0 = s1
        struct.pack_into("<HH", out, at, s0, s1)
    return bytes(out)
