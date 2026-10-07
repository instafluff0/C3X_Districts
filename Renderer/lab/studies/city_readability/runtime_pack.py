"""Read and write the generic compiled city library (city.bin).

The production CityCompositionRuntime pack is the preserved output of the
offline recipe pipeline; its normalized source intakes are no longer needed to
revise compositions. This codec keeps model geometry as opaque wire bytes and
exposes materials, model bounds/hulls and composition placements for Lab
recomposition. Version 5 adds the Lab readability fields described in
README.md; versions 2-4 decode unchanged.
"""
from __future__ import annotations

import struct
from pathlib import Path

LOOK_FIELDS = 8
# Instance flags (version 5).
SITE_OPTIONAL = 1   # may yield to a river, water, mountain or steep relief
ACCENT = 2          # era accent added by Lab recomposition (review only)
TREE = 4            # small planting rather than architecture
NO_EFFECT_MATERIAL = 0xFFFFFFFF
# Attached effect record (version 5): x, y, z (model space), kind, width (tile
# widths), height (world height units), seed, strength.
FLAME, SMOKE, NIGHT_LIGHT = 0, 1, 2


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.cursor = 8

    def take(self, count: int) -> bytes:
        if count < 0 or self.cursor + count > len(self.data):
            raise ValueError("truncated city library")
        start = self.cursor
        self.cursor += count
        return self.data[start:self.cursor]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def floats(self, count: int) -> list[float]:
        return list(struct.unpack(f"<{count}f", self.take(4 * count)))

    def string(self) -> str:
        return self.take(self.u32()).decode()


def decode(path: Path) -> dict:
    data = Path(path).read_bytes()
    magic = data[:8]
    if magic not in (b"C3XCITY2", b"C3XCITY3", b"C3XCITY4", b"C3XCITY5"):
        raise ValueError("not a city library")
    version = magic[7] - ord("0")
    r = Reader(data)
    counts = [r.u32() for _ in range(3)]
    look = r.floats(LOOK_FIELDS) if version >= 5 else [0.0] * LOOK_FIELDS
    effect_material = r.u32() if version >= 5 else NO_EFFECT_MATERIAL
    materials = []
    for _ in range(counts[0]):
        address, bits, ground = r.u32(), r.u32(), r.u32()
        materials.append({"address": address, "bits": bits, "ground": ground,
                          "textures": [r.string() for _ in range(7)]})
    models = []
    for _ in range(counts[1]):
        start = r.cursor
        parts = r.u32()
        bounds = r.floats(6)
        hull = [r.floats(2) for _ in range(r.u32())]
        part_materials = []
        vertex_count = 0
        for _ in range(parts):
            material = r.u32()
            vertices, indices = r.u32(), r.u32()
            r.take(vertices * 72 + indices * 4)
            part_materials.append(material)
            vertex_count += vertices
        models.append({"low": bounds[:3], "high": bounds[3:], "hull": hull,
                       "materials": part_materials, "vertex_count": vertex_count,
                       "wire": data[start:r.cursor]})
    templates = []
    for _ in range(counts[2]):
        t = {key: r.u32() for key in ("culture", "era", "size", "capital", "environment")}
        if version >= 4:
            t.update({key: r.u32() for key in ("variant", "walled", "owns_walls", "anchor_layout")})
        else:
            t.update(variant=0, walled=0, owns_walls=0, anchor_layout=0)
        t["authority"] = r.string()
        t["clearance"] = r.floats(4)
        instances = []
        for _ in range(r.u32()):
            model, capital = r.u32(), r.u32()
            values = r.floats(8)
            flags = r.u32() if version >= 5 else 0
            lights = [r.floats(12) for _ in range(r.u32())]
            effects = [r.floats(8) for _ in range(r.u32())] if version >= 5 else []
            instances.append({"model": model, "capital": capital, "scale": values[0],
                              "rotation": values[1], "offset": values[2:4],
                              "bounds": values[4:8], "flags": flags, "lights": lights,
                              "effects": effects})
        t["instances"] = instances
        t["paving"] = None
        if r.u32():
            material = r.u32()
            values = r.floats(6)
            nv, ni = r.u32(), r.u32()
            vertices = [r.floats(3) for _ in range(nv)]
            indices = list(struct.unpack(f"<{ni}I", r.take(4 * ni)))
            t["paving"] = {"material": material, "period": values[:2], "atlas": values[2:],
                           "vertices": vertices, "indices": indices}
        t["foundation"] = None
        if version >= 3 and r.u32():
            t["foundation"] = {"material": r.u32(), "values": r.floats(6)}
        templates.append(t)
    if r.cursor != len(data):
        raise ValueError("trailing bytes in city library")
    return {"version": version, "look": look, "effect_material": effect_material, "materials": materials,
            "models": models, "templates": templates}


def encode(library: dict, version: int = 5) -> bytes:
    if version not in (4, 5):
        raise ValueError("Lab recomposition writes version 4 or 5")
    out = bytearray(f"C3XCITY{version}".encode())

    def u32(value):
        out.extend(struct.pack("<I", int(value)))

    def floats(values):
        out.extend(struct.pack(f"<{len(values)}f", *values))

    def string(value):
        encoded = value.encode()
        u32(len(encoded))
        out.extend(encoded)

    u32(len(library["materials"]))
    u32(len(library["models"]))
    u32(len(library["templates"]))
    if version >= 5:
        look = list(library.get("look") or [])
        floats((look + [0.0] * LOOK_FIELDS)[:LOOK_FIELDS])
        u32(library.get("effect_material", NO_EFFECT_MATERIAL))
    for m in library["materials"]:
        u32(m["address"]); u32(m["bits"]); u32(m["ground"])
        for texture in m["textures"]:
            string(texture)
    for m in library["models"]:
        out.extend(m["wire"])
    for t in library["templates"]:
        for key in ("culture", "era", "size", "capital", "environment",
                    "variant", "walled", "owns_walls", "anchor_layout"):
            u32(t[key])
        string(t["authority"])
        floats(t["clearance"])
        u32(len(t["instances"]))
        for i in t["instances"]:
            u32(i["model"]); u32(i["capital"])
            floats([i["scale"], i["rotation"], *i["offset"], *i["bounds"]])
            if version >= 5:
                u32(i.get("flags", 0))
            u32(len(i["lights"]))
            for light in i["lights"]:
                floats(light)
            if version >= 5:
                u32(len(i.get("effects", [])))
                for effect in i.get("effects", []):
                    floats(effect)
        p = t["paving"]
        u32(1 if p else 0)
        if p:
            u32(p["material"]); floats([*p["period"], *p["atlas"]])
            u32(len(p["vertices"])); u32(len(p["indices"]))
            for vertex in p["vertices"]:
                floats(vertex)
            out.extend(struct.pack(f"<{len(p['indices'])}I", *p["indices"]))
        f = t["foundation"]
        u32(1 if f else 0)
        if f:
            u32(f["material"]); floats(f["values"])
    return bytes(out)


def model_wire(parts: list[dict], low: list[float], high: list[float],
               hull: list[list[float]]) -> bytes:
    """Wire bytes for a new model: parts carry material, 18-float vertices, indices."""
    out = bytearray(struct.pack("<I", len(parts)))
    out.extend(struct.pack("<6f", *low, *high))
    out.extend(struct.pack("<I", len(hull)))
    for point in hull:
        out.extend(struct.pack("<2f", *point))
    for part in parts:
        out.extend(struct.pack("<III", part["material"], len(part["vertices"]), len(part["indices"])))
        for vertex in part["vertices"]:
            if len(vertex) != 18:
                raise ValueError("city vertex wire contract is 18 floats")
            out.extend(struct.pack("<18f", *vertex))
        out.extend(struct.pack(f"<{len(part['indices'])}I", *part["indices"]))
    return bytes(out)
