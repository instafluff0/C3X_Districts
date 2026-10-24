#!/usr/bin/env python3
"""Compile the approved constructed-wonder kits into the generic WonderRuntime pack.

The art direction is the Lab study (Renderer/lab/studies/wonders/, approved for
the game on 2026-10-24); docs/constructed_wonder_rendering.md is the contract.
Wonders are city bodies (Civ VI lights them through their second and third UV
sets), so the pack is a city library (city.bin) the renderer appends to the
production one, plus its own textures:

- index.bin and groups/<n>.bin: one city library per wonder variant and facing
  (or era district), loaded on demand (docs, Residency). Each holds one
  composition per season mask, named by its
  authority: "wonder|<alias>|<variant>|<season mask>|<flags>". Variants are
  "complete" and "build:<progress>"; a "+alt" suffix is the turned composition
  for C3X's alternate direction. Flags: "r" = the kit stands across a river.
- the unused Wonder District per era: "district|<era>|complete|<season mask>|".
- season masks: foliage parts only are emitted per season (Summer and Winter,
  Autumn, Spring); every other part has mask 0, all seasons. Slender foliage
  (conifers) keeps its green all year.
- seating: a kit whose metadata asks for one ("foundation") carries the city
  terrace; others stand on the ground at their centre, as in the approved Lab.

Runtime code never sees a Civ VI identifier or branches on a wonder: aliases are
the C3X config names, lowercased.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.city_readability import runtime_pack as rp
from Renderer.lab.studies.wonders import city_pack, kits, plan
from Renderer.tools.asset_compiler import build_resource_compositions as rc

OUTPUT = ROOT / "Renderer/packs/WonderRuntime"
PRODUCTION = ROOT / "Renderer/packs/CityCompositionRuntime"
LEDGER = ROOT / "Renderer/inventory/vanilla_c3x_to_civ6_constructed_wonders.json"
CONFIG = ROOT / "default.districts_wonders_config.txt"
TEXTURE_PREFIX = "Renderer/packs/WonderRuntime/textures/"
# Resident texture budget (BC-compressed, full mip chains), set by the integration lead.
BUDGET = 64 * 1024 * 1024
# Largest side per material role, in schedules tried in order until the pack fits.
CAP_SCHEDULES = ({"base_color": 1024, "emissive": 1024, "": 512},
                 {"base_color": 512, "emissive": 512, "": 512},
                 {"base_color": 512, "emissive": 512, "": 256},
                 {"base_color": 256, "emissive": 256, "": 256})
PROP_CAP = 512
BUILDS = (.25, .5, .75)
# Vertices per group: a tile's prepared objects stay under half the preparation byte
# limit (render_core/content_preparation.h), so heavy build stages shed their smallest
# construction pieces (loose bricks, rubble) first.
GROUP_VERTICES = 60000
ALL, SUMMER_WINTER, AUTUMN, SPRING = 0, 0b0101, 0b0010, 0b1000
# Lab lamp kinds (studies/wonders kits.LAMP_EDIT) become the night-light kind with a
# mode: 1 + colour band (warm, white, red, green), + 4 when it blinks. The mode rides
# in the record's seed as seed + 64 * mode; kind 2's mode 0 is the default night light.
NIGHT_LIGHT, LAB_LAMP, LAB_BLINK = 2.0, 7.0, 8.0
BLOCK = {71: 8, 72: 8, 80: 8, 81: 8, 74: 16, 75: 16, 77: 16, 78: 16, 83: 16, 84: 16, 98: 16, 99: 16}


def capped(data: bytes, side: int) -> bytes:
    """A DDS (DX10 header, full mip chain) brought down to `side` texels on its
    longer edge by dropping its top mip levels; block data is copied as is."""
    height, width = struct.unpack_from("<II", data, 12)
    mips = max(1, struct.unpack_from("<I", data, 28)[0])
    dxgi = struct.unpack_from("<I", data, 128)[0]
    block = BLOCK.get(dxgi)
    skip = 0
    while max(width >> skip, height >> skip) > side and skip + 1 < mips:
        skip += 1
    if block is None or skip == 0:
        return data
    layout = rc.mip_offsets(width, height, mips, block)
    payload = data[148:]
    start = layout[skip][0]
    w, h = max(1, width >> skip), max(1, height >> skip)
    return rc.dds_header(w, h, mips - skip, dxgi, layout[skip][1] * layout[skip][2] * block) + payload[start:]


def resident_bytes(data: bytes) -> int:
    return len(data) - 148


class PackLibrary(city_pack.Library):
    """The wonder-only city library: its own materials, models and compositions,
    plus copies of production's effect material and era pavings (texture closure in
    the pack). Textures are capped by role and counted against the budget."""

    production_library: dict | None = None

    def __init__(self, caps: dict, files: dict[str, bytes]) -> None:
        if PackLibrary.production_library is None:
            PackLibrary.production_library = rp.decode(PRODUCTION / "city.bin")
        self.production = PackLibrary.production_library
        self.caps = caps
        self.files = files
        self.prop = False
        self.material_ids = {}
        self.library = {"version": self.production["version"], "look": list(self.production["look"]),
                        "effect_material": 0, "materials": [], "models": [], "templates": []}
        self.library["materials"].append(self.copied(self.production["materials"][self.production["effect_material"]]))

    def copied(self, material: dict) -> dict:
        textures = []
        for texture in material["textures"]:
            textures.append(self.add((PRODUCTION / "textures" / Path(texture).name).read_bytes(), "production")
                            if texture else "")
        return {**material, "textures": textures}

    def add(self, data: bytes, role: str = "") -> str:
        if role != "production":
            side = PROP_CAP if self.prop else self.caps.get(role, self.caps[""])
            data = capped(data, min(side, self.caps.get(role, self.caps[""])))
        name = hashlib.sha256(data).hexdigest() + ".dds"
        self.files[name] = data
        return TEXTURE_PREFIX + name

    def paving(self, era: int, half: float = .36) -> dict:
        """The era's production paving under a square plot, feathered at its edge."""
        source = next(t["paving"] for t in self.production["templates"] if t["era"] == era and t["paving"])
        material = len(self.library["materials"])
        self.library["materials"].append(self.copied(self.production["materials"][source["material"]]))
        n, feather = 12, .05
        vertices = []
        for j in range(n + 1):
            for i in range(n + 1):
                x, y = -half - feather + (2 * half + 2 * feather) * i / n, -half - feather + (2 * half + 2 * feather) * j / n
                edge = max(abs(x), abs(y))
                vertices.append([x, y, max(0.0, min(1.0, (half + feather - edge) / feather))])
        indices = [k for j in range(n) for i in range(n) for k in
                   (j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i,
                    j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i)]
        return {"material": material, "period": list(source["period"]), "atlas": list(source["atlas"]),
                "vertices": vertices, "indices": indices}


def config_wonders() -> dict:
    """C3X's default wonder configs: name -> fields (enable_img_alt_dir, ...)."""
    out, current = {}, None
    for line in CONFIG.read_text().splitlines():
        if line.startswith("#Wonder"):
            current = {}
        match = re.match(r"\s*([a-z_]+)\s*=\s*(.*?)\s*$", line)
        if match and current is not None:
            current[match.group(1)] = match.group(2)
            if match.group(1) == "name":
                out[match.group(2)] = current
    return out


def lamp(record: list) -> list:
    """A Lab lamp record as a night light with its mode in the seed."""
    kind, seed = record[3], record[6]
    if kind not in (LAB_LAMP, LAB_BLINK):
        return record
    mode = 1 + min(3, int(seed) // 16) + (4 if kind == LAB_BLINK else 0)
    return record[:3] + [NIGHT_LIGHT] + record[4:6] + [(seed % 16) * 4 + 64 * mode] + record[7:]


def thinned(pieces: list, budget: int, weights: list[int]) -> list:
    """Pieces within a vertex budget (each piece's vertices counted `weights` times:
    foliage draws once per season form): their connected parts largest first (by
    bounding area), regrouped per material; the smallest parts are left out."""
    total = sum(len(vertices) * weight for (_, _, vertices, _), weight in zip(pieces, weights))
    if total <= budget:
        return pieces
    parts = []
    for number, (path, material, vertices, indices) in enumerate(pieces):
        for part_vertices, part_indices in kits._connected(vertices, indices):
            extent = [max(p[a] for p, _, _ in part_vertices) - min(p[a] for p, _, _ in part_vertices) for a in range(3)]
            parts.append((extent[0] * extent[1] + extent[1] * extent[2] + extent[0] * extent[2], number, part_vertices, part_indices))
    parts.sort(key=lambda part: -part[0])
    kept_parts: dict[int, tuple[list, list]] = {}
    used = 0
    for _size, number, part_vertices, part_indices in parts:
        if used + len(part_vertices) * weights[number] > budget:
            continue
        vertices, indices = kept_parts.setdefault(number, ([], []))
        indices += [len(vertices) + i for i in part_indices]
        vertices += part_vertices
        used += len(part_vertices) * weights[number]
    return [(pieces[number][0], pieces[number][1], vertices, indices) for number, (vertices, indices) in sorted(kept_parts.items())]


def foliage(material: dict, sources: Path) -> bool:
    """Broadleaf foliage: a material Autumn recolours (natural_wonders.seasonal_foliage)
    that is not a conifer."""
    from Renderer.lab.studies.natural_wonders import kits as nw
    base = material.get("channels", {}).get("base_color")
    if material.get("conifer") or not isinstance(base, dict):
        return False
    texture = sources / base["texture"]
    return Path(nw.seasonal_foliage(texture, 1, .5, kits.OUT / "texture_cache")) != texture


def bodies(library: PackLibrary, cell: dict) -> tuple[dict[int, list], dict | None]:
    """A cell's parts as city bodies per season mask, and its foundation."""
    out: dict[int, list] = {}
    foundation = None
    facing = cell["facing"]
    # The cell's vertex budget, shared evenly by its parts.
    budget = GROUP_VERTICES // max(1, sum(not part["source"].startswith("plot:") for part in cell["parts"]))
    for part in cell["parts"]:
        if part["source"].startswith("plot:"):
            continue
        library.prop = bool(kits.prop(part["source"]))
        finished, cx, cy, scale, _low, _high = kits.layout(part)
        staging = {k: v for k, v in part.items() if k in ("built", "stage")}
        reference = kits.kept(finished, full=True)
        if part.get("built", 1.0) < 1.0 and not kits.staged(part["source"]) and reference:
            top = max(p[2] for _, _, vertices, _ in reference for p, _, _ in vertices)
            pieces = kits.kept({**finished, "keep": {**part.get("keep", {}), "clip_z": part["built"] * top}}, full=True)
        else:
            pieces = kits.kept({**finished, **staging}, full=True) if staging else reference
        if not pieces:
            continue
        sources = kits.use(part["source"])
        pieces = thinned(pieces, budget, [3 if foliage(piece[1], sources) else 1 for piece in pieces])
        leaves = [piece for piece in pieces if foliage(piece[1], sources)]
        rest = [piece for piece in pieces if piece not in leaves]
        effects = []
        if part.get("built", 1.0) >= 1.0 and part.get("stage", 1.0) >= 1.0 and not kits.prop(part["source"]):
            try:
                effects = [lamp(record) for record in kits.effect_records(part, finished, cx, cy, scale, facing)]
            except ValueError:
                effects = []
        if rest:
            kits.SEASON["index"] = 0
            model, bounds = kits.city_body(library, part, rest, cx, cy, scale, facing)
            out.setdefault(ALL, []).append((model, bounds, effects[:16], []))
            effects = []
            # A foundation (the city terrace, with its skirt and top plates) only where the
            # kit's metadata asks for one; otherwise bodies stand on the ground at their centre.
            if foundation is None and part.get("foundation") and not kits.prop(part["source"]):
                foundation = seat(library, rest, sources)
        for mask, season in ((SUMMER_WINTER, 0), (AUTUMN, 1), (SPRING, 3)):
            if leaves:
                kits.SEASON["index"] = season
                model, bounds = kits.city_body(library, part, leaves, cx, cy, scale, facing)
                out.setdefault(mask, []).append((model, bounds, effects[:16] if mask == SUMMER_WINTER else [], []))
        kits.SEASON["index"] = 0
    return out, foundation


def seat(library: PackLibrary, pieces: list, sources: Path) -> dict:
    """The kit's foundation: its lowest piece's material and a texel patch around its
    lowest vertex, in modules of a tenth of a tile by 12 height units (city_fidelity
    terraces the body on it where relief under the footprint varies)."""
    path, material, vertices, _ = min(pieces, key=lambda piece: min(p[2] for p, _, _ in piece[2]))
    (_x, _y, _z), _n, (uv0, uv1, uv2) = min(vertices, key=lambda vertex: vertex[0][2])
    has_uv1 = any(any(uvs[1]) for _, _, uvs in vertices)
    has_uv2 = any(any(uvs[2]) for _, _, uvs in vertices)
    material_id, _light = library.material(material, sources, has_uv1, has_uv2)
    u, v = uv0[0] % 1.0, uv0[1] % 1.0
    u, v = min(u, .94), min(v, .94)
    return {"material": material_id, "values": [u, v, u + .05, v + .05, .1, 12.0]}


def template(authority: str, instances: list, paving: dict | None, foundation: dict | None) -> dict:
    return {"culture": 0, "era": 0, "size": 0, "capital": 0, "environment": 0, "variant": 0, "walled": 0,
            "owns_walls": 0, "anchor_layout": 1, "authority": authority, "clearance": [0.0, 0.0, 0.0, 0.0],
            "instances": [{"model": model, "capital": 0, "scale": 1.0, "rotation": 0.0, "offset": [0.0, 0.0],
                           "bounds": bounds, "flags": 0, "lights": lights, "effects": effects}
                          for model, bounds, effects, lights in instances],
            "paving": paving, "foundation": foundation}


def cells() -> list[dict]:
    """Every cell of the pack: each wonder's variants and facings, then the Wonder
    District per era."""
    ledger = json.loads(LEDGER.read_text())["mappings"]
    configs = config_wonders()
    out = []
    for entry in ledger:
        key, alias = entry["c3x_key"], entry["c3x_name"].lower()
        pick = plan.PICKS[key][0]
        facing = plan.FACING.get(key, plan.FACE_SW)
        facings = [("", facing)]
        if configs.get(entry["c3x_name"], {}).get("enable_img_alt_dir") == "1":
            facings.append(("+alt", plan.FACE_SE if facing == plan.FACE_SW else plan.FACE_SW))
        flags = "r" if any(item.get("river_span") for item in pick["parts"]) else ""
        for suffix, turn in facings:
            out.append({"kind": "wonder", "alias": alias, "variant": "complete" + suffix, "facing": turn,
                        "parts": pick["parts"], "flags": flags})
            for built in BUILDS:
                out.append({"kind": "wonder", "alias": alias, "variant": f"build:{built}{suffix}", "facing": turn,
                            "parts": plan.staging(pick["parts"], built), "flags": flags})
    for era, name in enumerate(plan.WD_ERAS):
        out.append({"kind": "district", "alias": str(era), "variant": "complete", "facing": plan.FACE_SW,
                    "parts": plan.wonder_district(name), "flags": "", "era": era})
    return out


MAGIC = b"C3XWND1\0"
# Terrain stamps for kits that stand across a river (river_span), in the natural-wonder
# kit format (render_core/natural_wonder_pack.h) so one relief-slot registry serves both:
# a gorge whose channel follows the stamp's long axis, walls rising beyond it.
STAMP_FLAG, SNOW_FLAG, FOLLOW_RIVER_FLAG = 1, 2, 4
# Field half-widths across the axis: the channel holds the dam (fitted about .95 tile
# long: half a tile each side, .37 of the 2.6-tile span's half-width), crested walls beyond.
GORGE = {"span": 2.6, "units": 96.0, "region": 3, "size": 256,
         "channel": .34, "crest": .5, "outer": .76,
         "reach": .76}   # 1.0 tile over the 1.3-tile field half, less a texel


def gorge_fields(size: int, channel: float, crest: float, outer: float) -> tuple[bytes, bytes, bytes]:
    """Height, ground and special fields of a gorge stamp (R8, size squared). Across
    the axis (|v|, field half-widths) the channel floor sits at the river's level and
    ridge walls rise from `channel` to a rounded crest at `crest`, falling back to the
    host by `outer`. Along the axis the ridges undulate and fade at both ends, so they
    read as mountain flanks rather than slabs."""
    import numpy as np
    axis = (np.arange(size) + .5) / size * 2 - 1
    u, v = np.meshgrid(axis, axis)
    def smooth(a, b, x):
        t = np.clip((x - a) / (b - a), 0, 1)
        return t * t * (3 - 2 * t)
    across = np.abs(v)
    side = np.sign(v)
    rise = smooth(channel, crest, across) * (1 - smooth(crest, outer, across))
    # Each wall is a short run of peaks whose crest line undulates on its own
    # (deterministic, no random state).
    wave = .62 + .26 * np.cos(u * 7.0 + side * 1.3) ** 2 + .12 * np.sin(u * 13.1 + side * .4)
    # The renderer caps a follow-river kit's reach at 1.0 tile (span/2 = 1.3 tiles per
    # field half), so everything slopes away to zero within REACH of the field half,
    # as a mountain's flanks do, rather than ending in a cut.
    radius = np.sqrt(u * u + v * v)
    ridge = rise * wave * (1 - smooth(.28, GORGE["reach"], radius)) ** 1.4 * 2.1
    texture = .04 * np.sin(u * 37 + v * 11) * np.sin(v * 29 - u * 7)
    height = np.clip(ridge + texture * smooth(.1, .4, ridge), 0, 1)
    special = smooth(.05, .3, height)
    ground = 1 - smooth(.0, .25, height)
    to_bytes = lambda field: (np.clip(field, 0, 1) * 255 + .5).astype(np.uint8).tobytes()
    return to_bytes(height), to_bytes(ground), to_bytes(special)


def stamps_bytes(river_aliases: list[str]) -> bytes:
    """stamps.bin: one gorge kit per river-span wonder, aliased "wonder|<alias>"."""
    out = bytearray(b"C3XNWK1\0")
    out += struct.pack("<IIII", 1, len(river_aliases), len(river_aliases), 8)
    out += bytes(4 * 8 * 4)                       # no bodies, so no season texture slots
    fields = gorge_fields(GORGE["size"], GORGE["channel"], GORGE["crest"], GORGE["outer"])
    for alias in river_aliases:
        key = ("gorge-" + alias).encode()
        out += struct.pack("<I", len(key)) + key
        out += struct.pack("<II4fI", STAMP_FLAG | FOLLOW_RIVER_FLAG, GORGE["region"],
                           GORGE["span"], GORGE["units"], 0.0, 0.0, GORGE["size"])
        for field in fields:
            out += field
        out += struct.pack("<III", 0, 0, 0)       # effects, waves, bodies
    for number, alias in enumerate(river_aliases):
        name = f"wonder|{alias}".encode()
        out += struct.pack("<I", len(name)) + name + struct.pack("<I", number)
    return bytes(out)


def build_once(caps: dict, only: set | None = None) -> tuple[list, dict, dict]:
    """Every group (one cell: a wonder's variant and facing, or an era's district) as
    its own city library over one shared texture set."""
    files: dict[str, bytes] = {}
    groups, report = [], {"groups": {}, "caps": caps}
    for cell in cells():
        if only and cell["alias"] not in only:
            continue
        library = PackLibrary(caps, files)
        per_mask, foundation = bodies(library, cell)
        paving = library.paving(cell["era"]) if cell["kind"] == "district" else None
        key = f"{cell['kind']}|{cell['alias']}|{cell['variant']}"
        for mask, instances in sorted(per_mask.items()):
            library.library["templates"].append(template(f"{key}|{mask}|{cell['flags']}", instances,
                                                         paving if mask == ALL else None,
                                                         foundation if mask == ALL and cell["kind"] == "wonder" else None))
        if not library.library["templates"]:
            continue
        data = rp.encode(library.library, library.library["version"])
        textures = sorted({Path(t).name for m in library.library["materials"] for t in m["textures"] if t})
        groups.append((key, data, textures))
        report["groups"][key] = {"bytes": len(data), "vertices": sum(m["vertex_count"] for m in library.library["models"]),
                                 "textures": len(textures),
                                 "texture_bytes": sum(resident_bytes(files[name]) for name in textures),
                                 "effects": sum(len(i["effects"]) for t in library.library["templates"] for i in t["instances"])}
    report["texture_bytes"] = sum(resident_bytes(data) for data in files.values())
    report["textures"] = len(files)
    return groups, files, report


def index_bytes(groups: list) -> bytes:
    """index.bin: per group its key (kind|alias|variant), file and byte size, and the
    textures it uses (file names under textures/)."""
    out = bytearray(MAGIC)
    out += struct.pack("<I", len(groups))
    for number, (key, data, textures) in enumerate(groups):
        for text in (key, f"groups/{number}.bin"):
            encoded = text.encode()
            out += struct.pack("<I", len(encoded)) + encoded
        out += struct.pack("<II", len(data), len(textures))
        for name in textures:
            encoded = name.encode()
            out += struct.pack("<I", len(encoded)) + encoded
    return bytes(out)


def build(output: Path = OUTPUT, only: set | None = None) -> dict:
    for caps in CAP_SCHEDULES:
        groups, files, report = build_once(caps, only)
        print(f"caps {caps}: {report['texture_bytes'] / 2**20:.1f} MiB in {report['textures']} textures", flush=True)
        if report["texture_bytes"] <= BUDGET:
            break
    else:
        raise ValueError(f"wonder textures exceed {BUDGET / 2**20:.0f} MiB at every cap schedule")
    if output.exists():
        shutil.rmtree(output)
    (output / "textures").mkdir(parents=True)
    (output / "groups").mkdir()
    for name, data in files.items():
        (output / "textures" / name).write_bytes(data)
    for number, (_key, data, _textures) in enumerate(groups):
        (output / "groups" / f"{number}.bin").write_bytes(data)
    (output / "index.bin").write_bytes(index_bytes(groups))
    rivers = sorted({cell["alias"] for cell in cells() if "r" in cell["flags"] and (not only or cell["alias"] in only)})
    (output / "stamps.bin").write_bytes(stamps_bytes(rivers))
    report["stamps"] = rivers
    report["group_bytes"] = sum(len(data) for _k, data, _t in groups)
    report["largest_group_bytes"] = max(len(data) for _k, data, _t in groups)
    report["budget_bytes"] = BUDGET
    (output / "pack_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--only", action="append", help="alias (lowercase C3X name) to include; repeatable")
    parser.add_argument("--stamps", action="store_true", help="write stamps.bin only (the groups are unchanged)")
    args = parser.parse_args()
    if args.stamps:
        rivers = sorted({cell["alias"] for cell in cells() if "r" in cell["flags"]})
        (args.output / "stamps.bin").write_bytes(stamps_bytes(rivers))
        print("stamps:", rivers)
        return 0
    report = build(args.output, set(args.only) if args.only else None)
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
