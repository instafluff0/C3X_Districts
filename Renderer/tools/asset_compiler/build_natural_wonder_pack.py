#!/usr/bin/env python3
"""Compile the approved natural-wonder kits into the generic NaturalWonderRuntime pack.

The art direction is the Lab study (Renderer/lab/studies/natural_wonders/kits.py,
approved for all four seasons on 2026-10-23). This builder runs each kit once per
C3X season and writes what the renderer instantiates:

- wonders.bin: per kit its terrain stamp (span, height, turn, region kind, the
  three R8 fields), attached effect records, foam-collar ribbon points and body
  compositions with the seasons each draws in; then the config-name aliases and
  the per-season texture slot table.
- resource_runtime.bin + textures/: the kit bodies in the composition bundle
  format (its eight texture slots). Every season shares one mesh set and one atlas
  layout: a recoloured autumn or spring crown replaces its summer cell in a copy of
  that atlas (textures/season_<n>.dds, slot value 8 + n in the season table), so a
  season flip only rebinds textures. Parts that differ by season (winter's
  snowy pines) become compositions with their own season mask.

Runtime code never sees a Civ VI identifier: kits are keyed by generic names and
C3X config names reach them only through the aliases.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Renderer.tools.asset_compiler import build_resource_compositions as rc

OUTPUT = ROOT / "Renderer/packs/NaturalWonderRuntime"
MAGIC = b"C3XNWK1\0"
VERSION = 1
SEASONS = (0, 1, 2, 3)                 # C3X: Summer, Autumn, Winter, Spring
ALL_SEASONS = 0b1111
FLAG_STAMP, FLAG_SNOW = 1, 2
# Topology bits 28-31 carry a relief slot: 1 + the rank of a kit with a stamp or foam collars,
# in pack order (docs/natural_wonder_rendering.md, Facts).
RELIEF_SLOTS = 15
MAX_ASSETS = 256                       # the composition bundle loader's limit
DIGITS = 6                             # placement identity precision across seasons
# 2048 atlases (16 cells of 512): the seasonal crowns fill few cells, so a season swaps
# only small atlases. Same block formats as the resource composition atlases.
ATLAS_KINDS = {"opaque": (2048, 8, 8, 72, (71, 72)), "masked": (2048, 8, 16, 78, (77, 78))}
GROUPS = (("opaque", False), ("opaque", True), ("masked", False), ("masked", True))
# Runtime effect kinds: 4 geyser, 5 waterfall ribbon, 7 lake, 8 fjord channel, 9 spray. The
# study numbers spray 6, which the production effect layer gives Winter's flurry.
EFFECT_KINDS = {6.0: 9.0}


def season_groups(per_season: dict[int, list[dict]]) -> list[tuple[int, list[dict]]]:
    """Split one kit's per-season instance lists into (season mask, instances) groups:
    an instance drawn in every season lands in the ALL_SEASONS group, a part only some
    seasons have (a snowy pine, the summer pine it replaces) in its own group."""
    def identity(item: dict) -> tuple:
        return (item["model"],) + tuple(round(float(item[k]), DIGITS)
                                        for k in ("u", "v", "rotation", "scale", "lift", "ground_fit"))
    counts = {season: Counter(identity(item) for item in items) for season, items in per_season.items()}
    first = {}
    for season in sorted(per_season):
        for item in per_season[season]:
            first.setdefault(identity(item), item)
    groups: dict[int, list[dict]] = {}
    for key, item in first.items():
        most = max(counts[season][key] for season in per_season)
        for copy in range(most):
            mask = sum(1 << season for season in per_season if counts[season][key] > copy)
            groups.setdefault(mask, []).append(item)
    return sorted(groups.items(), key=lambda pair: (pair[0] != ALL_SEASONS, pair[0]))


def bc3_colour_as_bc1(block: bytes) -> bytes:
    """A BC3 block's colour half as an equivalent BC1 block. BC3 colour blocks are always
    four-colour; BC1 reads c0 <= c1 as three-colour, so such blocks swap their endpoints
    and indices (0<->1, 2<->3), which is exact."""
    c0, c1, bits = struct.unpack("<HHI", block[8:16])
    if c0 > c1:
        return block[8:16]
    if c0 == c1:
        return struct.pack("<HHI", c0, c1, 0) if c0 else struct.pack("<HHI", 1, 0, 0x55555555)
    swapped = 0
    for texel in range(16):
        index = (bits >> (2 * texel)) & 3
        swapped |= (index ^ 1) << (2 * texel)
    return struct.pack("<HHI", c1, c0, swapped)


def matching_variant(summer: Path, variant: Path, cache: Path) -> Path:
    """The season variant in the summer texture's size and format, so it can take the summer
    texture's atlas cell: larger top mips are dropped and a BC3 variant of a BC1 texture keeps
    only its colour blocks."""
    width, _, mips, dxgi, _ = rc.dds(summer)
    v_width, _, v_mips, v_dxgi, v_payload = rc.dds(variant)
    if (v_width, v_dxgi) == (width, dxgi):
        return variant
    v_block = 16 if v_dxgi in (77, 78) else 8
    layout = rc.mip_offsets(v_width, v_width, v_mips, v_block)
    start = next((level for level in range(v_mips) if (v_width >> level) == width), None)
    if start is None or v_mips - start < mips:
        raise ValueError(f"{variant.name}: no {width}px mip chain to match {summer.name}")
    levels = [v_payload[offset:offset + bw * bh * v_block] for offset, bw, bh in layout[start:start + mips]]
    block = 16 if dxgi in (77, 78) else 8
    if block == 8 and v_block == 16:
        levels = [b"".join(bc3_colour_as_bc1(level[i:i + 16]) for i in range(0, len(level), 16)) for level in levels]
    elif block != v_block:
        raise ValueError(f"{variant.name}: cannot match {summer.name}'s format")
    target = cache / f"{variant.stem}_as_{width}_{dxgi}.dds"
    return rc.write_dds(target, width, mips, dxgi, block, levels)


def crown_seed(model: str) -> float:
    """The study's per-mesh crown palette seed (kits.kit: one seed per mesh key)."""
    return (sum(map(ord, model)) % 3 + .5) / 3


def build_kits(kits, assets) -> dict[str, dict[int, dict]]:
    """Every approved kit in every season, from the study's own builder."""
    kits.STAMP = True
    cache = kits.OUT / "texture_cache"
    compiler = assets.Compiler()
    built: dict[str, dict[int, dict]] = {}
    try:
        for season in SEASONS:
            kits.SEASON = season
            for key in kits.PICKS:
                built.setdefault(key, {})[season] = kits.kit(key, compiler, cache)
    finally:
        kits.SEASON = 0
        compiler.save()
    return built


def cell_list(meshes: dict[str, dict], kits, cache: Path) -> tuple[list[tuple], dict[str, int]]:
    """Atlas cells in packing order, (class, summer texture, seed or None), and each mesh's
    cell. A texture whose crowns change with the season gets one cell per crown seed it is
    drawn with, since each seed recolours differently."""
    cells, index, mesh_cell = [], {}, {}
    for key, mesh in meshes.items():
        texture = Path(mesh["texture"])
        seed = None
        if not key.startswith("decal/") and any(
                kits.seasonal_foliage(texture, season, crown_seed(key), cache) != texture for season in kits.SEASON_CROWNS):
            seed = crown_seed(key)
        cell = ("masked" if mesh["masked"] else "opaque", texture, seed)
        if cell not in index:
            index[cell] = len(cells)
            cells.append(cell)
        mesh_cell[key] = index[cell]
    return cells, mesh_cell


def pack_cells(cells: list[tuple], season: int, kits, cache: Path) -> tuple[list[bytes], dict[int, tuple]]:
    """Atlas bytes for one season and every cell's (slot, u0, v0, extent). Cells that never
    change pack apart from the season-recoloured ones, so a season only replaces the small
    atlases holding crowns. Placement depends only on the cell order and texture sizes, so
    every season shares the summer layout."""
    atlases, place = [], {}
    for kind, seasonal in GROUPS:
        atlas_kind = ATLAS_KINDS[kind]
        members = [i for i, cell in enumerate(cells) if cell[0] == kind and (cell[2] is not None) == seasonal]
        if not members:
            continue
        paths = []
        for i in members:
            _, texture, seed = cells[i]
            if seed is not None and season in kits.SEASON_CROWNS:
                variant = kits.seasonal_foliage(texture, season, seed, cache)
                texture = matching_variant(texture, variant, cache) if variant != texture else texture
            paths.append(texture)
        more, _ = rc.pack_atlases(paths, atlas_kind, len(atlases))
        per_atlas = (atlas_kind[0] // rc.CELL) ** 2
        for order, i in enumerate(members):
            width = rc.dds(paths[order])[0]
            cell = order % per_atlas
            side = atlas_kind[0] // rc.CELL
            place[i] = (len(atlases) + order // per_atlas, (cell % side) * rc.CELL / atlas_kind[0],
                        (cell // side) * rc.CELL / atlas_kind[0], width / atlas_kind[0])
        atlases += more
    return atlases, place


def field_bytes(field: np.ndarray) -> bytes:
    return np.clip(np.asarray(field, np.float64) * 255 + .5, 0, 255).astype(np.uint8).tobytes()


def write_string(blob: bytearray, value: str) -> None:
    blob.extend(rc.bundle_string(value))


def kit_record(key: str, seasons: dict[int, dict], bodies: list[tuple[str, int]], kits) -> bytes:
    """One kit's wonders.bin record."""
    summer = seasons[0]
    stamp = summer.get("stamp")
    pick = kits.PICKS[key]
    blob = bytearray()
    write_string(blob, key)
    flags = (FLAG_STAMP if stamp is not None else 0) | (FLAG_SNOW if key in kits.SNOW_CAPPED else 0)
    blob.extend(struct.pack("<II", flags, int(stamp["kind"]) if stamp is not None else 0))
    blob.extend(struct.pack("<4f", *(float(stamp[k]) for k in ("span", "units", "turn")) if stamp is not None
                            else (0.0, 0.0, 0.0), float(pick.get("winter_wet", 0.0))))
    if stamp is not None:
        size = stamp["field"].shape[0]
        if any(stamp[name].shape != (size, size) for name in ("field", "ground", "special")) or not 0 < size <= 1024:
            raise ValueError(f"{key}: stamp fields must share one square size up to 1024")
        blob.extend(struct.pack("<I", size))
        for name in ("field", "ground", "special"):
            blob.extend(field_bytes(stamp[name]))
    else:
        blob.extend(struct.pack("<I", 0))
    effects = [value for record in (stamp or {}).get("effects", [])
               for value in [EFFECT_KINDS.get(float(record[0]), float(record[0]))] + list(record[1:])]
    if len(effects) % 10:
        raise ValueError(f"{key}: effect records must have ten values")
    blob.extend(struct.pack(f"<I{len(effects)}f", len(effects) // 10, *effects))
    waves = list(summer.get("waves") or [])
    if len(waves) % 5:
        raise ValueError(f"{key}: wave points must have five values")
    blob.extend(struct.pack(f"<I{len(waves)}f", len(waves) // 5, *waves))
    blob.extend(struct.pack("<I", len(bodies)))
    for name, mask in bodies:
        write_string(blob, name)
        blob.extend(struct.pack("<I", mask))
    return bytes(blob)


def check_veil_slot(slots: int, season_slots: list[list[int]]) -> None:
    """The feature shader blends any resource body that samples the last slot (rc.VEIL_SLOT)
    through its alpha as a veil. No wonder atlas may land there in any season, unless a kit
    is ever meant to be translucent (spray, mist), and then deliberately."""
    if slots > rc.VEIL_SLOT:
        raise ValueError(f"{slots} atlases would place a body atlas at the veil slot {rc.VEIL_SLOT}")
    for season, table in enumerate(season_slots):
        if table[rc.VEIL_SLOT] != rc.VEIL_SLOT:
            raise ValueError(f"season {season} maps an atlas into the veil slot {rc.VEIL_SLOT}")


def read_wonders(data: bytes) -> dict:
    """Parse wonders.bin (the layout the renderer's loader reads)."""
    at = 0

    def take(fmt: str):
        nonlocal at
        values = struct.unpack_from(fmt, data, at)
        at += struct.calcsize(fmt)
        return values

    def string() -> str:
        (length,) = take("<I")
        return take(f"<{length}s")[0].decode("utf-8")
    if data[:8] != MAGIC:
        raise ValueError("not a natural-wonder pack")
    at = 8
    version, kit_count, alias_count, slots = take("<IIII")
    if version != VERSION or slots != rc.TEXTURE_SLOTS:
        raise ValueError("unsupported natural-wonder pack version")
    season_slots = [list(take(f"<{slots}I")) for _ in SEASONS]
    kits_out = []
    for _ in range(kit_count):
        kit = {"key": string()}
        kit["flags"], kit["region_kind"] = take("<II")
        kit["span"], kit["units"], kit["turn"], kit["snow_floor"] = take("<4f")
        (size,) = take("<I")
        kit["field_size"] = size
        kit["fields"] = [take(f"<{size * size}s")[0] for _ in range(3)] if size else []
        (count,) = take("<I")
        kit["effects"] = [list(take("<10f")) for _ in range(count)]
        (count,) = take("<I")
        kit["waves"] = [list(take("<5f")) for _ in range(count)]
        (count,) = take("<I")
        kit["bodies"] = [(string(), take("<I")[0]) for _ in range(count)]
        kits_out.append(kit)
    aliases = [(string(), take("<I")[0]) for _ in range(alias_count)]
    if at != len(data):
        raise ValueError("trailing bytes in natural-wonder pack")
    return {"season_slots": season_slots, "kits": kits_out, "aliases": aliases}


def relief_slots(kits: list[dict]) -> dict[str, int]:
    """Each relief kit's topology slot from a parsed wonders.bin (read_wonders)."""
    relief = [kit["key"] for kit in kits if kit["flags"] & FLAG_STAMP or kit["waves"]]
    if len(relief) > RELIEF_SLOTS:
        raise ValueError(f"{len(relief)} relief kits exceed the {RELIEF_SLOTS} topology slots")
    return {key: slot for slot, key in enumerate(relief, 1)}


def check_season_invariants(key: str, seasons: dict[int, dict]) -> None:
    """Seasons change only crowns and pines: the stamp, effects and waves are summer's."""
    summer = seasons[0]
    for season, built in seasons.items():
        a, b = summer.get("stamp"), built.get("stamp")
        if (a is None) != (b is None) or (a is not None and (
                not np.array_equal(a["field"], b["field"]) or a["effects"] != b["effects"])):
            raise ValueError(f"{key}: season {season} changes the terrain stamp")
        if list(summer.get("waves") or []) != list(built.get("waves") or []):
            raise ValueError(f"{key}: season {season} changes the foam collars")


def build(output: Path = OUTPUT) -> dict:
    from Renderer.lab.studies.natural_wonders import kits, assets
    cache = kits.OUT / "texture_cache"
    built = build_kits(kits, assets)
    meshes: dict[str, dict] = {}
    compositions: list[tuple[str, list[dict]]] = []
    bodies: dict[str, list[tuple[str, int]]] = {}
    for key, seasons in built.items():
        check_season_invariants(key, seasons)
        for season, kit in seasons.items():
            for name, mesh in kit["meshes"].items():
                # Season variants share the summer mesh; the atlas swap recolours it.
                meshes.setdefault(name, seasons[0]["meshes"].get(name, mesh))
        for mask, instances in season_groups({s: kit["instances"] for s, kit in seasons.items()}):
            name = f"NW_{key}" if mask == ALL_SEASONS else f"NW_{key}~{mask}"
            compositions.append((name, instances))
            bodies.setdefault(key, []).append((name, mask))
    used = {item["model"] for _, instances in compositions for item in instances}
    meshes = {name: mesh for name, mesh in meshes.items() if name in used}
    if len(meshes) > MAX_ASSETS:
        raise ValueError(f"{len(meshes)} meshes exceed the pack's {MAX_ASSETS}")

    cells, mesh_cell = cell_list(meshes, kits, cache)
    summer_atlases, place = pack_cells(cells, 0, kits, cache)
    slots = len(summer_atlases)
    if slots > rc.TEXTURE_SLOTS:
        raise ValueError(f"{slots} atlases exceed the {rc.TEXTURE_SLOTS} texture slots")
    # Bundle texture list: the summer atlases padded to the eight slots, then each season
    # atlas that differs from summer's. season_slots[season][slot] names the texture bound
    # to that slot in that season (Winter and Summer bind the summer atlases).
    extra: list[bytes] = []
    season_slots = [list(range(rc.TEXTURE_SLOTS)) for _ in SEASONS]
    for season in SEASONS:
        if season not in kits.SEASON_CROWNS:
            continue
        atlases, season_place = pack_cells(cells, season, kits, cache)
        if season_place != place:
            raise ValueError(f"season {season} atlas layout differs from summer")
        for slot, data in enumerate(atlases):
            if data != summer_atlases[slot]:
                season_slots[season][slot] = rc.TEXTURE_SLOTS + len(extra)
                extra.append(data)
    # The bundle lists exactly the eight slots (the runtime bundle loader's contract);
    # season atlases are side files textures/season_<n>.dds that slot values 8+n name.
    check_veil_slot(slots, season_slots)
    names = [f"textures/atlas_{i}.dds" for i in range(slots)]
    names += ["textures/unused.dds"] * (rc.TEXTURE_SLOTS - slots)
    if len(extra) > rc.TEXTURE_SLOTS:
        raise ValueError(f"{len(extra)} season atlases exceed the {rc.TEXTURE_SLOTS} the runtime loads")

    payloads, asset_index = [], {}
    for name, mesh in meshes.items():
        atlas, u0, v0, extent = place[mesh_cell[name]]
        edge = .5 / (extent * ATLAS_KINDS["opaque"][0])
        vertices = [(p, n, (u0 + min(1 - edge, max(edge, uv[0])) * extent, v0 + min(1 - edge, max(edge, uv[1])) * extent))
                    for p, n, uv in mesh["vertices"]]
        asset_index[name] = len(payloads)
        payloads.append(rc.mesh_payload(name, atlas, vertices, mesh["indices"]))
    blob = bytearray(rc.MAGIC)
    blob.extend(struct.pack("<IIII", 2, len(names), len(payloads), 0))
    for texture in names:
        blob.extend(rc.bundle_string(texture))
    for payload in payloads:
        blob.extend(payload)
    blob.extend(struct.pack("<I", len(compositions)))
    for name, instances in compositions:
        blob.extend(rc.bundle_string(name))
        blob.extend(struct.pack("<III", 1, kits.ALL_TERRAINS, len(instances)))
        for item in instances:
            blob.extend(struct.pack("<I6f", asset_index[item["model"]], item["u"], item["v"], item["rotation"],
                                    item["scale"], item["lift"], item["ground_fit"]))
    blob.extend(struct.pack("<I", len(compositions)))
    for position, (name, _) in enumerate(compositions):
        blob.extend(rc.bundle_string(name))
        blob.extend(struct.pack("<I", position))

    keys = list(built)
    aliases = [(item["name"], keys.index(key)) for key, item in kits.definitions().items() if key in keys]
    wonders = bytearray(MAGIC)
    wonders.extend(struct.pack("<IIII", VERSION, len(keys), len(aliases), rc.TEXTURE_SLOTS))
    for season in SEASONS:
        wonders.extend(struct.pack(f"<{rc.TEXTURE_SLOTS}I", *season_slots[season]))
    for key in keys:
        wonders.extend(kit_record(key, built[key], bodies.get(key, []), kits))
    for name, index in aliases:
        write_string(wonders, name)
        wonders.extend(struct.pack("<I", index))

    relief = relief_slots(read_wonders(bytes(wonders))["kits"])
    if output.exists():
        shutil.rmtree(output)
    (output / "textures").mkdir(parents=True)
    for slot, data in enumerate(summer_atlases):
        (output / f"textures/atlas_{slot}.dds").write_bytes(data)
    if slots < rc.TEXTURE_SLOTS:
        (output / "textures/unused.dds").write_bytes(rc.dds_header(4, 4, 1, rc.MODEL_ATLAS[3], 8) + bytes(8))
    for i, data in enumerate(extra):
        (output / f"textures/season_{i}.dds").write_bytes(data)
    (output / "resource_runtime.bin").write_bytes(blob)
    (output / "wonders.bin").write_bytes(wonders)
    report = {
        "schema": "c3x.natural_wonder_pack_report.v1",
        "kits": {key: {"stamp": built[key][0].get("stamp") is not None,
                       "bodies": [{"composition": name, "seasons": mask} for name, mask in bodies.get(key, [])],
                       "effects": len((built[key][0].get("stamp") or {}).get("effects", [])),
                       "wave_points": len(built[key][0].get("waves") or []) // 5}
                 for key in keys},
        "aliases": {name: keys[index] for name, index in aliases},
        "meshes": len(payloads), "atlases": slots, "season_atlases": len(extra),
        "season_slots": {str(season): season_slots[season] for season in SEASONS},
        "relief_slots": relief,
    }
    (output / "pack_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps({k: report[k] for k in ("meshes", "atlases", "season_atlases", "season_slots")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
