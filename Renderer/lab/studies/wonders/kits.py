#!/usr/bin/env python3
"""Civ VI and Workshop wonder models for C3X wonders on the Renderer64 map (Lab only).

Civ VI keeps its constructed wonders only in WonderMovie.blp (base game, every
DLC and Workshop wonder mods alike): one build-movie model per wonder whose
bind pose holds the finished wonder plus its scaffolds, cranes and stone piles.
Stand-ins that are not Civ VI wonders (Government Plaza buildings, district
heroes, the Dam and Spaceport) come from the Landmarks ArtDefs.

This study reuses the district study's importer and Lab render path
(Renderer/lab/studies/districts/kits.py). Each source compiles into its own
Lab sub-pack, so imports run in parallel; `bake` merges the kept pieces of
each source into one composition per C3X wonder cell, fitted to the tile.

    python3 Renderer/lab/studies/wonders/kits.py compile [--source KEY ...] [--jobs 4]
    python3 Renderer/lab/studies/wonders/kits.py probe [--source KEY ...]
    python3 Renderer/lab/studies/wonders/kits.py render [--catalog NAME ...] [--zoom 256]
    python3 Renderer/lab/studies/wonders/kits.py crops

`probe` writes per-material point previews and statistics of each imported
model to Renderer/lab/out/wonders/probe/, for choosing which materials and
radius keep the finished wonder (plan.py `keep`). Nothing here changes
production packs, ledgers or the renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import zlib
import json
import math
import shutil
import struct
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.districts import kits as dk
from Renderer.tools.asset_compiler import build_resource_compositions as rc
from Renderer.lab.studies.wonders import city_pack

SOURCES = ROOT / "Renderer/packs/WonderKitSourcesLab"
CATALOG = ROOT / "Renderer/packs/WonderKitCatalogLab"
OUT = ROOT / "Renderer/lab/out/wonders"
WORKSHOP = Path.home() / "Library/Application Support/Steam/steamapps/workshop/content/289070"
# The district study's importer, baker and renderer, pointed at this study's packs.
dk.CATALOG, dk.OUT = CATALOG, OUT
# render() swaps dk.Compiler out; sources still compile with the district study's own class.
DistrictCompiler = dk.__dict__.setdefault("DistrictLabCompiler", dk.Compiler)

# Civ VI wonders (WonderMovie.blp): key -> (content root, entry, label, origin).
CIV6 = {
    "pyramids": ("Base", "BUILDING_PYRAMIDS", "Pyramids", "Civ VI base game"),
    "hanging_gardens": ("Base", "BUILDING_HANGING_GARDENS", "Hanging Gardens", "Civ VI base game"),
    "oracle": ("Base", "BUILDING_ORACLE", "Oracle", "Civ VI base game"),
    "great_library": ("Base", "BUILDING_GREAT_LIBRARY", "Great Library", "Civ VI base game"),
    "great_lighthouse": ("Base", "BUILDING_GREAT_LIGHTHOUSE", "Great Lighthouse", "Civ VI base game"),
    "colossus": ("Base", "BUILDING_COLOSSUS", "Colossus", "Civ VI base game"),
    "forbidden_city": ("Base", "BUILDING_FORBIDDEN_CITY", "Forbidden City", "Civ VI base game"),
    "hagia_sophia": ("Base", "BUILDING_HAGIA_SOPHIA", "Hagia Sophia", "Civ VI base game"),
    "oxford_university": ("Base", "BUILDING_OXFORD_UNIVERSITY", "Oxford University", "Civ VI base game"),
    "bolshoi_theatre": ("Base", "BUILDING_BOLOSHI_THEATRE", "Bolshoi Theatre", "Civ VI base game"),
    "ruhr_valley": ("Base", "BUILDING_RUHR_VALLEY", "Ruhr Valley", "Civ VI base game"),
    "broadway": ("Base", "BUILDING_BROADWAY", "Broadway", "Civ VI base game"),
    "large_rocket": ("Base", "BUILDING_LARGE_ROCKET", "Space race rocket", "Civ VI base game"),
    "statue_of_zeus": ("DLC/Byzantium_Gaul", "BUILDING_STATUE_OF_ZEUS", "Statue of Zeus", "Gathering Storm"),
    "biosphere": ("DLC/Byzantium_Gaul", "BUILDING_BIOSPHERE", "Biosphère", "New Frontier Pass"),
    "temple_of_artemis": ("DLC/Expansion1", "BUILDING_TEMPLE_OF_ARTEMIS", "Temple of Artemis", "Rise and Fall"),
    "st_basils": ("DLC/Expansion1", "BUILDING_ST_BASILS_CATHEDRAL", "St. Basil's Cathedral", "Rise and Fall"),
    "amundsen_scott": ("DLC/Expansion1", "BUILDING_AMUNDSEN_SCOTT_RESEARCH_STATION", "Amundsen-Scott Research Station",
                       "Rise and Fall"),
    "mausoleum": ("DLC/Macedonia_Persia", "BUILDING_HALICARNASSUS", "Mausoleum at Halicarnassus",
                  "Macedonia & Persia pack"),
}
# Civ VI district and building stand-ins (Landmarks ArtDefs): key -> (record args, label, origin).
LANDMARKS = {
    "cathedral": (("hero", "DISTRICT_HOLY_SITE", "BUILDING_CATHEDRAL"), "Cathedral (Holy Site)", "Civ VI base game"),
    "university": (("hero", "DISTRICT_CAMPUS", "BUILDING_UNIVERSITY"), "University (Campus)", "Civ VI base game"),
    "research_lab": (("hero", "DISTRICT_CAMPUS", "BUILDING_RESEARCH_LAB"), "Research Lab (Campus)", "Civ VI base game"),
    "military_academy": (("hero", "DISTRICT_ENCAMPMENT", "BUILDING_MILITARY_ACADEMY"), "Military Academy (Encampment)",
                         "Civ VI base game"),
    "stock_exchange": (("hero", "DISTRICT_COMMERCIAL_HUB", "BUILDING_STOCK_EXCHANGE", "Modern"),
                       "Stock Exchange (Commercial Hub)", "Civ VI base game"),
    "gov_spies": (("hero", "DISTRICT_GOVERNMENT", "BUILDING_GOV_SPIES"), "Intelligence Agency (Government Plaza)",
                  "Rise and Fall"),
    "gov_military": (("hero", "DISTRICT_GOVERNMENT", "BUILDING_GOV_MILITARY"), "War Department (Government Plaza)",
                     "Rise and Fall"),
    "gov_wide": (("hero", "DISTRICT_GOVERNMENT", "BUILDING_GOV_WIDE"), "Audience Chamber (Government Plaza)",
                 "Rise and Fall"),
    "dam_hydro": (("base", "DISTRICT_DAM", "HYDROELECTRIC_DAM", "Modern"), "Dam + Hydroelectric Dam", "Gathering Storm"),
    "spaceport": (("base", "DISTRICT_SPACEPORT", "LARGE_ROCKET", "Modern"), "Spaceport (large rocket stage)",
                  "Civ VI base game"),
    "observatory": (("base", "DISTRICT_OBSERVATORY", "LIBRARY, UNIVERSITY", "Industrial"),
                    "Observatory (Maya Campus)", "Maya & Gran Colombia pack"),
    "office_block_1": (("city", "DIS_CTY_MGG_Block_SQ_01"), "Modern city block 1", "Civ VI base game"),
    "office_block_2": (("city", "DIS_CTY_MGG_Block_SQ_02"), "Modern city block 2", "Civ VI base game"),
    "office_block_3": (("city", "DIS_CTY_MGG_Block_SQ_03"), "Modern city block 3", "Civ VI base game"),
    "broadcast_center": (("hero", "DISTRICT_ACROPOLIS", "BUILDING_BROADCAST_CENTER"), "Broadcast Center",
                         "Civ VI base game"),
    "bank_modern": (("hero", "DISTRICT_COMMERCIAL_HUB", "BUILDING_BANK", "Modern"), "Bank (Modern)",
                    "Civ VI base game"),
    "glass_tower_a": (("city", "DIS_CTY_MG_Bld_07"), "Modern glass tower", "Civ VI base game"),
    "glass_tower_b": (("city", "DIS_CTY_MGG_Bld_05"), "Modern glass tower", "Civ VI base game"),
    "glass_tower_c": (("city", "DIS_CTY_MG_Bld_01"), "Modern glass tower", "Civ VI base game"),
}
# Installed Workshop wonders: key -> (item, package path in the item, entry, label, origin).
MODS = {
    "notre_dame": ("3407707470", "Platforms/Windows/BLPs/WonderMovie.blp", "WON_Suk_NotreDame_Movie",
                   "Notre-Dame de Paris", "Workshop 3407707470 (Sukritact)"),
    "arecibo": ("2317145428", "Platforms/Windows/BLPs/WonderMovie.blp", "WON_CL_Arecibo",
                 "Arecibo Observatory", "Workshop 2317145428 (Albro, Deliverator23)"),
    "globe_theatre": ("1770688835", "Platforms/Windows/BLPs/WonderMovie.blp", "BUILDING_GLOBE_THEATRE",
                       "Globe Theatre", "Workshop 1770688835 (Deliverator23)"),
    "st_peters": ("2511437298", "Platforms/Windows/BLPs/WonderMovie.blp", "BUILDING_AL_STPETERSBASILICA_Movie",
                   "St. Peter's Basilica", "Workshop 2511437298 (Albro)"),
    "pentagon_mod": ("3337911322", "Platforms/Windows/BLPs/WonderMovie.blp", "WON_The_Pentagon_Movie",
                      "The Pentagon", "Workshop 3337911322 (IthildinX)"),
    "albro_ironworks": ("2612473657", "Platforms/Windows/BLPs/WonderMovie.blp", "NAT_WON_CL_IRONWORKS",
                         "Ironworks (National Wonders)", "Workshop 2612473657 (Albro)"),
    "albro_finance": ("2612473657", "Platforms/Windows/BLPs/WonderMovie.blp", "NAT_WON_CL_FINANCE",
                       "Financial Center (National Wonders)", "Workshop 2612473657 (Albro)"),
    "albro_college": ("2503076363", "Platforms/Windows/BLPs/WonderMovie.blp", "NAT_WON_CL_COLLEGE",
                       "National College (National Wonders)", "Workshop 2503076363 (Albro)"),
    "three_gorges": ("1710103474", "Platforms/Windows/BLPs/WonderMovie.blp", "Dujiangyan",
                      "Three Gorges Dam", "Workshop 1710103474 (WindFly)"),
    "ulugh_beg": ("3283744069", "Platforms/Windows/BLPs/WonderMovie.blp", "WON_C_Ulugh_Beg_Observatory_Movie_New",
                   "Ulugh Beg Observatory", "Workshop 3283744069 (c)"),
    "notre_dame_2024": ("3391287227", "Platforms/Windows/BLPs/WonderMovie.blp", "BUILDING_NOTRE_DAME_DE_PARIS",
                         "Notre-Dame de Paris (2024)", "Workshop 3391287227 (Phantagonist)"),
}


def keys() -> list[str]:
    return list(CIV6) + list(LANDMARKS) + list(MODS)


def record(key: str) -> dict:
    if key in CIV6:
        content, entry = CIV6[key][:2]
        return dk.raw(content, "WonderMovie", entry)
    if key in LANDMARKS:
        kind, *args = LANDMARKS[key][0]
        if kind == "hero":
            district, building, era = (*args, "Industrial")[:3]
            return dk.hero(district, building, era)
        if kind == "base":
            return dk.base(*args)
        return dk.city(args[0])
    item, package, entry = MODS[key][:3]
    return {**dk.raw(f"workshop/{item}", package.rsplit("/", 1)[-1].removesuffix(".blp"), entry),
            "path": str(WORKSHOP / item / package)}


def prop(key: str) -> tuple[str, str] | None:
    """A "prop:<sub-pack>:<entry>" source: a construction prop (crane, brick or wood
    pile) already imported with a wonder movie."""
    if not key.startswith("prop:"):
        return None
    _, pack, entry = key.split(":", 2)
    return pack, entry


def label(key: str) -> tuple[str, str]:
    if prop(key):
        return prop(key)[1], f"Civ VI construction prop ({CIV6[prop(key)[0]][2]} movie)"
    if key in CIV6:
        return CIV6[key][2:4]
    if key in LANDMARKS:
        return LANDMARKS[key][1:3]
    return MODS[key][3:5]


def is_movie(key: str) -> bool:
    return key in CIV6 or key in MODS


def use(key: str) -> Path:
    """Point the district study's importer at this source's own sub-pack."""
    dk.SOURCES = SOURCES / (prop(key)[0] if prop(key) else key)
    return dk.SOURCES


@lru_cache(maxsize=None)
def compiler(key: str):
    """The district study's Lab compiler with two more build-movie tolerances in
    _decode_skeletons (this process only): a package model holding more than one
    string array takes the one whose element count equals the bone count (as the
    natural-wonder study does), and a bone with no name (the Oracle's) gets a
    placeholder name."""
    use(key)
    found = DistrictCompiler()
    namespace = found.compile_asset.__globals__
    import inspect
    source = inspect.getsource(dk.cli._decode_skeletons)
    for old, new in (
            ('                raise ValueError("Compound skeleton bone name does not resolve")\n',
             '                name = f"unnamed_bone_{index}"\n'),
            ('        if len(bone_fields) != 1 or len(name_fields) != 1:\n',
             '        if len(bone_fields) == 1 and len(name_fields) > 1:\n'
             '            _count = package.allocations[bone_fields[0][1] - 1]["element_count"]\n'
             '            name_fields = [field for field in name_fields\n'
             '                           if package.allocations[field[1] - 1]["element_count"] == _count\n'
             '                           and len(package.bytes_for(field[1])) == 8 * _count]\n'
             '        if len(bone_fields) != 1 or len(name_fields) != 1:\n')):
        if old not in source:
            raise ValueError("compound_landmark_importer changed; update the wonder Lab patch of _decode_skeletons")
        source = source.replace(old, new)
    exec(compile("from __future__ import annotations\n" + source, "wonder_lab__decode_skeletons", "exec"), namespace)
    # The district study relaxes the bind-pose check for wonder movies on the shared
    # module while it compiles; this copy looks the check up there at call time.
    namespace["measure_bind_pose"] = lambda bones: dk.cli.measure_bind_pose(bones)
    # Workshop wonder packages (this process only): an empty or repeated attachment
    # parameter is skipped (Albro's packs), the first of several attachment-point
    # lists is the model's and a model may have none (Three Gorges), a missing
    # TerrainFollowMode is Pivot Height (Albro's packs), empty primitives are
    # skipped (the Pentagon). The district study's own patches of these functions
    # are applied again on top.
    for function, replacements in (
            (dk.cli._decode_component_attachment_values,
             (('    if (\n        values["ConnectionType"] not in ("NONE", "ROAD")',
               '    if False and (\n        values["ConnectionType"] not in ("NONE", "ROAD")'),
              ('    if set(values) != required:\n',
               '    values.setdefault("RandomizeAnims", False)\n    values.setdefault("TerrainFollowMode", "Pivot Height")\n'
               '    if set(values) != required:\n'),
              ('            if not parameter or parameter in values:\n',
               '            if not parameter or parameter in values:\n                continue\n            if False:\n'))),
            (dk.cli._decode_attachment_points,
             (("        if len(matches) != 1 and not (\n", "        if False and len(matches) != 1 and not (\n"),
              # A component attached at a name that is no bone (or several) hangs
              # at the movie skeleton's root (Globe Theatre, Ulugh Beg, Notre-Dame 2024).
              ('        has_component = component is not None and component["source_terminal"] is not None\n',
               '        has_component = component is not None and component["source_terminal"] is not None\n'
               '        if len(matches) > 1:\n            matches = matches[-1:]\n'
               '        if not matches and has_component and skeletons:\n'
               '            matches = [(len(skeletons) - 1, 0, skeletons[-1]["bones"][0])]\n'),
              ('    if len(fields) != 1:\n        raise ValueError("Compound landmark must have exactly one attachment-point list")',
               '    if not fields:\n        return [], {"list": None, "data": None, "count": 0, "points": []}'))),
            (dk.cli._extract_decal_component,
             (('    if len(vector_fields) != 1:\n        raise ValueError("Compound landmark must have exactly one decal vector")',
               '    if not vector_fields:\n        return None, {"vector": None, "status": "empty"}\n'
               '    if len(vector_fields) != 1:\n        raise ValueError("Compound landmark must have exactly one decal vector")'),)),
            (dk.cli._compile_asset,
             (("            if count < 1 or first_mesh + count > mesh_count:\n",
               "            if count < 1:\n                continue\n            if first_mesh + count > mesh_count:\n"),
              ("                primitive = primitives[primitive_index]\n",
               "                primitive = primitives[primitive_index]\n"
               "                if primitive[\"index_count\"] < 3:\n                    continue\n"),
              ("    if len(terrain_fields) != 1:\n        raise ValueError(\"Compound landmark must have exactly one terrain-edit vector\")\n"
               "    terrain_vector = terrain_fields[0][1]\n"
               "    terrain_edit, terrain_edit_evidence = _decode_terrain_edit(\n        package, terrain_vector, terrain_edit_policy\n    )\n",
               "    terrain_vector = terrain_fields[0][1] if terrain_fields else None\n"
               "    if terrain_fields:\n        terrain_edit, terrain_edit_evidence = _decode_terrain_edit(\n"
               "            package, terrain_vector, terrain_edit_policy)\n"
               "    else:\n        terrain_edit, terrain_edit_evidence = {\"status\": \"absent\"}, {\"status\": \"absent\"}\n")))):
        source = inspect.getsource(function)
        for old_text, new_text in replacements:
            if old_text not in source:
                raise ValueError(f"compound_landmark_importer changed; update the wonder Lab patch of {function.__name__}: {old_text[:60]!r}")
            source = source.replace(old_text, new_text)
        exec(compile("from __future__ import annotations\n" + source, "wonder_lab_" + function.__name__, "exec"),
             namespace)
    patched = namespace["_compile_asset"]
    # Civ VI material slots the shared importer leaves out: metalness, and the baked
    # light map (laid out by uv1) some wonders light their night with (the Oracle).
    dk.cli.MATERIAL_TEXTURE_SLOTS = {**dk.cli.MATERIAL_TEXTURE_SLOTS,
                                     "metalness": (0x2C, "Generic_Metalness", False),
                                     "light_map": (0x38, "Generic_LightMap", False)}

    def compile_asset(*args, **kwargs):
        # Civ VI's second and third UV sets: ambient occlusion and emissive (night
        # lights) are mapped by them, as production city bodies read them.
        return patched(*args, **{**kwargs, "auxiliary_uvs": True})
    found.compile_asset = compile_asset
    if key in MODS:
        # A Workshop wonder's attached models live in its own packages.
        item = WORKSHOP / MODS[key][0]
        for path in sorted(item.rglob("*.blp")):
            lowered = [part.lower() for part in path.relative_to(item).parts]
            if "windows" in lowered and "blps" in lowered:
                found.packages.append({"path": path, "relative": str(path), "content_root": str(item),
                                       "logical": Path(*path.relative_to(item).parts[lowered.index("blps") + 1:])
                                       .with_suffix("").as_posix().lower()})
    # Some Workshop packages open only with the relaxed temp-stripe inference
    # the natural-wonder study uses for them.
    strict_open = found.open

    def open_package(package_relative, entry):
        try:
            return strict_open(package_relative, entry)
        except ValueError as exc:
            if "Temp-stripe" not in str(exc):
                raise
            return dk.IndexedStaticPackage(dk.DEFAULT_ASSETS_ROOT / package_relative, entry, minimum_temp_support=2)
    found.open = open_package
    return found


def compile_one(key: str) -> str:
    use(key)
    found = compiler(key)
    # A recompile re-imports every asset (the manifest otherwise reuses them).
    for section in ("roots", "assets", "failures"):
        found.manifest[section].clear()
    try:
        found.root(record(key))
        return f"{key} compiled"
    except ValueError as exc:
        return f"{key} FAILED: {exc}"
    finally:
        found.save()


def landmarks(key: str) -> list[tuple[str, dict, list]]:
    """(asset, landmark document, transform) of a source: a wonder movie's own
    model only (its set dressing children are left out), a building with all of
    its attached children."""
    use(key)
    found = compiler(key)
    root = found.root(record(key))
    pieces = dk.components(found.manifest, root, dk.IDENTITY, [])
    # A movie's own model only, unless its geometry lives in attached models
    # (Workshop wonders such as Ulugh Beg's).
    root_triangles = sum(len(_geometry(str(dk.SOURCES / pieces[0][1]["components"]["geometry"][binding["geometry"]]))
                             ["topology"]["indices"]) // 3 for binding in pieces[0][1]["draw_bindings"])
    return pieces[:1] if is_movie(key) and root_triangles >= 200 else pieces


# ---------------------------------------------------------------- probe

PALETTE = ((230, 80, 70), (70, 160, 230), (240, 200, 60), (90, 200, 110), (200, 110, 220), (240, 140, 50),
           (60, 210, 210), (160, 160, 160), (250, 120, 170), (140, 100, 60), (120, 120, 250), (190, 230, 90))


def _splat(triangles, colours, size: int = 200):
    """Iso preview of triangles (Civ III's view: x down-right, y down-left, z up),
    drawn as area-weighted surface samples with a depth sort, as PNG rows."""
    import numpy as np
    if not len(triangles):
        return size, size, [bytes(size * 3)] * size
    t = np.asarray(triangles, dtype=np.float64)
    face = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = np.linalg.norm(face, axis=1)
    normal = np.nan_to_num(face / np.maximum(area, 1e-12)[:, None])
    rng = np.random.default_rng(1)
    count = 400000
    pick = rng.choice(len(t), count, p=area / area.sum()) if area.sum() > 0 else rng.integers(0, len(t), count)
    a, b = rng.random(count), rng.random(count)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    p = t[pick, 0] + a[:, None] * (t[pick, 1] - t[pick, 0]) + b[:, None] * (t[pick, 2] - t[pick, 0])
    sx, sy = p[:, 0] - p[:, 1], (-(p[:, 0] + p[:, 1]) * .5 - p[:, 2] * 1.1)
    depth = p[:, 0] + p[:, 1] + p[:, 2] * .2
    span = max(float(np.ptp(sx)), float(np.ptp(sy)) * 1.4, 1e-6)
    scale = (size - 20) / span
    ix = np.clip(((sx - (sx.min() + sx.max()) / 2) * scale + size / 2).astype(np.int32), 0, size - 1)
    iy = np.clip(((sy - (sy.min() + sy.max()) / 2) * scale + size / 2).astype(np.int32), 0, size - 1)
    light = np.clip(.35 + .65 * np.abs(normal[pick] @ np.asarray([.3, -.5, .8])), .2, 1.0)
    c = (np.asarray(colours, dtype=np.float64)[pick] * light[:, None]).astype(np.uint8)
    image = np.full((size, size, 3), 28, np.uint8)
    order = np.argsort(-depth)       # far first, near last
    image[iy[order], ix[order]] = c[order]
    return size, size, [image[row].tobytes() for row in range(size)]


def probe(key: str) -> dict:
    """Per-material piece statistics and previews of one imported source:
    all materials, then each material of its main model alone, in material order."""
    from Renderer.lab.studies.resources import audit
    folder = OUT / "probe"
    folder.mkdir(parents=True, exist_ok=True)
    summary, tiles, everything = {}, [], ([], [])
    for index, (asset, document, transform) in enumerate(landmarks(key)):
        by_material: dict[int, list] = {}
        for binding in document["draw_bindings"]:
            if "worked" in binding["states"]:
                mesh = dk.load(dk.SOURCES / document["components"]["geometry"][binding["geometry"]])
                by_material.setdefault(binding["material"], []).append(mesh)
        for material, meshes in sorted(by_material.items()):
            radii = sorted(math.hypot(sum(v["position"][0] for v in m["vertices"]) / len(m["vertices"]),
                                      sum(v["position"][1] for v in m["vertices"]) / len(m["vertices"]))
                           for m in meshes)
            triangles = []
            for m in meshes:
                points = [_apply(transform, v["position"]) for v in m["vertices"]]
                indices = m["topology"]["indices"]
                triangles += [[points[i] for i in indices[k:k + 3]] for k in range(0, len(indices) - 2, 3)]
            zs = [p[2] for tri in triangles for p in tri] or [0.0]
            summary[f"{index}:{material}"] = {
                "pieces": len(meshes), "triangles": len(triangles),
                "radius_p10_p50_p90": [round(radii[int(q * (len(radii) - 1))], 3) for q in (.1, .5, .9)],
                "z": [round(min(zs), 3), round(max(zs), 3)]}
            colour = PALETTE[material % len(PALETTE)]
            everything[0].extend(triangles)
            everything[1].extend([colour] * len(triangles))
            if index == 0:
                tiles.append(_splat(triangles, [colour] * len(triangles)))
    tiles.insert(0, _splat(*everything, size=400))
    height = 400
    rows = []
    for row in range(height):
        line = b""
        for tile in tiles:
            line += tile[2][row] if row < tile[1] else bytes(tile[0] * 3)
        rows.append(line)
    audit.write_png(folder / f"{key}.png", sum(tile[0] for tile in tiles), height, rows)
    report = {"key": key, "materials": summary}
    (folder / f"{key}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


# ---------------------------------------------------------------- bake

def _apply(transform: list, p) -> list:
    x, y, z = p
    return [x * transform[0] + y * transform[4] + z * transform[8] + transform[12],
            x * transform[1] + y * transform[5] + z * transform[9] + transform[13],
            x * transform[2] + y * transform[6] + z * transform[10] + transform[14]]


def _normal(transform: list, n) -> list:
    x, y, z = n
    out = [x * transform[0] + y * transform[4] + z * transform[8],
           x * transform[1] + y * transform[5] + z * transform[9],
           x * transform[2] + y * transform[6] + z * transform[10]]
    length = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / length for v in out]


@lru_cache(maxsize=None)
def _geometry(path: str) -> dict:
    return dk.load(Path(path))


@lru_cache(maxsize=None)
def posable(key: str) -> bool:
    """A build-movie source whose converted movie (stages.py) ends on the finished
    wonder: its final frame is the clean finished model."""
    from Renderer.lab.studies.wonders import stages
    if prop(key) or not is_movie(key) or not stages.ready(key):
        return False
    try:
        stages.span(key)
    except (ValueError, StopIteration):
        return False
    # The movie skeleton must carry most of the model (the Globe Theatre's mesh
    # sits on another skeleton its clip never moves).
    _asset, document, _transform = landmarks(key)[0]
    skeleton = stages.movie(key)[4]
    triangles = {binding["geometry"]: len(_geometry(str(dk.SOURCES / document["components"]["geometry"][
        binding["geometry"]]))["topology"]["indices"]) // 3 for binding in document["draw_bindings"]}
    on_movie = sum(triangles[b["geometry"]] for b in document["draw_bindings"] if b["skeleton"] == skeleton)
    return on_movie >= .5 * sum(triangles[b["geometry"]] for b in document["draw_bindings"])


@lru_cache(maxsize=None)
def staged(key: str) -> bool:
    """A posable source whose movie really builds it; a movie that only reveals
    the finished wonder at the end (or reuses another wonder's clip) is not."""
    from Renderer.lab.studies.wonders import stages
    if not posable(key):
        return False
    start, end = stages.span(key)
    return end - start > .1


def pieces(part: dict):
    """(model index, landmark document, draw binding, transform) of every piece a
    part draws. A build-movie source with a converted movie is posed at its
    `stage` (1.0, the finished wonder, by default): each piece on its own bone,
    pieces the movie hides at that moment left out, props on their bones.
    Otherwise the bind pose (with every scaffold and stone pile in it)."""
    key = part["source"]
    if prop(key):
        pack, entry = prop(key)
        use(pack)
        found = compiler(pack)
        asset = next(asset for asset, value in found.manifest["assets"].items() if value.get("source_entry") == entry)
        for index, (asset, document, transform) in enumerate(dk.components(found.manifest, asset, dk.IDENTITY, [])):
            for binding in document["draw_bindings"]:
                yield index, document, binding, transform
        return
    if posable(key):
        from Renderer.lab.studies.wonders import stages
        stage = stages.progress(key, part["built"]) if "built" in part and staged(key) else part.get("stage", 1.0)
        yield from stages.posed_pieces(key, stage)
        return
    # Children named in `drop_children` are left out: props whose attachment never
    # resolved to a bone all stand at the model's centre (the Globe's trees).
    drop = tuple(part.get("drop_children", ()))
    manifest = compiler(key).manifest["assets"]
    for index, (asset, document, transform) in enumerate(landmarks(key)):
        if index and drop and any(name in manifest[asset]["source_entry"] for name in drop):
            continue
        for binding in document["draw_bindings"]:
            yield index, document, binding, transform


def extent(part: dict) -> tuple[list, list]:
    """Bounds (low, high) of what a wonder is fitted by: the main model's raised
    pieces in the finished wonder, turned by the part's turn. Flat ground pieces
    (paving, sand) and attached props may reach past them."""
    finished = {k: v for k, v in part.items() if k not in ("built", "stage")}
    turn = finished.get("turn", 0.0)
    c, s = math.cos(turn), math.sin(turn)
    squash = finished.get("squash", 1.0)
    keep = finished.get("keep", {})
    materials, radius = keep.get("materials"), keep.get("radius")
    low, high = [9.0] * 3, [-9.0] * 3
    top = -9.0
    use(part["source"])
    for index, document, binding, transform in pieces(finished):
        if "worked" not in binding["states"] or \
                (materials is not None and binding["material"] not in materials):
            continue
        if index:
            # Attached children (a spaceport's rocket) count toward the height cap only.
            if not is_movie(part["source"]):
                mesh = _geometry(str(dk.SOURCES / document["components"]["geometry"][binding["geometry"]]))
                top = max(top, max(_apply(transform, v["position"])[2] for v in mesh["vertices"]) * squash)
            continue
        mesh = _geometry(str(dk.SOURCES / document["components"]["geometry"][binding["geometry"]]))
        points = [_apply(transform, v["position"]) for v in mesh["vertices"]]
        if max(p[2] for p in points) - min(p[2] for p in points) < .01:
            continue
        mx, my = sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)
        if radius is not None and math.hypot(mx, my) > radius:
            continue
        box = keep.get("box")
        if box is not None and not (box[0] <= mx <= box[2] and box[1] <= my <= box[3]):
            continue
        if keep.get("z_min") is not None and max(p[2] for p in points) < keep["z_min"]:
            continue
        for x, y, z in points:
            # Vertices standing above the ground only: a model may merge its ground
            # plane into the building mesh (the Pyramids' sand covers the hex).
            if z < .012:
                continue
            for axis, value in enumerate((x * c - y * s, x * s + y * c, z * squash)):
                low[axis], high[axis] = min(low[axis], value), max(high[axis], value)
    high[2] = max(high[2], top)
    return low, high


def kept(part: dict, full: bool = False) -> list[tuple[str, dict, list, list]]:
    """The pieces a part keeps, per texture: (material path, material, vertices, indices),
    vertices in kit units turned by the part's turn and squashed, at its own origin.
    `full` keeps one group per material instead, with turned normals and Civ VI's
    three UV sets (uv0, uv1, uv2) per vertex (city bodies, city_pack.py)."""
    keep = part.get("keep", {})
    materials, radius = keep.get("materials"), keep.get("radius")
    drop, low, high = set(keep.get("drop", ())), keep.get("z_min"), keep.get("clip_z")
    inside = part.get("inside")
    turn = part.get("turn", 0.0)
    c, s = math.cos(turn), math.sin(turn)
    squash = part.get("squash", 1.0)
    out: dict[str, list] = {}
    SOURCES_OF[0] = use(part["source"])
    for index, document, binding, transform in pieces(part):
        # Construction props (scaffolds, stones) exist only in Civ VI's construction state.
        if "worked" not in binding["states"] and not (prop(part["source"]) and "construction" in binding["states"]):
            continue
        tag = f"{index}:{binding['material']}" if index else binding["material"]
        if index == 0 and (materials is not None and binding["material"] not in materials or tag in drop):
            continue
        mesh = _geometry(str(dk.SOURCES / document["components"]["geometry"][binding["geometry"]]))
        points = [_apply(transform, v["position"]) for v in mesh["vertices"]]
        cx, cy = sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)
        if index == 0 and radius is not None and math.hypot(cx, cy) > radius:
            continue
        box = keep.get("box")
        if index == 0 and box is not None and not (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
            continue
        if inside is not None:
            # Only pieces standing on the Civ III tile: a Civ VI hex ring's outer props
            # and plaza pieces beyond the square tile are left out.
            bx, by, half = inside
            if max(abs(cx * c - cy * s - bx), abs(cx * s + cy * c - by)) > half:
                continue
        if low is not None and max(p[2] for p in points) < low:
            continue
        path = document["components"]["materials"][binding["material"]]
        if full:
            def turned(n):
                return [n[0] * c - n[1] * s, n[0] * s + n[1] * c, n[2]]
            vertices = [([x * c - y * s, x * s + y * c, z * squash], turned(_normal(transform, v["normal"])),
                         (list(v["uv0"]), list(v.get("uv1", (0.0, 0.0))), list(v.get("uv2", (0.0, 0.0)))))
                        for (x, y, z), v in zip(points, mesh["vertices"])]
        else:
            vertices = [([x * c - y * s, x * s + y * c, z * squash], _normal(transform, v["normal"]), list(v["uv0"]))
                        for (x, y, z), v in zip(points, mesh["vertices"])]
        if inside is not None:
            # Ground-level geometry (paving, paths, sand, skirts) reaching past the tile
            # is pressed to its edge; raised parts stand inside already.
            bx, by, half = inside
            vertices = [([min(bx + half, max(bx - half, x)), min(by + half, max(by - half, y)), z]
                         if z < .02 else [x, y, z], n, uv) for (x, y, z), n, uv in vertices]
        indices = mesh["topology"]["indices"]
        if high is not None:
            # A building rising from the ground: everything above the cut is pressed
            # down onto it (walls stop there, the roof lies on top as a lid).
            if min(p[2] for p in points) > high:
                continue
            vertices = [([x, y, min(z, high * squash)], n, uv) for (x, y, z), n, uv in vertices]
        if indices:
            # One mesh per texture, not per material: build-movie models repeat a
            # texture across many materials, and a catalog holds 256 meshes.
            channels = dk.load(dk.SOURCES / path).get("channels", {})
            texture = tuple((channels.get(name) or {}).get("texture") if isinstance(channels.get(name), dict) else None
                            for name in ("base_color", "opacity"))
            # Full groups keep slender pieces (twice as tall as wide) apart: a conifer
            # keeps its green when Autumn and Spring recolour broadleaf foliage (the
            # user's rule), even where it shares a texture with the shrubs.
            if not full:
                out.setdefault(texture, (path, [], False))[1].append((vertices, indices))
                continue
            for part_vertices, part_indices in _connected(vertices, indices):
                xs, ys, zs = ([p[a] for p, _, _ in part_vertices] for a in range(3))
                slender = max(zs) - min(zs) > 2 * max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
                out.setdefault((path, slender), (path, [], slender))[1].append((part_vertices, part_indices))
    result = []
    for path, meshes, slender in out.values():
        material = dk.load(dk.SOURCES / path)
        if slender:
            material = {**material, "conifer": True}
        if not isinstance(material.get("channels", {}).get("base_color"), dict):
            continue
        chunks = [([], [])]
        for vertices, indices in meshes:
            if len(chunks[-1][0]) + len(vertices) > 60000:
                chunks.append(([], []))
            chunk_vertices, chunk_indices = chunks[-1]
            chunk_indices += [len(chunk_vertices) + i for i in indices]
            chunk_vertices += vertices
        result += [(path, material, v, i) for v, i in chunks if i]
    return result


def _connected(vertices: list, indices: list) -> list:
    """A mesh's connected parts (by shared vertices), each with its own indices."""
    parent = list(range(len(vertices)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for t in range(0, len(indices), 3):
        a, b, c = (root(i) for i in indices[t:t + 3])
        parent[b] = a
        parent[root(c)] = a
    groups: dict[int, list[int]] = {}
    for t in range(0, len(indices), 3):
        groups.setdefault(root(indices[t]), []).extend(indices[t:t + 3])
    out = []
    for triangles in groups.values():
        used = sorted(set(triangles))
        remap = {old: new for new, old in enumerate(used)}
        out.append(([vertices[i] for i in used], [remap[i] for i in triangles]))
    return out


@lru_cache(maxsize=None)
def _alignment(signature: str) -> float:
    """Turn (radians, within 45 degrees) that squares a part's raised footprint to
    the Civ III tile: Civ VI authors wonders at hex (30/60 degree) angles."""
    from Renderer.lab.shared.cities.ground import footprint_alignment
    part = json.loads(signature)
    points = []
    use(part["source"])
    for index, document, binding, transform in pieces(part):
        if index or "worked" not in binding["states"]:
            continue
        mesh = _geometry(str(dk.SOURCES / document["components"]["geometry"][binding["geometry"]]))
        points += [(x, y) for x, y, z in (_apply(transform, v["position"]) for v in mesh["vertices"]) if z >= .012]
    return footprint_alignment(points) if len(points) >= 3 else 0.0


def layout(part: dict) -> tuple[dict, float, float, float, list, list]:
    """A finished part laid on the Civ III tile: (part with its square turn and tile
    box, centre x, centre y, scale, low, high). The part is turned square to the
    tile (unless `square` is False), fitted by its raised pieces so they span
    `fit` tiles (capped at `height` tall), and keeps only pieces whose centre
    stands on the tile."""
    finished = {k: v for k, v in part.items() if k not in ("built", "stage", "inside")}
    if part.get("square", True) and not prop(part["source"]):
        base = {k: v for k, v in finished.items() if k not in ("turn", "at", "fit", "height", "scale", "lift")}
        finished["turn"] = finished.get("turn", 0.0) + _alignment(json.dumps(base, sort_keys=True, default=str))
    low, high = extent(finished)
    if high[0] < low[0]:      # nothing raised: fit the whole thing
        reference = kept(finished)
        low = [min(p[a] for _, _, vertices, _ in reference for p, _, _ in vertices) for a in range(3)]
        high = [max(p[a] for _, _, vertices, _ in reference for p, _, _ in vertices) for a in range(3)]
    cx, cy = (low[0] + high[0]) / 2, (low[1] + high[1]) / 2
    span = max(high[0] - low[0], high[1] - low[1], 1e-6)
    scale = part.get("scale") or min(part.get("fit", .95) / span, part.get("height", 1.2) / max(high[2], 1e-6))
    if not prop(part["source"]) and not part.get("at"):
        finished["inside"] = [round(cx, 5), round(cy, 5), round(.5 / scale, 5)]
    return finished, cx, cy, scale, low, high


# The season and hour being baked and rendered (render() sets them per pass).
SEASON = {"index": 0}
SEASON_NAMES = {0: "summer", 1: "autumn", 2: "winter", 3: "spring"}


def fit_dds(path: Path, size: int, block: int, cache: Path) -> Path:
    """A square atlas-ready texture brought down to `size` texels by dropping
    mip levels (block copies, no recompression)."""
    width, height, mips, dxgi, payload = rc.dds(path)
    if width == size:
        return path
    layout = rc.mip_offsets(width, height, mips, block)
    skip = (width // size).bit_length() - 1
    levels = [b"".join(dk._block_rows(payload, *layout[min(skip + level, len(layout) - 1)], block))
              for level in range((size // 4).bit_length())]
    return rc.write_dds(cache / f"fit{size}_{path.parent.name}_{path.name}", size, len(levels), dxgi, block, levels)


def with_emissive(mesh: dict, material: dict, cache: Path, key: str) -> dict:
    """A composition mesh carrying Civ VI's emissive map (windows, lamps) for the
    atlas's lower half, at the albedo's size; and, in Autumn and Spring, broadleaf
    foliage recoloured with Renderer64's seasonal crown colours (which reach only
    natural forests in the shaders)."""
    channel = material.get("channels", {}).get("emissive")
    if isinstance(channel, dict) and not mesh["masked"]:
        emissive, _su, _sv = dk.square(SOURCES_OF[0] / channel["texture"], cache, 8)
        size = min(rc.dds(mesh["texture"])[0], rc.dds(emissive)[0])
        if rc.dds(emissive)[3] in rc.MODEL_ATLAS[4]:
            mesh = {**mesh, "texture": fit_dds(mesh["texture"], size, 8, cache),
                    "emissive": fit_dds(emissive, size, 8, cache)}
    if SEASON["index"] in (1, 3) and not key.startswith("decal/"):
        from Renderer.lab.studies.natural_wonders import kits as nw
        seed = (zlib.crc32(key.encode()) % 997) / 997
        graded = nw.seasonal_foliage(mesh["texture"], SEASON["index"], seed, cache)
        if graded != Path(mesh["texture"]):
            mesh = {**mesh, "texture": graded, "masked": True, "emissive": None}
    return mesh


SOURCES_OF = [SOURCES]   # the sub-pack whose material is being meshed (kept() sets it)


def plot_mesh(era: str, half: float, cache: Path) -> dict:
    """A paved construction plot (the city pack's era paving) as a ground decal."""
    texture = dk.PLATES / f"{era.lower()}.dds"
    n = 6
    vertices = [([-half + 2 * half * i / n, -half + 2 * half * j / n, 0.0], [0.0, 0.0, 1.0], [i / n, j / n])
                for j in range(n + 1) for i in range(n + 1)]
    indices = [index for j in range(n) for i in range(n) for index in
               (j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i,
                j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i)]
    return dk._mesh(vertices, indices, {"channels": {"base_color": {"texture": str(texture)}}}, cache)


def _meshes(meshes: dict, key: str, vertices: list, indices: list, material: dict, cache: Path) -> list[str]:
    """Composition meshes for one chunk, halved by triangles until each fits the
    runtime's bounds (repeating UVs split a mesh's vertices when it is made)."""
    if key in meshes or any(name.startswith(key + "/") for name in meshes):
        return [key] if key in meshes else sorted(name for name in meshes if name.startswith(key + "/"))
    try:
        meshes[key] = with_emissive(dk._mesh(vertices, indices, material, cache), material, cache, key)
        return [key]
    except ValueError as exc:
        if "runtime bounds" not in str(exc) or len(indices) < 6:
            raise
    half = len(indices) // 6 * 3
    keys = []
    for side, part in enumerate((indices[:half], indices[half:])):
        used = sorted(set(part))
        remap = {old: new for new, old in enumerate(used)}
        keys += _meshes(meshes, f"{key}/{side}", [vertices[i] for i in used], [remap[i] for i in part], material, cache)
    return keys


# City effect layer records per Civ VI effect class (effects.kind_of): kind (90 +
# kind in the shader), width, height (tile units) and strength. Lights are lamps
# (7) or blinkers (8) whose seed band picks the colour (kits.LAMP_EDIT).
EFFECT_LOOK = {"flame": (0, .12, .24, 1.3), "smoke": (1, .16, .5, .5), "light": (7, .12, .12, 1.6),
               # A beacon (the Great Lighthouse's) burns larger than a brazier and shows from its lantern.
               "beacon": (0, .24, .42, 1.6)}


def effect_records(part: dict, finished: dict, cx: float, cy: float, scale: float, facing: float) -> list[list]:
    """City effect records (city_pack.effect_record) of a finished part's Civ VI
    effects: each anchor turned and fitted as the part's bodies are."""
    from Renderer.lab.studies.wonders import effects
    turn, squash = finished.get("turn", 0.0), finished.get("squash", 1.0)
    c, s = math.cos(turn), math.sin(turn)
    cf, sf = math.cos(facing), math.sin(facing)
    du, dv = part.get("at", (0.0, 0.0))
    u = .5 + du + (-cx * cf + cy * sf) * scale
    v = .5 + dv + (-cx * sf - cy * cf) * scale
    inside = finished.get("inside")
    # Only effects on geometry the part draws: a bone of an undrawn prop (Apollo's
    # light mast) would light empty ground. Kept vertices are turned already.
    drawn = [p for _, _, vertices, _ in kept(finished) for p, _, _ in vertices]
    out, seen = [], set()
    found = effects.finished_effects(part["source"])
    missing = list(dict.fromkeys(effect["bone_hash"] for effect in found
                                 if effect["position"] is None and effect["kind"] == "flame"))
    if part.get("fires"):
        # Authored anchors (plan.py) stand in for fires whose bones the package lacks:
        # each on the highest drawn point within .02 of its model-space spot.
        found = [effect for effect in found if effect["position"] is not None]
        for fx, fy in part["fires"]:
            tx, ty = fx * c - fy * s, fx * s + fy * c
            tops = [p[2] for p in drawn if math.hypot(p[0] - tx, p[1] - ty) < .02]
            if tops:
                found.append({"name": "authored_fire", "kind": "flame", "bone_hash": None,
                              "position": [fx, fy, max(tops) / squash]})
    elif missing:
        # Otherwise they stand at the light map's brightest points, where Civ VI baked
        # their light (light_peaks); without a light map they are left out.
        peaks = light_peaks(json.dumps(finished, sort_keys=True, default=str), len(missing))
        for effect in found:
            if effect["position"] is None and effect["kind"] == "flame" and missing.index(effect["bone_hash"]) < len(peaks):
                x, y, z = peaks[missing.index(effect["bone_hash"])]
                # Peaks are turned already; undo the turn so the loop below can apply it.
                effect["position"] = [x * c + y * s, -x * s + y * c, z / squash]
    for number, effect in enumerate(effect for effect in found if effect["position"] is not None):
        beacon = effect["kind"] == "flame" and "lighthouse_fire" in effect["name"].lower()
        look = EFFECT_LOOK.get("beacon" if beacon else effect["kind"])
        # A completion trigger and the idle loop often light the same fire at one point.
        where = (effect["kind"], *(round(value, 3) for value in effect["position"]))
        # Below the ground: a bone of an attached model's own skeleton at rest.
        if look is None or where in seen or effect["position"][2] < -.005:
            continue
        seen.add(where)
        x, y, z = effect["position"]
        keep = part.get("keep", {})
        box, radius = keep.get("box"), keep.get("radius")
        # A part that keeps only some pieces (Apollo's rocket and its hangar) keeps their effects.
        if box is not None and not (box[0] <= x <= box[2] and box[1] <= y <= box[3]) or \
                radius is not None and math.hypot(x, y) > radius:
            continue
        x, y = x * c - y * s, x * s + y * c
        if not any(abs(p[0] - x) < .012 and abs(p[1] - y) < .012 and abs(p[2] - z * squash) < .03 for p in drawn):
            continue
        if inside is not None and max(abs(x - inside[0]), abs(y - inside[1])) > inside[2]:
            continue
        kind, width, height, strength = look
        n = effect["name"].lower()
        band = 0
        if kind == 7:
            kind = 8 if "blink" in n else 7
            band = 2 if "red" in n else 3 if "green" in n else 1 if ("white" in n or "spot" in n or "flood" in n) else 0
        seed = band * 16 + (number * 7) % 16
        lx, ly = (x * cf - y * sf) * scale, (x * sf + y * cf) * scale
        out.append(city_pack.effect_record(u + lx, 1 - v - ly, z * squash * scale + part.get("lift", 0.0),
                                           kind, width, height, seed, strength))
    return out


@lru_cache(maxsize=None)
def _light_texels(path: str):
    import numpy as np
    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    width, height = int(probe[0]), int(probe[1])
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).mean(axis=2)


@lru_cache(maxsize=None)
def light_peaks(signature: str, count: int) -> list:
    """The `count` brightest light sources of a finished part's baked light maps
    (sampled by uv1): its brightest vertices clustered within .03 kit units.
    Turned kit-space points."""
    part = json.loads(signature)
    sources = use(part["source"])
    points = []
    for _path, material, vertices, _indices in kept(part, full=True):
        light = material.get("channels", {}).get("light_map")
        if not isinstance(light, dict):
            continue
        texels = _light_texels(str(sources / light["texture"]))
        height, width = texels.shape
        for (x, y, z), _n, (_uv0, uv1, _uv2) in vertices:
            value = texels[min(height - 1, int((uv1[1] % 1) * height)), min(width - 1, int((uv1[0] % 1) * width))]
            points.append((value, x, y, z))
    if not points:
        return []
    points.sort(reverse=True)
    cut = points[max(0, int(len(points) * .005) - 1)][0]
    clusters = []
    for value, x, y, z in points:
        if value < cut:
            break
        for cluster in clusters:
            if math.hypot(x - cluster[1], y - cluster[2]) < .03:
                cluster[4] += 1
                break
        else:
            clusters.append([value, x, y, z, 1])
    # Identical lamps bake identical light: the brightest group of at least `count`
    # clusters within 2% of each other (the Hanging Gardens' four bowls), else the
    # brightest clusters.
    for first in clusters:
        group = [cluster for cluster in clusters if abs(cluster[0] - first[0]) <= .02 * first[0]]
        if len(group) >= count:
            group.sort(key=lambda cluster: -cluster[4])
            return [(x, y, z) for _value, x, y, z, _count in group[:count]]
    return [(x, y, z) for _value, x, y, z, _count in clusters[:count]]


def city_body(library, part: dict, pieces: list, cx: float, cy: float, scale: float, facing: float) -> tuple:
    """A part's kept pieces (kept(full=True)) as one city body in tile space, laid
    out exactly as the resource path laid them (append_instance): turned to the
    cell's facing, fitted, centred and raised to the resource path's height."""
    cf, sf = math.cos(facing), math.sin(facing)
    du, dv = part.get("at", (0.0, 0.0))
    u, v = .5 + du + (-cx * cf + cy * sf) * scale, .5 + dv + (-cx * sf - cy * cf) * scale
    lift = part.get("lift", 0.0)
    sources = use(part["source"])
    bodies = []
    for path, material, vertices, indices in pieces:
        out = []
        for (x, y, z), n, (uv0, uv1, uv2) in vertices:
            lx, ly = (x * cf - y * sf) * scale, (x * sf + y * cf) * scale
            nx, ny = n[0] * cf - n[1] * sf, n[0] * sf + n[1] * cf
            nz = n[2] / city_pack.LIFT
            length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            out.append((u + lx - .5, v + ly - .5, (z * scale + lift) * city_pack.LIFT,
                        [nx / length, ny / length, nz / length], uv0, uv1, uv2))
        colour = None
        if SEASON["index"] in (1, 3) and not material.get("conifer") and \
                isinstance(material.get("channels", {}).get("base_color"), dict):
            from Renderer.lab.studies.natural_wonders import kits as nw
            base = sources / material["channels"]["base_color"]["texture"]
            graded = nw.seasonal_foliage(base, SEASON["index"], (zlib.crc32(path.encode()) % 997) / 997,
                                         OUT / "texture_cache")
            colour = graded if Path(graded) != base else None
        has_uv1 = any(any(uvs[1]) for _, _, uvs in vertices)
        has_uv2 = any(any(uvs[2]) for _, _, uvs in vertices)
        material_id, light_map = library.material(material, sources, has_uv1, has_uv2, colour)
        if light_map:
            # A light map is drawn as the material's night emission, which the city
            # shader samples by uv2: its uv1 layout takes that place.
            out = [(x, y, z, n, uv0, uv1, uv1) for x, y, z, n, uv0, uv1, _uv2 in out]
        bodies.append((material_id, out, indices))
    return library.model(bodies)


def bake(catalog: dict, _compiler=None, _scale: float = 1.0, _ground: bool = True) -> dict:
    """Write the Lab packs for one catalog. Wonders are city bodies in the Lab city
    library (city_pack.py), one composition per cell named for the cell; the Lab
    resource pack keeps each cell's ground decals and names the cell's tile."""
    cache = OUT / "texture_cache"
    meshes: dict[str, dict] = {"clearance/wonder": marker_mesh(cache)}
    library = city_pack.Library()
    compositions, report = [], {}
    for cell in catalog["cells"]:
        # The tile's resource composition: ground decals and a marker that names the cell.
        instances = [{"model": "clearance/wonder", "u": .5, "v": .5, "rotation": 0.0, "scale": 1.0,
                      "lift": 0.0, "ground_fit": 0.0}]
        bodies, sources = [], []
        for number, part in enumerate(cell["parts"]):
            if part["source"].startswith("plot:"):
                # The Wonder District's paved construction plot, a ground decal on the tile.
                era = part["source"].split(":", 1)[1]
                key = f"decal/wonder_plot/{era}"
                if key not in meshes:
                    meshes[key] = plot_mesh(era, part.get("half", .36), cache)
                instances.insert(0, {"model": key, "u": .5, "v": .5, "rotation": 0.0, "scale": 1.0,
                                     "lift": 0.0, "ground_fit": 0.0})
                continue
            # A construction stage is fitted and centred like its finished wonder, so
            # every stage stands on the same footprint. Sources without a build movie
            # rise from the ground instead (cut at the built share of their height).
            finished, cx, cy, scale, low, high = layout(part)
            staging = {k: v for k, v in part.items() if k in ("built", "stage")}
            reference = kept(finished, full=True)
            if part.get("built", 1.0) < 1.0 and not staged(part["source"]) and reference:
                top = max(p[2] for _, _, vertices, _ in reference for p, _, _ in vertices)
                pieces = kept({**finished, "keep": {**part.get("keep", {}), "clip_z": part["built"] * top}}, full=True)
            else:
                pieces = kept({**finished, **staging}, full=True) if staging else reference
            if not pieces:
                continue
            sources.append(part["source"])
            span = max(high[0] - low[0], high[1] - low[1], 1e-6)
            facing = cell.get("facing", math.pi)
            model, bounds = city_body(library, part, pieces, cx, cy, scale, facing)
            effects = []
            if part.get("built", 1.0) >= 1.0 and part.get("stage", 1.0) >= 1.0 and not prop(part["source"]):
                try:
                    effects = effect_records(part, finished, cx, cy, scale, facing)
                except ValueError as exc:   # a package the trigger reader cannot open (some Workshop mods)
                    print(f"{part['source']}: no effects ({exc})", flush=True)
            bodies.append((model, bounds, effects, []))
            report.setdefault(cell["name"], {"label": cell["label"], "parts": []})["parts"].append(
                {"source": part["source"], "scale": round(scale, 3), "span_tiles": round(span * scale, 3),
                 "height_tiles": round(high[2] * scale, 3), "effects": len(effects)})
        report.setdefault(cell["name"], {"label": cell["label"], "parts": []})
        report[cell["name"]].update({"instances": len(instances) + len(bodies), "missing": [], "sources": sources})
        compositions.append((cell["name"], [(dk.ALL_TERRAINS, instances)]))
        if bodies:
            library.composition(cell["name"], bodies)
    write_pack(catalog["id"], meshes, compositions)
    library.save()
    summary = {"catalog": catalog["id"], "meshes": len(meshes), "cells": report}
    (CATALOG / "composition_report.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def marker_mesh(cache: Path) -> dict:
    """A one-triangle mesh for the cell's marker body: "clearance/" bodies are
    layout rules the renderer never draws, so the tile shows only its decals."""
    mesh = plot_mesh("Ancient", .36, cache)
    return {**mesh, "vertices": [([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [.5, .5]), ([.001, 0.0, 0.0], [0.0, 0.0, 1.0], [.5, .5]),
                                 ([0.0, .001, 0.0], [0.0, 0.0, 1.0], [.5, .5])], "indices": [0, 1, 2],
            "low": [0.0, 0.0, 0.0], "high": [.001, .001, 0.0]}


def paired_atlases(pairs: list) -> tuple[list, dict]:
    """Opaque model atlases whose upper half holds albedo cells and whose lower half
    holds each cell's emissive map at the same place (black where there is none):
    the Lab night shader samples uv + (0, .5) for a composition body's lights."""
    size, levels, block, dxgi_out, accepted = rc.MODEL_ATLAS
    per_row = size // rc.CELL
    per_atlas = per_row * per_row // 2
    atlases, placement = [], {}
    for start in range(0, len(pairs), per_atlas):
        layout = rc.mip_offsets(size, size, levels, block)
        blob = bytearray(layout[-1][0] + layout[-1][1] * layout[-1][2] * block)
        for slot, (albedo, emissive) in enumerate(pairs[start:start + per_atlas]):
            cx, cy = slot % per_row, slot // per_row
            for path, row in ((albedo, cy), (emissive, cy + per_row // 2)):
                if path is None:
                    continue
                width, height, mips, dxgi, payload = rc.dds(path)
                full_chain = (width // 4).bit_length()
                if dxgi not in accepted or width != height or width not in (128, 256, 512) or \
                        mips < min(levels, full_chain):
                    raise ValueError(f"Atlas source must be square 128-512 with a full mip chain: {path}")
                source = rc.mip_offsets(width, height, mips, block)
                for level in range(levels):
                    origin, atlas_bw, _ = layout[level]
                    offset, bw, bh = source[min(level, len(source) - 1, full_chain - 1)]
                    base_x, base_y = cx * (rc.CELL >> level) // 4, row * (rc.CELL >> level) // 4
                    for r in range(bh):
                        target = origin + ((base_y + r) * atlas_bw + base_x) * block
                        blob[target:target + bw * block] = payload[offset + r * bw * block: offset + (r + 1) * bw * block]
            width = rc.dds(albedo)[0]
            placement[(albedo, emissive)] = (len(atlases), cx * rc.CELL / size, cy * rc.CELL / size, width / size)
        atlases.append(rc.dds_header(size, size, levels, dxgi_out, layout[0][1] * layout[0][2] * block) + bytes(blob))
    return atlases, placement


def write_pack(catalog_id: str, meshes: dict, compositions: list) -> None:
    """The Lab composition pack (the district study's format: Renderer64's
    resource-composition bundle with its own atlases)."""
    if len(meshes) > 256:
        raise ValueError(f"{catalog_id}: {len(meshes)} meshes exceed the runtime asset limit; split the catalog")
    opaque = list(dict.fromkeys((mesh["texture"], mesh.get("emissive")) for mesh in meshes.values()
                                if not mesh["masked"]))
    cutout = list(dict.fromkeys(mesh["texture"] for mesh in meshes.values() if mesh["masked"]))
    atlases, pair_place = paired_atlases(opaque)
    place = {texture: where for (texture, _emissive), where in pair_place.items()}
    for mesh in meshes.values():
        if not mesh["masked"]:
            mesh["place"] = pair_place[(mesh["texture"], mesh.get("emissive"))]
    if cutout:
        more, cut_place = rc.pack_atlases(cutout, rc.MASKED_ATLAS, len(atlases))
        atlases += more
        place.update(cut_place)
    if len(atlases) > rc.TEXTURE_SLOTS:
        raise ValueError(f"{catalog_id}: {len(atlases)} atlases exceed the texture slots; split the catalog")
    slots = [f"textures/atlas_{i}.dds" for i in range(len(atlases))]
    padded = slots + ["textures/unused.dds"] * (rc.TEXTURE_SLOTS - len(slots))
    index, payloads = {}, []
    for key, mesh in meshes.items():
        atlas, u0, v0, extent = mesh.get("place") or place[mesh["texture"]]
        edge = .5 / (extent * rc.MODEL_ATLAS[0])
        vertices = [(p, n, (u0 + min(1 - edge, max(edge, uv[0])) * extent,
                            v0 + min(1 - edge, max(edge, uv[1])) * extent)) for p, n, uv in mesh["vertices"]]
        index[key] = len(payloads)
        payloads.append(rc.mesh_payload(key, atlas, vertices, mesh["indices"]))
    blob = bytearray(rc.MAGIC)
    blob.extend(struct.pack("<IIII", 2, rc.TEXTURE_SLOTS, len(payloads), 0))
    for texture in padded:
        blob.extend(rc.bundle_string(texture))
    for payload in payloads:
        blob.extend(payload)
    blob.extend(struct.pack("<I", len(compositions)))
    for name, variants in compositions:
        blob.extend(rc.bundle_string(name))
        blob.extend(struct.pack("<I", len(variants)))
        for mask, instances in variants:
            blob.extend(struct.pack("<II", mask, len(instances)))
            for item in instances:
                blob.extend(struct.pack("<I6f", index[item["model"]], item["u"], item["v"], item["rotation"],
                                        item["scale"], item["lift"], item["ground_fit"]))
    blob.extend(struct.pack("<I", len(compositions)))
    for position, (name, _) in enumerate(compositions):
        blob.extend(rc.bundle_string(name))
        blob.extend(struct.pack("<I", position))
    if CATALOG.exists():
        shutil.rmtree(CATALOG)
    (CATALOG / "textures").mkdir(parents=True)
    for position, data in enumerate(atlases):
        (CATALOG / f"textures/atlas_{position}.dds").write_bytes(data)
    (CATALOG / "textures/unused.dds").write_bytes(rc.dds_header(4, 4, 1, rc.MODEL_ATLAS[3], 8) + bytes(8))
    (CATALOG / "resource_runtime.bin").write_bytes(blob)


class _NoCompiler:
    """dk.render builds and saves a district Compiler; this study bakes from its own sub-packs."""
    def save(self) -> None:
        pass


# Lab night lights for composition bodies (resource_body.hlsl): the body's emissive
# map sits in its atlas's lower half (paired_atlases); at night it is added like a
# city's windows (night activation x emissive scale x 1.45). Lab scenes hold no
# other composition bodies, so the patch is safe in every frame of this study.
EMISSIVE_EDITS = (
    ("    else { albedo=0; emissive=0; }\n",
     "    else { albedo=0; emissive=0; }\n"
     "    if (resource_weight > 0.5)   // Lab wonders: emissive under the albedo cell\n"
     "        emissive = sample_reused_resource_slot(input.material_index - 21.0,\n"
     "            float2(input.uv.x, input.uv.y + 0.5)).rgb;\n"),
    ("saturate(city_weight + mine_weight + raised_infrastructure_weight) *",
     "saturate(city_weight + mine_weight + raised_infrastructure_weight + resource_weight) *"),
)
SHADER_PARTS = ("Renderer/native", "Renderer/lab/shared", "Renderer/sandbox")
# Lab lamps for Civ VI light effects (effects.py) in the city effect layer: a lamp
# (7) or blinker (8); 90 + kind stays below the body materials (100+). Seed 0-15 warm, 16-31 white, 32-47 red, 48-63 green; lit
# at night, a faint point by day.
LAMP_EDIT = (
    " float2 g=float2(q.x,(q.y-.5)*2);\n float glow=exp(-dot(g,g)*3)*night*strength;\n",
    " if(kind>6.5 && kind<8.5){\n"
    "  float band=floor(seed/16);\n"
    "  float3 tint=band<.5?float3(1,.68,.32):band<1.5?float3(1,.93,.8):band<2.5?float3(1,.16,.08):float3(.4,1,.45);\n"
    "  float on=kind>7.5?step(.45,frac(Q8_CITY_TIME*.6+seed*.137)):1;\n"
    "  float2 l=float2(q.x,(q.y-.5)*2);float r2=dot(l,l);\n"
    "  float lamp=(exp(-r2*40)*1.4+exp(-r2*5)*.35*night)*strength*on*(.25+.75*night);\n"
    "  clip(lamp-.004);\n"
    "  return float4(tint*(1.2+2.5*lamp),saturate(lamp));\n"
    " }\n"
    " float2 g=float2(q.x,(q.y-.5)*2);\n float glow=exp(-dot(g,g)*3)*night*strength;\n")


@lru_cache(maxsize=None)
def shader_root(season: int) -> Path:
    """A private copy of the checkout's runtime shaders with the Lab night-light
    patch, its season pinned as the natural-wonder study pins it (seasons.py, read
    only). The checkout is never written."""
    from Renderer.lab.shared.seasons import seasons
    root = OUT / "roots" / f"root-{SEASON_NAMES[season]}"
    if root.exists():
        shutil.rmtree(root)
    for part in SHADER_PARTS:
        for path in (ROOT / part).rglob("*"):
            relative = path.relative_to(ROOT)
            if path.is_file() and path.suffix.lower() in (".hlsl", ".hlsli", ".h", ".json") and \
                    not {"build", ".cache", "__pycache__"} & set(relative.parts):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    # The shared material code is generated into every city_fidelity shader that
    # draws bodies (resource_body, city, feature, rigid_feature, ...): patch each.
    patched = []
    for body in sorted((root / "Renderer/native").rglob("*.hlsl")):
        text = body.read_text()
        if all(text.count(old) == 1 for old, _new in EMISSIVE_EDITS):
            for old, new in EMISSIVE_EDITS:
                text = text.replace(old, new)
            body.write_text(text)
            patched.append(body.name)
    if "resource_body.hlsl" not in patched:
        raise ValueError("resource_body.hlsl changed; update the wonder night patch")
    lamps = []
    for path in sorted((root / "Renderer").rglob("*.hlsl")):
        text = path.read_text()
        if "float4 q8_city_effect(FeaturePixelInput p){" in text:
            if text.count(LAMP_EDIT[0]) != 1:
                raise ValueError(f"{path.name}: city effect glow changed; update the wonder lamp patch")
            path.write_text(text.replace(*LAMP_EDIT))
            lamps.append(path.name)
    if not lamps:
        raise ValueError("no city effect shader found for the wonder lamps")
    if season:
        for relative in seasons.TARGETS:
            path = root / relative
            if not path.is_file() or relative.startswith("Renderer/sandbox/"):
                continue                     # game-only open water (the Lab draws q3 water)
            text = seasons.apply(relative, path.read_text())
            for index in ("int(C3XSeason.x + 0.5)", "int(game_resource_look.z + 0.5)"):
                text = text.replace(f"#define C3X_SEASON_INDEX {index}", f"#define C3X_SEASON_INDEX {season}")
            if f"#define C3X_SEASON_INDEX {season}" not in text and "C3X_SEASON_INDEX" in text:
                raise ValueError(f"{relative}: season index define not pinned")
            path.write_text(text)
    return root


def render(catalogs: dict, zooms, only=None) -> None:
    """dk.render with this study's bake, once per season among the catalogs' shots
    (each shot's `season`, Summer by default), through this study's shader root.
    A frame whose catalog changed since it was drawn is removed first."""
    from Renderer import renderer
    for key, spec in catalogs.items():
        if only and key not in only:
            continue
        stamp = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()
        for shot in dk.shots(spec, zooms):
            folder = dk.frame(key, shot)
            marker = folder / "spec.sha1"
            if folder.is_dir() and (not marker.is_file() or marker.read_text() != stamp):
                shutil.rmtree(folder)
    dk.bake, dk.Compiler = bake, _NoCompiler
    native = renderer.native_render
    seasons_used = sorted({shot.get("season", 0) for spec in catalogs.values() for shot in dk.shots(spec, zooms)})
    try:
        for season in seasons_used:
            SEASON["index"] = season
            root = shader_root(season)
            env = {"C3X_RENDERER_SHADER_SOURCE_ROOT": "..\\..\\" + renderer.relative(root).replace("/", "\\"),
                   "C3X_RENDERER_PREVIEW_SEASON": str(season),
                   # The Lab city library: production cities plus the catalog's wonders (city_pack.py).
                   "C3X_RENDERER_CITY_PACK": renderer.relative(city_pack.PACK).replace("/", "\\")}
            # Wonder cells as city bodies need the Lab tree (effects_tree.py); its preview
            # is copied out so a rebuild of the tree never races a render.
            from Renderer.lab.studies.wonders import effects_tree
            dll, tree_preview = effects_tree.binaries()
            preview = OUT / "effects_preview.exe"
            shutil.copy2(tree_preview, preview)

            def seasonal(*args, extra_env=None, **kwargs):
                return native(*args, extra_env={**(extra_env or {}), **env}, **{**kwargs, "candidate": dll, "preview": preview})
            renderer.native_render = seasonal
            subset = {key: {**spec, "shots": [shot for shot in dk.shots(spec, zooms) if shot.get("season", 0) == season]}
                      for key, spec in catalogs.items()}
            subset = {key: spec for key, spec in subset.items() if spec["shots"]}
            dk.render(subset, zooms, only)
    finally:
        renderer.native_render = native
        SEASON["index"] = 0
    for key, spec in catalogs.items():
        stamp = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()
        for shot in dk.shots(spec, zooms):
            folder = dk.frame(key, shot)
            if (folder / "cells.json").is_file():
                (folder / "spec.sha1").write_text(stamp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("compile", "probe", "render", "crops", "list"))
    parser.add_argument("--source", action="append")
    parser.add_argument("--catalog", action="append")
    parser.add_argument("--zoom", action="append", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    chosen = args.source or keys()
    if args.action == "list":
        for key in chosen:
            print(key, *label(key), sep=" | ")
        return 0
    if args.action == "compile":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for line in pool.map(compile_one, chosen):
                print(line, flush=True)
        return 0
    if args.action == "probe":
        for key in chosen:
            try:
                print(key, json.dumps(probe(key)["materials"]), flush=True)
            except ValueError as exc:
                print(f"{key}: {exc}", file=sys.stderr, flush=True)
        return 0
    from Renderer.lab.studies.wonders import plan
    catalogs = plan.everything()
    zooms = tuple(args.zoom or (128, 256))
    if args.action == "render":
        render(catalogs, zooms, args.catalog)
    else:
        dk.crops(catalogs, zooms)
    return 0


if __name__ == "__main__":
    # One module instance: stages.py imports this module by its package name.
    from Renderer.lab.studies.wonders import kits as module
    raise SystemExit(module.main())
