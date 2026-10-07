#!/usr/bin/env python3
"""Lab gallery of Civ VI mine and quarry kits, and their parts, on grassland, hills and mountains.

Imports each kit Civ VI ships (base game kits, the older quarry kits the
ArtDefs no longer reference, Gathering Storm's mountain tunnel) whole and
renders it through the production D3D11 renderer as a Lab-only resource
composition on Civ III grassland, hills and mountains, enlarged and at Civ
VI's own size. A second set shows each signature part (entrances, buildings,
headframes, spoil, carts, quarry pits) alone, enlarged to a common footprint,
for composing a C3X mine that reads at a glance. Each attached
component keeps Civ VI's "pivot height" terrain follow: it is its own instance
standing on the ground under its pivot. Civ VI also flattens the terrain under
a mine (FlattenTerrain); C3X cannot, so slopes show unflattened ground.

Nothing here changes production packs, shared art direction or the resource
roster file: the source and catalog packs are Lab-only folders, and the Lab
batches are supplied to the roster in-process. Outputs are disposable, under
Renderer/lab/out/mines/kits/.

    python3 Renderer/lab/studies/mines/kit_gallery.py            # compile, bake, render, page
    python3 Renderer/lab/studies/mines/kit_gallery.py --page-only
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT, _package_index, _resolve_package
from Renderer.tools.asset_compiler.compound_landmark_importer import _compile_asset, _matrix_multiply
from Renderer.tools.asset_compiler.improvement_asset_importer import _asset_id, _content_root, _shared_roots
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler import build_resource_compositions as rc

SOURCES = ROOT / "Renderer/packs/MineKitSourcesLab"
CATALOG = ROOT / "Renderer/packs/MineKitCatalogLab"
OUT = ROOT / "Renderer/lab/out/mines/kits"
BASE = "Base/Platforms/Windows/BLPs/landmarks/tilebases.blp"
RISE = "DLC/Expansion1/platforms/windows/BLPs/landmarks/tilebases.blp"
STORM = "DLC/Expansion2/Platforms/Windows/BLPs/landmarks/tilebases.blp"
IDENTITY = [1.0 if row == column else 0.0 for row in range(4) for column in range(4)]
TERRAINS = ("Grassland", "Hills", "Mountains")

# Kit rows: (label, package, entry, era family, how Civ VI uses it). Rise and
# Fall's mine kits (IMP_Mine_*_EXP1, also used by Gathering Storm) are these
# base kits minus one sledge or ore cart, so they are not repeated.
# Part rows: (label, component entry, era family, note); each part is shown
# alone, scaled to a common footprint.
CATALOGS = {
    "mines": {"kind": "kit", "title": "Mine kits", "rows": [
        ("Mine AN 01", BASE, "IMP_Mine_AN_01", "Pre-industrial", "Ancient, Classical and default era"),
        ("Mine AN 02", BASE, "IMP_Mine_AN_02", "Pre-industrial", "Ancient and Classical"),
        ("Mine AN 03", BASE, "IMP_Mine_AN_03", "Pre-industrial", "Ancient and Classical"),
        ("Mine IND", BASE, "IMP_Mine_IND", "Industrial", "Industrial and Modern"),
        ("Mine IND 02", BASE, "IMP_Mine_IND_02", "Industrial", "Industrial and Modern"),
        ("Mine IND 03", BASE, "IMP_Mine_IND_03", "Industrial", "Industrial and Modern"),
    ], "blurb": "Civ VI's three pre-industrial and three industrial mine kits. Civ VI picks one of the three at "
                "random per tile within the era family. Rise and Fall's versions are the same kits minus one "
                "sledge or ore cart."},
    "quarries": {"kind": "kit", "title": "Quarry kits", "rows": [
        ("Quarry ANC", BASE, "IMP_QuarryREDO_ANC", "Pre-industrial", "Every era before Industrial"),
        ("Quarry IND", BASE, "IMP_QuarryREDO_IND", "Industrial", "Industrial and Modern"),
        ("Mountain tunnel", STORM, "IMP_Mountain_Tunnel", "Industrial",
         "Gathering Storm tunnel portal; not a mine, shown as a mountain-face idea"),
    ], "blurb": "The two quarry kits Civ VI uses, one per era family, and Gathering Storm's mountain tunnel "
                "portal. The quarry's cut-stone pit is authored to sit in ground Civ VI flattens and lowers; "
                "on unflattened ground it stands up as a terraced block."},
    "quarries-old": {"kind": "kit", "title": "Older quarry kits", "rows": [
        ("Old quarry AN 01", BASE, "IMP_Quarry_AN_01", "Pre-industrial", "Older kit, not referenced by any ArtDef"),
        ("Old quarry AN 02", BASE, "IMP_Quarry_AN_02", "Pre-industrial", "Older kit, not referenced by any ArtDef"),
        ("Old quarry AN 03", BASE, "IMP_Quarry_AN_03", "Pre-industrial", "Older kit, not referenced by any ArtDef"),
        ("Old quarry IND", BASE, "IMP_Quarry_IND", "Industrial", "Older kit, not referenced by any ArtDef"),
        ("Old quarry IND 01", BASE, "IMP_Quarry_IND_01", "Industrial", "Older kit, not referenced by any ArtDef"),
    ], "blurb": "Five quarry kits Civ VI still ships but no ArtDef uses: stepped stone hills with cranes, sleds "
                "and stone blocks."},
    "parts-mine": {"kind": "part", "title": "Mine parts: entrances, headframe, spoil and carts", "rows": [
        ("Timber entrance", "IMP_MINE_ANC_Entrance", "Pre-industrial", "every pre-industrial kit"),
        ("Industrial entrance", "IMP_MINE_IND_Entrance", "Industrial", "every industrial kit"),
        ("Headframe tower", "IMP_MINE_IND_Tower", "Industrial", "Mine IND"),
        ("Large spoil heap", "IMP_MINE_Rocks_Lg", "Pre-industrial", "Mine AN 02 and IND 02"),
        ("Ore sledge", "IMP_Mine_Sledge", "Pre-industrial", "base-game AN 01 and AN 02"),
        ("Ore cart", "IMP_Mine_ANC_Car", "Pre-industrial", "every pre-industrial kit"),
        ("Loaded ore car", "IMP_Mine_IND_Car_Rocks", "Industrial", "base-game industrial kits"),
    ], "blurb": "The pieces that say \"mine\" at a distance, each alone and enlarged to a common footprint."},
    "parts-mine-buildings": {"kind": "part", "title": "Mine parts: main buildings", "rows": [
        ("Early building A", "IMP_MINE_ANC_Bld_A", "Pre-industrial", "Mine AN 01"),
        ("Early building B", "IMP_MINE_ANC_Bld_B", "Pre-industrial", "Mine AN 02"),
        ("Early building C", "IMP_MINE_ANC_Bld_C", "Pre-industrial", "Mine AN 03"),
        ("Industrial building A", "IMP_MINE_IND_Bld_A", "Industrial", "Mine IND"),
        ("Industrial building B", "IMP_MINE_IND_Bld_B", "Industrial", "Mine IND 02"),
        ("Industrial building C", "IMP_MINE_IND_Bld_C", "Industrial", "Mine IND 03"),
    ], "blurb": "The main building of every mine kit, alone and enlarged to a common footprint."},
    "parts-quarry": {"kind": "part", "title": "Quarry parts: current kits", "rows": [
        ("Quarry pit", "IMP_QuarryREDO_Terrain", "Pre-industrial", "both current quarry kits"),
        ("Quarry building A", "IMP_QuarryREDO_ANC_BuildingA", "Pre-industrial", "Quarry ANC"),
        ("Quarry building B", "IMP_QuarryREDO_ANC_BuildingB", "Pre-industrial", "Quarry ANC"),
        ("Ind. quarry building A", "IMP_QuarryREDO_IND_BuildingA", "Industrial", "Quarry IND"),
        ("Ind. quarry building B", "IMP_QuarryREDO_IND_BuildingB", "Industrial", "Quarry IND"),
        ("Water tower", "IMP_QuarryREDO_WaterTower", "Industrial", "Quarry IND"),
        ("Stepped stone hill", "IMP_Quarry_AN_Hill_LG", "Pre-industrial", "Old quarry AN 03"),
    ], "blurb": "The quarry pit, buildings and landmarks, each alone and enlarged to a common footprint."},
    "parts-quarry-old": {"kind": "part", "title": "Quarry parts: older kits", "rows": [
        ("Old ind. building A", "IMP_Quarry_IND_BldgA", "Industrial", "old industrial quarries"),
        ("Old ind. building B", "IMP_Quarry_IND_BldgB", "Industrial", "old industrial quarries"),
        ("Quarry truck", "IMP_Quarry_IND_Truck", "Industrial", "old industrial quarries"),
        ("Stone ramp", "IMP_Quarry_IND_Ramp", "Industrial", "Old quarry IND"),
        ("Stone workshop", "IMP_Quarry_AN_Bldg", "Pre-industrial", "old pre-industrial quarries"),
        ("Lean-to", "IMP_Quarry_AN_Leanto", "Pre-industrial", "old pre-industrial quarries"),
        ("Industrial crane", "PROP_Crane_IND_LG", "Industrial", "old industrial quarries"),
    ], "blurb": "Parts of the unused older quarry kits, each alone and enlarged to a common footprint."},
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def compile_kits(assets_root: Path = DEFAULT_ASSETS_ROOT) -> dict[str, str]:
    """Normalize every kit and its attached components into the Lab source pack.
    Returns kit entry -> root asset id. Resource-conditional parts stay out."""
    packages = _package_index(assets_root)
    package_bytes: dict[str, bytes] = {}
    opened: dict[str, IndexedStaticPackage] = {}
    texture_cache: dict = {}
    assets: dict[str, dict] = {}
    report = {"skipped": [], "optional_failures": []}

    def ensure(package_relative: str, entry: str, stack: tuple = ()) -> str:
        asset_id = _asset_id(package_relative, entry)
        if asset_id in assets:
            return asset_id
        if asset_id in stack:
            raise ValueError(f"component cycle at {entry}")
        package = opened.get(package_relative) or IndexedStaticPackage(assets_root / package_relative, entry)
        opened[package_relative] = package
        try:
            manifest_asset, evidence = _compile_asset(package, _shared_roots(assets_root, package_relative), SOURCES,
                                                      entry, asset_id, 100.0, texture_cache,
                                                      terrain_edit_policy="preserve_unresolved")
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            raise ValueError(f"{entry}: {exc}") from exc
        document_path = SOURCES / manifest_asset["landmark"]
        document = load(document_path)
        source_points = {point["id"]: point for point in evidence["attachments"]["points"]}
        for point in document["attachment_points"]:
            if point["binding_status"] == "source_condition_unmapped":
                report["skipped"].append({"kit_part": entry, "reason": "resource-conditional"})
            if point["binding_status"] != "component_unresolved":
                continue
            terminal = source_points[point["id"]]["component_source"]
            resolution = _resolve_package(packages, terminal["package"], _content_root(package_relative),
                                          terminal["entry"], package_bytes)
            try:
                if resolution["status"] != "resolved":
                    raise ValueError(f"unresolved child {terminal['entry']}")
                try:
                    point["component_asset"] = ensure(resolution["package_path"], terminal["entry"],
                                                      stack + (asset_id,))
                except ValueError:
                    # An expansion package names a base-game part without defining it
                    # (the expansion depends on the base content): use the base part.
                    if resolution["package_path"] == BASE:
                        raise
                    point["component_asset"] = ensure(BASE, terminal["entry"], stack + (asset_id,))
                point["binding_status"] = "resolved"
            except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                if point.get("selection", {}).get("cull") != "optional":
                    raise
                point["binding_status"] = "component_compile_unresolved"
                report["optional_failures"].append({"parent": entry, "child": terminal["entry"], "reason": str(exc)})
        document_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        assets[asset_id] = {**manifest_asset, "source_entry": entry}
        return asset_id

    roots = {}
    for spec in CATALOGS.values():
        for _label, package, entry, _family, _use in spec["rows"] if spec["kind"] == "kit" else ():
            try:
                roots[entry] = ensure(package, entry)
            except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                report.setdefault("kit_failures", []).append({"kit": entry, "reason": str(exc)})
                print(f"skipped {entry}: {exc}", file=sys.stderr)
    (SOURCES / "manifest.json").write_text(json.dumps(
        {"schema": "c3x.asset_pack.v0", "name": "MineKitSourcesLab",
         "source_policy": "Local licensed-source import; derived art is not redistributable.",
         "assets": dict(sorted(assets.items())), "roots": roots}, indent=2, sort_keys=True) + "\n")
    (SOURCES / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return roots


def components(asset_id: str, transform: list[float], manifest: dict, found: list) -> list:
    """Every component instance of a kit with its kit-space transform (bind pose)."""
    from Renderer.preview.render_improvement_sheet import _skeleton_worlds
    landmark = load(SOURCES / manifest["assets"][asset_id]["landmark"])
    found.append((asset_id, landmark, transform))
    worlds = [_skeleton_worlds(load(SOURCES / path)) for path in landmark["components"]["skeletons"]]
    for point in landmark["attachment_points"]:
        if point["binding_status"] == "resolved":
            components(point["component_asset"], _matrix_multiply(worlds[point["skeleton"]][point["bone"]], transform),
                       manifest, found)
    return found


def yaw_scale(transform: list[float]) -> tuple[float, float] | None:
    """(yaw, scale) when a row-vector transform is an upright rotation with uniform scale."""
    scale = math.hypot(transform[0], transform[1])
    if scale <= 0 or any(abs(transform[i]) > 1e-3 * scale for i in (2, 6, 8, 9)) or \
            abs(transform[10] - scale) > 1e-3 * scale or abs(transform[5] - transform[0]) > 1e-3 * scale or \
            abs(transform[4] + transform[1]) > 1e-3 * scale:
        return None
    return math.atan2(transform[1], transform[0]), scale


def square(path: Path, cache: Path) -> tuple[Path, float, float]:
    """An atlas-ready square BC1 texture no larger than a cell: a wide texture
    repeats its rows to fill the square (block copies, no recompression).
    Returns the texture and the UV scale that maps the original into it."""
    width, height, mips, dxgi, payload = rc.dds(path)
    if width == height:
        return rc.fitted(path, cache), 1.0, 1.0
    layout = rc.mip_offsets(width, height, mips, 8)
    skip = 0
    while max(width, height) >> skip > rc.CELL:
        skip += 1
    size = max(width, height) >> skip
    levels = []
    for level in range(size.bit_length() - 2):   # down to one 4x4 block
        offset, bw, bh = layout[min(skip + level, len(layout) - 1)]
        side = max(1, (size >> level) // 4)
        rows = [payload[offset + row * bw * 8: offset + (row + 1) * bw * 8] for row in range(bh)]
        levels.append(b"".join((rows[row % bh] * (side // bw + 1))[:side * 8] for row in range(side)))
    target = rc.write_dds(cache / f"square_{path.name}", size, len(levels), dxgi, 8, levels)
    return target, min(1.0, width / height), min(1.0, height / width)


PART_FOOTPRINT = .3   # a part alone is enlarged until its half extent is this many tiles


def bake(catalog: str, layouts: dict) -> dict:
    """Write the Lab composition pack for one catalog: every kit component a
    separate instance (its own ground contact), one variant per terrain."""
    manifest = load(SOURCES / "manifest.json")
    by_entry = {asset["source_entry"]: asset_id for asset_id, asset in manifest["assets"].items()}
    spec = CATALOGS[catalog]
    cache = OUT / "texture_cache"
    meshes: dict[str, dict] = {}   # asset id -> texture, vertices, indices
    compositions, report = [], {}
    for row in spec["rows"]:
        label, entry = row[0], row[2] if spec["kind"] == "kit" else row[1]
        if spec["kind"] == "kit":
            found = components(manifest["roots"][entry], IDENTITY, manifest, [])
        else:
            found = [(by_entry[entry], load(SOURCES / manifest["assets"][by_entry[entry]]["landmark"]), IDENTITY)]
        parts = []   # (asset id, pivot xyz, yaw, scale, radius)
        for component, landmark, transform in found:
            fitted = yaw_scale(transform)
            linear = IDENTITY if fitted else transform[:12] + [0.0, 0.0, 0.0, 1.0]
            for binding in landmark["draw_bindings"]:
                if "worked" not in binding["states"]:
                    continue
                geometry = landmark["components"]["geometry"][binding["geometry"]]
                material = load(SOURCES / landmark["components"]["materials"][binding["material"]])
                colour = material["channels"]["base_color"]["texture"]
                key = f"minekit/{geometry.rsplit('/', 1)[-1].removesuffix('.json')}"
                if not fitted:
                    key += "/" + format(abs(hash(tuple(round(v, 4) for v in linear))) % 16 ** 8, "08x")
                if key not in meshes:
                    mesh = load(SOURCES / geometry)
                    vertices = []
                    for vertex in mesh["vertices"]:
                        x, y, z = vertex["position"]
                        nx, ny, nz = vertex["normal"]
                        position = [x * linear[0] + y * linear[4] + z * linear[8],
                                    x * linear[1] + y * linear[5] + z * linear[9],
                                    x * linear[2] + y * linear[6] + z * linear[10]]
                        normal = [nx * linear[0] + ny * linear[4] + nz * linear[8],
                                  nx * linear[1] + ny * linear[5] + nz * linear[9],
                                  nx * linear[2] + ny * linear[6] + nz * linear[10]]
                        length = math.sqrt(sum(value * value for value in normal)) or 1.0
                        vertices.append((position, [value / length for value in normal], list(vertex["uv0"])))
                    indices = mesh["topology"]["indices"]
                    if any(value < 0 or value > 1 for _, _, uv in vertices for value in uv):
                        vertices, indices = rc.within_texture(vertices, indices)
                    texture, su, sv = square(SOURCES / colour, cache)
                    vertices = [(p, n, [uv[0] * su, uv[1] * sv]) for p, n, uv in vertices]
                    meshes[key] = {"texture": texture, "vertices": vertices, "indices": indices,
                                   "radius": max(math.hypot(p[0], p[1]) for p, _, _ in vertices)}
                yaw, scale = fitted or (0.0, 1.0)
                parts.append((key, transform[12:15], yaw, scale, meshes[key]["radius"] * scale))
        factor, ox, oy = 1.0, 0.0, 0.0
        if spec["kind"] == "part":
            # A part's pivot can sit at one end; centre its footprint on the tile
            # and size it by its half extent.
            points = [(px + (x * math.cos(yaw) - y * math.sin(yaw)) * scale,
                       py + (x * math.sin(yaw) + y * math.cos(yaw)) * scale)
                      for key, (px, py, _), yaw, scale, _ in parts for (x, y, _), _, _ in meshes[key]["vertices"]]
            low = [min(point[axis] for point in points) for axis in (0, 1)]
            high = [max(point[axis] for point in points) for axis in (0, 1)]
            ox, oy = (low[0] + high[0]) / 2, (low[1] + high[1]) / 2
            reach = max(high[0] - low[0], high[1] - low[1]) / 2
            factor = min(6.0, max(1.0, round(PART_FOOTPRINT / reach * 2) / 2))
            parts = [(key, (px - ox, py - oy, pz), yaw, scale, radius)
                     for key, (px, py, pz), yaw, scale, radius in parts]
        variants = []
        for terrain in TERRAINS:
            layout = layouts[terrain]
            size, (cu, cv), facing = layout["scale"] * factor, layout["centre"], layout["facing"]
            cosine, sine = math.cos(facing), math.sin(facing)
            instances = []
            for key, (px, py, pz), yaw, scale, radius in parts:
                instances.append({"model": key, "u": cu + (px * cosine - py * sine) * size,
                                  "v": cv + (px * sine + py * cosine) * size, "rotation": facing + yaw,
                                  "scale": size * scale, "lift": pz * size, "ground_fit": 0.0})
            variants.append((1 << rc.TERRAIN_INDEX[terrain], instances))
        compositions.append((label, variants))
        report[label] = {"entry": entry, "instances": len(parts), "factor": factor}
    if len(meshes) > 256:
        raise ValueError(f"{catalog}: {len(meshes)} meshes exceed the runtime asset limit")
    textures = list(dict.fromkeys(mesh["texture"] for mesh in meshes.values()))
    atlases, place = rc.pack_atlases(textures, rc.MODEL_ATLAS)
    if len(atlases) > rc.TEXTURE_SLOTS:
        raise ValueError(f"{catalog}: {len(atlases)} atlases exceed the texture slots")
    slots = [f"textures/atlas_{i}.dds" for i in range(len(atlases))]
    padded = slots + ["textures/unused.dds"] * (rc.TEXTURE_SLOTS - len(slots))
    index, payloads = {}, []
    for key, mesh in meshes.items():
        atlas, u0, v0, extent = place[mesh["texture"]]
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
        import shutil
        shutil.rmtree(CATALOG)
    (CATALOG / "textures").mkdir(parents=True)
    for position, data in enumerate(atlases):
        (CATALOG / f"textures/atlas_{position}.dds").write_bytes(data)
    (CATALOG / "textures/unused.dds").write_bytes(rc.dds_header(4, 4, 1, rc.MODEL_ATLAS[3], 8) + bytes(8))
    (CATALOG / "resource_runtime.bin").write_bytes(blob)
    summary = {"catalog": catalog, "meshes": len(meshes), "textures": len(textures), "kits": report}
    (CATALOG / "composition_report.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


# Kits face Civ VI's camera (their -y side) toward C3X's viewer, the +u+v
# diagonal. Each part stands on the ground under its own pivot, as Civ VI's
# "pivot height" does. Hills centre the kit on the crown; on mountains it stands
# at the camera-facing foot, as the earlier mine study placed it. "authored" is
# Civ VI's own size (one hex = one tile): its 3D parts cover only the middle of
# the tile, because Civ VI fills the rest with ground decals. "large" is the
# readable trial; parts multiply it by their own enlargement.
FACING = 3 * math.pi / 4
LAYOUT_SETS = {
    "large": {"Grassland": {"centre": (.5, .5), "scale": 2.0, "facing": FACING},
              "Hills": {"centre": (.5, .5), "scale": 1.8, "facing": FACING},
              "Mountains": {"centre": (.78, .78), "scale": 1.4, "facing": FACING}},
    "authored": {"Grassland": {"centre": (.5, .5), "scale": 1.0, "facing": FACING},
                 "Hills": {"centre": (.5, .5), "scale": 1.0, "facing": FACING},
                 "Mountains": {"centre": (.72, .72), "scale": 1.0, "facing": FACING}},
    "part": {"Grassland": {"centre": (.5, .5), "scale": 1.0, "facing": FACING},
             "Hills": {"centre": (.5, .5), "scale": .9, "facing": FACING},
             "Mountains": {"centre": (.78, .78), "scale": .75, "facing": FACING}},
}
RENDERS = {"kit": (("large", 256), ("large", 128), ("authored", 256)), "part": (("part", 256), ("part", 128))}


def lab_cases() -> list[str]:
    """Make the catalogs Lab roster batches for this process only (one row per
    kit, one column per terrain); the shared roster file is left alone."""
    from Renderer.lab.studies.resources import roster
    for spec in CATALOGS.values():
        # The renderer passes resource names through a 24-byte field; seven rows
        # keep each catalog's top row clear of the 32-tile Lab map's edge.
        if len(spec["rows"]) > 7 or any(len(row[0]) > 23 for row in spec["rows"]):
            raise ValueError(f"{spec['title']}: at most 7 rows with names of 23 characters or fewer")
    original = roster.native()
    data = {**original,
            "batches": {**original["batches"], **{f"catalog-{name}": [row[0] for row in spec["rows"]]
                                                  for name, spec in CATALOGS.items()}},
            "batch_terrains": {**original.get("batch_terrains", {}),
                               **{f"catalog-{name}": list(TERRAINS) for name in CATALOGS}}}
    roster.native = lambda: data
    roster.native_layout.cache_clear()
    roster.CASES = roster.cases()
    return [f"native-catalog-{name}" for name in CATALOGS]


def frame(case: str, tag: str, zoom: int) -> Path:
    return OUT / f"{case}-{tag}-z{zoom}"


def render(only=None) -> None:
    from Renderer import renderer
    cases = lab_cases()
    renderer.prepare_sources(["resources"])
    renderer.ensure_candidate(["resources"])
    for case in cases:
        catalog = case.removeprefix("native-catalog-")
        if only and catalog not in only:
            continue
        renders = RENDERS[CATALOGS[catalog]["kind"]]
        for tag in dict.fromkeys(tag for tag, _ in renders):
            summary = bake(catalog, LAYOUT_SETS[tag])
            print(json.dumps(summary), flush=True)
            for zoom in (zoom for render_tag, zoom in renders if render_tag == tag):
                renderer.native_render("resources", case, 12, zoom, frame(case, tag, zoom),
                                       resource_pack=CATALOG.name)
                (frame(case, tag, zoom) / "factors.json").write_text(json.dumps(summary["kits"], indent=1) + "\n")
                census = (frame(case, tag, zoom) / "native.log").read_text(errors="replace")
                missing = [row[0] for row in CATALOGS[catalog]["rows"] if f" {row[0]} replaced=1" not in census]
                if missing:
                    raise ValueError(f"{case} {tag} z{zoom}: no composition drawn for {missing}")


def jpeg_uri(png: Path) -> str:
    import subprocess
    target = png.with_suffix(".jpg")
    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "86", str(png), "--out", str(target)],
                   check=True, capture_output=True)
    return "data:image/jpeg;base64," + base64.b64encode(target.read_bytes()).decode()


def page() -> Path:
    from Renderer.lab.studies.resources import audit, roster
    cases = lab_cases()
    crops = OUT / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    factors = {}
    sections = []
    for case in cases:
        catalog = case.removeprefix("native-catalog-")
        spec = CATALOGS[catalog]
        renders = RENDERS[spec["kind"]]
        if spec["kind"] == "part":
            report = load(frame(case, "part", 256) / "factors.json")
            factors.update({label: item["factor"] for label, item in report.items()})
        images = {(tag, zoom): audit.read_bmp(frame(case, tag, zoom) / f"{case}-h12-z{zoom}.bmp")
                  for tag, zoom in renders}
        overview = crops / f"{catalog}-overview.png"
        audit.write_png(overview, *images[renders[1]])
        cells: dict[tuple, str] = {}
        for dx, dy, name, terrain in roster.placements(case):
            for (tag, zoom), image in images.items():
                cx, cy = image[0] // 2 + dx * zoom // 2, image[1] // 2 + dy * zoom // 4
                piece = audit.crop(image, cx - zoom * 5 // 8, cy - zoom * 13 // 16, zoom * 5 // 4, zoom * 17 // 16)
                slug = "".join(c if c.isalnum() else "-" for c in f"{name}-{terrain}-{tag}-{zoom}".lower())
                audit.write_png(crops / f"{slug}.png", *piece)
                cells[(name, terrain, tag, zoom)] = jpeg_uri(crops / f"{slug}.png")

        def strip(label: str, tag: str, zoom: int, width: int, caption: bool) -> str:
            return "".join(
                f'<figure><img src="{cells[(label, terrain, tag, zoom)]}" alt="{html.escape(label)} on '
                f'{terrain.lower()}" width="{width}">' + (f"<figcaption>{terrain}</figcaption>" if caption else "")
                + "</figure>" for terrain in TERRAINS)
        cards = []
        for row in spec["rows"]:
            label, family = row[0], row[3] if spec["kind"] == "kit" else row[2]
            entry, note = (row[2], row[4]) if spec["kind"] == "kit" else (row[1], "Used in " + row[3])
            unused = "not referenced" in note or "not a mine" in note
            badges = (f'<span class="badge {family.split("-")[0].lower()}">{family}</span>' +
                      (f'<span class="badge {"unused" if unused else "now"}">'
                       f'{"Shipped, unused" if unused else "Used by Civ VI"}</span>' if spec["kind"] == "kit" else
                       f'<span class="badge scale">{factors[label]:g}&times; Civ VI size on grassland</span>'))
            views = (f'<div class="views">{strip(label, renders[0][0], 256, 320, True)}</div>'
                     f'<div class="views small"><span class="muted">Gameplay zoom, same size</span>'
                     f'{strip(label, renders[1][0], 128, 160, False)}</div>')
            if spec["kind"] == "kit":
                views += (f'<div class="views small"><span class="muted">Civ VI authored size, close-up</span>'
                          f'{strip(label, "authored", 256, 160, False)}</div>')
            cards.append(f'<article class="card"><header><h3>{html.escape(label)}</h3>{badges}</header>'
                         f'<p>{html.escape(note)} · <code>{entry}</code></p>{views}</article>')
        sections.append(
            f'<section id="{catalog}"><h2>{html.escape(spec["title"])}</h2>'
            f'<p class="muted">{html.escape(spec["blurb"])}</p>'
            f'<details><summary>Whole Lab frame at gameplay zoom</summary>'
            f'<img class="overview" src="{jpeg_uri(overview)}" alt="{html.escape(spec["title"])} overview">'
            f'</details><div class="grid">{"".join(cards)}</div></section>')
    contents = " · ".join(f'<a href="#{name}">{html.escape(spec["title"])}</a>' for name, spec in CATALOGS.items())
    target = OUT / "gallery.html"
    target.write_text(PAGE.replace("{contents}", contents).replace("{sections}", "".join(sections)))
    return target


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Civ VI Mine Kits</title><style>
:root{--bg:#f6f5f1;--fg:#1f2421;--muted:#5d665f;--card:#fff;--line:#dcdcd4;--pre:#8a5a12;--prebg:#f6ead6;
--ind:#2f5d8a;--indbg:#dfeaf6;--ok:#2e7d4f;--okbg:#e3f2e8;--no:#6b6f6c;--nobg:#ecece8}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161a18;--fg:#e4e8e3;--muted:#9aa49c;
--card:#1f2421;--line:#333b36;--pre:#e8b86a;--prebg:#3a2c16;--ind:#8fbbe8;--indbg:#1a2a3a;--ok:#7fd3a0;
--okbg:#1d3a29;--no:#b4bab5;--nobg:#2a2f2c}}
:root[data-theme="dark"]{--bg:#161a18;--fg:#e4e8e3;--muted:#9aa49c;--card:#1f2421;--line:#333b36;--pre:#e8b86a;
--prebg:#3a2c16;--ind:#8fbbe8;--indbg:#1a2a3a;--ok:#7fd3a0;--okbg:#1d3a29;--no:#b4bab5;--nobg:#2a2f2c}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:19px;margin:36px 0 4px}p{max-width:900px}.muted{color:var(--muted)}code{font-size:12px}
ul{max-width:900px;padding-left:20px}details{margin:8px 0 14px}summary{cursor:pointer;color:var(--muted)}
.overview{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line);margin-top:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,1010px),1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card header{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.card h3{margin:0;font-size:17px}
.card p{margin:6px 0 10px;color:var(--muted)}
.badge{font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px}
.badge.pre{color:var(--pre);background:var(--prebg)}.badge.industrial{color:var(--ind);background:var(--indbg)}
.badge.now{color:var(--ok);background:var(--okbg)}.badge.unused{color:var(--no);background:var(--nobg)}
.views{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end}.views figure{margin:0}
.views img{display:block;border-radius:6px;max-width:100%;height:auto}
figcaption{font-size:12px;color:var(--muted);margin-top:2px}
.views.small{margin-top:8px;align-items:center}.views.small .muted{font-size:12px;width:100%}
.badge.scale{color:var(--muted);background:var(--nobg)}nav{margin:10px 0}nav a{color:inherit}
</style></head><body><main>
<h1>Civ VI mines and quarries in C3X</h1>
<div class="muted">Lab render through the production D3D11 renderer · Lab-only pack MineKitCatalogLab · not
promoted · {date}</div>
<p>Every Civ VI mine and quarry kit, and the parts they are built from, set on Civ III grassland, hills and
mountains at noon. This is raw material for a C3X mine that reads as a mine at a glance, as Civ III's dark
timbered pit mouth does.</p>
<nav class="muted">{contents}</nav>
<ul>
<li><b>Size.</b> Kits are shown enlarged (2&times; on grassland, 1.8&times; on hills, 1.4&times; at the mountain
foot) so their buildings read at Civ III's distance; the bottom strip of each kit card is Civ VI's own size
(one hex = one tile). At that size the 3D parts cover only the middle of the tile, because Civ VI surrounds them
with dirt decals and views them from a much closer camera. Each part card states its own enlargement: parts are
scaled to a common footprint so small pieces such as the entrances can be judged.</li>
<li><b>Eras.</b> Civ VI has two art families. Its ArtDefs use the pre-industrial kits for Ancient, Classical and
the default era, and the industrial kits for Industrial and Modern. The current C3X mines follow the same split:
Civ III Ancient and Middle Ages use pre-industrial, Industrial and Modern use industrial.</li>
<li><b>Ground contact.</b> Each part stands on the ground under its own pivot, as Civ VI's "pivot height" rule
does. Civ VI also flattens the ground under a mine; C3X does not, so on hills and mountains large buildings cut
into the slope.</li>
<li><b>Not shown.</b> Civ VI's ground decals (dirt, gravel and spoil stamps) are atlas stamps whose stamp
coordinates the current importer does not recover, so they are left out; C3X would add its own ground scar.
Parts Civ VI shows only for a specific resource are left out, as are night lights.</li>
<li><b>Mountains.</b> Civ VI allows no mine on a mountain. Here the kit stands at the camera-facing foot of the
mountain, the placement the earlier C3X mine study used; enlarged, it spills past the tile edge.</li>
</ul>
{sections}
</main></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compile-only", action="store_true", help="only import the Civ VI kits")
    parser.add_argument("--page-only", action="store_true", help="rebuild the page from existing renders")
    parser.add_argument("--catalog", action="append", choices=sorted(CATALOGS), help="render only these catalogs")
    args = parser.parse_args()
    if not args.page_only:
        roots = compile_kits()
        if args.compile_only:
            print(json.dumps(roots, indent=1))
            return 0
        render(only=args.catalog)
    import datetime
    target = page()
    target.write_text(target.read_text().replace("{date}", datetime.date.today().isoformat()))
    print(f"Wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
