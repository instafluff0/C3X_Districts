#!/usr/bin/env python3
"""Civ VI district kits composed for C3X districts on the Renderer64 map (Lab only).

Civ VI builds a district from one base variant, chosen by the set of buildings
present, the art era and (rarely) the culture, plus one "hero building" model
per building, authored in place in the hex. `census()` reads every installed
Landmarks*.artdef; the catalogs in `plan.py` pick base variants and heroes for
each C3X district cell (era row x building column, or a culture/alternative
cell). Each catalog is compiled into a Lab-only source pack, baked into a
Lab-only composition pack and rendered through the production renderer as
Lab resource compositions on Civ III terrain (the same composition format
Renderer64 draws in game). Crops land in Renderer/lab/out/districts/.

Nothing here changes production packs, the shared roster file or ledgers.

    python3 Renderer/lab/studies/districts/kits.py compile
    python3 Renderer/lab/studies/districts/kits.py render [--catalog NAME ...] [--zoom 128 --zoom 256]
    python3 Renderer/lab/studies/districts/kits.py crops
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT, _package_index, _resolve_package
from Renderer.tools.asset_compiler import compound_landmark_importer as cli
from Renderer.tools.asset_compiler import clutter_blp_extractor as cbe
from Renderer.tools.asset_compiler.improvement_asset_importer import _asset_id, _content_root, _shared_roots
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler import build_resource_compositions as rc

SOURCES = ROOT / "Renderer/packs/DistrictKitSourcesLab"
CATALOG = ROOT / "Renderer/packs/DistrictKitCatalogLab"
OUT = ROOT / "Renderer/lab/out/districts"
IDENTITY = [1.0 if row == column else 0.0 for row in range(4) for column in range(4)]
# Fronts face south-west or south-east, as C3X's sprites and Renderer64's city
# buildings do: Civ VI's camera-facing (-y) side turned toward +v (SW) or +u (SE).
FACING = {"SW": math.pi, "SE": math.pi / 2, "tile": 0.0}   # "tile": kit axes are the tile u/v axes
ALL_TERRAINS = (1 << 14) - 1
# Civ VI authored size fills about a third of a Civ III tile; C3X districts fill the tile.
SCALE = float(os.environ.get("C3X_LAB_DISTRICT_SCALE", "2.5"))
PLATES = ROOT / "Renderer/packs/CityAccentsLab/ground"
CITY_STYLE = ROOT / "Renderer/lab/studies/city_readability/style.json"
# Tile-drafted districts pack their buildings inside the tile, so they are drawn
# a little smaller than Civ VI's hex ring at SCALE (2.5 x 0.8 = 2.0).
DRAFT_RATIO = .8
# Civ III era -> Civ VI art eras, most specific first (Civ VI falls back to DEFAULT).
ART_ERAS = {"Ancient": ("ARTERA_ANCIENT", "DEFAULT"),
            "Medieval": ("ARTERA_CLASSICAL", "DEFAULT", "ARTERA_ANCIENT"),
            "Industrial": ("ARTERA_INDUSTRIAL", "ARTERA_CLASSICAL", "DEFAULT"),
            "Modern": ("ARTERA_MODERN", "ARTERA_INDUSTRIAL", "ARTERA_CLASSICAL", "DEFAULT")}


def load(path: Path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------- census

def _fields(element) -> dict:
    out = {}
    values = element.find("m_Fields/m_Values")
    for value in (values if values is not None else ()):
        name = value.find("m_ParamName")
        name = name.get("text") if name is not None else "?"
        kind = value.get("class", "")
        if kind.endswith("BLPEntryValue"):
            out[name] = {"entry": value.find("m_EntryName").get("text"),
                         "package": value.find("m_BLPPackage").get("text")}
        elif kind.endswith("ArtDefReferenceValue"):
            out[name] = value.find("m_ElementName").get("text")
        elif kind.endswith("CollectionValue"):
            out[name] = [item.find("m_ElementName").get("text") for item in value.find("m_Values")
                         if item.find("m_ElementName") is not None]
        else:
            for key in ("m_Value", "m_bValue", "m_fValue", "m_nValue"):
                item = value.find(key)
                if item is not None:
                    out[name] = item.get("text") if item.get("text") is not None else item.text
                    break
    return out


def _tokens(value: str) -> frozenset:
    return frozenset(token.strip().upper().removeprefix("BUILDING_").replace(" ", "_")
                     for token in (value or "").split(",") if token.strip() and token.strip().upper() != "EMPTY")


@lru_cache(maxsize=None)
def census() -> tuple[dict, ...]:
    """Every district/landmark variant with an asset: collection, owner name,
    child collection, building set, hero, era, culture, entry, package, content root."""
    cache = OUT / "census.json"
    if cache.is_file():
        return tuple(load(cache))
    rows = []
    for path in sorted(DEFAULT_ASSETS_ROOT.rglob("Landmarks*.artdef")):
        root = ET.parse(path).getroot()
        content = path.relative_to(DEFAULT_ASSETS_ROOT).as_posix().split("/ArtDefs")[0]
        for collection in root.find("m_RootCollections"):
            kind = collection.find("m_CollectionName").get("text")
            if kind not in ("Districts", "Landmarks"):
                continue
            for owner in collection:
                if owner.tag != "Element" or owner.find("m_ChildCollections") is None:
                    continue
                for child in owner.find("m_ChildCollections"):
                    child_name = child.find("m_CollectionName").get("text")
                    for variant in child:
                        if variant.tag != "Element":
                            continue
                        f = _fields(variant)
                        if not isinstance(f.get("Asset"), dict):
                            continue
                        rows.append({"collection": kind, "name": owner.find("m_Name").get("text"),
                                     "child": child_name, "set": f.get("Set_HeroBuildings", ""),
                                     "hero": f.get("Tag_HeroBuilding", ""), "era": f.get("Tag_Era", "DEFAULT"),
                                     "culture": f.get("Tag_Culture", "DEFAULT"), "entry": f["Asset"]["entry"],
                                     "package": f["Asset"]["package"], "content": content})
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=0) + "\n")
    return tuple(rows)


def _pick(rows: list[dict], era: str, what: str) -> dict:
    if not rows:
        raise KeyError(f"no Civ VI variant for {what}")
    for art_era in ART_ERAS[era]:
        found = [row for row in rows if row["era"] == art_era]
        if found:
            # Prefer the base game's record, then the latest expansion.
            return sorted(found, key=lambda row: (row["content"] != "Base", row["content"]))[0]
    return rows[0]


def base(district: str, buildings: str, era: str, culture: str = "DEFAULT") -> dict:
    """The base variant Civ VI would choose for this building set, era and culture."""
    wanted = _tokens(buildings)
    rows = [row for row in census() if row["collection"] == "Districts" and row["name"] == district and
            row["child"] == "BaseVariants" and _tokens(row["set"]) == wanted and row["culture"] == culture]
    return _pick(rows, era, f"{district} [{buildings}] {era}")


def hero(district: str, building: str, era: str, culture: str = "DEFAULT") -> dict:
    rows = [row for row in census() if row["collection"] == "Districts" and row["name"] == district and
            row["child"] == "BuildingVariants" and row["hero"] == building]
    # A civilization-unique building (Japan's Electronics Factory) has only its own culture tag.
    exact = [row for row in rows if row["culture"] == culture] or [row for row in rows if row["culture"] == "DEFAULT"]
    return _pick(exact or rows, era, f"{district} {building} {era}")


def landmark(name: str, era: str, entry: str | None = None) -> dict:
    rows = [row for row in census() if row["collection"] == "Landmarks" and row["name"] == name and
            (entry is None or row["entry"] == entry)]
    return _pick(rows, era, f"{name} {era}")


def raw(content: str, package: str, entry: str) -> dict:
    return {"collection": "raw", "name": entry, "child": "", "set": "", "hero": "", "era": "", "culture": "",
            "entry": entry, "package": package, "content": content}


def component(entry: str) -> dict:
    """A component already imported as part of another kit (a Civ VI tree,
    a power-plant stack), as a record of that asset."""
    manifest = load(SOURCES / "manifest.json")
    asset = next(key for key, value in manifest["assets"].items() if value.get("source_entry") == entry)
    return {**raw("", "", entry), "asset": asset}


@lru_cache(maxsize=None)
def city_buildings() -> dict:
    """The city pack's houses by culture and era (recovered from its frozen
    recipes), plus Civ VI's modern glass towers, with their packages."""
    return load(Path(__file__).with_name("city_buildings.json"))


@lru_cache(maxsize=None)
def city_tops() -> dict:
    """Height (kit units) of every city-pack house, measured once."""
    path = OUT / "city_tops.json"
    return load(path) if path.is_file() else {}


def city(entry: str) -> dict:
    """A Civ VI city-generator building as a raw record."""
    path = city_buildings()["packages"][entry]
    return {**raw(path.split("/Platforms/")[0], path.split("/BLPs/")[-1].removesuffix(".blp"), entry), "path": path}


# ---------------------------------------------------------------- compile

def _lab_compiler():
    """compound_landmark_importer._compile_asset with Lab-only tolerances.

    District kits trip checks the shared importer keeps strict: bases list a
    trailing model with no meshes; parks, canals and dams use attachment
    policies (cull modes, terrain follow, road connections) it rejects; water
    parks name effect bones they no longer have. Patched copies of those three
    functions share one namespace, so the shared file is unchanged."""
    namespace = dict(vars(cli))
    patches = (
        (cli._decode_component_attachment_values,
         [('    if (\n        values["ConnectionType"] not in ("NONE", "ROAD")',
           '    if False and (\n        values["ConnectionType"] not in ("NONE", "ROAD")'),
          ('    if set(values) != required:\n',
           '    values.setdefault("RandomizeAnims", False)\n    if set(values) != required:\n')]),
        (cli._decode_attachment_points,
         [("        if len(matches) != 1 and not (\n", "        if False and len(matches) != 1 and not (\n")]),
        (cli._compile_asset,
         [("            if count < 1 or first_mesh + count > mesh_count:\n",
           "            if count < 1:\n                continue\n            if first_mesh + count > mesh_count:\n")]),
    )
    for function, replacements in patches:
        source = inspect.getsource(function)
        for old, new in replacements:
            if old not in source:
                raise ValueError(f"compound_landmark_importer changed; update the Lab patch of {function.__name__}")
            source = source.replace(old, new)
        exec(compile("from __future__ import annotations\n" + source, "district_lab_" + function.__name__, "exec"),
             namespace)
    return namespace["_compile_asset"]


@contextmanager
def _lab_relaxations(wonder_movie: bool = False):
    """Skin weights renormalized and base colour optional, for the shared
    helpers the patched compiler calls (restored afterwards). Constructed
    wonders exist only in WonderMovie.blp: its entries read as landmarks, and
    their construction-animation bones have no clean bind pose."""
    saved = cli.decode_skin_influences, cli.MATERIAL_TEXTURE_SLOTS, cli.measure_bind_pose, cbe.TYPE_CITY_BLOCK
    skin = saved[0]
    cli.decode_skin_influences = lambda raw, offset, bones, normalize_nonzero_weights=False: skin(raw, offset, bones, True)
    cli.MATERIAL_TEXTURE_SLOTS = {**saved[1], "base_color": saved[1]["base_color"][:2] + (False,)}
    if wonder_movie:
        cli.measure_bind_pose = lambda bones: (0.0, "")
        cbe.TYPE_CITY_BLOCK = "WonderMoviePackageEntry"
    try:
        yield
    finally:
        cli.decode_skin_influences, cli.MATERIAL_TEXTURE_SLOTS, cli.measure_bind_pose, cbe.TYPE_CITY_BLOCK = saved


class Compiler:
    def __init__(self) -> None:
        self.packages = _package_index(DEFAULT_ASSETS_ROOT)
        self.package_bytes: dict[str, bytes] = {}
        self.opened: dict[str, IndexedStaticPackage] = {}
        self.texture_cache: dict = {}
        self.manifest = load(SOURCES / "manifest.json") if (SOURCES / "manifest.json").is_file() else \
            {"schema": "c3x.asset_pack.v0", "name": SOURCES.name, "assets": {}, "roots": {}, "failures": {},
             "source_policy": "Local licensed-source import; derived art is not redistributable."}
        self.compile_asset = _lab_compiler()
        self.manifest["failures"] = {}   # retry earlier failures each run

    def resolve(self, record: dict) -> str:
        if record.get("path"):
            return record["path"]
        resolution = _resolve_package(self.packages, record["package"], record["content"], record["entry"],
                                      self.package_bytes)
        if resolution["status"] != "resolved":
            raise ValueError(f"{record['entry']}: {resolution['status']}")
        return resolution["package_path"]

    def root(self, record: dict) -> str:
        if record.get("asset"):
            return record["asset"]
        key = f"{record['content']}|{record['package']}|{record['entry']}"
        if key in self.manifest["roots"]:
            return self.manifest["roots"][key]
        if key in self.manifest["failures"]:
            raise ValueError(self.manifest["failures"][key])
        try:
            asset = self.ensure(self.resolve(record), record["entry"])
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            self.manifest["failures"][key] = str(exc)
            raise ValueError(str(exc)) from exc
        self.manifest["roots"][key] = asset
        return asset

    def ensure(self, package_relative: str, entry: str, stack: tuple = ()) -> str:
        asset_id = _asset_id(package_relative, entry)
        if asset_id in self.manifest["assets"]:
            return asset_id
        if asset_id in stack:
            raise ValueError(f"component cycle at {entry}")
        package = self.opened.get(package_relative) or self.open(package_relative, entry)
        self.opened[package_relative] = package
        try:
            with _lab_relaxations(package_relative.endswith("WonderMovie.blp")):
                manifest_asset, evidence = self.compile_asset(
                    package, _shared_roots(DEFAULT_ASSETS_ROOT, package_relative), SOURCES, entry, asset_id, 100.0,
                    self.texture_cache, terrain_edit_policy="preserve_unresolved")
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            raise ValueError(f"{entry}: {exc}") from exc
        document_path = SOURCES / manifest_asset["landmark"]
        document = load(document_path)
        source_points = {point["id"]: point for point in evidence["attachments"]["points"]}
        for point in document["attachment_points"]:
            if point["binding_status"] != "component_unresolved":
                continue
            terminal = source_points[point["id"]]["component_source"]
            resolution = _resolve_package(self.packages, terminal["package"], _content_root(package_relative),
                                          terminal["entry"], self.package_bytes)
            try:
                if resolution["status"] != "resolved":
                    raise ValueError(f"unresolved child {terminal['entry']}")
                point["component_asset"] = self.ensure(resolution["package_path"], terminal["entry"],
                                                       stack + (asset_id,))
                point["binding_status"] = "resolved"
            except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                # Lab: a child that does not import (trees, shrubs) is left out.
                point["binding_status"] = "component_compile_unresolved"
                point["compile_failure"] = str(exc)[:200]
        document_path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        self.manifest["assets"][asset_id] = {**manifest_asset, "source_entry": entry}
        return asset_id

    def open(self, package_relative: str, entry: str) -> IndexedStaticPackage:
        """Open a package; its base is inferred from a name it holds once, so an
        entry named twice (a hero and its own root bone) borrows another name."""
        logical = package_relative.lower().split("/blps/")[-1].removesuffix(".blp")
        names = [entry] + [row["entry"] for row in census() if row["package"].lower() == logical]
        # Wall packages have no Landmarks.artdef rows; the city-wall intake names entries they hold once.
        adjuncts = load(ROOT / "Renderer/tools/asset_compiler/city_adjunct_sets.json")["assets"]
        names += [item["source_entry"] for item in adjuncts if item["source_package"] == package_relative]
        for name in dict.fromkeys(names):
            try:
                return IndexedStaticPackage(DEFAULT_ASSETS_ROOT / package_relative, name)
            except ValueError as exc:
                if "Expected target name once" not in str(exc):
                    raise
        raise ValueError(f"{package_relative}: no uniquely named entry to open it with")

    def save(self) -> None:
        SOURCES.mkdir(parents=True, exist_ok=True)
        if (SOURCES / "manifest.json").is_file():
            # Another run (a long wonder import) may have added assets meanwhile.
            disk = load(SOURCES / "manifest.json")
            for key in ("assets", "roots"):
                self.manifest[key] = {**disk.get(key, {}), **self.manifest[key]}
        (SOURCES / "manifest.json").write_text(json.dumps(self.manifest, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------- bake

def _skeleton_worlds(skeleton: dict) -> list:
    from Renderer.preview.render_improvement_sheet import _skeleton_worlds as worlds
    return worlds(skeleton)


def components(manifest: dict, asset_id: str, transform: list[float], found: list) -> list:
    landmark = load(SOURCES / manifest["assets"][asset_id]["landmark"])
    found.append((asset_id, landmark, transform))
    worlds = [_skeleton_worlds(load(SOURCES / path)) for path in landmark["components"]["skeletons"]]
    for point in landmark["attachment_points"]:
        if point["binding_status"] == "resolved":
            components(manifest, point["component_asset"],
                       cli._matrix_multiply(worlds[point["skeleton"]][point["bone"]], transform), found)
    return found


def yaw_scale(transform: list[float]) -> tuple[float, float] | None:
    scale = math.hypot(transform[0], transform[1])
    if scale <= 0 or any(abs(transform[i]) > 1e-3 * scale for i in (2, 6, 8, 9)) or \
            abs(transform[10] - scale) > 1e-3 * scale or abs(transform[5] - transform[0]) > 1e-3 * scale or \
            abs(transform[4] + transform[1]) > 1e-3 * scale:
        return None
    return math.atan2(transform[1], transform[0]), scale


def _wall_alignment(entries: list, meshes: dict) -> float | None:
    """The turn that puts a building's walls on the tile axes: the strongest
    direction (modulo 90 degrees) of its side-facing normals above the ground
    (pads, curbs and tarmac edges at hex angles do not vote), or None when no
    wall direction dominates (round or flat pieces). Annexes and conveyors
    that skew a minimum-area rectangle do not move it."""
    histogram = [0.0] * 90
    for key, _, yaw, _ in entries:
        mesh = meshes[key]
        if "walls" not in mesh:
            bins = [0.0] * 90
            for (_, _, z), (nx, ny, _), _ in mesh["vertices"]:
                h = math.hypot(nx, ny)
                if h > .85 and z > .012:
                    bins[int(math.degrees(math.atan2(ny, nx)) % 90) % 90] += h
            mesh["walls"] = bins
        shift = int(round(math.degrees(yaw)))
        for degree, weight in enumerate(mesh["walls"]):
            histogram[(degree + shift) % 90] += weight
    total = sum(histogram)
    if total <= 0:
        return None
    smooth = [sum(histogram[(d + k) % 90] for k in (-2, -1, 0, 1, 2)) for d in range(90)]
    peak = max(range(90), key=smooth.__getitem__)
    if smooth[peak] < .25 * total:
        return None
    return -math.radians(peak)


def _grid_snap(entries: list, meshes: dict, facing: float, force: bool = False) -> list:
    """Turn one component (a building) the way Renderer64 turns city buildings:
    its minimum-area footprint rectangle aligned with the tile axes, and its
    front (Civ VI's camera-facing -y) toward whichever of south-west or
    south-east is nearer. It turns about its own footprint centre, so the
    district layout keeps its places. Long thin pieces (walls, fences,
    palisades, canal sides) keep their authored angle so they still join."""
    from Renderer.lab.shared.cities.ground import footprint_alignment
    points = []
    for key, (px, py, _), yaw, scale in entries:
        c, s = math.cos(yaw), math.sin(yaw)
        points += [(px + (x * c - y * s) * scale, py + (x * s + y * c) * scale)
                   for (x, y, _), _, _ in meshes[key]["vertices"]]
    if len(points) < 3:
        return entries
    # Long thin pieces (wall and fence runs) align on their length; buildings on
    # their walls. Stakes and posts have no wall direction of their own.
    rectangle = footprint_alignment(points)
    c, s = math.cos(rectangle), math.sin(rectangle)
    turned = [(x * c - y * s, x * s + y * c) for x, y in points]
    width = max(p[0] for p in turned) - min(p[0] for p in turned)
    depth = max(p[1] for p in turned) - min(p[1] for p in turned)
    long_thin = min(width, depth) > 1e-6 and max(width, depth) / min(width, depth) > 2.5
    align = rectangle if long_thin else _wall_alignment(entries, meshes)
    if align is None:
        align = rectangle
    if min(width, depth) < 1e-6 or (not force and max(width, depth) / min(width, depth) > 3.0):
        return entries
    front = entries[0][2] - math.pi / 2                  # Civ VI -y in kit space
    targets = (math.pi / 2 - facing, -facing)           # world +v (SW), +u (SE) in kit space
    def miss(turn):
        angle = front + align + turn
        return min(abs(math.atan2(math.sin(angle - t), math.cos(angle - t))) for t in targets)
    delta = align + min((k * math.pi / 2 for k in range(4)), key=miss)
    cx = (min(p[0] for p in points) + max(p[0] for p in points)) / 2
    cy = (min(p[1] for p in points) + max(p[1] for p in points)) / 2
    c, s = math.cos(delta), math.sin(delta)
    return [[key, [cx + (px - cx) * c - (py - cy) * s, cy + (px - cx) * s + (py - cy) * c, pz], yaw + delta, scale]
            for key, (px, py, pz), yaw, scale in entries]


def _box(entries: list, meshes: dict) -> tuple[list, list, float]:
    """Axis-aligned footprint (kit units) and top height of one component."""
    low, high, top = [math.inf, math.inf], [-math.inf, -math.inf], 0.0
    for key, (px, py, pz), yaw, scale in entries:
        mesh = meshes[key]
        c, s = math.cos(yaw), math.sin(yaw)
        for x in (mesh["low"][0], mesh["high"][0]):
            for y in (mesh["low"][1], mesh["high"][1]):
                u, v = px + (x * c - y * s) * scale, py + (x * s + y * c) * scale
                low, high = [min(low[0], u), min(low[1], v)], [max(high[0], u), max(high[1], v)]
        top = max(top, pz + mesh["high"][2] * scale)
    return low, high, top


def _role(name: str, hero: bool) -> str:
    """What a Civ VI district component is, from its source entry name."""
    lowered = name.lower()
    if hero:
        return "hero"
    if any(key in lowered for key in ("_wall", "fence", "palisade")):
        return "wall"
    if "tower" in lowered and "aero" not in lowered:
        return "tower"
    if any(key in lowered for key in ("tree_", "shrub", "bush", "plant")):
        return "plant"
    if "_base_" in lowered:
        return "ground"          # a hex ground platform inside the kit; it cannot fit a diamond tile
    if any(key in lowered for key in ("hippodrome", "arena", "colosseum", "stadium", "mission_control")):
        return "landmark"
    if any(key in lowered for key in ("statue", "fountain", "obelisk", "square", "plaza")):
        return "centre"
    if any(key in lowered for key in ("filler", "bld", "shop", "tent", "shed", "house", "hall", "hangar")):
        return "building"
    return "prop"


def _turned(item: dict, turn: float, factor: float, target: tuple, size: float) -> dict:
    """The component turned about its footprint centre and moved to target (tile units)."""
    cx, cy = item["centre"]
    c, s = math.cos(turn), math.sin(turn)
    tx, ty = target[0] / size, target[1] / size
    return {**item, "entries": [
        [key, [tx + ((px - cx) * c - (py - cy) * s) * factor, ty + ((px - cx) * s + (py - cy) * c) * factor,
               pz * factor], yaw + turn, scale * factor]
        for key, (px, py, pz), yaw, scale in item["entries"]]}


GRID = .01      # drafting occupancy grid, tile units


def _draft(groups: list, meshes: dict, facing: float, size: float, rng: random.Random,
           layout: dict | None = None, template: dict | None = None) -> tuple[list, dict]:
    """Re-draft a Civ VI hex kit onto the square Civ III tile, densely.

    Civ VI lays a district out on a hex: buildings in a ring that spills past
    the C3X diamond. Here its pieces are packed onto the tile's square grid by
    role (from the Civ VI entry name), as one seeded, irregular cluster:
    hero buildings first, toward the back; a statue or fountain near the
    middle; then every kit building, tallest first, each grown onto the
    cluster; then copies of the kit's own buildings until the tile is about
    as full as `density` asks; a few props; and the kit's trees scattered into
    what is left. Real wall kits (many wall pieces) become a closed square ring
    along the tile edges with towers at its corners; a stray fence or two is
    dropped, and towers elsewhere (smokestacks, cooling and water towers) are
    buildings. Each piece is squared to the tile with its front toward
    south-west or south-east, chosen at random. Units are tile units (kit
    units x size)."""
    import numpy as np
    layout = layout or {}
    # A column keeps its layout from era to era: pieces go back to where the
    # first era's draft put them (template), so only the models change.
    template = template or {}
    record = {"majors": [], "buildings": [], "props": [], "trees": []}

    def anchored(kind, index, fallback):
        spots = template.get(kind, [])
        return spots[index] if index < len(spots) else fallback

    front = (math.cos(math.pi / 4 - facing), math.sin(math.pi / 4 - facing))   # tile (+u,+v) in kit axes
    targets = (math.pi / 2 - facing, -facing)                                   # SW, SE fronts in kit axes
    roles: dict[str, list] = {}
    for group in groups:
        low, high, top = _box(group["entries"], meshes)
        w, d = (high[0] - low[0]) * size, (high[1] - low[1]) * size
        if w <= 0 or d <= 0:
            continue
        role = group.get("role") or _role(group.get("name", ""), group["hero"])
        if group.get("fixed") and role in ("building", "prop", "hero"):
            role = "landmark" if role != "hero" else role
        if role in ("building", "prop") and w * d >= .012 and not group.get("role"):
            role = "landmark"    # a large structure of the kit (tanks, halls) ranks with the heroes
        roles.setdefault(role, []).append({**group, "centre": ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2),
                                           "w": w, "d": d, "top": top * size, "area": w * d, "role": role,
                                           "explicit": bool(group.get("role"))})
    if layout.get("walls") == "explicit":
        # A catalog's own wall kit (city walls round an Encampment) replaces the kit's.
        for role in ("wall", "tower"):
            roles[role] = [item for item in roles.get(role, []) if item["explicit"]]
        walled = bool(roles["wall"])
    else:
        walled = len(roles.get("wall", [])) >= 6
    if not walled:
        roles["building"] = roles.get("building", []) + roles.pop("tower", [])
    edge = .47                                   # the wall ring, just inside the tile edge
    inner = edge - .06 if walled else layout.get("limit", .47)
    drafted, boxes = [], []
    if walled:
        # The kit's longest wall piece repeated round the tile, each overlapping the
        # next so the ring is closed; towers at the corners.
        piece = max(roles["wall"], key=lambda w: max(w["w"], w["d"]))
        length = max(piece["w"], piece["d"])
        along_x = piece["w"] >= piece["d"]
        count = max(1, round(2 * edge / length))
        spacing = 2 * edge / count
        stretch = spacing / length * 1.02          # whole pieces, just touching: a closed run
        gate = (roles.get("gate") or [None])[0]
        normals = ((0, -1), (0, 1), (-1, 0), (1, 0))
        front_side = max(range(4), key=lambda side: normals[side][0] * front[0] + normals[side][1] * front[1])
        for side in range(4):
            horizontal = side < 2
            fixed = -edge if side % 2 == 0 else edge
            turn = 0.0 if horizontal == along_x else math.pi / 2
            for k in range(count):
                run = -edge + spacing * (k + .5)
                at = (run, fixed) if horizontal else (fixed, run)
                if gate and side == front_side and k == count // 2:
                    # The gate in the middle of the side facing the viewer.
                    drafted.append(_turned(gate, 0.0 if horizontal == (gate["w"] >= gate["d"]) else math.pi / 2,
                                           1.0, at, size))
                    continue
                drafted.append(_turned(piece, turn, stretch, at, size))
        for k, tower in zip(range(4), (roles.get("tower") or [])[:1] * 4):
            corner = ((-edge, -edge), (edge, -edge), (-edge, edge), (edge, edge))[k]
            drafted.append(_turned(tower, 0.0, 1.0, corner, size))
    n = int(round(1 / GRID)) + 1
    axis = -.5 + np.arange(n) * GRID
    gx, gy = np.meshgrid(axis, axis)
    occupied = np.zeros((n, n), bool)
    coarse = np.array([[rng.uniform(-1, 1) for _ in range(6)] for _ in range(6)])
    noise = coarse[np.minimum(np.arange(n) * 6 // n, 5)[:, None], np.minimum(np.arange(n) * 6 // n, 5)[None, :]]

    def mark(box, pad):
        occupied[(gx >= box[0] - pad) & (gx <= box[2] + pad) & (gy >= box[1] - pad) & (gy <= box[3] + pad)] = True

    for box in layout.get("reserve", ()):
        # Kept clear (a street or a square): paved, never built on.
        mark(box, 0.0)
    pond = layout.get("pond")
    # A pond in the middle, or toward the front when temples need the back.
    pond_at = (front[0] * layout.get("pond_offset", 0.0), front[1] * layout.get("pond_offset", 0.0))
    if pond:
        occupied[np.hypot(gx - pond_at[0], gy - pond_at[1]) < pond + .03] = True

    def place(item, attract, limit, pad=.008, spin=None, factor=1.0):
        yaw = item["entries"][0][2] - math.pi / 2
        near = [abs(math.atan2(math.sin(yaw - t), math.cos(yaw - t))) for t in targets]
        turns = [spin] if spin is not None else [0.0, -math.pi / 2 if near[0] <= near[1] else math.pi / 2]
        rng.shuffle(turns)
        table = np.pad(occupied.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        rows, cols = np.arange(n)[:, None], np.arange(n)[None, :]
        best = None
        for k, turn in enumerate(turns):
            if spin is not None:
                hw = hd = max(item["w"], item["d"]) * factor * .4     # a tree's crown, not its box
            else:
                quarter = abs(math.sin(turn)) > .5
                hw = (item["d"] if quarter else item["w"]) * factor / 2
                hd = (item["w"] if quarter else item["d"]) * factor / 2
            rx, ry = int(math.ceil((hw + pad) / GRID)), int(math.ceil((hd + pad) / GRID))
            y0, y1 = np.clip(rows - ry, 0, n), np.clip(rows + ry + 1, 0, n)
            x0, x1 = np.clip(cols - rx, 0, n), np.clip(cols + rx + 1, 0, n)
            free = (table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]) == 0
            free &= (np.abs(gx) + hw <= limit) & (np.abs(gy) + hd <= limit)
            if not free.any():
                continue
            score = np.where(free, np.hypot(gx - attract[0], gy - attract[1]) + .035 * noise + .004 * k, np.inf)
            index = np.unravel_index(np.argmin(score), score.shape)
            if best is None or score[index] < best[0]:
                best = (score[index], float(gx[index]), float(gy[index]), turn, hw, hd)
        if best is None:
            return None
        _, x, y, turn, hw, hd = best
        box = (x - hw, y - hd, x + hw, y + hd)
        mark(box, pad)
        drafted.append(_turned(item, turn, factor, (x, y), size))
        return box

    def cluster(radius):
        if not boxes:
            return (rng.uniform(-.1, .1), rng.uniform(-.1, .1))
        x0, y0, x1, y1 = rng.choice(boxes)
        angle = rng.uniform(0, 2 * math.pi)
        return ((x0 + x1) / 2 + radius * math.cos(angle), (y0 + y1) / 2 + radius * math.sin(angle))

    def coverage():
        inside = (np.abs(gx) <= inner) & (np.abs(gy) <= inner)
        return occupied[inside].mean()

    kept = 0
    majors = sorted(roles.get("hero", []), key=lambda h: -h["area"]) + \
        sorted(roles.get("landmark", []), key=lambda h: -h["area"])
    for index, hero in enumerate(majors):
        if index == 0:
            attract = (-.2 * front[0] + rng.uniform(-.12, .12), -.2 * front[1] + rng.uniform(-.12, .12))
        else:
            angle = rng.uniform(0, 2 * math.pi)
            attract = (.24 * math.cos(angle), .24 * math.sin(angle))
        attract = anchored("majors", index, attract)
        # Civ VI's largest heroes (airport, factory) would fill the tile alone at 2x;
        # a fixed-size landmark (the Colosseum) keeps its size everywhere.
        factor = 1.0 if hero.get("fixed") else \
            max(.55, min(1.0, math.sqrt(layout.get("hero_area", .2) / hero["area"])))
        box = place(hero, attract, max(inner, .5), factor=factor)
        if box:
            boxes.append(box)
            record["majors"].append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            kept += 1
    for centre in ([] if pond else roles.get("centre", [])[:1]):
        box = place(centre, (rng.uniform(-.06, .06), rng.uniform(-.06, .06)), inner)
        if box:
            boxes.append(box)
            kept += 1
    # One of each kit building; pieces a catalog adds explicitly (stacks) all stand.
    distinct = list({item["name"]: item for item in roles.get("building", []) if not item["explicit"]}.values())
    explicit = [item for item in roles.get("building", []) if item["explicit"]]
    grow = layout.get("building_scale", 1.0)       # fewer, larger pieces (fair tents)
    for building in sorted(distinct + explicit, key=lambda b: (-round(b["top"], 2), -b["area"])):
        box = place(building, anchored("buildings", len(record["buildings"]), cluster(.16)), inner,
                    factor=grow if not building["explicit"] else 1.0)
        if box:
            boxes.append(box)
            record["buildings"].append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            kept += 1
    # Fill: copies of the kit's own buildings (and any filler houses) until the
    # tile is as dense as asked.
    pool = distinct + list({item["name"]: item for item in roles.get("filler", [])}.values())
    copies: dict[str, int] = {}
    misses, density = 0, layout.get("density", .5)
    while pool and misses < 10 and kept < 48 and coverage() < density:
        item = rng.choice(pool)
        box = place(item, anchored("buildings", len(record["buildings"]), cluster(.14)), inner,
                    factor=grow if item["role"] == "building" else 1.0)
        if box:
            boxes.append(box)
            record["buildings"].append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            kept += 1
            misses = 0
            copies[item["name"]] = copies.get(item["name"], 0) + 1
            if copies[item["name"]] >= layout.get("copies", 4):     # no block of one repeated building
                pool.remove(item)
        else:
            misses += 1
    props = list({item["name"]: item for item in roles.get("prop", []) if not item["explicit"]}.values())
    # A few of the kit's props, largest first; props a catalog adds explicitly all stand.
    chosen = sorted(props, key=lambda p: -p["area"])[:layout.get("props", 5)] + \
        sorted((item for item in roles.get("prop", []) if item["explicit"]), key=lambda p: -p["area"])
    for prop in chosen:
        if prop.get("near") == "front":
            # A gate stands at the district's front, toward the viewer.
            attract = (.34 * front[0] + rng.uniform(-.08, .08), .34 * front[1] + rng.uniform(-.08, .08))
        elif boxes:
            x0, y0, x1, y1 = rng.choice(boxes)
            attract = ((x0 + x1) / 2 + rng.uniform(-.08, .08), (y0 + y1) / 2 + rng.uniform(-.08, .08))
        else:
            attract = (rng.uniform(-.2, .2), rng.uniform(-.2, .2))
        box = place(prop, anchored("props", len(record["props"]), attract), inner, pad=.004)
        if box:
            boxes.append(box)
            record["props"].append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            kept += 1
    # Trees: scattered at random into the gaps (never a ring of them), or a loose
    # ring round a pond.
    plants = list({item["name"]: item for item in roles.get("plant", [])}.values())
    count = layout.get("trees", rng.randint(4, 8) if plants else 0)
    for k in range(count if plants else 0):
        if pond:
            angle = 2 * math.pi * (k + rng.uniform(-.3, .3)) / max(count, 1)
            radius = pond + rng.uniform(.07, .13)
            attract = (pond_at[0] + radius * math.cos(angle), pond_at[1] + radius * math.sin(angle))
        else:
            attract = (rng.uniform(-inner, inner), rng.uniform(-inner, inner))
        box = place(rng.choice(plants), anchored("trees", len(record["trees"]), attract), inner, pad=.004,
                    spin=rng.uniform(0, 2 * math.pi), factor=rng.uniform(.8, 1.15))
        if box:
            record["trees"].append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            kept += 1
    return drafted, {"status": "tile_draft", "kept": kept, "of": sum(len(v) for v in roles.values()),
                     "walls": walled, "boxes": boxes, "limit": edge - .03 if walled else .48, "pond": pond,
                     "pond_at": pond_at, "template": record,
                     "reserved": [tuple(box) for box in layout.get("reserve", ())]}


def _centre(placed) -> tuple[float, float]:
    """Footprint centre (x, y) of placed meshes: (mesh, pivot, yaw, scale)."""
    points = []
    for mesh, (px, py, _), yaw, scale in placed:
        for x in (mesh["low"][0], mesh["high"][0]):
            for y in (mesh["low"][1], mesh["high"][1]):
                points.append((px + (x * math.cos(yaw) - y * math.sin(yaw)) * scale,
                               py + (x * math.sin(yaw) + y * math.cos(yaw)) * scale))
    return ((min(p[0] for p in points) + max(p[0] for p in points)) / 2,
            (min(p[1] for p in points) + max(p[1] for p in points)) / 2)


def footprint_centre(compiler: "Compiler", record: dict) -> tuple[float, float]:
    """Where a part (for example another hero building) stands in the hex."""
    found = components(compiler.manifest, compiler.root(record), IDENTITY, [])
    placed = []
    for _component, landmark, transform in found:
        for binding in landmark["draw_bindings"]:
            if "worked" not in binding["states"]:
                continue
            mesh = load(SOURCES / landmark["components"]["geometry"][binding["geometry"]])
            xs = [v["position"][0] for v in mesh["vertices"]]
            ys = [v["position"][1] for v in mesh["vertices"]]
            fitted = yaw_scale(transform) or (0.0, 1.0)
            placed.append(({"low": [min(xs), min(ys)], "high": [max(xs), max(ys)]}, transform[12:15], *fitted))
    if not placed:
        raise ValueError(f"{record['entry']}: no geometry to locate")
    return _centre(placed)


def kit_turn(record: dict) -> float:
    """The turn that squares one Civ VI part (an airport's tarmac, authored in
    place in the hex) to the tile; applied to a whole kit, it keeps Civ VI's
    own layout while lining it up with the Civ III tile."""
    from Renderer.lab.shared.cities.ground import footprint_alignment
    compiler = Compiler()
    points = []
    for _component, landmark, transform in components(compiler.manifest, compiler.root(record), IDENTITY, [])[:1]:
        for binding in landmark["draw_bindings"]:
            if "worked" in binding["states"]:
                mesh = load(SOURCES / landmark["components"]["geometry"][binding["geometry"]])
                points += [(v["position"][0] * transform[0] + v["position"][1] * transform[4] + transform[12],
                            v["position"][0] * transform[1] + v["position"][1] * transform[5] + transform[13])
                           for v in mesh["vertices"]]
    return footprint_alignment(points)


def _block_rows(payload: bytes, offset: int, bw: int, bh: int, block: int) -> list[bytes]:
    return [payload[offset + row * bw * block: offset + (row + 1) * bw * block] for row in range(bh)]


def square(path: Path, cache: Path, block: int) -> tuple[Path, float, float]:
    """An atlas-ready square texture of 128-512 texels with a full mip chain.
    Wide or tall textures repeat their short side; small ones repeat whole
    (block copies, no recompression). Returns the texture and its UV scale."""
    width, height, mips, dxgi, payload = rc.dds(path)
    if width == height and 128 <= width <= rc.CELL and mips >= (width // 4).bit_length():
        return path, 1.0, 1.0
    layout = rc.mip_offsets(width, height, mips, block)
    skip = 0
    while max(width, height) >> skip > rc.CELL:
        skip += 1
    size = max(128, max(width, height) >> skip)
    levels = []
    for level in range((size // 4).bit_length()):
        source = min(skip + level, len(layout) - 1)
        offset, bw, bh = layout[source]
        rows = _block_rows(payload, offset, bw, bh, block)
        side = max(1, (size >> level) // 4)
        levels.append(b"".join((rows[row % bh] * (side // bw + 1))[:side * block] for row in range(side)))
    target = rc.write_dds(cache / f"sq{size}_{path.parent.name}_{path.name}", size, len(levels), dxgi, block, levels)
    return target, (width >> skip) / size, (height >> skip) / size


def _hull(points: list) -> list:
    """Convex hull of 2D points (monotone chain)."""
    points = sorted(set(points))
    if len(points) < 3:
        return points
    def chain(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - \
                    (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return chain(points) + chain(points[::-1])


def _footprints(groups: list, meshes: dict, size: float) -> list:
    """Boxes (tile units, kit axes: centre, half sizes, angle) under every
    standing mesh of the placed buildings, measured on the placed outline of
    its parts above the ground: drafted buildings are squared to the tile, so
    the box follows their walls even when Civ VI authored the mesh turned.
    Flat pads, props, trees and walls are left out."""
    boxes = []
    for group in groups:
        if group.get("role") in ("plant", "wall", "tower", "ground", "prop"):
            continue
        for key, (px, py, _), yaw, scale in group["entries"]:
            mesh = meshes[key]
            if (mesh["high"][2] - mesh["low"][2]) * scale * size < .01:
                continue
            if "hull" not in mesh:
                # Above the ground only: Civ VI sinks a wider foundation skirt below it.
                mesh["hull"] = _hull([(round(x, 4), round(y, 4)) for (x, y, z), _, _ in mesh["vertices"] if z > .003])
            if not mesh["hull"]:
                continue
            c, s = math.cos(yaw), math.sin(yaw)
            xs = [(px + (x * c - y * s) * scale) * size for x, y in mesh["hull"]]
            ys = [(py + (x * s + y * c) * scale) * size for x, y in mesh["hull"]]
            boxes.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (max(xs) - min(xs)) / 2,
                          (max(ys) - min(ys)) / 2, 0.0))
    return boxes


def _paving_mesh(boxes: list, limit: float, texture: Path, size: float, seed: str, reach: float = .12,
                 **tight) -> dict:
    """The city pack's ground plate for a drafted district (the field of
    city_readability recompose.paving, with its style settings), cut close:
    paving only just round each building's own footprint (turned boxes), short
    paths between near neighbours (a minimum spanning tree without its long
    edges) and no plaza, kept inside the wall ring or the tile. This
    composition path alpha-tests rather than feathering, so the plate edge is
    the field's zero contour (marching squares). Boxes are tile units in kit
    axes; the mesh is kit units, its UVs repeating every paving period."""
    import numpy as np
    settings = {**load(CITY_STYLE)["paving"], **tight}
    period = settings["period"]
    step = period / round(period / settings["step"])     # no quad straddles a texture repeat
    span = int(math.ceil(.5 / step))
    axis = np.arange(-span, span + 1) * step
    gx, gy = np.meshgrid(axis, axis)
    d = np.full(gx.shape, np.inf)
    for cx, cy, hx, hy, angle in boxes:
        c, s = math.cos(angle), math.sin(angle)
        lx, ly = (gx - cx) * c + (gy - cy) * s, (gy - cy) * c - (gx - cx) * s
        qx, qy = np.abs(lx) - hx, np.abs(ly) - hy
        field = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
        d = np.minimum(d, field - settings["margin"])
    centres = [(cx, cy) for cx, cy, _, _, _ in boxes]
    connected = {0} if centres else set()
    while len(connected) < len(centres):
        length, a, b = min(((centres[i][0] - centres[j][0]) ** 2 + (centres[i][1] - centres[j][1]) ** 2, i, j)
                           for i in connected for j in range(len(centres)) if j not in connected)
        connected.add(b)
        if length > reach * reach:
            continue                     # no path across open ground
        (ax, ay), (bx, by) = centres[a], centres[b]
        t = np.clip(((gx - ax) * (bx - ax) + (gy - ay) * (by - ay)) / (length or 1.0), 0, 1)
        d = np.minimum(d, np.hypot(gx - ax - t * (bx - ax), gy - ay - t * (by - ay)) - settings["lane"])
    rng = random.Random(zlib.crc32(seed.encode()))
    count = len(axis)
    jitter = np.array([[(rng.random() - .5) * 2 * settings["noise"] for _ in range(count // 3 + 2)]
                       for _ in range(count // 3 + 2)])
    d = d + jitter[np.arange(count)[:, None] // 3, np.arange(count)[None, :] // 3]
    d = np.maximum(d, np.maximum(np.abs(gx), np.abs(gy)) - limit)
    vertices, indices = [], []
    for iy in range(count - 1):
        for ix in range(count - 1):
            corners = [(iy, ix), (iy, ix + 1), (iy + 1, ix + 1), (iy + 1, ix)]
            values = [d[c] for c in corners]
            if min(values) >= 0:
                continue
            points = [(gx[c], gy[c]) for c in corners]
            polygon = []
            for k in range(4):
                (pa, va), (pb, vb) = (points[k], values[k]), (points[(k + 1) % 4], values[(k + 1) % 4])
                if va < 0:
                    polygon.append(pa)
                if (va < 0) != (vb < 0):
                    t = va / (va - vb)
                    polygon.append((pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])))
            ox = math.floor(axis[ix] / period + 1e-6) * period
            oy = math.floor(axis[iy] / period + 1e-6) * period
            first = len(vertices)
            vertices += [((x / size, y / size, .003), (0.0, 0.0, 1.0), [(x - ox) / period, (y - oy) / period])
                         for x, y in polygon]
            for k in range(1, len(polygon) - 1):
                indices += [first, first + k, first + k + 1]
    return {"texture": texture, "masked": True, "vertices": vertices, "indices": indices,
            "low": [min(p[a] for p, _, _ in vertices) for a in range(3)],
            "high": [max(p[a] for p, _, _ in vertices) for a in range(3)]}


def _water_texture(cache: Path) -> Path:
    """Still pond water: deep teal in the middle, lighter toward the shore
    (BC3, opaque, solid 4 x 4 blocks; Civ VI's own water is shader-only)."""
    target = cache / "pond_water.dds"
    if target.is_file():
        return target
    levels = []
    for level in range(6):                       # 128 .. 4
        blocks = max(1, (128 >> level) // 4)
        data = bytearray()
        for by in range(blocks):
            for bx in range(blocks):
                r = min(1.0, math.hypot((bx + .5) / blocks * 2 - 1, (by + .5) / blocks * 2 - 1))
                red, green, blue = (int(d + (e - d) * r ** 1.6) for d, e in zip((24, 58, 66), (64, 104, 94)))
                colour = (red >> 3) << 11 | (green >> 2) << 5 | blue >> 3
                data += bytes((255, 255, 0, 0, 0, 0, 0, 0)) + struct.pack("<HHI", colour, colour, 0)
        levels.append(bytes(data))
    return rc.write_dds(target, 128, len(levels), 78, 16, levels)


def _pond_mesh(radius: float, size: float, cache: Path, seed: str, at: tuple = (0.0, 0.0)) -> dict:
    """A round pond with an irregular shore at the tile centre (tile units in,
    kit units out), drawn as a terrain-following ground decal."""
    rng = random.Random(zlib.crc32(("pond" + seed).encode()))
    steps = 32
    wobble = [rng.uniform(-1, 1) for _ in range(6)]
    outline = []
    for k in range(steps):
        angle = 2 * math.pi * k / steps
        r = radius * (1 + .1 * sum(w * math.sin((j + 2) * angle + j) for j, w in enumerate(wobble)) / 3)
        outline.append((r * math.cos(angle), r * math.sin(angle)))
    reach = radius * 1.25
    vertices = [((at[0] / size, at[1] / size, .002), (0.0, 0.0, 1.0), [.5, .5])]
    vertices += [(((at[0] + x) / size, (at[1] + y) / size, .002), (0.0, 0.0, 1.0),
                  [.5 + x / (2 * reach), .5 + y / (2 * reach)]) for x, y in outline]
    indices = []
    for k in range(steps):
        indices += [0, 1 + k, 1 + (k + 1) % steps]
    return {"texture": _water_texture(cache), "masked": True, "vertices": vertices, "indices": indices,
            "low": [min(p[a] for p, _, _ in vertices) for a in range(3)],
            "high": [max(p[a] for p, _, _ in vertices) for a in range(3)]}


def _snow_texture(cache: Path) -> Path:
    """Patchy snow for a ski mountain: white, opaque at the middle and thinning
    in broken patches toward the edge (BC3 with per-texel alpha)."""
    target = cache / "ski_snow.dds"
    if target.is_file():
        return target
    rng = random.Random("ski-snow")
    lattice = [[rng.random() for _ in range(9)] for _ in range(9)]

    def noise(u, v):
        x, y = u * 8, v * 8
        i, j = min(int(x), 7), min(int(y), 7)
        fx, fy = x - i, y - j
        top = lattice[j][i] * (1 - fx) + lattice[j][i + 1] * fx
        bottom = lattice[j + 1][i] * (1 - fx) + lattice[j + 1][i + 1] * fx
        return top * (1 - fy) + bottom * fy

    levels = []
    for level in range(7):                       # 256 .. 4
        side = 256 >> level
        blocks = max(1, side // 4)
        data = bytearray()
        for by in range(blocks):
            for bx in range(blocks):
                alphas, shade = [], 0.0
                for py in range(4):
                    for px in range(4):
                        u, v = (bx * 4 + px + .5) / (blocks * 4), (by * 4 + py + .5) / (blocks * 4)
                        r = math.hypot(u * 2 - 1, v * 2 - 1)
                        n = noise(u, v)
                        alphas.append(max(0.0, min(1.0, (1.0 - r * 1.15 + (n - .5) * .8) * 2.2)))
                        shade += n / 16
                # Alpha block: a0 = 255, a1 = 0 and six steps between (index 0 = 255, 1 = 0).
                order = [0, 2, 3, 4, 5, 6, 7, 1]
                bits = 0
                for k, a in enumerate(alphas):
                    bits |= order[round((1 - a) * 7)] << (3 * k)
                data += bytes((255, 0)) + bits.to_bytes(6, "little")
                grey = int(226 + 22 * shade)
                colour = (min(255, grey - 6) >> 3) << 11 | (min(255, grey - 2) >> 2) << 5 | min(255, grey + 8) >> 3
                data += struct.pack("<HHI", colour, colour, 0)
        levels.append(bytes(data))
    return rc.write_dds(target, 256, len(levels), 78, 16, levels)


def _snow_mesh(radius: float, scale: float, cache: Path) -> dict:
    """A dense disc of snow over a mountain (tile units in, units of `scale`
    out): a ground decal, so each vertex lies on the drawn mountain."""
    steps = 28
    vertices, indices = [], []
    for j in range(steps + 1):
        for i in range(steps + 1):
            u, v = i / steps, j / steps
            x, y = (u * 2 - 1) * radius, (v * 2 - 1) * radius
            vertices.append(((x / scale, y / scale, .002), (0.0, 0.0, 1.0), [u, v]))
    for j in range(steps):
        for i in range(steps):
            a = j * (steps + 1) + i
            if math.hypot((i + .5) / steps * 2 - 1, (j + .5) / steps * 2 - 1) > 1.02:
                continue
            indices += [a, a + 1, a + steps + 1, a + 1, a + steps + 2, a + steps + 1]
    return {"texture": _snow_texture(cache), "masked": True, "vertices": vertices, "indices": indices,
            "low": [min(p[a] for p, _, _ in vertices) for a in range(3)],
            "high": [max(p[a] for p, _, _ in vertices) for a in range(3)]}


def _runway_texture(cache: Path) -> Path:
    """An airfield runway (BC3, opaque): asphalt with white edge lines, a dashed
    centre line and threshold bars at both ends; it runs along texture u."""
    target = cache / "runway.dds"
    if target.is_file():
        return target

    def texel(u, v):
        if v < .06 or v > .94:
            return (238, 238, 232)                                   # edge lines
        if (u < .07 or u > .93) and int(v * 12) % 2 == 1:
            return (238, 238, 232)                                   # threshold bars
        if abs(v - .5) < .035 and int(u * 16) % 2 == 0 and .1 < u < .9:
            return (232, 232, 226)                                   # dashed centre line
        return (58, 60, 64) if (int(u * 64) + int(v * 16)) % 7 else (64, 66, 70)

    levels = []
    for level in range(7):                       # 256 .. 4
        blocks = max(1, (256 >> level) // 4)
        data = bytearray()
        for by in range(blocks):
            for bx in range(blocks):
                red, green, blue = texel((bx + .5) / blocks, (by + .5) / blocks)
                colour = (red >> 3) << 11 | (green >> 2) << 5 | blue >> 3
                data += bytes((255, 255, 0, 0, 0, 0, 0, 0)) + struct.pack("<HHI", colour, colour, 0)
        levels.append(bytes(data))
    return rc.write_dds(target, 256, len(levels), 78, 16, levels)


def _runway_mesh(box: tuple, along_u: bool, cache: Path) -> dict:
    """A runway strip over box (tile units round the tile centre), subdivided
    along its length so the decal follows the ground."""
    x0, y0, x1, y1 = box
    steps = 16
    vertices, indices = [], []
    for k in range(steps + 1):
        t = k / steps
        for side in (0.0, 1.0):
            x, y = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * side) if along_u else (x0 + (x1 - x0) * side, y0 + (y1 - y0) * t)
            vertices.append(((x, y, .002), (0.0, 0.0, 1.0), [t, side]))
    for k in range(steps):
        a = 2 * k
        indices += [a, a + 2, a + 1, a + 1, a + 2, a + 3]
    return {"texture": _runway_texture(cache), "masked": True, "vertices": vertices, "indices": indices,
            "low": [min(p[a] for p, _, _ in vertices) for a in range(3)],
            "high": [max(p[a] for p, _, _ in vertices) for a in range(3)]}


def _runway_box(instances: list, meshes: dict, width: float = .09):
    """Where a runway fits beside an airfield's buildings: the longest strip
    along one tile axis that touches none of them, as close to them as it can lie."""
    boxes = []
    for item in instances:
        mesh = meshes[item["model"]]
        if (mesh["high"][2] - mesh["low"][2]) * item["scale"] < .01:
            continue
        if "hull" not in mesh:     # what stands above the ground (an airport's alpha-cut tarmac does not)
            mesh["hull"] = _hull([(round(x, 4), round(y, 4)) for (x, y, z), _, _ in mesh["vertices"] if z > .003])
        if not mesh["hull"]:
            continue
        c, s = math.cos(item["rotation"]), math.sin(item["rotation"])
        xs = [item["u"] - .5 + (x * c - y * s) * item["scale"] for x, y in mesh["hull"]]
        ys = [item["v"] - .5 + (x * s + y * c) * item["scale"] for x, y in mesh["hull"]]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    for length in (.86, .76, .66, .56):
        best = None
        for along_u in (True, False):
            for k in range(-44, 45):
                offset = k / 100
                for slide in (0.0, -.1, .1):
                    lo, hi = slide - length / 2, slide + length / 2
                    if lo < -.47 or hi > .47:
                        continue
                    strip = (lo, offset - width / 2, hi, offset + width / 2) if along_u else \
                        (offset - width / 2, lo, offset + width / 2, hi)
                    gaps = [max(b[0] - strip[2], strip[0] - b[2], b[1] - strip[3], strip[1] - b[3]) for b in boxes]
                    if gaps and min(gaps) < .015:
                        continue
                    score = (min(gaps) if gaps else 0.0) + abs(offset) * .2
                    if best is None or score < best[0]:
                        best = (score, strip, along_u)
        if best:
            return best[1], best[2]
    return None


def _opaque_bc3(path: Path, cache: Path) -> Path:
    target = cache / f"opaque_{path.name}"
    if not target.is_file():
        data = bytearray(path.read_bytes())
        for offset in range(148, len(data) - 15, 16):
            data[offset:offset + 8] = b"\xff\xff\x00\x00\x00\x00\x00\x00"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(data))
    return target


def _material_textures(material: dict) -> tuple[Path, Path | None]:
    channels = material["channels"]
    colour = SOURCES / channels["base_color"]["texture"]
    opacity = channels.get("opacity")
    opacity = SOURCES / opacity["texture"] if isinstance(opacity, dict) else None
    if opacity is not None and rc.fully_opaque(opacity):
        opacity = None
    return colour, opacity


def _bc1_rgb(path: Path):
    """Top-level RGB texels of a BC1 texture (numpy, uint8)."""
    import numpy as np
    width, height, _, dxgi, payload = rc.dds(path)
    if dxgi not in (71, 72):
        raise ValueError(f"BC1 texture required: {path.name}")
    bw, bh = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    blocks = np.frombuffer(payload, np.uint8, bw * bh * 8).reshape(bh, bw, 8)
    ends = np.ascontiguousarray(blocks[..., :4]).view("<u2").astype(np.int32)
    def rgb(v):
        return np.stack([(v >> 11 & 31) * 255 // 31, (v >> 5 & 63) * 255 // 63, (v & 31) * 255 // 31], -1)
    c0, c1 = rgb(ends[..., 0]), rgb(ends[..., 1])
    four = (ends[..., 0] > ends[..., 1])[..., None]
    palette = np.stack([c0, c1, np.where(four, (2 * c0 + c1) // 3, (c0 + c1) // 2),
                        np.where(four, (c0 + 2 * c1) // 3, 0)], 2)
    bits = np.ascontiguousarray(blocks[..., 4:]).view("<u4")[..., 0]
    index = (bits[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
    texels = np.take_along_axis(palette, index[..., None].astype(np.int64), 2)
    return texels.reshape(bh, bw, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)[:height, :width]


def _bc1_encode(image) -> bytes:
    """BC1 blocks of an RGB image (sides multiples of 4): luma-extreme endpoints."""
    import numpy as np
    h, w = image.shape[:2]
    blocks = image.reshape(h // 4, 4, w // 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(h // 4, w // 4, 16, 3)
    luma = (blocks * np.array([.299, .587, .114])).sum(-1)
    pick = lambda at: np.take_along_axis(blocks, at[..., None, None], 2)[..., 0, :]
    def q565(c):
        c = np.clip(np.rint(c), 0, 255).astype(np.int32)
        return (c[..., 0] * 31 + 127) // 255 << 11 | (c[..., 1] * 63 + 127) // 255 << 5 | (c[..., 2] * 31 + 127) // 255
    c0, c1 = q565(pick(luma.argmax(-1))), q565(pick(luma.argmin(-1)))
    c0, c1 = np.maximum(c0, c1), np.minimum(c0, c1)
    def rgb(v):
        return np.stack([(v >> 11 & 31) * 255 // 31, (v >> 5 & 63) * 255 // 63, (v & 31) * 255 // 31], -1)
    e0, e1 = rgb(c0).astype(float), rgb(c1).astype(float)
    palette = np.stack([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3], 2)
    index = ((blocks[..., :, None, :] - palette[..., None, :, :]) ** 2).sum(-1).argmin(-1)
    index = np.where((c0 == c1)[..., None], 0, index).astype(np.uint32)
    out = np.zeros((h // 4, w // 4, 8), np.uint8)
    out[..., 0:2] = c0.astype("<u2")[..., None].view(np.uint8)
    out[..., 2:4] = c1.astype("<u2")[..., None].view(np.uint8)
    out[..., 4:8] = (index << (2 * np.arange(16, dtype=np.uint32))).sum(-1).astype("<u4")[..., None].view(np.uint8)
    return out.tobytes()


def _emissive_square(emissive: Path, size: int, su: float, sv: float, cache: Path) -> Path:
    """Civ VI's emissive map laid out like its colour's atlas-ready square
    (the colour fills su x sv of it and repeats), BC1 with a full mip chain."""
    import numpy as np
    w, h = max(4, round(su * size)), max(4, round(sv * size))
    target = cache / f"glow{size}_{w}x{h}_{emissive.parent.name}_{emissive.name}"
    if target.is_file():
        return target
    source = _bc1_rgb(emissive).astype(float)
    eh, ew = source.shape[:2]
    if eh % h == 0 and ew % w == 0:
        image = source.reshape(h, eh // h, w, ew // w, 3).mean((1, 3))
    else:
        def axis(n, m):
            x = np.clip((np.arange(n) + .5) * m / n - .5, 0, m - 1)
            low = np.floor(x).astype(int)
            return low, np.minimum(low + 1, m - 1), x - low
        y0, y1, fy = axis(h, eh)
        x0, x1, fx = axis(w, ew)
        rows = source[y0] * (1 - fy)[:, None, None] + source[y1] * fy[:, None, None]
        image = rows[:, x0] * (1 - fx)[None, :, None] + rows[:, x1] * fx[None, :, None]
    return _write_bc1(np.tile(image, (-(-size // h), -(-size // w), 1))[:size, :size], target)


def _write_bc1(image, target: Path) -> Path:
    """A square RGB image as BC1 (sRGB) with a full box-filtered mip chain."""
    levels = []
    while True:
        levels.append(_bc1_encode(image))
        if image.shape[0] <= 4:
            break
        image = image.reshape(image.shape[0] // 2, 2, image.shape[1] // 2, 2, 3).mean((1, 3))
    return rc.write_dds(target, image.shape[0] << (len(levels) - 1), len(levels), 72, 8, levels)


def _tint_code(tint) -> str:
    colours = tint if isinstance(tint[0], (tuple, list)) else (tint,)
    return "".join(f"{round(v * 100):02}" for colour in colours for v in colour)


def _tinted(path: Path, tint: tuple, cache: Path) -> Path:
    """Civ VI's tint for a BC3 colour texture: its alpha marks the panels the
    game recolours per building (on Civ VI's houses, the roof planks). Masked
    texels take the first colour, multiplied into their own shading. A second
    colour paints the pale, unsaturated texels outside the mask (white
    weatherboard walls) for a Lab house palette, and saturated orange brick
    (chimneys) is toned down beside it. BC1 out."""
    import numpy as np
    roof, wall = (tint[0], tint[1]) if isinstance(tint[0], (tuple, list)) else (tint, None)
    target = cache / f"tint{_tint_code(tint)}_{path.name}"
    if target.is_file():
        return target
    width, height, _, dxgi, payload = rc.dds(path)
    if dxgi not in (77, 78) or width != height:
        raise ValueError(f"Square BC3 colour required: {path.name}")
    bw, bh = width // 4, height // 4
    blocks = np.frombuffer(payload, np.uint8, bw * bh * 16).reshape(bh, bw, 16)
    a0, a1 = blocks[..., 0].astype(int)[..., None], blocks[..., 1].astype(int)[..., None]
    k = np.arange(1, 7)
    palette = np.where(a0 > a1, np.concatenate([a0, a1, ((7 - k) * a0 + k * a1) // 7], -1),
                       np.concatenate([a0, a1, ((5 - k[:4]) * a0 + k[:4] * a1) // 5, 0 * a0, 0 * a0 + 255], -1))
    bits = sum(blocks[..., 2 + i].astype(np.uint64) << np.uint64(8 * i) for i in range(6))
    index = (bits[..., None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64)) & np.uint64(7)).astype(np.int64)
    alpha = np.take_along_axis(palette, index, 2).reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(height, width)
    colour = rc.write_dds(cache / f"bc1_{path.name}", width, 1, 72, 8, [np.ascontiguousarray(blocks[..., 8:]).tobytes()])
    rgb = _bc1_rgb(colour).astype(float)
    mask = alpha[..., None] / 255.0
    out = rgb * (1 + mask * (np.array(roof) - 1))
    if wall is not None:
        high, low = rgb.max(-1, keepdims=True) / 255, rgb.min(-1, keepdims=True) / 255
        saturation = (high - low) / np.maximum(high, 1e-6)
        luma = (rgb * np.array([.2126, .7152, .0722])).sum(-1, keepdims=True) / 255
        ramp = lambda x, a, b: np.clip((x - a) / (b - a), 0, 1)
        pale = ramp(luma, .55, .75) * (1 - ramp(saturation, .10, .22)) * (1 - mask)
        r, g, b = rgb[..., :1], rgb[..., 1:2], rgb[..., 2:]
        brick = ramp(saturation, .40, .60) * (r > g) * (g > b) * ((r - g) > (g - b) * .6) * (1 - mask)
        grey = luma * 255
        out = out * (1 + pale * (np.array(wall) - 1))
        out = out * (1 - brick) + brick * (.72 * (grey + (out - grey) * .55))
    return _write_bc1(np.clip(out, 0, 255), target)


def _dark_square(size: int, cache: Path) -> Path:
    """An unlit (black) BC1 square for atlas cells without an emissive map."""
    target = cache / f"glow_black_{size}.dds"
    if not target.is_file():
        count = (size // 4).bit_length()
        rc.write_dds(target, size, count, 72, 8, [bytes(max(1, (size >> level) // 4) ** 2 * 8) for level in range(count)])
    return target


def _tint_of(part: dict, material: dict) -> tuple | None:
    """A part's tint where its material has a tint mask (BC3 colour)."""
    if not part.get("tint"):
        return None
    width, height, _, dxgi, _ = rc.dds(_material_textures(material)[0])
    return tuple(tuple(c) if isinstance(c, (tuple, list)) else c for c in part["tint"]) \
        if dxgi in (77, 78) and width == height else None


def _tint_key(part: dict, material: dict) -> str:
    tint = _tint_of(part, material)
    return f"/t{_tint_code(tint)}" if tint else ""


def _mesh(vertices: list, indices: list, material: dict, cache: Path, tint: tuple | None = None) -> dict:
    """A composition mesh: UVs kept within the texture, the texture made
    atlas-ready (cutout or opaque) and its UVs scaled to match. Civ VI's
    emissive map (night lights) gets the same square layout; a tint colours a
    BC3 colour's masked wall panels as Civ VI does."""
    colour, opacity = _material_textures(material)
    if any(value < 0 or value > 1 for _, _, uv in vertices for value in uv):
        vertices, indices = rc.within_texture(vertices, indices)
    masked = False
    if tint:
        texture, su, sv = square(_tinted(colour, tint, cache), cache, 8)
    elif rc.dds(colour)[3] in (77, 78):
        # BC3 colour: its alpha is a tint mask on building bodies, not a
        # cutout. Keep the colour blocks, force alpha opaque (BC3 atlas).
        texture, su, sv = square(_opaque_bc3(colour, cache), cache, 16)
        masked = True
    elif opacity is not None:
        try:
            texture, su, sv = square(rc.masked(colour, opacity, cache), cache, 16)
            masked = True
        except ValueError:
            texture, su, sv = square(colour, cache, 8)
    else:
        texture, su, sv = square(colour, cache, 8)
    glow = material["channels"].get("emissive")
    if isinstance(glow, dict) and glow.get("texture"):
        try:
            glow = _emissive_square(SOURCES / glow["texture"], rc.dds(texture)[0], su, sv, cache)
        except ValueError:
            glow = None
    else:
        glow = None
    vertices = [(p, n, [uv[0] * su, uv[1] * sv]) for p, n, uv in vertices]
    if len(vertices) > 100000 or len(indices) > 300000:
        raise ValueError(f"mesh of {len(vertices)} vertices exceeds the runtime bounds")
    return {"texture": texture, "masked": masked, "emissive": glow, "vertices": vertices, "indices": indices,
            "low": [min(p[a] for p, _, _ in vertices) for a in range(3)],
            "high": [max(p[a] for p, _, _ in vertices) for a in range(3)]}


def bake(catalog: dict, compiler: Compiler, district_scale: float = SCALE, ground: bool = True) -> dict:
    """Write the Lab composition pack for one catalog: one composition per cell,
    every component its own instance standing on its own ground."""
    manifest = compiler.manifest
    cache = OUT / "texture_cache"
    meshes: dict[str, dict] = {}
    compositions, report = [], {}
    templates: dict[str, dict] = {}      # a column's first-era layout, reused by its later eras
    for cell in catalog["cells"]:
        parts, missing, groups = [], [], []
        facing = cell.get("facing", catalog.get("facing", "SW"))
        facing = FACING[facing] if isinstance(facing, str) else facing
        size = cell.get("scale", catalog.get("scale", 1.0)) * district_scale
        # A drafted district re-lays every piece on the tile, so every piece is squared to it.
        # "force" drafts a cell with no Civ VI district base (a neighborhood of city houses).
        draft_flag = cell.get("draft", catalog.get("draft", False))
        drafting = bool(draft_flag and cell["parts"] and
                        (draft_flag == "force" or cell["parts"][0]["record"]["collection"] == "Districts"))
        for part_index, part in enumerate(cell["parts"]):
            try:
                root = compiler.root(part["record"])
            except ValueError as exc:
                missing.append(f"{part['record']['entry']}: {exc}")
                continue
            found = components(manifest, root, IDENTITY, [])
            merge = part.get("merge")
            if merge:
                found = found[:1]    # the wonder alone, without its movie set dressing
            instances, instances_of_component, spans, name = [], [], [], ""
            for component, landmark, transform in found:
                if instances_of_component:
                    start = len(instances)
                    instances += _grid_snap(instances_of_component, meshes, facing, drafting) \
                        if cell.get("snap", catalog.get("snap", True)) else instances_of_component
                    spans.append((start, len(instances), name))
                instances_of_component = []
                name = manifest["assets"][component]["source_entry"]
                fitted = yaw_scale(transform)
                linear = IDENTITY if fitted else transform[:12] + [0.0, 0.0, 0.0, 1.0]
                if merge:
                    # A wonder-movie model is thousands of animated pieces: one mesh per
                    # kept material (chunked to the runtime bounds); pieces beyond the
                    # radius (scaffolds, stone piles) are left out.
                    pieces: dict[int, list] = {}
                    for binding in landmark["draw_bindings"]:
                        if binding["material"] not in merge["materials"] or \
                                cell.get("state", "worked") not in binding["states"]:
                            continue
                        mesh = load(SOURCES / landmark["components"]["geometry"][binding["geometry"]])
                        points = [vertex["position"] for vertex in mesh["vertices"]]
                        if math.hypot(sum(p[0] for p in points) / len(points),
                                      sum(p[1] for p in points) / len(points)) > merge["radius"]:
                            continue
                        chunks = pieces.setdefault(binding["material"], [([], [])])
                        if len(chunks[-1][0]) + len(points) > 60000:
                            chunks.append(([], []))
                        vertices, indices = chunks[-1]
                        indices += [len(vertices) + index for index in mesh["topology"]["indices"]]
                        squash = part.get("squash", 1.0)
                        vertices += [([vertex["position"][0], vertex["position"][1], vertex["position"][2] * squash],
                                      vertex["normal"], list(vertex["uv0"])) for vertex in mesh["vertices"]]
                    for material_index, chunks in pieces.items():
                        material = load(SOURCES / landmark["components"]["materials"][material_index])
                        for k, (vertices, indices) in enumerate(chunks):
                            key = f"district/{root.rsplit('/', 1)[-1]}/m{material_index}c{k}" + \
                                (f"/s{part['squash']}" if part.get("squash") else "") + _tint_key(part, material)
                            if key not in meshes:
                                meshes[key] = _mesh(vertices, indices, material, cache, _tint_of(part, material))
                            instances_of_component.append([key, list(transform[12:15]), 0.0, 1.0])
                    continue
                for binding in landmark["draw_bindings"]:
                    if cell.get("state", "worked") not in binding["states"]:
                        continue
                    geometry = landmark["components"]["geometry"][binding["geometry"]]
                    material = load(SOURCES / landmark["components"]["materials"][binding["material"]])
                    if not isinstance(material.get("channels", {}).get("base_color"), dict):
                        continue
                    key = f"district/{geometry.rsplit('/', 1)[-1].removesuffix('.json')}"
                    if not fitted:
                        key += "/" + format(abs(hash(tuple(round(v, 4) for v in linear))) % 16 ** 8, "08x")
                    clip = part.get("clip_z")
                    if clip is not None:
                        key += f"/z{clip}"
                    if part.get("squash"):
                        key += f"/s{part['squash']}"
                    key += _tint_key(part, material)
                    if part.get("conform"):
                        # Renderer64 seats every vertex of a "city/walls/" asset on the
                        # ground under it (mountain faces included): a gondola climbs the
                        # drawn mountain instead of keeping Civ VI's slope.
                        key = "city/walls/" + key
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
                        if part.get("squash"):
                            # A squatter building (Civ VI's arenas stand tall beside Civ III art).
                            vertices = [([x, y, z * part["squash"]], n, uv) for (x, y, z), n, uv in vertices]
                        ramp = part.get("ramp")
                        if ramp:
                            # Take out Civ VI's authored slope (zero at the bottom station,
                            # `rise` at the top) so only height above the ground is left.
                            bottom, top, rise = ramp
                            vertices = [([x, y, max(-.006, z - rise * min(1.0, max(0.0, (bottom - x) / (bottom - top))))],
                                         n, uv) for (x, y, z), n, uv in vertices]
                        indices = mesh["topology"]["indices"]
                        if clip is not None:
                            # Only what stands below the cut (a rig's deck without its derrick).
                            indices = [index for t in range(0, len(indices), 3)
                                       if max(vertices[i][0][2] for i in indices[t:t + 3]) <= clip
                                       for index in indices[t:t + 3]]
                            if not indices:
                                continue
                        meshes[key] = _mesh(vertices, indices, material, cache, _tint_of(part, material))
                    yaw, scale = fitted or (0.0, 1.0)
                    instances_of_component.append([key, list(transform[12:15]), yaw, scale])
            if instances_of_component:
                start = len(instances)
                instances += _grid_snap(instances_of_component, meshes, facing, drafting) \
                    if cell.get("snap", catalog.get("snap", True)) else instances_of_component
                spans.append((start, len(instances), name))
            if part.get("place"):
                # Explicit piece placement (Great Wall runs): kit-space offset and turn.
                ox, oy, turn = part["place"]
                c, s_ = math.cos(turn), math.sin(turn)
                instances = [[key, [ox + px * c - py * s_, oy + px * s_ + py * c, pz], yaw + turn, scale]
                             for key, (px, py, pz), yaw, scale in instances]
            if part.get("lift"):
                # Raise a part whose flat ground (an airport's tarmac) sits at or under the ground.
                instances = [[key, [px, py, pz + part["lift"]], yaw, scale] for key, (px, py, pz), yaw, scale in instances]
            factor = part.get("scale", 1.0)
            move_to = part.get("move_to")
            if move_to is not None or part.get("centre") or factor != 1.0:
                # Re-seat a hero in another hero's slot (or centre a lone part):
                # move and scale this part about its footprint centre.
                cx, cy = _centre([(meshes[key], position, yaw, scale) for key, position, yaw, scale in instances])
                if isinstance(move_to, dict):
                    move_to = footprint_centre(compiler, move_to)
                tx, ty = move_to if move_to is not None else (0.0, 0.0) if part.get("centre") else (cx, cy)
                for instance in instances:
                    instance[1] = [tx + (instance[1][0] - cx) * factor, ty + (instance[1][1] - cy) * factor,
                                   instance[1][2] * factor]
                    instance[3] *= factor
            if part.get("whole") and spans:
                # A model whose parts stay together (a ruined block): one piece.
                spans = [(spans[0][0], spans[-1][1], part["record"]["entry"])]
            groups += [{"entries": instances[a:b], "name": n, "role": part.get("role"), "near": part.get("near"),
                        "fixed": part.get("fixed"),
                        "hero": part_index > 0 and not part.get("place") and not part.get("filler"),
                        "fit": part.get("ground_fit", 0.0)} for a, b, n in spans]
        draft = None
        if groups and drafting:
            # District kits only; improvements (parks, farms, rigs) keep their compact layout.
            size *= DRAFT_RATIO
            key = cell.get("layout_key", cell["name"])
            groups, draft = _draft(groups, meshes, facing, size, random.Random(zlib.crc32(key.encode())),
                                   cell.get("layout", catalog.get("layout")), templates.get(key))
            templates.setdefault(key, draft["template"])
        parts = [(key, tuple(position), yaw, scale, group.get("fit", 0.0)) for group in groups
                 for key, position, yaw, scale in group["entries"]]
        if not parts:
            report[cell["name"]] = {"label": cell["label"], "instances": 0, "missing": missing}
            continue
        cosine, sine = math.cos(facing), math.sin(facing)
        cu = cv = .5
        # ground_fit (tile units) seats a piece on the lowest ground under its footprint.
        instances = [{"model": key, "u": cu + (px * cosine - py * sine) * size,
                      "v": cv + (px * sine + py * cosine) * size, "rotation": facing + yaw,
                      "scale": size * scale, "lift": pz * size, "ground_fit": fit}
                     for key, (px, py, pz), yaw, scale, fit in parts]
        footprints = _footprints(groups, meshes, size) if draft else []
        footprints += [((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, 0.0)
                       for x0, y0, x1, y1 in (draft or {}).get("reserved", ())]
        if ground and footprints and cell.get("era"):
            # Civ VI paints district ground with its terrain system (absent from the
            # landmark); the city pack's era paving stands in, laid like a city plate.
            # A "decal/" asset follows the terrain under every vertex (a flat mesh
            # would sink under Renderer64's relief) and casts no shadow.
            key = f"decal/paving/{cell['name']}"
            # Surgical, unlike a city plate: paving hugs each building's own
            # footprint; open ground between buildings stays grass.
            meshes[key] = _paving_mesh(footprints, draft["limit"], PLATES / f"{cell['era'].lower()}.dds",
                                       size, cell["name"], margin=.01, lane=.008, noise=.006)
            instances.insert(0, {"model": key, "u": cu, "v": cv, "rotation": facing, "scale": size,
                                 "lift": 0.0, "ground_fit": 0.0})
        if draft and draft.get("pond"):
            key = f"decal/pond/{cell['name']}"
            meshes[key] = _pond_mesh(draft["pond"], size, cache, cell["name"], draft.get("pond_at", (0.0, 0.0)))
            instances.insert(0, {"model": key, "u": cu, "v": cv, "rotation": facing, "scale": size,
                                 "lift": 0.0, "ground_fit": 0.0})
        if cell.get("runway"):
            # Civ VI paints its runway as a decal the import cannot place; draw one
            # along a tile axis beside the airfield's buildings.
            found = _runway_box(instances, meshes)
            if found:
                key = f"decal/runway/{cell['name']}"
                meshes[key] = _runway_mesh(found[0], found[1], cache)
                instances.insert(0, {"model": key, "u": cu, "v": cv, "rotation": 0.0, "scale": 1.0,
                                     "lift": 0.0, "ground_fit": 0.0})
        if cell.get("snow_cover"):
            # Civ VI's ski slopes are snow-covered; a peak-centred decal at a large
            # instance scale drapes the mountain (a small one would read as too steep).
            key = f"decal/snow/{cell['name']}"
            meshes[key] = _snow_mesh(cell["snow_cover"], 1.5, cache)
            instances.insert(0, {"model": key, "u": cu, "v": cv, "rotation": 0.0, "scale": 1.5,
                                 "lift": 0.0, "ground_fit": 0.0})
        if draft:
            draft = {k: v for k, v in draft.items() if k not in ("boxes", "limit", "reserved", "pond_at", "template")}
        compositions.append((cell["name"], [(ALL_TERRAINS, instances)]))
        report[cell["name"]] = {"label": cell["label"], "instances": len(parts), "missing": missing,
                                "sources": [part["record"]["entry"] for part in cell["parts"]], "draft": draft}
    if len(meshes) > 256:
        raise ValueError(f"{catalog['id']}: {len(meshes)} meshes exceed the runtime asset limit; split the catalog")
    opaque = list(dict.fromkeys(mesh["texture"] for mesh in meshes.values() if not mesh["masked"]))
    cutout = list(dict.fromkeys(mesh["texture"] for mesh in meshes.values() if mesh["masked"]))
    atlases, place = rc.pack_atlases(opaque, rc.MODEL_ATLAS)
    if cutout:
        more, cut_place = rc.pack_atlases(cutout, rc.MASKED_ATLAS, len(atlases))
        atlases += more
        place.update(cut_place)
    if len(atlases) > rc.TEXTURE_SLOTS:
        raise ValueError(f"{catalog['id']}: {len(atlases)} atlases exceed the texture slots; split the catalog")
    # Night lights: emissive_N.dds repeats atlas_N.dds's cell layout (BC1 for every
    # atlas); only atlases holding at least one emissive map are written.
    glow_of = {}
    for mesh in meshes.values():
        if mesh.get("emissive") and mesh["texture"] not in glow_of:
            glow_of[mesh["texture"]] = mesh["emissive"]
    glows, lit = [], []
    per_atlas = (rc.MODEL_ATLAS[0] // rc.CELL) ** 2
    for group in (opaque, cutout):
        cells = [glow_of[t] if t in glow_of and rc.dds(glow_of[t])[0] == rc.dds(t)[0] else
                 _dark_square(rc.dds(t)[0], cache) for t in group]
        if group:
            glows += rc.pack_atlases(cells, rc.MODEL_ATLAS)[0]
            lit += [any(cell.name.startswith("glow") and not cell.name.startswith("glow_black")
                        for cell in cells[k:k + per_atlas]) for k in range(0, len(cells), per_atlas)]
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
    for position, data in enumerate(glows):
        if lit[position]:
            (CATALOG / f"textures/emissive_{position}.dds").write_bytes(data)
    (CATALOG / "textures/unused.dds").write_bytes(rc.dds_header(4, 4, 1, rc.MODEL_ATLAS[3], 8) + bytes(8))
    (CATALOG / "resource_runtime.bin").write_bytes(blob)
    summary = {"catalog": catalog["id"], "meshes": len(meshes), "atlases": len(atlases), "lit_atlases": sum(lit), "cells": report}
    (CATALOG / "composition_report.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


# ---------------------------------------------------------------- layout and render

TERRAIN = {"Grassland": (2, 2), "Plains": (1, 1), "Desert": (0, 0), "Hills": (2, 5), "Mountains": (2, 6),
           "Forest": (2, 7), "Coast": (11, 11), "Sea": (12, 12), "Tundra": (3, 3)}


def layout(catalog: dict) -> list[tuple[int, int, dict]]:
    """(dx, dy, cell): row/column grid two tiles apart, centred on the view."""
    if catalog.get("tiles"):
        # A scene of many tiles sharing compositions (Great Wall runs): (dx, dy, name, terrain).
        return [(dx, dy, {"name": name, "terrain": terrain, "row": 0, "col": index})
                for index, (dx, dy, name, terrain) in enumerate(catalog["tiles"])]
    if all("dx" in cell for cell in catalog["cells"]):
        # A context scene: explicit raw offsets around a city at the centre.
        return [(cell["dx"], cell["dy"], cell) for cell in catalog["cells"]]
    # Rows sit a tile and a half apart (six raw units) so a district's front
    # never reaches the crop of the row in front of it; a row whose raw y is odd
    # shifts half a tile sideways to keep Civ III tile parity.
    rows = max(cell["row"] for cell in catalog["cells"]) + 1
    columns = max(cell["col"] for cell in catalog["cells"]) + 1
    if rows > 4 or columns > 7:
        raise ValueError(f"{catalog['id']}: at most 4 rows x 7 columns fit the Lab map")
    cells = []
    for cell in catalog["cells"]:
        dx, dy = 4 * cell["col"] - 2 * (columns - 1), 6 * cell["row"] - 3 * (rows - 1)
        cells.append((dx + (dx + dy) % 2, dy, cell))
    return cells


def _water(catalog: dict) -> tuple[set, set]:
    """Coastal cells stand on coast with a small bay toward the viewer; sea
    cells (offshore platforms) stand in open sea."""
    tiles, sea = set(), set()
    for dx, dy, cell in layout(catalog):
        if cell.get("terrain") == "Coast":
            for ox, oy in ((0, 0), (1, 1), (-1, 1), (0, 2), (2, 2), (-2, 2), (1, 3), (-1, 3)):
                tiles.add((dx + ox, dy + oy))
        elif cell.get("terrain") == "Sea":
            sea |= {(dx + ox, dy + oy) for ox in range(-4, 5) for oy in range(-4, 5)
                    if abs(ox) + abs(oy) <= 4 and (ox + oy) % 2 == 0}
    return tiles, sea


def install_cases(catalogs: dict) -> None:
    """Register the catalogs as Lab roster cases for this process only."""
    from Renderer.lab.studies.resources import roster
    if getattr(roster, "district_catalogs", None) is not None:
        # Already registered in this process: extend the catalog map only.
        roster.district_catalogs.update(catalogs)
        data = roster.native()
        data["batches"].update({f"district-{key}": [cell["name"] for cell in spec["cells"]]
                                for key, spec in catalogs.items()})
        roster.CASES = roster.cases()
        return
    roster.district_catalogs = catalogs = dict(catalogs)
    original_native, original_layout = roster.native, roster.native_layout
    original_terrain, original_viewport = roster.terrain, roster.viewport
    data = {**original_native(), "batches": {**original_native()["batches"],
                                              **{f"district-{key}": [cell["name"] for cell in spec["cells"]]
                                                 for key, spec in catalogs.items()}}}
    roster.native = lambda: data
    roster.native_layout.cache_clear()

    def native_layout(case):
        key = case.removeprefix("native-district-")
        if case.startswith("native-district-") and key in catalogs:
            # Ghost cells only mark grid places (city sheets); they carry no composition.
            return tuple((dx, dy, cell["name"], cell.get("terrain", "Grassland"))
                         for dx, dy, cell in layout(catalogs[key]) if not cell.get("ghost"))
        return original_layout(case)

    def terrain(case, dx, dy, base, real):
        key = case.removeprefix("native-district-")
        if case.startswith("native-district-") and key in catalogs:
            water, sea = _water(catalogs[key])
            for x, y, cell in layout(catalogs[key]):
                if (x, y) == (dx, dy):
                    # Civ III's snow-capped bonus bit (0x10) for a snowy mountain.
                    return (*TERRAIN[cell.get("terrain", "Grassland")], 0x10) if cell.get("snow") else \
                        TERRAIN[cell.get("terrain", "Grassland")]
            if (dx, dy) in sea:
                return TERRAIN["Sea"]
            if (dx, dy) in water:
                return TERRAIN["Coast"] if dy < max(y for _, y in water) else TERRAIN["Sea"]
            return TERRAIN["Grassland"]
        return original_terrain(case, dx, dy, base, real)

    def viewport(case, zoom):
        key = case.removeprefix("native-district-")
        if case.startswith("native-district-") and key in catalogs:
            cells = layout(catalogs[key])
            return ((max(abs(x) for x, _, _ in cells) + 3) * zoom,
                    (max(abs(y) for _, y, _ in cells) // 2 + 3) * zoom)
        return original_viewport(case, zoom)

    roster.native_layout = native_layout
    roster.terrain = terrain
    roster.viewport = viewport
    roster.CASES = roster.cases()


def shots(spec: dict, zooms=(128, 256)) -> list[dict]:
    """Every frame of a catalog: a tag, zoom, hour, district scale and ground switch."""
    if spec.get("shots"):
        return [{"tag": "", "hour": 12, "scale": SCALE, "ground": True, **shot} for shot in spec["shots"]]
    return [{"tag": "", "zoom": zoom, "hour": 12, "scale": SCALE, "ground": True} for zoom in zooms]


def frame(catalog_id: str, shot: dict) -> Path:
    return OUT / "renders" / f"{catalog_id}{shot['tag']}-z{shot['zoom']}-h{shot['hour']:02}"


def vm_render(call, tolerated_frame: Path | None, key: str) -> None:
    """Take turns on the shared VM: wait while another session's prlctl exec runs,
    and retry once when a render collides. The tile-overlay path (context
    scenes) skips the roster ownership witness, so its frame is kept."""
    import subprocess
    import time
    for attempt in range(2):
        for _ in range(240):
            busy = subprocess.run(["pgrep", "-f", "prlctl exec"], capture_output=True, text=True).stdout.split()
            if not busy:
                break
            time.sleep(5)
        started = time.time()
        try:
            call()
            return
        except ValueError as exc:
            log = tolerated_frame.parent / "native.log" if tolerated_frame is not None else None
            # Only a frame this run wrote counts (an older one may be left in the folder).
            if log is not None and tolerated_frame.is_file() and log.is_file() and \
                    tolerated_frame.stat().st_mtime >= started - 1 and \
                    "0 fallback, output=" in log.read_text(errors="replace"):
                # City scenes skip the roster ownership witness; the frame itself completed.
                print(f"note {key}: {exc}", flush=True)
                return
            if attempt:
                raise
            print(f"retry {key}: {exc}", flush=True)
            time.sleep(10)


def current(marker: dict, summary: dict | None = None) -> bool:
    """A frame of the current layout: v11 adds ground cover to drafted cells,
    so a v10 frame with no drafted cell is still current."""
    cells = (summary or marker)["cells"].values()
    if marker.get("shot", {}).get("hour", 12) >= 20 and not marker.get("lights"):
        return False   # a night frame from before the emissive night lights
    return marker.get("v") == 11 or marker.get("v") == 10 and not any(cell.get("draft") for cell in cells)


NIGHT_TREE = OUT / "tree-night"


def night_renderer():
    """The night tree's DLL, preview and shader root (night_tree.py): the current
    candidate plus emissive night lights; None until that tree is built."""
    from Renderer import renderer
    import shutil
    dll = NIGHT_TREE / "Renderer/native/build/candidate/C3XRenderer.dll"
    built = NIGHT_TREE / "Renderer/lab/.cache/native_preview.exe"
    if not (dll.is_file() and built.is_file()):
        return None
    preview = OUT / "night_preview.exe"
    if not preview.is_file() or preview.read_bytes() != built.read_bytes():
        shutil.copy2(built, preview)
    return dll, preview, {"C3X_RENDERER_SHADER_SOURCE_ROOT": "..\\..\\" + renderer.relative(NIGHT_TREE).replace("/", "\\")}


def render(catalogs: dict, zooms=(128, 256), only=None) -> None:
    from Renderer import renderer
    import shutil
    install_cases(catalogs)
    # A pinned copy of the current candidate: peers rebuild the shared one, and
    # this study neither prepares category assets nor rebuilds the candidate.
    dll = OUT / "dll/C3XRenderer.dll"
    if not dll.is_file():
        dll.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    night = night_renderer()
    if night:
        dll, preview, shader_env = night
    else:
        preview, shader_env = None, {}
    compiler = Compiler()
    for key, spec in catalogs.items():
        if only and key not in only:
            continue
        case = f"native-district-{key}"
        extra = None
        if spec.get("cities"):
            # Production cities beside the districts: the Lab tile-overlay path
            # places both cities and named compositions.
            from Renderer.lab.studies.resources import roster
            extra = {"C3X_LAB_TILE_OVERLAYS": "1", "C3X_LAB_TILE_RESOURCES": roster.spec(case),
                     "C3X_LAB_TILE_CITIES": ";".join(",".join(str(v) for v in city) for city in spec["cities"])}
        baked = None
        for shot in shots(spec, zooms):
            if baked != (shot["scale"], shot["ground"]):
                summary = bake(spec, compiler, shot["scale"], shot["ground"])
                compiler.save()
                baked = (shot["scale"], shot["ground"])
                print(json.dumps({k: v for k, v in summary.items() if k != "cells"}), flush=True)
            empty = [name for name, item in summary["cells"].items() if not item["instances"]]
            target = frame(key, shot)
            hour, zoom = shot["hour"], shot["zoom"]
            done = target / "cells.json"
            if done.is_file() and load(done).get("shot", {}).get("tag") == shot["tag"] and current(load(done), summary) and \
                    (target / f"{case}-h{hour:02}-z{zoom}.bmp").is_file():
                continue
            # A city-only scene has no compositions; an empty Lab pack does not load.
            pack = CATALOG.name if any(item["instances"] for item in summary["cells"].values()) else ""
            try:
                vm_render(lambda: renderer.native_render("resources", case, hour, zoom, target,
                                                         resource_pack=pack, candidate=dll, preview=preview,
                                                         extra_env={**shader_env, **(extra or {})} or None),
                          target / f"{case}-h{hour:02}-z{zoom}.bmp" if extra else None, key)
            except ValueError as exc:
                # A frame that fails twice (the shared VM crashes now and then) is left
                # stale for the next run; the remaining catalogs still render.
                print(f"FAILED {key} z{zoom} h{hour}: {exc}", flush=True)
                continue
            (target / "cells.json").write_text(json.dumps({**summary, "shot": shot, "v": 11, "lights": bool(night)}, indent=1) + "\n")
            census_log = (target / "native.log").read_text(errors="replace")
            missing = [cell["name"] for cell in spec["cells"]
                       if cell["name"] not in empty and f" {cell['name']} replaced=1" not in census_log]
            if missing and not extra:
                print(f"WARNING {key} z{zoom}: no composition drawn for {missing}", flush=True)


CITY_CULTURES = ("american", "european", "roman", "middle_eastern", "asian")


def neighborhoods(zooms=(128, 256)) -> None:
    """The production city pack's culture ladders (four eras x Town / walled Town /
    City / Metropolis): the art a city-art Neighborhood would borrow."""
    from Renderer import renderer
    from Renderer.lab.studies.city_readability import cases as city_cases
    dll = OUT / "dll/C3XRenderer.dll"
    for culture in CITY_CULTURES:
        case = f"city-ladder-{culture}"
        centre = city_cases.centre(case)
        env = {"C3X_RENDERER_PREVIEW_OBJECTS": "", "C3X_RENDERER_PREVIEW_CITY": "",
               "C3X_LAB_TILE_CITIES": city_cases.city_spec(case, centre),
               "C3X_LAB_TILE_RESOURCES": city_cases.resources("cities", case, centre)}
        for zoom in zooms:
            target = OUT / "renders" / f"nbh-{culture}-z{zoom}"
            if (target / f"{case}-h12-z{zoom}.bmp").is_file():
                continue
            vm_render(lambda: renderer.native_render("cities", case, 12, zoom, target, candidate=dll,
                                                     center=(16 + centre[0], 16 + centre[1]), extra_env=env),
                      None, case)


def scene_box(spec: dict, image, zoom: int) -> tuple[int, int, int, int]:
    """Crop of a scene around the view centre: (left, top, width, height)."""
    width, height = spec.get("crop", (5.0, 2.75))
    return (image[0] // 2 - int(zoom * width / 2), image[1] // 2 - int(zoom * (height / 2 + .25)),
            int(zoom * width), int(zoom * height))


def crops(catalogs: dict, zooms=(128, 256)) -> dict:
    """Crop every cell (or a context scene) from its frames into Renderer/lab/out/districts/crops/."""
    from Renderer.lab.studies.resources import audit
    folder = OUT / "crops"
    folder.mkdir(parents=True, exist_ok=True)
    index = {}
    for key, spec in catalogs.items():
        for shot in shots(spec, zooms):
            zoom, hour = shot["zoom"], shot["hour"]
            bmp = frame(key, shot) / f"native-district-{key}-h{hour:02}-z{zoom}.bmp"
            if not bmp.is_file():
                continue
            image = audit.read_bmp(bmp)
            label = f"{shot['tag']}-z{zoom}-h{hour:02}"
            if spec.get("cities") or spec.get("tiles"):
                name = f"{key}{label}.png"
                audit.write_png(folder / name, *audit.crop(image, *scene_box(spec, image, zoom)))
                index.setdefault(key, {})[label] = name
                continue
            audit.write_png(folder / f"{key}{label}-overview.png", *image)
            for dx, dy, cell in layout(spec):
                cx, cy = image[0] // 2 + dx * zoom // 2, image[1] // 2 + dy * zoom // 4
                piece = audit.crop(image, cx - zoom * 11 // 16, cy - zoom * 15 // 16, zoom * 11 // 8, zoom * 5 // 4)
                name = f"{key}{label}-{cell['row']}-{cell['col']}.png"
                audit.write_png(folder / name, *piece)
                index.setdefault(key, {}).setdefault(label, {})[f"{cell['row']},{cell['col']}"] = name
    (folder / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


def main() -> int:
    from Renderer.lab.studies.districts import plan
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("census", "compile", "render", "crops", "list", "neighborhoods"))
    parser.add_argument("--catalog", action="append")
    parser.add_argument("--zoom", action="append", type=int)
    args = parser.parse_args()
    catalogs = plan.catalogs()
    if args.catalog:
        unknown = set(args.catalog) - set(catalogs)
        if unknown:
            parser.error(f"unknown catalogs {sorted(unknown)}")
    zooms = tuple(args.zoom or (128, 256))
    if args.action == "census":
        print(len(census()))
    elif args.action == "list":
        for key, spec in catalogs.items():
            print(key, len(spec["cells"]), spec["title"])
    elif args.action == "compile":
        compiler = Compiler()
        for key, spec in catalogs.items():
            if args.catalog and key not in args.catalog:
                continue
            for cell in spec["cells"]:
                for part in cell["parts"]:
                    try:
                        compiler.root(part["record"])
                    except ValueError as exc:
                        print(f"{key} {cell['name']}: {exc}", file=sys.stderr)
            compiler.save()
            print(key, "compiled", flush=True)
    elif args.action == "render":
        render(catalogs, zooms, args.catalog)
    elif args.action == "neighborhoods":
        neighborhoods(zooms)
    else:
        crops(catalogs, zooms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
