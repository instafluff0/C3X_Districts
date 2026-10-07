#!/usr/bin/env python3
"""Import the Civ VI resource art that composition profiles name.

Source-specific importer for build_resource_compositions.py. It resolves each
requested Resources.artdef entry (Base, then DLC) through its clutter set and
its terrain/feature ClutterVariants, then extracts every rock model with its
authored vertical origin, so Civ VI's own burial depth survives, and every
ground decal. The output, ResourceCompositionSources, is an ignored
intermediate pack: normalized meshes/materials/textures, decal records, and
per-source placement sets with generic terrain conditions. The runtime never
reads it; source names stay in this pack and its report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Renderer.tools.asset_compiler import clutter_blp_extractor
from Renderer.tools.asset_compiler.generic_decal_compiler import build_decal
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.resource_pack_builder import DEFAULT_ASSETS_ROOT

PROFILES = ROOT / "Renderer/inventory/resource_composition_profiles.json"
OUTPUT = ROOT / "Renderer/packs/ResourceCompositionSources"
SCHEMA = "c3x.resource_composition_sources.v0"
# Generic names for Civ VI variant conditions; unknown conditions are kept verbatim.
FEATURES = {"FEATURE_FOREST": "forest", "FEATURE_JUNGLE": "jungle", "FEATURE_MARSH": "marsh",
            "FEATURE_FLOODPLAINS": "flood_plain"}
# Generic rocks and bushes that Civ VI composes with resources; trees and jungle
# clumps belong to the forest/jungle features and stay ancillary.
ACCESSORIES = ("Boulder", "Shrub")
# Named entries come from the clutter package, else from the terrain asset set
# (the oasis's rock ring and plants).
PACKAGES = ("environment/clutter.blp", "terrain/TerrainAssetSet_Base.blp")


def _text(element: ET.Element | None) -> str:
    return "" if element is None else element.get("text", element.text or "")


def _value(value: ET.Element):
    kind = value.get("class", "")
    if kind.endswith("BLPEntryValue"):
        return _text(value.find("m_EntryName"))
    if kind.endswith("ArtDefReferenceValue"):
        return _text(value.find("m_ElementName"))
    for child in value:
        if child.tag in ("m_Value", "m_fValue", "m_nValue", "m_bValue"):
            return _text(child)
    return None


@lru_cache(maxsize=None)
def _roots(path: Path) -> dict[str, ET.Element]:
    entries = {}
    for collection in ET.parse(path).getroot().findall("./m_RootCollections/Element"):
        for element in collection.findall("./Element"):
            entries[_text(element.find("m_Name"))] = element
    return entries


def _definition(assets: Path, artdef: str, name: str) -> tuple[str, ET.Element]:
    """Last definition wins, as DLC ArtDefs extend Base."""
    found = None
    for path in [assets / "Base/ArtDefs" / artdef] + sorted((assets / "DLC").glob(f"*/ArtDefs/{artdef}")):
        if path.is_file() and name in _roots(path):
            found = ("Base" if path.parts[-3] == "Base" else "DLC/" + path.parts[-3], _roots(path)[name])
    if found is None:
        raise KeyError(f"{artdef} has no entry {name}")
    return found


def _children(element: ET.Element, collection: str) -> list[dict]:
    result = []
    for child in element.findall("./m_ChildCollections/Element"):
        if _text(child.find("m_CollectionName")) == collection:
            for item in child.findall("./Element"):
                values = {_text(v.find("m_ParamName")): _value(v) for v in item.findall("./m_Fields/m_Values/Element")}
                result.append({"name": _text(item.find("m_Name")), **values})
    return result


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _condition(variant: dict) -> dict:
    terrain = (variant.get("Terrain") or "").removeprefix("TERRAIN_").lower()
    return {"feature": FEATURES.get(variant.get("Feature") or "", (variant.get("Feature") or "").lower() or None),
            "terrain": terrain.removesuffix("_hills") or None, "hills": terrain.endswith("_hills")}


def requested_sources(profiles: dict) -> list[str]:
    names = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("source", "decal_source", "accent_source") and isinstance(item, str) and item:
                    names.add(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(profiles.get("resources", {}))
    visit(profiles.get("alternates", {}))
    for catalog in profiles.get("catalogs", {}).values():
        visit(catalog.get("entries", {}))
    return sorted(names)


def build_terrain_asset(package: IndexedStaticPackage, shared: Path, output: Path, entry: str,
                        asset_id: str) -> tuple[dict, dict]:
    """A Civ VI terrain asset (an oasis's rock ring or its plants): one named
    vertex/index buffer drawn as one batch. The authored origin is kept, so the
    pieces of one set line up when they share a placement."""
    cbe = clutter_blp_extractor

    def named(type_name: str, vertex: bool) -> dict:
        array = package.unique_allocation(type_name)
        for index in range(package.allocations[array - 1]["element_count"]):
            item = cbe.decode_buffer_entry(package, array, index, vertex)
            if item["name"] == entry:
                return item
        raise ValueError(f"{entry}: no {type_name}")
    vertex, index = named(cbe.TYPE_VERTEX_BUFFER, True), named(cbe.TYPE_INDEX_BUFFER, False)
    primitive = {"first_index": 0, "index_count": index["count"], "base_vertex": 0, "vertex_count": vertex["count"]}
    mesh, evidence = cbe.normalize_mesh(
        package.big_data(vertex["offset"], vertex["bytes"]), package.big_data(index["offset"], index["bytes"]),
        vertex, index, primitive, asset_id, allow_wrapping_uvs=True, drop_degenerate_triangles=True,
        preserve_vertical_origin=True)
    centre = [value / cbe.SOURCE_UNITS_PER_TILE for value in evidence["normalization"]["horizontal_center"]]
    for item in mesh["vertices"]:
        item["position"] = [round(item["position"][0] + centre[0], 8), round(item["position"][1] + centre[1], 8),
                            item["position"][2]]
    for bound in mesh["bounds"].values():
        bound[0], bound[1] = round(bound[0] + centre[0], 8), round(bound[1] + centre[1], 8)
    # The batch whose name hash matches names a material: base colour, then opacity.
    batches = package.unique_allocation("FOWMeshVisSystem::PackageBatch")
    raw = package.bytes_for(batches)
    step = len(raw) // package.allocations[batches - 1]["element_count"]
    material_index = next(struct.unpack_from("<I", raw, at + 52)[0] for at in range(0, len(raw), step)
                          if struct.unpack_from("<I", raw, at + 60)[0] == vertex["name_hash"])
    raw = package.bytes_for(package.unique_allocation("FOWMeshVisSystem::PackageMaterial"))
    base, opacity = struct.unpack_from("<II", raw, material_index * 24)
    textures = package.unique_allocation(cbe.TYPE_TEXTURE)
    stem = _slug(entry)
    material = {"schema": "c3x.material.v0", "name": stem, "alpha_mode": "opaque",
                "status": "normalized_local_import"}
    for role, texture in (("base_color", base), ("opacity", opacity)):
        if texture == 0xFFFFFFFF:
            continue
        name = cbe.decode_texture_entry(package, textures, texture)["name"]
        relative = f"textures/features/{stem}_{role}.dds"
        cbe.extract_civbig_texture(shared / name, output / relative)
        material[role] = {"texture": relative, "uv_channel": "uv0"}
        evidence[role] = name
    if "opacity" in material:
        material["alpha_mode"], material["alpha_cutoff"] = "mask", 0.5
    cbe.write_json(output / f"meshes/features/{stem}.json", mesh)
    cbe.write_json(output / f"materials/features/{stem}.json", material)
    return {"type": "feature", "mesh": f"meshes/features/{stem}.json",
            "material": f"materials/features/{stem}.json"}, evidence


def requested_entries(profiles: dict) -> list[str]:
    """Named Civ VI models a profile places directly ("extra_pieces", "set_piece")."""
    entries = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "extra_pieces":
                    entries.update(extra["entry"] for extra in item)
                elif key == "set_piece":
                    entries.update(item["entries"])
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(profiles.get("resources", {}))
    visit(profiles.get("alternates", {}))
    return sorted(entries)


def build(output: Path = OUTPUT, assets: Path = DEFAULT_ASSETS_ROOT, profiles_path: Path = PROFILES) -> dict:
    requested = requested_sources(json.loads(profiles_path.read_text()))
    sources, entries = {}, {}       # entry -> (origin, kind) for extraction
    for name in requested:
        origin, resource = _definition(assets, "Resources.artdef", name)
        clutter = [item["XrefName"] for item in _children(resource, "Clutter") if item.get("XrefName")]
        if len(clutter) != 1:
            raise ValueError(f"{name}: expected one base clutter set, found {clutter}")
        record = {"origin": origin, "base": clutter[0], "variants": [], "sets": {}}
        variants = [(item["XrefName"], _condition(item)) for item in _children(resource, "ClutterVariants")]
        for set_name in [clutter[0]] + [set_name for set_name, _ in variants]:
            if set_name in record["sets"]:
                continue
            set_origin, element = _definition(assets, "Clutter.artdef", set_name)
            placements = []
            for plant in _children(element, "Plants"):
                entry = plant.get("Asset")
                if not entry:
                    continue
                accessory = entry.startswith(ACCESSORIES)
                ancillary = not entry.startswith("RES_") and not accessory
                placements.append({
                    "asset": None if ancillary else "source/" + _slug(entry), "source_entry": entry,
                    "kind": "ancillary" if ancillary else "accessory" if accessory else
                            "decal" if "decal" in entry.lower() else "model",
                    "scale": float(plant.get("Scale") or 1), "count": int(plant.get("Count") or 0),
                    "min_count": int(plant.get("MinCount") or 0),
                    "scale_variation": float(plant.get("ScaleVariation") or 0),
                    "center": (plant.get("IsCenterModel") or "").lower() == "true"})
                if not ancillary:
                    entries.setdefault(entry, set_origin)
            record["sets"][set_name] = {
                "origin": set_origin, "placements": placements,
                "blocks": [item.get("Set") for item in _children(element, "Block") if item.get("Set")]}
        record["variants"] = [{"when": condition, "set": set_name} for set_name, condition in variants]
        sources[name] = record

    for entry in requested_entries(json.loads(profiles_path.read_text())):
        entries.setdefault(entry, "Base")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    packages: dict[tuple[str, str], IndexedStaticPackage] = {}
    manifest_assets, evidence, texture_cache = {}, {}, {}
    # Resource entries anchor each package's string table; generic boulders cannot.
    for entry, origin in sorted(entries.items(), key=lambda item: (not item[0].startswith("RES_"), item[0])):
        found = None
        for candidate, blp in ((c, b) for c in dict.fromkeys([origin, "Base"]) for b in PACKAGES):
            blps = assets / ("Base" if candidate == "Base" else candidate) / "Platforms"
            blps = next((p for p in (blps / "Windows/BLPs", blps / "windows/BLPs") if p.is_dir()), None)
            if blps is None:
                continue
            try:
                package = packages.get((candidate, blp))
                if package is None:
                    package = packages[(candidate, blp)] = IndexedStaticPackage(blps / blp, entry)
                package.select_direct_string(entry)
            except (OSError, KeyError, ValueError):
                continue
            found = (package, blps / "SHARED_DATA", blp)
            break
        if found is None:
            evidence[entry] = {"status": "missing"}
            continue
        package, shared, blp = found
        asset_id = "source/" + _slug(entry)
        try:
            if blp != PACKAGES[0]:
                asset, report = build_terrain_asset(package, shared, output, entry, asset_id)
            elif "decal" in entry.lower():
                asset, report = build_decal(package, shared, output, entry, asset_id,
                                            clutter_blp_extractor.SOURCE_UNITS_PER_TILE, texture_cache)
            else:
                spec = {"source_name": entry, "asset_id": asset_id, "manifest_key": asset_id,
                        "stem": _slug(entry), "group": "resource"}
                asset, report = clutter_blp_extractor.build_feature(
                    package, shared, output, spec, allow_wrapping_uvs=True, allow_optional_maps=True,
                    preserve_vertical_origin=True)
            manifest_assets[asset_id] = asset
            evidence[entry] = {"status": "normalized", "asset": asset_id}
        except (OSError, ValueError, KeyError) as error:
            evidence[entry] = {"status": "unsupported", "reason": str(error)[:240]}
    manifest = {"schema": SCHEMA, "importer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "requested": requested + requested_entries(json.loads(profiles_path.read_text())),
                "assets": manifest_assets, "sources": sources}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    report = {"schema": SCHEMA + ".report", "assets_root": str(assets), "entries": evidence,
              "summary": {"sources": len(sources), "entries": len(evidence),
                          "normalized": sum(e["status"] == "normalized" for e in evidence.values())}}
    (output.parent / (output.name + "_report.json")).write_text(json.dumps(report, indent=1) + "\n")
    return manifest


def ensure(output: Path = OUTPUT, profiles_path: Path = PROFILES) -> dict:
    """Rebuild only when the requested sources or this importer changed."""
    try:
        manifest = json.loads((output / "manifest.json").read_text())
        profiles = json.loads(profiles_path.read_text())
        if manifest.get("schema") == SCHEMA and \
                manifest.get("requested") == requested_sources(profiles) + requested_entries(profiles) and \
                manifest.get("importer") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            return manifest
    except (OSError, ValueError):
        pass
    return build(output, profiles_path=profiles_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output, args.assets_root)
    report = json.loads((args.output.parent / (args.output.name + "_report.json")).read_text())
    print(json.dumps(report["summary"], indent=1))
    for entry, item in report["entries"].items():
        if item["status"] != "normalized":
            print(entry, item)
    return 0 if manifest["assets"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
