#!/usr/bin/env python3
"""Inventory the installed Civ VI dune controls and desert sand decals.

This is a read-only evidence probe. It does not convert or redistribute source
art, and it deliberately distinguishes reflected package facts from inferred
engine behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.clutter_blp_extractor import (
    StaticPackage,
    TYPE_MATERIAL,
    TYPE_MESH,
    TYPE_MODEL,
    TYPE_PRIM_GROUP,
    TYPE_TEXTURE,
    decode_texture_entry,
    landmark_base_model,
)


MAC_ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
WINDOWS_ASSETS_ROOT = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets"
)
SAND_ASSETS = tuple(f"TER_Desert_Decal{index}" for index in range(10, 15))
TYPE_DECAL_VECTOR = "LandmarkPackageEntry::DecalDesc2VectorEntry"
TYPE_DECAL = "DecalDesc2"
TYPE_TERRAIN_EDIT_VECTOR = "LandmarkPackageEntry::TerrainEditDesc3VectorEntry"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_text(element: ET.Element, tag: str, attribute: str = "text") -> str | None:
    child = element.find(tag)
    return None if child is None else child.get(attribute)


def _named_element(root: ET.Element, name: str) -> ET.Element:
    matches = [
        element
        for element in root.iter("Element")
        if _direct_text(element, "m_Name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one ArtDef element named {name}, found {len(matches)}")
    return matches[0]


def _field_values(element: ET.Element) -> dict[str, Any]:
    result: dict[str, Any] = {}
    values = element.find("m_Fields/m_Values")
    if values is None:
        return result
    for value in values.findall("Element"):
        parameter = _direct_text(value, "m_ParamName")
        if not parameter:
            continue
        if value.find("m_EntryName") is not None:
            result[parameter] = _direct_text(value, "m_EntryName") or ""
        elif value.find("m_ElementName") is not None:
            result[parameter] = _direct_text(value, "m_ElementName") or ""
        elif value.find("m_fValue") is not None:
            result[parameter] = float(value.findtext("m_fValue", "0"))
        elif value.find("m_nValue") is not None:
            result[parameter] = int(value.findtext("m_nValue", "0"))
        elif value.find("m_bValue") is not None:
            result[parameter] = value.findtext("m_bValue", "false").lower() == "true"
        elif value.find("m_Value") is not None:
            result[parameter] = _direct_text(value, "m_Value") or ""
    return result


def parse_dune_style(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    collections = [
        element
        for element in root.iter("Element")
        if _direct_text(element, "m_CollectionName") == "DuneDesertHills"
    ]
    if len(collections) != 1:
        raise ValueError(f"Expected one DuneDesertHills collection, found {len(collections)}")
    defaults = [
        element
        for element in collections[0].findall("Element")
        if _direct_text(element, "m_Name") == "Default"
    ]
    if len(defaults) != 1:
        raise ValueError(f"Expected one DuneDesertHills/Default entry, found {len(defaults)}")
    fields = _field_values(defaults[0])
    expected = ("DesertHillsMtl", "DuneBase", "DuneHeight", "DuneWidth", "DuneNoise", "DuneAngle")
    missing = [name for name in expected if name not in fields]
    if missing:
        raise ValueError(f"DuneDesertHills is missing fields: {missing}")

    layer_refs = []
    for element in root.iter("Element"):
        values = _field_values(element)
        if values.get("DesertHillsLayer") == "Default":
            layer_refs.append(_direct_text(element, "m_Name"))
    return {
        "material": fields["DesertHillsMtl"],
        "parameters": {name: fields[name] for name in expected[1:]},
        "referenced_by_layers": [name for name in layer_refs if name],
    }


def parse_sand_placements(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    desert = _named_element(root, "CLUTTER_DESERT")
    placements = []
    for element in desert.iter("Element"):
        name = _direct_text(element, "m_Name")
        fields = _field_values(element)
        asset = fields.get("Asset")
        if name and "Sand" in name and asset in SAND_ASSETS:
            placements.append({
                "name": name,
                "asset": asset,
                "scale": fields.get("Scale"),
                "count": fields.get("Count"),
                "scale_variation": fields.get("ScaleVariation"),
                "show_decal": fields.get("ShowDecal"),
                "priority": fields.get("Priority"),
                "rotate_mode": fields.get("RotateMode"),
                "allow_overlap": fields.get("AllowOverlap"),
            })
    placements.sort(key=lambda item: item["asset"])
    if [item["asset"] for item in placements] != list(SAND_ASSETS):
        raise ValueError("CLUTTER_DESERT does not contain the expected five sand decals")
    return placements


def probe_sand_package(path: Path) -> list[dict[str, Any]]:
    package = StaticPackage(path, SAND_ASSETS[0])
    texture_array = package.unique_allocation(TYPE_TEXTURE)
    conventional_types = (TYPE_MODEL, TYPE_MESH, TYPE_PRIM_GROUP, TYPE_MATERIAL)
    result = []
    for asset in SAND_ASSETS:
        package.select_direct_string(asset)
        _landmark, landmark_user_data, base_model = landmark_base_model(package)
        user_raw = package.bytes_for(landmark_user_data)
        decal_vector = struct.unpack_from("<Q", user_raw, 0x10)[0]
        terrain_edit_vector = struct.unpack_from("<Q", user_raw, 0x18)[0]
        if package.type_name(decal_vector) != TYPE_DECAL_VECTOR:
            raise ValueError(f"{asset} does not have the expected decal vector")
        if package.type_name(terrain_edit_vector) != TYPE_TERRAIN_EDIT_VECTOR:
            raise ValueError(f"{asset} does not have the expected terrain-edit vector")
        vector_raw = package.bytes_for(decal_vector)
        decal = struct.unpack_from("<Q", vector_raw, 0x08)[0]
        if package.type_name(decal) != TYPE_DECAL:
            raise ValueError(f"{asset} does not resolve to a DecalDesc2")
        decal_raw = package.bytes_for(decal)
        textures = {
            role: decode_texture_entry(
                package,
                texture_array,
                struct.unpack_from("<I", decal_raw, offset)[0],
            )
            for role, offset in (("base_color", 0x50), ("height", 0x54), ("fog", 0x5C))
        }
        geometry = {
            target_type: len(package.pointer_fields(base_model, target_type))
            for target_type in conventional_types
        }
        result.append({
            "asset": asset,
            "decal_count": package.allocations[decal - 1]["element_count"],
            "terrain_edit_vector_empty": not any(package.bytes_for(terrain_edit_vector)),
            "conventional_container_counts": geometry,
            "textures": {
                role: {"name": entry["name"], "class": entry["class"]}
                for role, entry in textures.items()
            },
            "bounds_evidence": list(struct.unpack_from("<8f", decal_raw, 0x14)),
        })
    return result


def run_probe(assets_root: Path) -> dict[str, Any]:
    terrain_style = assets_root / "Base" / "ArtDefs" / "TerrainStyle.artdef"
    clutter_artdef = assets_root / "Base" / "ArtDefs" / "Clutter.artdef"
    clutter_blp = (
        assets_root / "Base" / "Platforms" / "Windows" / "BLPs" / "environment" / "clutter.blp"
    )
    for source in (terrain_style, clutter_artdef, clutter_blp):
        if not source.is_file():
            raise FileNotFoundError(source)
    return {
        "schema": "c3x.dune_source_probe.v0",
        "sources": {
            "terrain_style": {"path": str(terrain_style), "sha256": sha256_file(terrain_style)},
            "clutter_artdef": {"path": str(clutter_artdef), "sha256": sha256_file(clutter_artdef)},
            "clutter_package": {"path": str(clutter_blp), "sha256": sha256_file(clutter_blp)},
        },
        "dune_style": parse_dune_style(terrain_style),
        "sand_placements": parse_sand_placements(clutter_artdef),
        "sand_package": probe_sand_package(clutter_blp),
        "interpretation": {
            "confirmed": "Dedicated dune controls and height-bearing decals; no conventional model container for the five sand decals.",
            "inferred": "The engine likely synthesizes macro dune relief procedurally; the exact formula is not present in ArtDef/package data.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assets_root = args.assets_root
    if assets_root is None:
        assets_root = MAC_ASSETS_ROOT if MAC_ASSETS_ROOT.is_dir() else WINDOWS_ASSETS_ROOT
    report = run_probe(assets_root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
