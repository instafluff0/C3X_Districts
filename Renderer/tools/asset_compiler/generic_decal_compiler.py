#!/usr/bin/env python3
"""Compile proven source surface decals into a source-independent C3X pack.

The source adapter is deliberately fail-closed. It accepts one reflected decal
descriptor, no conventional model/mesh/material containers, and an empty
terrain-edit vector. Source identifiers and paths are confined to the external
build report; runtime-facing JSON contains stable C3X IDs and normalized DDS
roles only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.clutter_blp_extractor import (
    StaticPackage,
    TYPE_MATERIAL,
    TYPE_MESH,
    TYPE_MODEL,
    TYPE_PRIM_GROUP,
    TYPE_TEXTURE,
    decode_texture_entry,
    extract_civbig_texture,
    landmark_base_model,
    sha256_bytes,
    write_json,
)
from Renderer.tools.asset_compiler.grassland_pack_builder import (
    validate_runtime_independence,
)


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = Path(__file__).with_name("decal_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "DecalsNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "decals" / "decal_build.json"
MAC_ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
WINDOWS_ASSETS_ROOT = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets"
)

TYPE_DECAL_VECTOR = "LandmarkPackageEntry::DecalDesc2VectorEntry"
TYPE_DECAL = "DecalDesc2"
TYPE_TERRAIN_EDIT_VECTOR = "LandmarkPackageEntry::TerrainEditDesc3VectorEntry"
DECAL_BYTES = 108
SAFE_ID = re.compile(r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]?[a-z0-9]+)*)*$")
CONVENTIONAL_CONTAINER_TYPES = (TYPE_MODEL, TYPE_MESH, TYPE_PRIM_GROUP, TYPE_MATERIAL)
TEXTURE_SLOTS = {
    "base_color": {"offset": 0x50, "class": "Decal_BaseColor", "required": True},
    "height": {"offset": 0x54, "class": "Decal_Heightmap", "required": True},
    "specular": {"offset": 0x58, "class": "Decal_Spec", "required": False},
    "fog_color": {"offset": 0x5C, "class": "Decal_FOWColor", "required": False},
}


def _is_relative_safe_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\\" not in value


def load_mapping(path: Path) -> dict[str, Any]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema") != "c3x.source_decal_mapping.v0":
        raise ValueError("Unsupported decal mapping schema")
    units = mapping.get("source_units_per_tile")
    if not isinstance(units, (int, float)) or not math.isfinite(units) or units <= 0:
        raise ValueError("source_units_per_tile must be a positive finite number")
    sources = mapping.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"package", "shared_data", "artdef"}:
        raise ValueError("Mapping sources must define package, shared_data, and artdef")
    if not all(isinstance(value, str) and _is_relative_safe_path(value) for value in sources.values()):
        raise ValueError("Mapping source paths must be safe forward-slash relative paths")

    groups = mapping.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Mapping must define at least one decal group")
    seen_groups: set[str] = set()
    seen_assets: set[str] = set()
    seen_sources: set[str] = set()
    for group in groups:
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or not SAFE_ID.fullmatch(group_id):
            raise ValueError(f"Invalid decal group ID: {group_id!r}")
        if group_id in seen_groups:
            raise ValueError(f"Duplicate decal group ID: {group_id}")
        seen_groups.add(group_id)
        for field in ("artdef_set", "collection"):
            if not isinstance(group.get(field), str) or not group[field]:
                raise ValueError(f"Decal group {group_id} has no {field}")
        assets = group.get("assets")
        if not isinstance(assets, list) or not assets:
            raise ValueError(f"Decal group {group_id} has no assets")
        for asset in assets:
            source_asset = asset.get("source_asset")
            asset_id = asset.get("asset_id")
            if not isinstance(source_asset, str) or not source_asset:
                raise ValueError(f"Decal group {group_id} has an invalid source asset")
            if not isinstance(asset_id, str) or not SAFE_ID.fullmatch(asset_id):
                raise ValueError(f"Invalid normalized decal ID: {asset_id!r}")
            if source_asset in seen_sources:
                raise ValueError(f"Duplicate source decal: {source_asset}")
            if asset_id in seen_assets:
                raise ValueError(f"Duplicate normalized decal ID: {asset_id}")
            seen_sources.add(source_asset)
            seen_assets.add(asset_id)
    return mapping


def _artdef_value(value: ET.Element) -> str | None:
    child = next((child for child in value if child.tag != "m_ParamName"), None)
    if child is None:
        return None
    return child.attrib.get("text", child.text)


def _parse_bool(value: str, field: str) -> bool:
    lowered = value.lower()
    if lowered not in ("true", "false"):
        raise ValueError(f"ArtDef {field} is not a Boolean: {value}")
    return lowered == "true"


def read_artdef_group(
    path: Path,
    set_name: str,
    collection_name: str,
    source_to_asset: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = ET.parse(path).getroot()
    matches = [
        element
        for element in root.iter("Element")
        if element.find("m_Name") is not None
        and element.find("m_Name").attrib.get("text") == set_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one ArtDef decal set {set_name}, found {len(matches)}")
    collections = [
        element
        for element in matches[0].findall("./m_ChildCollections/Element")
        if element.find("m_CollectionName") is not None
        and element.find("m_CollectionName").attrib.get("text") == collection_name
    ]
    if len(collections) != 1:
        raise ValueError(f"Expected one {collection_name} collection in {set_name}")

    by_source: dict[str, dict[str, Any]] = {}
    all_sources: list[str] = []
    for item in collections[0].findall("Element"):
        values: dict[str, str] = {}
        for value in item.findall("./m_Fields/m_Values/Element"):
            parameter = value.find("m_ParamName")
            decoded = _artdef_value(value)
            if parameter is not None and decoded is not None:
                values[parameter.attrib["text"]] = decoded
        source_asset = values.get("Asset")
        if source_asset is None:
            continue
        all_sources.append(source_asset)
        if source_asset not in source_to_asset:
            continue
        if source_asset in by_source:
            raise ValueError(f"ArtDef source decal occurs more than once in {set_name}: {source_asset}")
        required = ("Scale", "Count", "ScaleVariation")
        missing = [field for field in required if field not in values]
        if missing:
            raise ValueError(f"Incomplete ArtDef placement for {source_asset}: {missing}")
        by_source[source_asset] = {
            "asset": source_to_asset[source_asset],
            "scale": float(values["Scale"]),
            "count": int(values["Count"]),
            "scale_variation": float(values["ScaleVariation"]),
            "low_end_reduction": float(values.get("LowendReduction", 0.0)),
            "show_decal": _parse_bool(values.get("ShowDecal", "false"), "ShowDecal"),
            "priority": int(values.get("Priority", 0)),
            "width": float(values.get("Width", 0.0)),
            "rotate_mode": values.get("RotateMode", "RotateZ"),
            "is_center_model": _parse_bool(values.get("IsCenterModel", "false"), "IsCenterModel"),
            "allow_overlap": _parse_bool(values.get("AllowOverlap", "false"), "AllowOverlap"),
            "min_count": int(values.get("MinCount", 0)),
        }
    missing_sources = [source for source in source_to_asset if source not in by_source]
    if missing_sources:
        raise ValueError(f"Mapped decals absent from ArtDef set {set_name}: {missing_sources}")
    placements = [by_source[source] for source in source_to_asset]
    return placements, {
        "set": set_name,
        "collection": collection_name,
        "mapped_sources": list(source_to_asset),
        "collection_source_count": len(all_sources),
    }


def _round_values(values: tuple[float, ...]) -> list[float]:
    return [round(value, 8) for value in values]


def decode_decal_descriptor(
    raw: bytes,
    texture_resolver: Callable[[int], dict[str, Any]],
    source_units_per_tile: float,
    required_roles: tuple[str, ...] = ("base_color", "height"),
) -> dict[str, Any]:
    if len(raw) != DECAL_BYTES:
        raise ValueError(f"Expected a {DECAL_BYTES}-byte decal descriptor, found {len(raw)}")
    values = struct.unpack_from("<8f", raw, 0x14)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Decal descriptor contains a non-finite bound")
    footprint_source = values[:4]
    content_source = values[4:]
    for label, bounds in (("footprint", footprint_source), ("content", content_source)):
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError(f"Decal {label} bounds are not ordered")

    accepted: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for role, slot in TEXTURE_SLOTS.items():
        index = struct.unpack_from("<I", raw, slot["offset"])[0]
        entry = texture_resolver(index)
        if entry.get("class") != slot["class"]:
            evidence[role] = {
                "status": "class_mismatch",
                "index": index,
                "expected_class": slot["class"],
                "actual": entry,
            }
            if role in required_roles:
                raise ValueError(
                    f"Required decal {role} slot has class {entry.get('class')}, "
                    f"expected {slot['class']}"
                )
            continue
        accepted[role] = entry
        evidence[role] = {"status": "accepted", "index": index, "source": entry}
    scale = 1.0 / source_units_per_tile
    return {
        "footprint_bounds": _round_values(tuple(value * scale for value in footprint_source)),
        "content_bounds": _round_values(tuple(value * scale for value in content_source)),
        "source_bounds": {
            "footprint": _round_values(footprint_source),
            "content": _round_values(content_source),
        },
        "textures": accepted,
        "texture_slots": evidence,
    }


def _relative_asset_document(asset_id: str) -> str:
    return "decals/" + asset_id.replace("/", "_").replace(".", "_") + ".json"


def _texture_target(role: str, source: Path) -> str:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return f"textures/decals/{role}_{digest}.dds"


def build_decal(
    package: StaticPackage,
    shared_data: Path,
    pack: Path,
    source_asset: str,
    asset_id: str,
    source_units_per_tile: float,
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    package.select_direct_string(source_asset)
    landmark, landmark_user_data, base_model = landmark_base_model(package)
    geometry = {
        target_type: len(package.pointer_fields(base_model, target_type))
        for target_type in CONVENTIONAL_CONTAINER_TYPES
    }
    if any(geometry.values()):
        raise ValueError(f"Source asset {source_asset} mixes decal and conventional geometry")

    decal_fields = package.pointer_fields(landmark_user_data, TYPE_DECAL_VECTOR)
    terrain_fields = package.pointer_fields(landmark_user_data, TYPE_TERRAIN_EDIT_VECTOR)
    if len(decal_fields) != 1 or len(terrain_fields) != 1:
        raise ValueError(
            f"Source asset {source_asset} must have one decal vector and one terrain-edit vector"
        )
    decal_vector = decal_fields[0][1]
    terrain_vector = terrain_fields[0][1]
    terrain_raw = package.bytes_for(terrain_vector)
    if any(terrain_raw):
        raise ValueError(f"Source asset {source_asset} has unsupported terrain-edit data")
    decal_pointers = package.pointer_fields(decal_vector, TYPE_DECAL)
    if len(decal_pointers) != 1:
        raise ValueError(f"Source asset {source_asset} must resolve to one decal descriptor array")
    decal_pointer = decal_pointers[0][1]
    decal_count = package.allocations[decal_pointer - 1]["element_count"]
    if decal_count < 1:
        raise ValueError(f"Source asset {source_asset} has no decal descriptors")

    texture_array = package.unique_allocation(TYPE_TEXTURE)
    relative_documents = []
    descriptor_reports = []
    for descriptor_index in range(decal_count):
        descriptor = decode_decal_descriptor(
            package.array_element(decal_pointer, descriptor_index),
            lambda index: decode_texture_entry(package, texture_array, index),
            source_units_per_tile,
        )
        runtime_channels: dict[str, dict[str, Any]] = {}
        texture_report: dict[str, dict[str, Any]] = {}
        for role, entry in descriptor["textures"].items():
            cache_key = (entry["name"], entry["class"])
            source = shared_data / entry["name"]
            if not source.is_file():
                raise ValueError(f"Missing standalone decal texture for {source_asset}: {source}")
            relative = _texture_target(role, source)
            if cache_key not in texture_cache:
                info = extract_civbig_texture(source, pack / relative)
                texture_cache[cache_key] = (relative, info)
            cached_relative, info = texture_cache[cache_key]
            if cached_relative != relative:
                raise ValueError("One source texture resolved inconsistently across decal roles")
            runtime_channels[role] = {
                "texture": relative,
                "format": info["format_name"],
                "color_space": info["color_space"],
                "address_u": "clamp",
                "address_v": "clamp",
            }
            texture_report[role] = {
                **entry,
                **info,
                "source": str(source),
                "runtime_texture": relative,
            }

        component_id = asset_id if decal_count == 1 else f"{asset_id}/part_{descriptor_index + 1:02d}"
        decal_document = {
            "schema": "c3x.decal.v0",
            "asset_id": component_id,
            "coordinate_system": {
                "handedness": "right",
                "up_axis": "+Z",
                "horizontal_axes": ["X", "Y"],
                "position_unit": "tile",
                "uv_origin": "upper_left",
            },
            "footprint": {
                "bounds_xy": descriptor["footprint_bounds"],
                "content_bounds_xy": descriptor["content_bounds"],
            },
            "uv_rect": [0.0, 0.0, 1.0, 1.0],
            "channels": runtime_channels,
            "render": {
                "projection": "terrain_surface",
                "blend_mode": "alpha",
                "depth_bias_policy": "terrain_decal",
            },
            "provenance": {
                "kind": "local_normalized_import",
                "adapter": "c3x.decal.v0",
                "source_format_dependency": None,
            },
        }
        relative_document = _relative_asset_document(component_id)
        write_json(pack / relative_document, decal_document)
        relative_documents.append(relative_document)
        descriptor_reports.append(
            {
                "index": descriptor_index,
                "source_bounds": descriptor["source_bounds"],
                "normalized_bounds": {
                    "footprint": descriptor["footprint_bounds"],
                    "content": descriptor["content_bounds"],
                },
                "texture_slots": descriptor["texture_slots"],
                "textures": texture_report,
            }
        )
    manifest_asset = (
        {"type": "decal", "decal": relative_documents[0]}
        if decal_count == 1
        else {"type": "compound_decal", "decals": relative_documents}
    )
    return manifest_asset, {
        "source_asset": source_asset,
        "asset_id": asset_id,
        "pointer_chain": {
            "landmark": landmark,
            "landmark_user_data": landmark_user_data,
            "base_model": base_model,
            "decal_vector": decal_vector,
            "terrain_edit_vector": terrain_vector,
            "decal_array": decal_pointer,
        },
        "conventional_container_counts": geometry,
        "terrain_edit_vector_empty": True,
        "descriptor_count": decal_count,
        "descriptors": descriptor_reports,
    }


def _ensure_report_outside_pack(report_path: Path, pack: Path) -> None:
    try:
        report_path.resolve().relative_to(pack.resolve())
    except ValueError:
        return
    raise ValueError("Source build report must be outside the runtime pack")


def compile_decal_pack(
    assets_root: Path,
    mapping_path: Path,
    pack: Path,
    report_path: Path,
) -> dict[str, Any]:
    mapping = load_mapping(mapping_path)
    _ensure_report_outside_pack(report_path, pack)
    sources = {name: assets_root / relative for name, relative in mapping["sources"].items()}
    for name, path in sources.items():
        if not path.is_file() and name != "shared_data":
            raise FileNotFoundError(path)
        if name == "shared_data" and not path.is_dir():
            raise FileNotFoundError(path)

    first_source = mapping["groups"][0]["assets"][0]["source_asset"]
    package = StaticPackage(sources["package"], first_source)
    assets: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    asset_reports = []
    artdef_reports = []
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for group in mapping["groups"]:
        source_to_asset = {
            asset["source_asset"]: asset["asset_id"] for asset in group["assets"]
        }
        placements, artdef_report = read_artdef_group(
            sources["artdef"], group["artdef_set"], group["collection"], source_to_asset
        )
        artdef_reports.append(artdef_report)
        variants = []
        for asset in group["assets"]:
            asset_id = asset["asset_id"]
            manifest_asset, asset_report = build_decal(
                package,
                sources["shared_data"],
                pack,
                asset["source_asset"],
                asset_id,
                float(mapping["source_units_per_tile"]),
                texture_cache,
            )
            assets[asset_id] = manifest_asset
            variants.append(asset_id)
            asset_reports.append(asset_report)
        groups[group["group_id"]] = {
            "variants": variants,
            "placements": placements,
            "status": "normalized_verified_set",
        }

    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "DecalsNormalized",
        "display_name": "Normalized Terrain Decals",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "assets": assets,
        "decal_groups": groups,
    }
    write_json(pack / "manifest.json", manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime pack is source-dependent: " + "; ".join(independence_errors))

    report = {
        "schema": "c3x.source_decal_build.v0",
        "mapping": {"path": str(mapping_path), "sha256": sha256_bytes(mapping_path.read_bytes())},
        "sources": {
            name: {
                "path": str(path),
                "sha256": sha256_bytes(path.read_bytes()) if path.is_file() else None,
            }
            for name, path in sources.items()
        },
        "allocation_table": {
            "package_offset": package.table_offset,
            "allocation_count": len(package.allocations),
            "stripe_bases": package.stripe_bases,
        },
        "normalization": {
            "source_units_per_tile": mapping["source_units_per_tile"],
            "runtime_units": "tile",
            "admission_profile": "single_surface_decal_no_geometry_no_terrain_edit",
        },
        "artdef_groups": artdef_reports,
        "assets": asset_reports,
        "outputs": {
            "pack": str(pack),
            "assets": len(assets),
            "groups": len(groups),
            "unique_textures": len(texture_cache),
        },
        "runtime_independence": "passed",
    }
    write_json(report_path, report)
    return report


def default_assets_root() -> Path:
    return MAC_ASSETS_ROOT if MAC_ASSETS_ROOT.is_dir() else WINDOWS_ASSETS_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=default_assets_root())
    parser.add_argument("--mapping", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = compile_decal_pack(args.assets_root, args.mapping, args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, struct.error, ET.ParseError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"Compiled {report['outputs']['assets']} decals in {report['outputs']['groups']} groups "
        f"with {report['outputs']['unique_textures']} unique textures"
    )
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
