#!/usr/bin/env python3
"""Compile route ArtDefs and decal materials into generic path-graph styles."""

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
    TYPE_TEXTURE,
    decode_texture_entry,
)
from Renderer.tools.asset_compiler import c3x_asset_compiler, civblp_material_resolver
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = Path(__file__).with_name("route_style_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "RouteStylesNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "route_styles" / "build.json"
MAC_ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
WINDOWS_ASSETS_ROOT = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets"
)
SAFE_ID = re.compile(r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]?[a-z0-9]+)*)*$")
SAFE_TOKEN = re.compile(r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*$")
TYPE_ROUTE_MATERIAL = "RouteDecalMaterialPackageEntry"
MATERIAL_SLOTS = {
    "base_color": (0x48, "Decal_BaseColor", True),
    "height": (0x4C, "Decal_Heightmap", True),
    "specular": (0x50, "Decal_Spec", False),
    "fog_color": (0x54, "Decal_FOWColor", False),
}
SEGMENT_KINDS = {
    "TILED_PATH": "tiled_path",
    "FADEOUT": "fadeout",
    "TRANSITION": "transition",
}
ROUTE_STATES = {"NORMAL": "normal", "PILLAGED": "pillaged"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\\" not in value


def load_mapping(path: Path) -> dict[str, Any]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema") != "c3x.source_route_style_mapping.v0":
        raise ValueError("Unsupported route-style mapping schema")
    catalogs = mapping.get("catalogs")
    if not isinstance(catalogs, list) or not catalogs:
        raise ValueError("Route-style mapping has no catalogs")
    source_keys: set[tuple[str, str]] = set()
    asset_ids: set[str] = set()
    for catalog in catalogs:
        artdefs = catalog.get("artdefs")
        if (
            not isinstance(artdefs, list)
            or not artdefs
            or not all(isinstance(value, str) and _safe_relative(value) for value in artdefs)
            or len(artdefs) != len(set(artdefs))
        ):
            raise ValueError("Route-style artdefs must be unique safe relative paths")
        material_sources = catalog.get("material_sources")
        if not isinstance(material_sources, list) or not material_sources:
            raise ValueError("Route-style catalog has no material sources")
        source_packages: set[str] = set()
        for material_source in material_sources:
            value = material_source.get("source_package")
            if not isinstance(value, str) or not _safe_relative(value):
                raise ValueError("Route-style source_package must be a safe relative path")
            if set(material_source) != {"source_package"}:
                raise ValueError("Route-style material source has unknown fields")
            if material_source["source_package"] in source_packages:
                raise ValueError("Route-style catalog repeats a material source package")
            source_packages.add(material_source["source_package"])
        scale = catalog.get("source_units_per_tile")
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Route-style source_units_per_tile must be positive and finite")
        styles = catalog.get("styles")
        if not isinstance(styles, list) or not styles:
            raise ValueError("Route-style catalog has no styles")
        for style in styles:
            source_route = style.get("source_route")
            asset_id = style.get("asset_id")
            if not isinstance(source_route, str) or not source_route:
                raise ValueError("Route-style entry has no source route")
            if not isinstance(asset_id, str) or not SAFE_ID.fullmatch(asset_id):
                raise ValueError(f"Invalid normalized route-style ID: {asset_id!r}")
            for field in ("route_kind", "style_stage"):
                if not isinstance(style.get(field), str) or not SAFE_TOKEN.fullmatch(style[field]):
                    raise ValueError(f"Invalid route-style {field}: {style.get(field)!r}")
            source_key = (catalog["artdefs"][-1], source_route)
            if source_key in source_keys:
                raise ValueError(f"Duplicate source route: {source_route}")
            if asset_id in asset_ids:
                raise ValueError(f"Duplicate normalized route-style ID: {asset_id}")
            source_keys.add(source_key)
            asset_ids.add(asset_id)
    return mapping


def _root_collections(root: ET.Element) -> dict[str, ET.Element]:
    container = root.find("m_RootCollections")
    if container is None:
        raise ValueError("Route ArtDef has no root collections")
    result: dict[str, ET.Element] = {}
    for collection in container.findall("Element"):
        name = collection.find("m_CollectionName")
        if name is None or "text" not in name.attrib:
            raise ValueError("Route ArtDef contains an unnamed root collection")
        key = name.attrib["text"]
        if key in result:
            raise ValueError(f"Duplicate route ArtDef root collection: {key}")
        result[key] = collection
    return result


def _items(collection: ET.Element) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for item in collection.findall("Element"):
        name = item.find("m_Name")
        if name is None or "text" not in name.attrib:
            raise ValueError("Route ArtDef collection contains an unnamed item")
        key = name.attrib["text"]
        if key in result:
            raise ValueError(f"Duplicate route ArtDef item: {key}")
        result[key] = item
    return result


def _child_collection(item: ET.Element, name: str) -> ET.Element:
    children = item.find("m_ChildCollections")
    matches = [] if children is None else [
        collection
        for collection in children.findall("Element")
        if collection.find("m_CollectionName") is not None
        and collection.find("m_CollectionName").attrib.get("text") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one nested route collection {name}, found {len(matches)}")
    return matches[0]


def _decode_parameter(value: ET.Element) -> tuple[str, Any]:
    parameter = value.find("m_ParamName")
    if parameter is None or "text" not in parameter.attrib:
        raise ValueError("Route ArtDef value has no parameter name")
    name = parameter.attrib["text"]
    if value.find("m_ElementName") is not None:
        decoded = ("reference", value.find("m_ElementName").attrib["text"])
    elif value.find("m_EntryName") is not None:
        decoded = ("package_entry", value.find("m_EntryName").attrib["text"])
    elif value.find("m_x") is not None and value.find("m_y") is not None:
        decoded = ("coord2", [float(value.find("m_x").text), float(value.find("m_y").text)])
    elif value.find("m_fValue") is not None:
        decoded = ("float", float(value.find("m_fValue").text))
    elif value.find("m_nValue") is not None:
        decoded = ("int", int(value.find("m_nValue").text))
    elif value.find("m_bValue") is not None:
        raw = value.find("m_bValue").text
        if raw not in ("true", "false"):
            raise ValueError(f"Route ArtDef Boolean {name} is malformed")
        decoded = ("bool", raw == "true")
    elif value.find("m_Value") is not None:
        node = value.find("m_Value")
        decoded = ("string", node.attrib.get("text", node.text))
    else:
        raise ValueError(f"Unsupported route ArtDef parameter representation: {name}")
    return name, decoded


def _parameters(item: ET.Element) -> dict[str, tuple[str, Any]]:
    values = item.find("./m_Fields/m_Values")
    result: dict[str, tuple[str, Any]] = {}
    for value in [] if values is None else values.findall("Element"):
        name, decoded = _decode_parameter(value)
        if name in result:
            raise ValueError(f"Duplicate route ArtDef parameter: {name}")
        result[name] = decoded
    return result


def _required(
    parameters: dict[str, tuple[str, Any]], name: str, expected_kind: str
) -> Any:
    if name not in parameters:
        raise ValueError(f"Route ArtDef is missing required parameter {name}")
    kind, value = parameters[name]
    if kind != expected_kind:
        raise ValueError(f"Route ArtDef parameter {name} is {kind}, expected {expected_kind}")
    return value


def _merged_collection_items(paths: list[Path]) -> dict[str, dict[str, ET.Element]]:
    merged: dict[str, dict[str, ET.Element]] = {}
    for path in paths:
        root = ET.parse(path).getroot()
        for name, collection in _root_collections(root).items():
            merged.setdefault(name, {}).update(_items(collection))
    return merged


def read_route_style(path: Path | list[Path], source_route: str) -> dict[str, Any]:
    paths = [path] if isinstance(path, Path) else path
    if not paths:
        raise ValueError("Route ArtDef merge has no inputs")
    collections = _merged_collection_items(paths)
    required_collections = (
        "RoutePieces",
        "Route Descriptions",
        "RouteTypes",
        "GameCoreRouteTranslator",
    )
    missing = [name for name in required_collections if name not in collections]
    if missing:
        raise ValueError(f"Route ArtDef lacks required collections: {missing}")
    translators = collections["GameCoreRouteTranslator"]
    if source_route not in translators:
        raise ValueError(f"Route ArtDef has no translator for {source_route}")
    translator_params = _parameters(translators[source_route])
    route_type = _required(translator_params, "RouteSystem Type", "reference")

    route_types = collections["RouteTypes"]
    descriptions = collections["Route Descriptions"]
    if route_type not in route_types or route_type not in descriptions:
        raise ValueError(f"Route type {route_type} lacks type or description data")
    priority = _required(_parameters(route_types[route_type]), "Priority", "int")
    description = descriptions[route_type]
    description_params = _parameters(description)
    if _required(description_params, "Type", "reference") != route_type:
        raise ValueError(f"Route description {route_type} references a different type")
    tile_uvs = _required(description_params, "TileUVs", "bool")
    width = _required(description_params, "Width", "float")
    blocker_width = _required(description_params, "BlockerWidth", "float")
    if width <= 0 or blocker_width < 0:
        raise ValueError(f"Route description {route_type} has invalid widths")

    piece_items = collections["RoutePieces"]
    decoded_pieces: dict[str, dict[str, Any]] = {}
    decoded_segments = []
    seen_segment_keys: set[tuple[str, str]] = set()
    for segment_name, segment in _items(_child_collection(description, "Route Segments")).items():
        params = _parameters(segment)
        source_kind = _required(params, "Segment Type", "string")
        source_state = _required(params, "State", "string")
        if source_kind not in SEGMENT_KINDS or source_state not in ROUTE_STATES:
            raise ValueError(
                f"Unsupported route segment kind/state: {source_kind}/{source_state}"
            )
        kind = SEGMENT_KINDS[source_kind]
        state = ROUTE_STATES[source_state]
        if (state, kind) in seen_segment_keys:
            raise ValueError(f"Duplicate route segment recipe: {state}/{kind}")
        seen_segment_keys.add((state, kind))
        layers = []
        for layer_name, layer in _items(_child_collection(segment, "Layers")).items():
            layer_params = _parameters(layer)
            height = _required(layer_params, "Height", "float")
            piece_name = _required(layer_params, "RoutePiece", "reference")
            if not math.isfinite(height) or piece_name not in piece_items:
                raise ValueError(
                    f"Route layer {layer_name} has invalid height {height} or missing piece "
                    f"{piece_name} in {source_route}/{route_type}"
                )
            if piece_name not in decoded_pieces:
                piece_params = _parameters(piece_items[piece_name])
                decoded_pieces[piece_name] = {
                    "source_piece": piece_name,
                    "source_material": _required(
                        piece_params, "DefaultMaterial", "package_entry"
                    ),
                    "uv_1": _required(piece_params, "DefaultUV_X1Y1", "coord2"),
                    "uv_2": _required(piece_params, "DefaultUV_X2Y2", "coord2"),
                }
            layers.append(
                {
                    "source_layer": layer_name,
                    "height": height,
                    "source_piece": piece_name,
                }
            )
        if not layers:
            raise ValueError(f"Route segment {segment_name} has no layers")
        decoded_segments.append(
            {
                "source_segment": segment_name,
                "state": state,
                "kind": kind,
                "layers": layers,
            }
        )
    if not decoded_segments:
        raise ValueError(f"Route description {route_type} has no segments")
    return {
        "source_route": source_route,
        "source_type": route_type,
        "priority": priority,
        "tile_uvs": tile_uvs,
        "width": width,
        "blocker_width": blocker_width,
        "segments": decoded_segments,
        "pieces": list(decoded_pieces.values()),
        "artdef_collection_counts": {
            name: len(collections[name]) for name in required_collections
        },
    }


def decode_route_material_slots(
    raw: bytes, texture_resolver: Callable[[int], dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if len(raw) != 96:
        raise ValueError("Route decal material entry must be 96 bytes")
    channels: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for role, (offset, expected_class, required) in MATERIAL_SLOTS.items():
        index = struct.unpack_from("<I", raw, offset)[0]
        texture = texture_resolver(index)
        if texture["class"] != expected_class:
            evidence[role] = {
                "status": "class_mismatch",
                "expected_class": expected_class,
                "source": texture,
            }
            if required:
                raise ValueError(
                    f"Route material {role} class is {texture['class']}, "
                    f"expected {expected_class}"
                )
            continue
        channels[role] = texture
        evidence[role] = {"status": "accepted", "source": texture}
    return channels, evidence


def decode_route_material_entry(
    package: IndexedStaticPackage, source_material: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    name_pointer = package.select_direct_string(source_material)
    references = package.references_to(name_pointer, TYPE_ROUTE_MATERIAL)
    if len(references) != 1 or references[0][1] != 0x38:
        raise ValueError(f"Route material {source_material} does not resolve uniquely")
    entry_pointer = references[0][0]
    texture_array = package.unique_allocation(TYPE_TEXTURE)
    channels, evidence = decode_route_material_slots(
        package.bytes_for(entry_pointer),
        lambda index: decode_texture_entry(package, texture_array, index),
    )
    return channels, {
        "entry": entry_pointer,
        "name_pointer": name_pointer,
        "texture_slots": evidence,
    }


def _decode_embedded_texture(
    package: IndexedStaticPackage, texture_array: int, index: int
) -> tuple[bytes, dict[str, Any]]:
    count = package.allocations[texture_array - 1]["element_count"]
    records = [package.array_element(texture_array, item) for item in range(count)]
    metadata_offset, _metadata_evidence = (
        civblp_material_resolver.infer_texture_metadata_offset(records)
    )
    resource_offset_field, resource_size_field, _resource_evidence = (
        civblp_material_resolver.infer_embedded_resource_fields(
            records,
            metadata_offset,
            len(package.data) - package.big_data_file_offset,
        )
    )
    record = records[index]
    metadata = civblp_material_resolver.decode_texture_metadata(record, metadata_offset)
    relative_offset = struct.unpack_from("<Q", record, resource_offset_field)[0]
    byte_count = struct.unpack_from("<Q", record, resource_size_field)[0]
    expected = civblp_material_resolver.expected_bc_bytes(
        metadata["width"],
        metadata["height"],
        metadata["mip_count"],
        metadata["format"]["block_bytes"],
    )
    if byte_count != expected:
        raise ValueError("Route texture embedded byte count disagrees with its mip chain")
    payload = package.big_data(relative_offset, byte_count)
    return payload, {
        "width": metadata["width"],
        "height": metadata["height"],
        "mip_count": metadata["mip_count"],
        "dxgi_format": metadata["format"]["dxgi"],
        "format_name": metadata["format"]["name"],
        "color_space": metadata["format"]["color_space"],
        "payload_bytes": byte_count,
        "relative_offset": relative_offset,
        "metadata_offset": metadata_offset,
        "resource_offset_field": resource_offset_field,
        "resource_size_field": resource_size_field,
    }


def _compile_material(
    package: IndexedStaticPackage,
    pack: Path,
    source_material: str,
    material_index: int,
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]],
) -> tuple[str, tuple[int, int], dict[str, Any]]:
    source_channels, evidence = decode_route_material_entry(package, source_material)
    channels = {}
    texture_evidence = {}
    base_dimensions: tuple[int, int] | None = None
    texture_array = package.unique_allocation(TYPE_TEXTURE)
    for role, entry in source_channels.items():
        payload, info = _decode_embedded_texture(package, texture_array, entry["index"])
        key = (str(package.source), str(entry["index"]))
        digest = _sha256(payload)[:16]
        relative = f"textures/routes/{role}_{digest}.dds"
        if key not in texture_cache:
            dds_info = {
                "width": info["width"],
                "height": info["height"],
                "mip_count": info["mip_count"],
                "dxgi_format": info["dxgi_format"],
                "payload_bytes": info["payload_bytes"],
            }
            dds = c3x_asset_compiler.make_dds_dx10_header(dds_info) + payload
            output = pack / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(dds)
            texture_cache[key] = (
                relative,
                {
                    **info,
                    "payload_sha256": _sha256(payload),
                    "dds_sha256": _sha256(dds),
                },
            )
        cached_relative, cached_info = texture_cache[key]
        if cached_relative != relative:
            raise ValueError("Route texture cache resolved one source inconsistently")
        channels[role] = {
            "texture": relative,
            "format": cached_info["format_name"],
            "color_space": cached_info["color_space"],
            "address_u": "clamp",
            "address_v": "clamp",
        }
        texture_evidence[role] = {**entry, **cached_info}
        dimensions = (cached_info["width"], cached_info["height"])
        if base_dimensions is None:
            base_dimensions = dimensions
        elif dimensions != base_dimensions:
            raise ValueError("Route material channels do not share atlas dimensions")
    if base_dimensions is None:
        raise ValueError("Route material has no accepted texture channels")
    relative_document = f"materials/routes/material_{material_index:02d}.json"
    _write_json(
        pack / relative_document,
        {
            "schema": "c3x.material.v0",
            "name": f"route_material_{material_index:02d}",
            "channels": channels,
            "alpha_mode": "blend",
            "provenance": {
                "kind": "local_normalized_import",
                "adapter": "c3x.route_style.v0",
                "source_format_dependency": None,
            },
        },
    )
    return relative_document, base_dimensions, {
        **evidence,
        "source_material": source_material,
        "textures": texture_evidence,
    }


def _normalized_uv(point: list[float], dimensions: tuple[int, int]) -> list[float]:
    width, height = dimensions
    if width < 2 or height < 2 or len(point) != 2:
        raise ValueError("Route atlas dimensions or UV point are invalid")
    if not all(math.isfinite(value) for value in point):
        raise ValueError("Route atlas UV point is non-finite")
    if point[0] < 0 or point[0] > width - 1 or point[1] < 0 or point[1] > height - 1:
        raise ValueError(f"Route atlas UV point {point} lies outside {dimensions}")
    return [round(point[0] / (width - 1), 8), round(point[1] / (height - 1), 8)]


def _compile_style(
    source: dict[str, Any],
    mapping: dict[str, Any],
    source_units_per_tile: float,
    materials: dict[str, tuple[str, tuple[int, int]]],
    pack: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    piece_indices = {
        piece["source_piece"]: index for index, piece in enumerate(source["pieces"])
    }
    pieces = []
    for source_piece in source["pieces"]:
        material_name = source_piece["source_material"]
        if material_name not in materials:
            raise ValueError(f"Route piece references uncompiled material {material_name}")
        material_path, dimensions = materials[material_name]
        pieces.append(
            {
                "id": f"piece_{len(pieces):02d}",
                "material": material_path,
                "uv_endpoints": [
                    _normalized_uv(source_piece["uv_1"], dimensions),
                    _normalized_uv(source_piece["uv_2"], dimensions),
                ],
                "atlas_dimensions": list(dimensions),
            }
        )
    segments = []
    for segment in source["segments"]:
        segments.append(
            {
                "state": segment["state"],
                "kind": segment["kind"],
                "layers": [
                    {
                        "height_tiles": round(layer["height"] / source_units_per_tile, 8),
                        "piece": f"piece_{piece_indices[layer['source_piece']]:02d}",
                    }
                    for layer in segment["layers"]
                ],
            }
        )
    asset_id = mapping["asset_id"]
    stem = asset_id.replace("/", "_").replace(".", "_")
    relative_document = f"route_styles/{stem}.json"
    document = {
        "schema": "c3x.route_style.v0",
        "asset_id": asset_id,
        "classification": {
            "route_kind": mapping["route_kind"],
            "style_stage": mapping["style_stage"],
        },
        "topology": {
            "input": "connected_centerline_graph",
            "connection_order": "renderer_defined",
            "terrain_conforming": True,
            "junction_policy": "compose_incident_branches",
        },
        "width_tiles": round(source["width"] / source_units_per_tile, 8),
        "tile_uvs": source["tile_uvs"],
        "states": sorted({segment["state"] for segment in segments}),
        "pieces": pieces,
        "segment_recipes": segments,
        "render": {
            "projection": "terrain_surface",
            "depth_bias_policy": "route_surface",
            "missing_component_policy": "fallback_whole_route_family",
        },
        "provenance": {
            "kind": "local_normalized_import",
            "adapter": "c3x.route_style.v0",
            "source_format_dependency": None,
        },
    }
    _write_json(pack / relative_document, document)
    evidence = {
        "source_route": source["source_route"],
        "source_type": source["source_type"],
        "asset_id": asset_id,
        "priority": source["priority"],
        "source_width": source["width"],
        "source_blocker_width": source["blocker_width"],
        "artdef_collection_counts": source["artdef_collection_counts"],
        "segments": source["segments"],
        "pieces": source["pieces"],
    }
    return {"type": "route_style", "style": relative_document}, evidence


def compile_route_styles(
    assets_root: Path,
    mapping_path: Path,
    pack: Path,
    report_path: Path,
) -> dict[str, Any]:
    mapping = load_mapping(mapping_path)
    try:
        report_path.resolve().relative_to(pack.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Route-style source report must be outside the runtime pack")
    assets = {}
    reports = []
    catalog_reports = []
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    material_cache: dict[tuple[str, str], tuple[str, tuple[int, int]]] = {}
    material_counter = 0
    for catalog in mapping["catalogs"]:
        artdef_paths = [assets_root / path for path in catalog["artdefs"]]
        material_sources = [
            {
                "package": assets_root / source["source_package"],
            }
            for source in catalog["material_sources"]
        ]
        for required in artdef_paths + [source["package"] for source in material_sources]:
            if not required.is_file():
                raise FileNotFoundError(required)
        decoded = [
            read_route_style(artdef_paths, style["source_route"])
            for style in catalog["styles"]
        ]
        material_names = list(
            dict.fromkeys(
                piece["source_material"]
                for source in decoded
                for piece in source["pieces"]
            )
        )
        materials: dict[str, tuple[str, tuple[int, int]]] = {}
        material_reports = []
        assignments: dict[int, list[str]] = {}
        source_bytes = [source["package"].read_bytes() for source in material_sources]
        for material_name in material_names:
            marker = material_name.encode("ascii") + b"\0"
            matches = [index for index, data in enumerate(source_bytes) if data.count(marker) == 1]
            if len(matches) != 1:
                raise ValueError(
                    f"Route material {material_name} resolves to {len(matches)} source packages"
                )
            assignments.setdefault(matches[0], []).append(material_name)
        package_reports = []
        for source_index, assigned_names in assignments.items():
            material_source = material_sources[source_index]
            package = IndexedStaticPackage(material_source["package"], assigned_names[0])
            for material_name in assigned_names:
                cache_key = (str(material_source["package"]), material_name)
                if cache_key in material_cache:
                    path, dimensions = material_cache[cache_key]
                    evidence = {
                        "source_material": material_name,
                        "status": "reused",
                        "material": path,
                    }
                else:
                    path, dimensions, evidence = _compile_material(
                        package,
                        pack,
                        material_name,
                        material_counter,
                        texture_cache,
                    )
                    material_counter += 1
                    material_cache[cache_key] = (path, dimensions)
                materials[material_name] = (path, dimensions)
                material_reports.append(evidence)
            package_reports.append(
                {
                    "package": str(material_source["package"]),
                    "package_sha256": _sha256(source_bytes[source_index]),
                    "allocation_count": len(package.allocations),
                    "assigned_materials": assigned_names,
                }
            )
        for source, style_mapping in zip(decoded, catalog["styles"]):
            manifest_asset, evidence = _compile_style(
                source,
                style_mapping,
                float(catalog["source_units_per_tile"]),
                materials,
                pack,
            )
            assets[style_mapping["asset_id"]] = manifest_asset
            reports.append(evidence)
        catalog_reports.append(
            {
                "artdefs": [
                    {"path": str(path), "sha256": _sha256(path.read_bytes())}
                    for path in artdef_paths
                ],
                "packages": package_reports,
                "materials": material_reports,
                "transition_doodads": "separate_geometry_import_required",
            }
        )
    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "RouteStylesNormalized",
        "display_name": "Normalized Route Styles",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "assets": assets,
    }
    _write_json(pack / "manifest.json", manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime route pack is source-dependent: " + "; ".join(independence_errors))
    report = {
        "schema": "c3x.source_route_style_build.v0",
        "mapping": {"path": str(mapping_path), "sha256": _sha256(mapping_path.read_bytes())},
        "catalogs": catalog_reports,
        "styles": reports,
        "outputs": {
            "pack": str(pack),
            "styles": len(assets),
            "materials": material_counter,
            "textures": len(texture_cache),
            "road_styles": sum(
                style["route_kind"] == "road"
                for catalog in mapping["catalogs"]
                for style in catalog["styles"]
            ),
            "railroad_styles": sum(
                style["route_kind"] == "railroad"
                for catalog in mapping["catalogs"]
                for style in catalog["styles"]
            ),
        },
        "runtime_independence": "passed",
        "runtime_integration": "not_enabled",
    }
    _write_json(report_path, report)
    return report


def default_assets_root() -> Path:
    return MAC_ASSETS_ROOT if MAC_ASSETS_ROOT.is_dir() else WINDOWS_ASSETS_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=default_assets_root())
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = compile_route_styles(args.assets_root, args.mapping, args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    outputs = report["outputs"]
    print(
        f"Compiled {outputs['styles']} route styles: {outputs['road_styles']} road, "
        f"{outputs['railroad_styles']} railroad; {outputs['materials']} materials"
    )
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
