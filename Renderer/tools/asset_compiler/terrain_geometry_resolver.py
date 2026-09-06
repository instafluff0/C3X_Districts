#!/usr/bin/env python3
"""Resolve Civ VI flat-terrain geometry to a generic C3X mesh contract.

Flat base terrain is selected from ArtDef semantics.  The local cooked-package
inventory is evidence only: no cooked geometry or texture payload is decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_CIV6_BASE = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets\Base"
)
RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESH = RENDERER_ROOT / "samples" / "geometry" / "flat_terrain_patch.json"
DEFAULT_REPORT = RENDERER_ROOT / "docs" / "civ6_grassland_geometry_uv.json"
DEFAULT_TERRAIN = "TERRAIN_GRASS"
DEFAULT_MATERIAL = "ART_DEF_TERRAIN_MATERIAL_GRASSLAND"
LOOSE_GEOMETRY_EXTENSIONS = (".fgx", ".cn6", ".glb", ".gltf", ".fbx", ".obj")
GEOMETRY_SIGNALS = (
    "BLP::IndexBufferEntry",
    "BLP::VertexBufferEntry",
    "FGXModel::ContainerDesc::Mesh",
)


def canonical_json_bytes(data: Any) -> bytes:
    return (json.dumps(data, indent=2) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_text(element: ET.Element, tag: str) -> str | None:
    child = element.find(tag)
    return None if child is None else child.get("text")


def find_root_collection(root: ET.Element, name: str) -> ET.Element:
    matches = [
        item
        for item in root.findall("./m_RootCollections/Element")
        if child_text(item, "m_CollectionName") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one root collection {name}, found {len(matches)}")
    return matches[0]


def find_collection_entry(collection: ET.Element, name: str) -> ET.Element:
    matches = [item for item in collection.findall("./Element") if child_text(item, "m_Name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} entry, found {len(matches)}")
    return matches[0]


def field_value(entry: ET.Element, collection_name: str, parameter_name: str) -> str:
    values = []
    for collection in entry.findall("./m_ChildCollections/Element"):
        if child_text(collection, "m_CollectionName") != collection_name:
            continue
        for value in collection.findall("./Element/m_Fields/m_Values/Element"):
            if child_text(value, "m_ParamName") == parameter_name:
                raw = value.find("m_Value")
                if raw is not None and raw.get("text") is not None:
                    values.append(raw.get("text"))
    if len(values) != 1:
        raise ValueError(
            f"Expected one {collection_name}.{parameter_name} value, found {len(values)}"
        )
    return values[0]


def blp_references(entry: ET.Element, recursive: bool = False) -> list[dict[str, str]]:
    references = []
    query = (
        ".//Element[@class='AssetObjects..BLPEntryValue']"
        if recursive
        else "./m_Fields/m_Values/Element[@class='AssetObjects..BLPEntryValue']"
    )
    for value in entry.findall(query):
        reference = {
            key: child_text(value, tag) or ""
            for key, tag in (
                ("entry_name", "m_EntryName"),
                ("xlp_class", "m_XLPClass"),
                ("xlp_path", "m_XLPPath"),
                ("blp_package", "m_BLPPackage"),
                ("library_name", "m_LibraryName"),
                ("parameter_name", "m_ParamName"),
            )
        }
        references.append(reference)
    return references


def resolve_artdef_chain(
    terrains_path: Path,
    terrain_style_path: Path,
    terrain_name: str = DEFAULT_TERRAIN,
    material_name: str = DEFAULT_MATERIAL,
) -> dict[str, Any]:
    terrains_root = ET.parse(terrains_path).getroot()
    terrain_entry = find_collection_entry(find_root_collection(terrains_root, "Terrain"), terrain_name)
    terrain_type = field_value(terrain_entry, "TerrainType", "XrefName")
    terrain_subtype = field_value(terrain_entry, "TerrainSubType", "XrefName")
    if terrain_type != "Flat":
        raise ValueError(f"Procedural flat patch cannot represent terrain type {terrain_type}")

    style_root = ET.parse(terrain_style_path).getroot()
    style_collection_name = f"Standard{terrain_type}"
    style_entry = find_collection_entry(
        find_root_collection(style_root, style_collection_name), "Default"
    )
    style_references = blp_references(style_entry)
    material_matches = [
        item
        for item in style_references
        if item["entry_name"] == material_name and item["xlp_class"] == "TerrainMaterial"
    ]
    if len(material_matches) != 1:
        raise ValueError(
            f"Expected one explicit {material_name} reference in {style_collection_name}, "
            f"found {len(material_matches)}"
        )
    geometry_references = [
        item for item in style_references if item["xlp_class"] in {"TerrainElement", "Asset"}
    ]
    grassland_relief = [
        item
        for item in geometry_references
        if item["parameter_name"] == "GrasslandElement" and item["entry_name"]
    ]

    hills_entry = find_collection_entry(find_root_collection(style_root, "StandardHills"), "Default")
    hills_geometry = [
        item
        for item in blp_references(hills_entry)
        if item["xlp_class"] in {"TerrainElement", "Asset"}
    ]

    if len(grassland_relief) != 1:
        raise ValueError("StandardFlat does not contain one explicit GrasslandElement relief reference")
    if not hills_geometry:
        raise ValueError("StandardHills comparison contains no explicit geometry reference")

    return {
        "terrain_entry": terrain_name,
        "terrain_type": terrain_type,
        "terrain_subtype": terrain_subtype,
        "style_collection": style_collection_name,
        "style_entry": "Default",
        "material_reference": material_matches[0],
        "authored_relief_reference": grassland_relief[0],
        "all_style_geometry_references": geometry_references,
        "comparison": {
            "style_collection": "StandardHills",
            "geometry_references": hills_geometry,
        },
        "confidence": "high",
        "evidence": [
            f"{terrain_name} explicitly declares TerrainType={terrain_type} and TerrainSubType={terrain_subtype}",
            f"{style_collection_name}/Default explicitly binds {material_name}",
            f"{style_collection_name}/Default separately declares {grassland_relief[0]['entry_name']} as GrasslandElement relief",
            "the terrain entry itself is Flat, while authored non-flat relief is explicitly represented by TerrainElement references",
        ],
    }


def make_flat_patch() -> dict[str, Any]:
    return {
        "schema": "c3x.normalized_mesh.v0",
        "asset_id": "terrain.flat.unit_patch",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Z",
            "horizontal_axes": ["X", "Y"],
            "position_unit": "tile",
            "uv0_origin": "upper_left",
        },
        "topology": {
            "primitive": "triangles",
            "front_face": "counter_clockwise",
            "indices": [0, 1, 2, 0, 2, 3],
        },
        "vertices": [
            {"position": [-0.5, -0.5, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 1.0]},
            {"position": [0.5, -0.5, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 1.0]},
            {"position": [0.5, 0.5, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [1.0, 0.0]},
            {"position": [-0.5, 0.5, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": [0.0, 0.0]},
        ],
        "bounds": {"minimum": [-0.5, -0.5, 0.0], "maximum": [0.5, 0.5, 0.0]},
        "material_slots": [{"slot": 0, "name": "terrain_surface", "triangle_start": 0, "triangle_count": 2}],
        "provenance": {
            "kind": "procedural",
            "generator": "terrain_geometry_resolver.make_flat_patch",
            "source_format_dependency": None,
        },
    }


def validate_normalized_mesh(mesh: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if mesh.get("schema") != "c3x.normalized_mesh.v0":
        errors.append("unsupported mesh schema")
    vertices = mesh.get("vertices", [])
    indices = mesh.get("topology", {}).get("indices", [])
    if len(vertices) != 4 or len(indices) != 6:
        errors.append("flat patch must contain four vertices and two triangles")
        return errors
    if any(not isinstance(index, int) or index < 0 or index >= len(vertices) for index in indices):
        errors.append("triangle index is out of range")
    if any(len(vertex.get("position", [])) != 3 for vertex in vertices):
        errors.append("each vertex must have a three-component position")
    if any(vertex.get("normal") != [0.0, 0.0, 1.0] for vertex in vertices):
        errors.append("flat patch normals must point along +Z")
    uvs = [vertex.get("uv0") for vertex in vertices]
    if any(uv is None or len(uv) != 2 or any(value < 0.0 or value > 1.0 for value in uv) for uv in uvs):
        errors.append("UV0 coordinates must be two-component values within 0..1")
    if set(tuple(uv) for uv in uvs) != {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}:
        errors.append("UV0 must cover the full unit square exactly once")
    for start in range(0, len(indices), 3):
        a, b, c = [vertices[index]["position"] for index in indices[start : start + 3]]
        signed_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if signed_area <= 0:
            errors.append("triangle winding is not counter-clockwise around +Z")
            break
    return errors


def inspect_civblp(path: Path) -> dict[str, Any]:
    actual_size = path.stat().st_size
    with path.open("rb") as source:
        header = source.read(28)
        if len(header) != 28 or header[:6] != b"CIVBLP":
            raise ValueError(f"Not a supported CIVBLP header: {path}")
        version = struct.unpack_from("<H", header, 6)[0]
        package_offset, package_size, big_data_offset, big_data_count, declared_size = struct.unpack_from(
            "<5I", header, 8
        )
        if declared_size != actual_size or package_offset + package_size != big_data_offset:
            raise ValueError(f"Inconsistent CIVBLP bounds: {path}")
        source.seek(package_offset)
        metadata = source.read(package_size)
    if len(metadata) != package_size:
        raise ValueError(f"Could not read complete package metadata: {path}")
    decoded = metadata.decode("ascii", errors="ignore")
    return {
        "name": path.name,
        "file_bytes": actual_size,
        "version": version,
        "package_metadata_bytes": package_size,
        "big_data_entry_count": big_data_count,
        "geometry_signals": [signal for signal in GEOMETRY_SIGNALS if signal in decoded],
        "contains_grassland_material_entry": DEFAULT_MATERIAL in decoded,
        "read_policy": "header and package metadata only",
    }


def build_report(civ6_base: Path, mesh: dict[str, Any]) -> dict[str, Any]:
    artdefs = civ6_base / "ArtDefs"
    terrain_dir = civ6_base / "Platforms" / "Windows" / "BLPs" / "terrain"
    chain = resolve_artdef_chain(artdefs / "Terrains.artdef", artdefs / "TerrainStyle.artdef")
    packages = [
        inspect_civblp(terrain_dir / name)
        for name in (
            "TerrainAssetSet_Base.blp",
            "TerrainElementSet_Base.blp",
            "TerrainMaterialSet_Base.blp",
        )
    ]
    loose_files = sorted(
        str(path.relative_to(civ6_base)).replace("\\", "/")
        for path in civ6_base.rglob("*")
        if path.is_file() and path.suffix.lower() in LOOSE_GEOMETRY_EXTENSIONS
    )
    mesh_errors = validate_normalized_mesh(mesh)
    if mesh_errors:
        raise ValueError("Invalid normalized flat mesh: " + "; ".join(mesh_errors))
    mesh_hash = hashlib.sha256(canonical_json_bytes(mesh)).hexdigest()
    return {
        "schema": "c3x.terrain_geometry_resolution.v0",
        "target": DEFAULT_TERRAIN,
        "material_target": DEFAULT_MATERIAL,
        "source_root": str(civ6_base),
        "source_provenance": {
            "terrains_artdef_sha256": sha256_file(artdefs / "Terrains.artdef"),
            "terrain_style_artdef_sha256": sha256_file(artdefs / "TerrainStyle.artdef"),
        },
        "artdef_resolution": chain,
        "cooked_inventory": packages,
        "loose_geometry_inventory": {
            "extensions": list(LOOSE_GEOMETRY_EXTENSIONS),
            "file_count": len(loose_files),
            "files": loose_files,
        },
        "selection": {
            "mode": "procedural_flat_grid",
            "normalized_mesh": "samples/geometry/flat_terrain_patch.json",
            "normalized_mesh_sha256": mesh_hash,
            "material_binding": "docs/civ6_grassland_material_binding.json",
            "uv_channel": "uv0",
            "uv_domain": "per_tile_unit_square",
            "confidence": "high",
            "rule": "TerrainType=Flat selects the source-agnostic unit grid for the base surface; the explicit StandardFlat material binds to that grid, while separately declared TerrainElement/Asset references remain authored relief rather than the flat base topology.",
            "evidence": [
                "the ArtDef chain explicitly selects StandardFlat and the grassland material",
                "StandardFlat separates the base material from an explicitly named GrasslandElement relief reference",
                "the normalized unit grid has validated CCW topology, +Z normals, and full-range UV0",
                "the runtime-facing mesh contains no Civ VI path, pointer, or source-format dependency",
            ],
        },
        "tool_inventory": {
            "civnexus6_source_reference": (RENDERER_ROOT / "third_party" / "CivNexus6" / "NexusBuddy.csproj").is_file(),
            "civnexus6_built_executable": any((RENDERER_ROOT / "third_party" / "CivNexus6").glob("**/CivNexus6.exe")),
            "fgxviewer_executable": any((RENDERER_ROOT / "third_party" / "CivNexus6").glob("**/fgxviewer.exe")),
            "texconv_executable": any((RENDERER_ROOT / "third_party" / "CivNexus6").glob("**/texconv.exe")),
            "blender_on_path": shutil.which("blender") is not None,
            "note": "The checked-in converter references remain appropriate for authored models, but flat base terrain needs no FGX conversion.",
        },
        "limitations": [
            "This step establishes geometry topology and a declared UV domain; texture payload extraction and visual calibration belong to M1.5.",
            "The per-tile unit UV domain is a C3X normalization decision, not a claim that Civ VI stores an equivalent authored flat mesh.",
            "Hills, mountains, cliffs, and landmarks have explicit geometry-bearing paths and are outside this grassland-flat step.",
        ],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(data))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve Civ VI flat terrain to normalized C3X geometry")
    parser.add_argument("--civ6-base", type=Path, default=DEFAULT_CIV6_BASE)
    parser.add_argument("--mesh-output", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        mesh = make_flat_patch()
        report = build_report(args.civ6_base, mesh)
        write_json(args.mesh_output, mesh)
        write_json(args.report, report)
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.mesh_output}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
