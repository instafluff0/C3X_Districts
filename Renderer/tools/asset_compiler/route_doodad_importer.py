#!/usr/bin/env python3
"""Compile Civ VI route bridge bodies and transition rules into a generic pack."""

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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.compound_landmark_importer import _compile_asset
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = Path(__file__).with_name("route_doodad_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "RouteDoodadsNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "route_doodads" / "build.json"
MAC_ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
WINDOWS_ASSETS_ROOT = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets"
)
SAFE_ID = re.compile(
    r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]?[a-z0-9]+)*)*$"
)


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
    if mapping.get("schema") != "c3x.source_route_doodad_mapping.v0":
        raise ValueError("Unsupported route-doodad mapping schema")
    route_types = mapping.get("route_types")
    if (
        not isinstance(route_types, dict)
        or not route_types
        or not all(
            isinstance(source, str)
            and source
            and isinstance(asset, str)
            and SAFE_ID.fullmatch(asset)
            for source, asset in route_types.items()
        )
        or len(set(route_types.values())) != len(route_types)
    ):
        raise ValueError("Route-doodad route_types must be a unique source-to-runtime map")
    catalogs = mapping.get("catalogs")
    if not isinstance(catalogs, list) or not catalogs:
        raise ValueError("Route-doodad mapping has no catalogs")
    source_entries: set[tuple[str, str]] = set()
    asset_ids: set[str] = set()
    for catalog in catalogs:
        for field in ("artdef", "source_package"):
            if not isinstance(catalog.get(field), str) or not _safe_relative(catalog[field]):
                raise ValueError(f"Route-doodad {field} must be a safe relative path")
        shared = catalog.get("shared_data")
        if (
            not isinstance(shared, list)
            or not shared
            or not all(isinstance(value, str) and _safe_relative(value) for value in shared)
            or len(shared) != len(set(shared))
        ):
            raise ValueError("Route-doodad shared_data must be an ordered unique path list")
        scale = catalog.get("source_units_per_tile")
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Route-doodad source_units_per_tile must be positive and finite")
        if not isinstance(catalog.get("allow_declared_size_mismatch"), bool):
            raise ValueError("Route-doodad header exception must be an explicit Boolean")
        assets = catalog.get("assets")
        if not isinstance(assets, list) or not assets:
            raise ValueError("Route-doodad catalog has no assets")
        for asset in assets:
            source_entry = asset.get("source_entry")
            asset_id = asset.get("asset_id")
            if not isinstance(source_entry, str) or not source_entry:
                raise ValueError("Route-doodad asset has no source entry")
            if not isinstance(asset_id, str) or not SAFE_ID.fullmatch(asset_id):
                raise ValueError(f"Invalid route-doodad asset ID: {asset_id!r}")
            source_key = (catalog["source_package"], source_entry)
            if source_key in source_entries or asset_id in asset_ids:
                raise ValueError("Route-doodad mapping repeats a source entry or runtime asset ID")
            source_entries.add(source_key)
            asset_ids.add(asset_id)
    return mapping


def _field_value(value: ET.Element) -> tuple[str, str]:
    parameter = value.find("m_ParamName")
    if parameter is None or not parameter.attrib.get("text"):
        raise ValueError("Route transition value has no parameter name")
    child = next((item for item in value if item.tag != "m_ParamName"), None)
    if child is None:
        raise ValueError("Route transition value has no payload")
    raw = child.attrib.get("text", child.text)
    if raw is None:
        raise ValueError("Route transition value payload is empty")
    return parameter.attrib["text"], raw


def read_transition_records(
    artdef: Path,
    route_types: dict[str, str],
    source_assets: dict[str, str],
    source_units_per_tile: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = ET.parse(artdef).getroot()
    collections = [
        collection
        for collection in root.findall("./m_RootCollections/Element")
        if collection.find("m_CollectionName") is not None
        and collection.find("m_CollectionName").attrib.get("text")
        == "Route Transition Doodads"
    ]
    if len(collections) != 1:
        raise ValueError(f"Expected one Route Transition Doodads collection in {artdef}")
    runtime_records = []
    evidence = []
    for item in collections[0].findall("Element"):
        name_node = item.find("m_Name")
        source_name = "" if name_node is None else name_node.attrib.get("text", "")
        values = dict(_field_value(value) for value in item.findall("./m_Fields/m_Values/Element"))
        required = {
            "Origin",
            "Destination",
            "Type",
            "Asset",
            "TransitionLength",
            "ContourToRoad",
            "ScaleToGap",
        }
        if set(values) != required:
            raise ValueError(
                f"Route transition {source_name or '<unnamed>'} has unexpected fields: "
                f"{sorted(set(values) ^ required)}"
            )
        origin = route_types.get(values["Origin"])
        destination = route_types.get(values["Destination"])
        asset = source_assets.get(values["Asset"])
        if origin is None or destination is None or asset is None:
            raise ValueError(f"Route transition {source_name} has an unmapped route or body")
        if values["Type"] != "BRIDGE":
            raise ValueError(f"Unsupported route transition type {values['Type']}")
        length = float(values["TransitionLength"])
        if not math.isfinite(length) or length <= 0:
            raise ValueError(f"Route transition {source_name} has an invalid length")
        booleans = []
        for field in ("ContourToRoad", "ScaleToGap"):
            if values[field] not in ("true", "false"):
                raise ValueError(f"Route transition {source_name} has invalid {field}")
            booleans.append(values[field] == "true")
        runtime_records.append(
            {
                "origin_style": origin,
                "destination_style": destination,
                "kind": "bridge",
                "asset": asset,
                "length_tiles": round(length / source_units_per_tile, 8),
                "contour_to_route": booleans[0],
                "scale_to_gap": booleans[1],
            }
        )
        evidence.append({"source_record": source_name, "source_values": values})
    if not runtime_records:
        raise ValueError(f"Route transition catalog is empty: {artdef}")
    return runtime_records, evidence


def compile_route_doodads(
    assets_root: Path,
    mapping_path: Path = DEFAULT_MAPPING,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    mapping = load_mapping(mapping_path)
    try:
        report_path.resolve().relative_to(pack.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Route-doodad source report must be outside the runtime pack")
    assets = {}
    transitions = []
    package_reports = []
    asset_reports = []
    transition_reports = []
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for catalog in mapping["catalogs"]:
        source_path = assets_root / catalog["source_package"]
        artdef = assets_root / catalog["artdef"]
        shared_roots = [assets_root / value for value in catalog["shared_data"]]
        missing = next(
            (
                path
                for path in [source_path, artdef, *shared_roots]
                if not (path.is_file() if path in (source_path, artdef) else path.is_dir())
            ),
            None,
        )
        if missing is not None:
            raise FileNotFoundError(missing)
        source_assets = {
            asset["source_entry"]: asset["asset_id"] for asset in catalog["assets"]
        }
        package = IndexedStaticPackage(
            source_path,
            catalog["assets"][0]["source_entry"],
            allow_declared_size_mismatch=catalog["allow_declared_size_mismatch"],
        )
        package_reports.append(
            {
                "source": str(source_path),
                "source_sha256": _sha256(source_path.read_bytes()),
                "allocation_count": len(package.allocations),
                "header": package.header,
            }
        )
        for asset_mapping in catalog["assets"]:
            asset_id = asset_mapping["asset_id"]
            manifest_asset, evidence = _compile_asset(
                package,
                shared_roots,
                pack,
                asset_mapping["source_entry"],
                asset_id,
                float(catalog["source_units_per_tile"]),
                texture_cache,
            )
            assets[asset_id] = manifest_asset
            asset_reports.append(evidence)
        runtime_records, source_records = read_transition_records(
            artdef,
            mapping["route_types"],
            source_assets,
            float(catalog["source_units_per_tile"]),
        )
        transitions.extend(runtime_records)
        transition_reports.extend(
            {"artdef": str(artdef), **record} for record in source_records
        )
    if len(transitions) != len(set(json.dumps(item, sort_keys=True) for item in transitions)):
        raise ValueError("Route transition catalog contains duplicate normalized records")
    transition_path = "route_transitions.json"
    _write_json(
        pack / transition_path,
        {
            "schema": "c3x.route_transition_catalog.v0",
            "transitions": transitions,
            "provenance": {
                "kind": "local_normalized_import",
                "adapter": "c3x.route_doodad.v0",
                "source_format_dependency": None,
            },
        },
    )
    _write_json(
        pack / "manifest.json",
        {
            "schema": "c3x.asset_pack.v0",
            "name": "RouteDoodadsNormalized",
            "display_name": "Normalized Route Bridge Bodies",
            "source_policy": "Local licensed-source import; derived art is not redistributable.",
            "assets": assets,
            "transition_catalog": transition_path,
        },
    )
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime pack is source-dependent: " + "; ".join(independence_errors))
    report = {
        "schema": "c3x.source_route_doodad_build.v0",
        "mapping": {"path": str(mapping_path), "sha256": _sha256(mapping_path.read_bytes())},
        "packages": package_reports,
        "assets": asset_reports,
        "transitions": transition_reports,
        "outputs": {
            "pack": str(pack),
            "assets": len(assets),
            "transitions": len(transitions),
            "geometry_parts": sum(len(item["geometry"]) for item in asset_reports),
            "materials": sum(len(item["materials"]) for item in asset_reports),
            "decal_descriptors": sum(item["decal"]["count"] for item in asset_reports),
            "textures": len({relative for relative, _info in texture_cache.values()}),
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
        report = compile_route_doodads(
            args.assets_root,
            args.mapping,
            args.pack,
            args.report,
        )
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    outputs = report["outputs"]
    print(
        f"Compiled {outputs['assets']} route bridge bodies, "
        f"{outputs['transitions']} transition rules, and "
        f"{outputs['decal_descriptors']} endpoint decals"
    )
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
