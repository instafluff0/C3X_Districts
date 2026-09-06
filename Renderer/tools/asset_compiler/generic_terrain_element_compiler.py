#!/usr/bin/env python3
"""Compile reflected terrain-element records into a generic normalized pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.terrain_relief_builder import (
    inspect_terrain_element_package,
    make_r8_dds,
)


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = Path(__file__).with_name("terrain_element_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "TerrainElementsNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "terrain_elements" / "expansion2_build.json"
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_mapping(path: Path) -> dict[str, Any]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema") != "c3x.source_terrain_element_mapping.v0":
        raise ValueError("Unsupported terrain-element mapping schema")
    source_package = mapping.get("source_package")
    if (
        not isinstance(source_package, str)
        or not source_package
        or PurePosixPath(source_package).is_absolute()
        or ".." in PurePosixPath(source_package).parts
        or "\\" in source_package
    ):
        raise ValueError("Terrain-element source package must be a safe relative path")
    elements = mapping.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ValueError("Terrain-element mapping has no elements")
    source_names: set[str] = set()
    asset_ids: set[str] = set()
    for item in elements:
        source_name = item.get("source_entry")
        asset_id = item.get("asset_id")
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("Terrain-element mapping has an invalid source entry")
        if not isinstance(asset_id, str) or not SAFE_ID.fullmatch(asset_id):
            raise ValueError(f"Invalid normalized terrain-element ID: {asset_id!r}")
        if source_name in source_names:
            raise ValueError(f"Duplicate terrain-element source entry: {source_name}")
        if asset_id in asset_ids:
            raise ValueError(f"Duplicate normalized terrain-element ID: {asset_id}")
        source_names.add(source_name)
        asset_ids.add(asset_id)
    return mapping


def _document_path(asset_id: str) -> str:
    return "terrain_elements/" + asset_id.replace("/", "_").replace(".", "_") + ".json"


def _texture_path(asset_id: str, role: str, level: int) -> str:
    stem = asset_id.replace("/", "_").replace(".", "_")
    return f"textures/terrain_elements/{stem}/{role}_lod{level}.dds"


def _report_outside_pack(report: Path, pack: Path) -> None:
    try:
        report.resolve().relative_to(pack.resolve())
    except ValueError:
        return
    raise ValueError("Source terrain-element report must be outside the runtime pack")


def compile_terrain_elements(
    assets_root: Path,
    mapping_path: Path,
    pack: Path,
    report_path: Path,
) -> dict[str, Any]:
    mapping = load_mapping(mapping_path)
    _report_outside_pack(report_path, pack)
    source_path = assets_root / mapping["source_package"]
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    resources, elements, package_report = inspect_terrain_element_package(source_path)
    mapped_sources = {item["source_entry"] for item in mapping["elements"]}
    absent = sorted(mapped_sources - set(elements))
    unmapped = sorted(set(elements) - mapped_sources)
    if absent:
        raise ValueError("Mapped terrain-element entries are absent: " + ", ".join(absent))
    if unmapped:
        raise ValueError("Terrain-element package has unmapped entries: " + ", ".join(unmapped))

    manifest_assets = {}
    source_evidence = []
    with source_path.open("rb") as source:
        for mapping_item in mapping["elements"]:
            source_name = mapping_item["source_entry"]
            asset_id = mapping_item["asset_id"]
            element = elements[source_name]
            runtime_lods: dict[int, dict[str, Any]] = {}
            channel_presence = {}
            for role, lods in element["channels"].items():
                channel_presence[role] = True
                for lod in lods:
                    level = lod["level"]
                    source.seek(package_report["big_data_offset"] + lod["relative_offset"])
                    pixels = source.read(lod["bytes"])
                    if len(pixels) != lod["bytes"]:
                        raise ValueError(f"Short terrain-element payload read: {source_name}/{role}/{level}")
                    relative = _texture_path(asset_id, role, level)
                    target = pack / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    discrete = role == "region_ids"
                    target.write_bytes(
                        make_r8_dds(lod["width"], lod["height"], pixels, 62 if discrete else 61)
                    )
                    runtime_lods.setdefault(level, {"level": level, "channels": {}})["channels"][role] = {
                        "texture": relative,
                        "format": "R8_UINT" if discrete else "R8_UNORM",
                        "width": lod["width"],
                        "height": lod["height"],
                    }
                    source_evidence.append({
                        "source_entry": source_name,
                        "asset_id": asset_id,
                        "role": role,
                        "lod": level,
                        "source_resource": lod["name"],
                        "source_index": lod["index"],
                        "source_sha256": _sha256(pixels),
                        "runtime_texture": relative,
                    })
            for role in ("height", "blend", "region_ids", "noise_2d"):
                channel_presence.setdefault(role, False)
            document = {
                "schema": "c3x.terrain_element.v0",
                "asset_id": asset_id,
                "grid_dimensions": element["grid_dimensions"],
                "height_calibration": {
                    "base_height": element["parameters"]["base_height"],
                    "height_scale": element["parameters"]["height_scale"],
                    "units": "source_authored",
                },
                "noise_scale": element["parameters"]["noise_scale"],
                "channel_presence": channel_presence,
                "lods": [runtime_lods[level] for level in sorted(runtime_lods)],
                "channel_semantics": {
                    "height": "authored_macro_height",
                    "blend": "continuous_footprint_or_height_blend",
                    "region_ids": "discrete_material_or_region_identifier",
                    "noise_2d": "authored_noise_field",
                },
                "provenance": {
                    "kind": "local_normalized_import",
                    "adapter": "c3x.terrain_element.v0",
                    "source_format_dependency": None,
                },
            }
            relative_document = _document_path(asset_id)
            _write_json(pack / relative_document, document)
            manifest_assets[asset_id] = {"type": "terrain_element", "element": relative_document}

    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "TerrainElementsNormalized",
        "display_name": "Normalized Terrain Elements",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "assets": manifest_assets,
    }
    _write_json(pack / "manifest.json", manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime pack is source-dependent: " + "; ".join(independence_errors))
    report = {
        "schema": "c3x.source_terrain_element_build.v0",
        "mapping": {"path": str(mapping_path), "sha256": _sha256(mapping_path.read_bytes())},
        "source": {"path": str(source_path), "sha256": _sha256(source_path.read_bytes())},
        "package": package_report,
        "outputs": {
            "pack": str(pack),
            "elements": len(manifest_assets),
            "textures": len(source_evidence),
        },
        "source_evidence": source_evidence,
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
        report = compile_terrain_elements(
            args.assets_root, args.mapping, args.pack, args.report
        )
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Compiled {report['outputs']['elements']} normalized terrain elements "
        f"and {report['outputs']['textures']} LOD textures"
    )
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
