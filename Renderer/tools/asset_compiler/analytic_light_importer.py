#!/usr/bin/env python3
"""Normalize a conservative reflected analytic-light library offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = Path(__file__).with_name("analytic_light_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "AnalyticLightsNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "ambient_effects" / "analytic_light_import.json"
SAFE_ID = re.compile(r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]?[a-z0-9]+)*)*$")
TYPE_ENTRY = "LightPackageEntry"
TYPE_VALUE_POINTER = "BLP::BLPPtr<BLP::Value>"
TYPE_FLOAT = "BLP::FloatValue"
TYPE_RGB = "BLP::RGBValue"
TYPE_STRING_VALUE = "BLP::StringValue"
TYPE_BOOL = "BLP::BoolValue"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mapping(path: Path = DEFAULT_MAPPING) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "c3x.source_analytic_light_mapping.v0":
        raise ValueError("Unsupported analytic-light mapping schema")
    package = document.get("source_package")
    if not isinstance(package, str) or not package.endswith(".blp") or ".." in Path(package).parts:
        raise ValueError("Analytic-light source package is invalid")
    scale = document.get("source_units_per_tile")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Analytic-light source scale must be positive and finite")
    lights = document.get("lights")
    if not isinstance(lights, list) or not lights:
        raise ValueError("Analytic-light mapping contains no lights")
    source_entries: set[str] = set()
    asset_ids: set[str] = set()
    for item in lights:
        if set(item) != {"source_entry", "asset_id", "family"}:
            raise ValueError("Analytic-light mapping record is invalid")
        if not isinstance(item["source_entry"], str) or not item["source_entry"]:
            raise ValueError("Analytic-light source entry is invalid")
        if not isinstance(item["asset_id"], str) or not SAFE_ID.fullmatch(item["asset_id"]):
            raise ValueError("Analytic-light asset ID is invalid")
        if not isinstance(item["family"], str) or not item["family"]:
            raise ValueError("Analytic-light family is invalid")
        if item["source_entry"] in source_entries or item["asset_id"] in asset_ids:
            raise ValueError("Analytic-light mapping contains a duplicate")
        source_entries.add(item["source_entry"])
        asset_ids.add(item["asset_id"])
    excluded = document.get("excluded_source_entries")
    if not isinstance(excluded, list) or not all(
        isinstance(item, dict)
        and set(item) == {"source_entry", "reason"}
        and all(isinstance(value, str) and value for value in item.values())
        for item in excluded
    ):
        raise ValueError("Analytic-light excluded-source inventory is invalid")
    if source_entries.intersection(item["source_entry"] for item in excluded):
        raise ValueError("Analytic-light entry cannot be both mapped and excluded")
    return document


def _decode_value(package: IndexedStaticPackage, pointer: int) -> tuple[str, Any]:
    raw = package.bytes_for(pointer)
    value_type = package.type_name(pointer)
    if len(raw) not in (32, 40):
        raise ValueError("Analytic-light value has an unsupported layout")
    parameter = package.string_value(struct.unpack_from("<Q", raw, 0x10)[0])
    if not parameter:
        raise ValueError("Analytic-light value has no parameter name")
    if value_type == TYPE_FLOAT and len(raw) == 32:
        value = struct.unpack_from("<f", raw, 0x18)[0]
        if not math.isfinite(value):
            raise ValueError("Analytic-light float is non-finite")
    elif value_type == TYPE_RGB and len(raw) == 40:
        value = list(struct.unpack_from("<3f", raw, 0x18))
        if not all(math.isfinite(channel) and 0 <= channel <= 255 for channel in value):
            raise ValueError("Analytic-light RGB value is invalid")
    elif value_type == TYPE_STRING_VALUE and len(raw) == 32:
        value = package.string_value(struct.unpack_from("<Q", raw, 0x18)[0])
        if not value:
            raise ValueError("Analytic-light string value is empty")
    elif value_type == TYPE_BOOL and len(raw) == 32:
        encoded = struct.unpack_from("<Q", raw, 0x18)[0]
        if encoded not in (0, 1):
            raise ValueError("Analytic-light Boolean value is invalid")
        value = bool(encoded)
    else:
        raise ValueError(f"Unsupported analytic-light value type: {value_type}")
    return parameter, value


def _decode_entry(package: IndexedStaticPackage, source_entry: str) -> dict[str, Any]:
    name_pointer = package.select_direct_string(source_entry)
    references = package.references_to(name_pointer, TYPE_ENTRY)
    if len(references) != 1 or references[0][1] != 0x38:
        raise ValueError(f"Analytic-light entry reference is ambiguous: {source_entry}")
    entry_pointer = references[0][0]
    raw = package.bytes_for(entry_pointer)
    if len(raw) != 104:
        raise ValueError("Analytic-light entry has an unsupported layout")
    values_pointer = struct.unpack_from("<Q", raw, 0x50)[0]
    if package.type_name(values_pointer) != TYPE_VALUE_POINTER:
        raise ValueError("Analytic-light entry has no typed value table")
    values_raw = package.bytes_for(values_pointer)
    allocation = package.allocations[values_pointer - 1]
    if len(values_raw) != allocation["element_count"] * 8:
        raise ValueError("Analytic-light value table is irregular")
    values: dict[str, Any] = {}
    for index in range(allocation["element_count"]):
        pointer = struct.unpack_from("<Q", values_raw, index * 8)[0]
        parameter, value = _decode_value(package, pointer)
        if parameter in values:
            raise ValueError(f"Analytic-light parameter repeats: {parameter}")
        values[parameter] = value
    expected = {"Color", "Radius", "Intensity", "Attenuation", "TimeOfDay", "ApplyLightMapWeight"}
    if set(values) != expected:
        raise ValueError(f"Analytic-light parameters differ from the proven profile: {sorted(values)}")
    if values["Radius"] <= 0 or values["Attenuation"] < 0:
        raise ValueError("Analytic-light range or attenuation is invalid")
    if values["TimeOfDay"] not in ("All", "Night"):
        raise ValueError("Analytic-light time-of-day policy is unsupported")
    return {"entry_pointer": entry_pointer, "values_pointer": values_pointer, "values": values}


def _source_entry_names(package: IndexedStaticPackage) -> set[str]:
    names: set[str] = set()
    for pointer in range(1, len(package.allocations) + 1):
        if package.type_name(pointer) != TYPE_ENTRY:
            continue
        raw = package.bytes_for(pointer)
        if len(raw) != 104:
            raise ValueError("Analytic-light package contains an unsupported entry layout")
        name = package.direct_string(struct.unpack_from("<Q", raw, 0x38)[0])
        if not name or name in names:
            raise ValueError("Analytic-light package entry names are invalid or duplicate")
        names.add(name)
    return names


def compile_lights(
    assets_root: Path,
    mapping_path: Path = DEFAULT_MAPPING,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    mapping = load_mapping(mapping_path)
    source = assets_root / mapping["source_package"]
    package = IndexedStaticPackage(
        source,
        mapping["lights"][0]["source_entry"],
        minimum_temp_support=2,
    )
    mapped_names = {item["source_entry"] for item in mapping["lights"]}
    excluded_names = {
        item["source_entry"] for item in mapping["excluded_source_entries"]
    }
    source_names = _source_entry_names(package)
    if source_names != mapped_names | excluded_names:
        raise ValueError(
            "Analytic-light mapping does not account for the complete source package: "
            f"missing={sorted(source_names - mapped_names - excluded_names)}, "
            f"stale={sorted((mapped_names | excluded_names) - source_names)}"
        )
    scale = float(mapping["source_units_per_tile"])
    lights: dict[str, Any] = {}
    evidence = []
    for mapped in mapping["lights"]:
        decoded = _decode_entry(package, mapped["source_entry"])
        values = decoded["values"]
        lights[mapped["asset_id"]] = {
            "kind": "point",
            "family": mapped["family"],
            "color_srgb": [round(channel / 255.0, 8) for channel in values["Color"]],
            "range_tiles": round(values["Radius"] / scale, 8),
            "intensity": round(values["Intensity"], 8),
            "attenuation": round(values["Attenuation"], 8),
            "activation_policy": values["TimeOfDay"].lower(),
            "apply_light_map_weight": values["ApplyLightMapWeight"],
            "binding_status": "resource_only_no_attachment",
            "calibration_status": "typed_parameters_unapproved_for_c3x_rendering",
        }
        evidence.append({"mapping": mapped, **decoded})
    manifest = {
        "schema": "c3x.analytic_light_pack.v0",
        "pack_id": "local.analytic_lights.normalized",
        "lights": dict(sorted(lights.items())),
        "attachment_bindings": [],
        "runtime_source_dependency": None,
        "runtime_status": "not_enabled",
    }
    _write_json(pack / "manifest.json", manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError(
            "Runtime analytic-light pack is source-dependent: "
            + "; ".join(independence_errors)
        )
    report = {
        "schema": "c3x.source_analytic_light_import.v0",
        "source": str(source),
        "source_sha256": _sha256(source),
        "mapping": {"path": str(mapping_path), "sha256": _sha256(mapping_path)},
        "lights": evidence,
        "excluded_source_entries": mapping["excluded_source_entries"],
        "summary": {
            "source_entries": len(source_names),
            "converted_lights": len(lights),
            "excluded_entries": len(mapping["excluded_source_entries"]),
            "typed_parameters_per_light": 6,
            "attachment_bindings": 0,
        },
        "runtime_independence": "passed",
        "runtime_integration": "not_enabled",
    }
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = compile_lights(args.assets_root, args.mapping, args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, struct.error, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
