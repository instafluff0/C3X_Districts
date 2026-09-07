#!/usr/bin/env python3
"""Compile every installed standard-game Civ VI palace into a generic local pack."""

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

from Renderer.tools.asset_compiler.artdef_graph_resolver import (
    DEFAULT_ASSETS_ROOT,
    _package_index,
    _resolve_package,
)
from Renderer.tools.asset_compiler.compound_landmark_importer import _compile_asset
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.improvement_asset_importer import _content_root, _shared_roots
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.palace_asset_probe import scan


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY = Path(__file__).with_name("palace_import_strategy.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "CityPalacesNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "cities" / "palace_build.json"
SAFE_ID = re.compile(
    r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]?[a-z0-9]+)*)*$"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _palace_asset_id(package_relative: str, entry: str) -> str:
    digest = _sha256((package_relative + "\0" + entry).encode("utf-8"))[:16]
    asset_id = f"city/palace/root/{digest}"
    if not SAFE_ID.fullmatch(asset_id):
        raise ValueError(f"Could not form safe palace asset ID for {entry}")
    return asset_id


def _dependency_asset_id(package_relative: str, entry: str) -> str:
    digest = _sha256((package_relative + "\0" + entry).encode("utf-8"))[:16]
    return f"city/palace_component/{digest}"


def load_strategy(path: Path = DEFAULT_STRATEGY) -> dict[str, Any]:
    strategy = json.loads(path.read_text(encoding="utf-8"))
    if strategy.get("schema") != "c3x.source_palace_import_strategy.v0":
        raise ValueError("Unsupported palace import strategy schema")
    scale = strategy.get("source_units_per_tile")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Palace source_units_per_tile must be positive and finite")
    inventory = strategy.get("inventory")
    if not isinstance(inventory, dict) or inventory.get("building_name") != "BUILDING_PALACE":
        raise ValueError("Palace strategy must use the exact BUILDING_PALACE binding")
    if inventory.get("include_scenarios") is not False:
        raise ValueError("General palace intake must exclude scenario-only bindings")
    for field in ("expected_standard_bindings", "expected_unique_standard_assets"):
        if not isinstance(inventory.get(field), int) or inventory[field] < 1:
            raise ValueError(f"Palace strategy has invalid {field}")
    representatives = inventory.get("required_representative_entries")
    if not isinstance(representatives, list) or not representatives or not all(
        isinstance(value, str) and value for value in representatives
    ):
        raise ValueError("Palace strategy needs representative source roots")
    runtime = strategy.get("runtime_selection")
    if not isinstance(runtime, dict):
        raise ValueError("Palace strategy has no runtime selection contract")
    if runtime.get("capital_source") != "authoritative_civ3_is_capital":
        raise ValueError("Palace visibility must use authoritative Civ III capital state")
    if runtime.get("source_selectors_are") != "provenance_only":
        raise ValueError("Civ VI palace selectors cannot become runtime selectors")
    if runtime.get("hard_coded_civ6_civilization_ids") is not False:
        raise ValueError("Palace selection cannot hard-code Civ VI civilization IDs")
    if strategy.get("runtime_integration") != "not_enabled":
        raise ValueError("Palace intake must remain offline-only")
    return strategy


def compile_palaces(
    assets_root: Path,
    strategy_path: Path = DEFAULT_STRATEGY,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    strategy = load_strategy(strategy_path)
    try:
        report_path.resolve().relative_to(pack.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Palace source report must be outside the runtime pack")

    inventory = scan(assets_root)
    expected = strategy["inventory"]
    summary = inventory["summary"]
    if inventory["status"] != "passed":
        raise ValueError("Installed palace inventory contains unresolved or malformed bindings")
    if summary["standard_bindings"] != expected["expected_standard_bindings"]:
        raise ValueError(
            "Installed standard palace binding count changed: "
            f"expected {expected['expected_standard_bindings']}, found {summary['standard_bindings']}"
        )
    roots = inventory["accepted_standard_assets"]
    if len(roots) != expected["expected_unique_standard_assets"]:
        raise ValueError(
            "Installed standard palace asset count changed: "
            f"expected {expected['expected_unique_standard_assets']}, found {len(roots)}"
        )
    root_entries = {root["source_entry"] for root in roots}
    missing = sorted(set(expected["required_representative_entries"]) - root_entries)
    if missing:
        raise ValueError("Installed palace inventory is missing representatives: " + ", ".join(missing))

    root_ids = {
        (root["source_package"], root["source_entry"]): _palace_asset_id(
            root["source_package"], root["source_entry"]
        )
        for root in roots
    }
    if len(root_ids) != len(set(root_ids.values())):
        raise ValueError("Palace root asset IDs are not unique")

    packages = _package_index(assets_root)
    package_bytes: dict[str, bytes] = {}
    package_cache: dict[str, IndexedStaticPackage] = {}
    package_reports: dict[str, dict[str, Any]] = {}
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    assets: dict[str, Any] = {}
    evidence_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    rejected_optional_dependencies = []
    unresolved_required_dependencies = []
    skipped_source_conditions = []
    visiting: set[tuple[str, str]] = set()

    def ensure_asset(package_relative: str, entry: str) -> str:
        key = (package_relative, entry)
        asset_id = root_ids.get(key, _dependency_asset_id(*key))
        if key in evidence_by_key:
            return asset_id
        if key in visiting:
            raise ValueError(f"Palace component dependency cycle at {entry}")
        source_path = assets_root / package_relative
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        package = package_cache.get(package_relative)
        if package is None:
            package = IndexedStaticPackage(source_path, entry)
            package_cache[package_relative] = package
            package_reports[package_relative] = {
                "source": str(source_path),
                "source_sha256": _sha256(package.data),
                "allocation_count": len(package.allocations),
                "header": package.header,
            }
        visiting.add(key)
        try:
            manifest_asset, evidence = _compile_asset(
                package,
                _shared_roots(assets_root, package_relative),
                pack,
                entry,
                asset_id,
                float(strategy["source_units_per_tile"]),
                texture_cache,
                terrain_edit_policy="preserve_unresolved",
                auxiliary_uvs=True,
                omit_empty_material_draws=True,
            )
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            visiting.discard(key)
            raise ValueError(f"Could not compile palace component {entry}: {exc}") from exc
        document_path = pack / manifest_asset["landmark"]
        document = json.loads(document_path.read_text(encoding="utf-8"))
        source_points = {point["id"]: point for point in evidence["attachments"]["points"]}
        for point in document["attachment_points"]:
            source_point = source_points[point["id"]]
            if point["binding_status"] == "component_unresolved":
                terminal = source_point["component_source"]
                resolution = _resolve_package(
                    packages,
                    terminal["package"],
                    _content_root(package_relative),
                    terminal["entry"],
                    package_bytes,
                )
                if resolution["status"] != "resolved":
                    optional = point.get("selection", {}).get("cull") == "optional"
                    point["binding_status"] = (
                        "component_source_unresolved_optional"
                        if optional
                        else "component_source_unresolved_required"
                    )
                    target = (
                        rejected_optional_dependencies
                        if optional
                        else unresolved_required_dependencies
                    )
                    target.append(
                        {
                            "parent_package": package_relative,
                            "parent_entry": entry,
                            "child_entry": terminal["entry"],
                            "reason": resolution,
                        }
                    )
                    continue
                try:
                    child_id = ensure_asset(resolution["package_path"], terminal["entry"])
                except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                    optional = point.get("selection", {}).get("cull") == "optional"
                    point["binding_status"] = (
                        "component_compile_unresolved_optional"
                        if optional
                        else "component_compile_unresolved_required"
                    )
                    target = (
                        rejected_optional_dependencies
                        if optional
                        else unresolved_required_dependencies
                    )
                    target.append(
                        {
                            "parent_package": package_relative,
                            "parent_entry": entry,
                            "child_package": resolution["package_path"],
                            "child_entry": terminal["entry"],
                            "reason": str(exc),
                        }
                    )
                    continue
                point["component_asset"] = child_id
                point["binding_status"] = "resolved"
            elif point["binding_status"] == "source_condition_unmapped":
                skipped_source_conditions.append(
                    {
                        "source_package": package_relative,
                        "source_entry": entry,
                        "attachment": source_point,
                    }
                )
        _write_json(document_path, document)
        assets[asset_id] = manifest_asset
        evidence_by_key[key] = evidence
        visiting.remove(key)
        return asset_id

    catalog_roots = []
    for root in roots:
        asset_id = ensure_asset(root["source_package"], root["source_entry"])
        catalog_roots.append(
            {
                "asset_id": asset_id,
                "source_selectors": root["selectors"],
                "source_content": _content_root(root["source_package"]),
            }
        )

    catalog_path = "palace_catalog.json"
    _write_json(
        pack / catalog_path,
        {
            "schema": "c3x.city_palace_catalog.v0",
            "palaces": catalog_roots,
            "conversion_status": (
                "root_library_normalized_with_unresolved_required_attachments"
                if unresolved_required_dependencies
                else "complete"
            ),
            "runtime_selection": strategy["runtime_selection"],
            "provenance": {
                "kind": "local_normalized_import",
                "adapter": "c3x.city_palace_component.v0",
                "source_format_dependency": None,
            },
            "runtime_integration": "not_enabled",
        },
    )
    _write_json(
        pack / "manifest.json",
        {
            "schema": "c3x.asset_pack.v0",
            "name": "CityPalacesNormalized",
            "display_name": "Normalized City Palace Library",
            "source_policy": "Local licensed-source import; derived art is not redistributable.",
            "assets": dict(sorted(assets.items())),
            "palace_catalog": catalog_path,
            "runtime_integration": "not_enabled",
        },
    )
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime palace pack is source-dependent: " + "; ".join(independence_errors))

    evidence = list(evidence_by_key.values())
    materials = [material for item in evidence for material in item["materials"]]
    attachments = [point for item in evidence for point in item["attachments"]["points"]]
    report = {
        "schema": "c3x.source_palace_build.v0",
        "strategy": {"path": str(strategy_path), "sha256": _sha256(strategy_path.read_bytes())},
        "inventory_summary": summary,
        "packages": [package_reports[key] for key in sorted(package_reports)],
        "assets": [
            {"source_package": key[0], **value}
            for key, value in sorted(evidence_by_key.items())
        ],
        "root_catalog": [
            {
                **catalog,
                "source_package": root["source_package"],
                "source_entry": root["source_entry"],
                "artdefs": root["artdefs"],
            }
            for root, catalog in zip(roots, catalog_roots)
        ],
        "skipped_source_conditions": skipped_source_conditions,
        "rejected_optional_dependencies": rejected_optional_dependencies,
        "unresolved_required_dependencies": unresolved_required_dependencies,
        "outputs": {
            "pack": str(pack),
            "root_palaces": len(catalog_roots),
            "compiled_components_with_dependencies": len(assets),
            "geometry_parts": sum(len(item["geometry"]) for item in evidence),
            "materials": len(materials),
            "emissive_materials": sum(
                material.get("texture_slots", {}).get("emissive", {}).get("status") == "accepted"
                for material in materials
            ),
            "attachment_points": len(attachments),
            "rejected_optional_dependencies": len(rejected_optional_dependencies),
            "unresolved_required_dependencies": len(unresolved_required_dependencies),
            "textures": len({relative for relative, _info in texture_cache.values()}),
        },
        "runtime_independence": "passed",
        "runtime_integration": "not_enabled",
    }
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = compile_palaces(args.assets_root, args.strategy, args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, struct.error, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
