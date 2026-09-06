#!/usr/bin/env python3
"""Compile representative mine and farm graphs into a generic local pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.artdef_graph_resolver import (
    DEFAULT_ASSETS_ROOT,
    DEFAULT_RESOURCE_MAPPING,
    _package_index,
    _resolve_package,
    build_resource_improvement_graphs,
)
from Renderer.tools.asset_compiler.compound_landmark_importer import _compile_asset
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY = Path(__file__).with_name("improvement_render_strategy.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "ImprovementsNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "improvements" / "build.json"
SAFE_ID = re.compile(
    r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]?[a-z0-9]+)*)*$"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_source_path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
        and "\\" not in value
    )


def load_strategy(path: Path = DEFAULT_STRATEGY) -> dict[str, Any]:
    strategy = json.loads(path.read_text(encoding="utf-8"))
    if strategy.get("schema") != "c3x.source_improvement_strategy.v0":
        raise ValueError("Unsupported improvement strategy schema")
    scale = strategy.get("source_units_per_tile")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Improvement source_units_per_tile must be positive and finite")
    mine = strategy.get("mine")
    farm = strategy.get("farm")
    if not isinstance(mine, dict) or not _safe_source_path(mine.get("source_package")):
        raise ValueError("Mine strategy needs a safe source package")
    mine_eras = mine.get("eras")
    if not isinstance(mine_eras, list) or not mine_eras:
        raise ValueError("Mine strategy has no eras")
    mine_coverage = []
    for era in mine_eras:
        if not isinstance(era.get("id"), str) or not SAFE_ID.fullmatch(era["id"]):
            raise ValueError("Mine era has an invalid generic ID")
        entries = era.get("source_entries")
        if not isinstance(entries, list) or not entries or not all(isinstance(value, str) and value for value in entries):
            raise ValueError("Mine era has no source entries")
        mine_coverage.extend(era.get("civ3_eras", []))
    if sorted(mine_coverage) != list(range(4)):
        raise ValueError("Mine strategy must cover Civ III eras 0 through 3 exactly once")
    if not isinstance(farm, dict) or not isinstance(farm.get("packages"), dict):
        raise ValueError("Farm strategy has no source packages")
    if not farm["packages"] or not all(_safe_source_path(value) for value in farm["packages"].values()):
        raise ValueError("Farm strategy contains an unsafe source package")
    farm_coverage = []
    for era in farm.get("eras", []):
        if not isinstance(era.get("id"), str) or not SAFE_ID.fullmatch(era["id"]):
            raise ValueError("Farm era has an invalid generic ID")
        for field in ("building_entries", "tile_entries"):
            values = era.get(field)
            if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
                raise ValueError(f"Farm era has no {field}")
        farm_coverage.extend(era.get("civ3_eras", []))
    if sorted(farm_coverage) != list(range(4)):
        raise ValueError("Farm strategy must cover Civ III eras 0 through 3 exactly once")
    crop_ids = []
    for crop in farm.get("crop_styles", []):
        crop_id = crop.get("id")
        package = crop.get("package")
        if not isinstance(crop_id, str) or not SAFE_ID.fullmatch(crop_id):
            raise ValueError("Farm crop style has an invalid generic ID")
        if package not in farm["packages"]:
            raise ValueError("Farm crop style uses an unknown source package")
        entries = crop.get("source_entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Farm crop style has no source entries")
        crop_ids.append(crop_id)
    if not crop_ids or len(crop_ids) != len(set(crop_ids)) or "default" not in crop_ids:
        raise ValueError("Farm crop styles must be unique and include default")
    runtime = farm.get("runtime", {})
    if runtime.get("civ3_adjacency_bits") != 4 or runtime.get("civ3_adjacency_masks") != 16:
        raise ValueError("Farm strategy must preserve the 16-mask Civ III irrigation contract")
    if strategy.get("runtime_integration") != "not_enabled":
        raise ValueError("Improvement intake must remain offline-only")
    return strategy


def _content_root(package_relative: str) -> str:
    parts = Path(package_relative).parts
    lowered = [part.lower() for part in parts]
    if "platforms" not in lowered:
        raise ValueError(f"Improvement package has no Platforms path component: {package_relative}")
    return Path(*parts[: lowered.index("platforms")]).as_posix()


def _shared_roots(assets_root: Path, package_relative: str) -> list[Path]:
    relative = Path(package_relative)
    lowered = [part.lower() for part in relative.parts]
    if "blps" not in lowered:
        raise ValueError(f"Improvement package has no BLPs path component: {package_relative}")
    index = lowered.index("blps")
    local = assets_root / Path(*relative.parts[: index + 1]) / "SHARED_DATA"
    base = assets_root / "Base" / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA"
    result = [local]
    if local != base:
        result.append(base)
    missing = next((path for path in result if not path.is_dir()), None)
    if missing is not None:
        raise FileNotFoundError(missing)
    return result


def _asset_id(package_relative: str, entry: str) -> str:
    digest = _sha256((package_relative + "\0" + entry).encode("utf-8"))[:16]
    return f"improvement/component/{digest}"


def _graph_summary(report: dict[str, Any], graph_id: str) -> dict[str, Any]:
    graph = next(item for item in report["graphs"] if item["graph_id"] == graph_id)
    terminals = [item for item in graph["terminals"] if item["scope"] == "map_visual"]
    return {
        "nodes": len(graph["nodes"]),
        "terminals": len(terminals),
        "unique_assets": len({(item["package_path"], item["entry"]) for item in terminals}),
        "classes": dict(sorted(Counter(item["class"] for item in terminals).items())),
        "packages": dict(sorted(Counter(item["package_path"] for item in terminals).items())),
    }


def compile_improvements(
    assets_root: Path,
    strategy_path: Path = DEFAULT_STRATEGY,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_REPORT,
    compile_discovered_library: bool = False,
) -> dict[str, Any]:
    strategy = load_strategy(strategy_path)
    try:
        report_path.resolve().relative_to(pack.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Improvement source report must be outside the runtime pack")
    graph_report = build_resource_improvement_graphs(assets_root, DEFAULT_RESOURCE_MAPPING)
    packages = _package_index(assets_root)
    package_bytes: dict[str, bytes] = {}
    package_cache: dict[str, IndexedStaticPackage] = {}
    package_reports: dict[str, dict[str, Any]] = {}
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    assets: dict[str, Any] = {}
    evidence_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    visiting: set[tuple[str, str]] = set()
    skipped_source_conditions = []
    rejected_optional_dependencies = []
    rejected_optional_roots = []
    rejected_discovered_roots = []

    def ensure_asset(package_relative: str, entry: str) -> str:
        key = (package_relative, entry)
        asset_id = _asset_id(*key)
        if key in evidence_by_key:
            return asset_id
        if key in visiting:
            raise ValueError(f"Improvement component dependency cycle at {entry}")
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
            )
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            visiting.discard(key)
            raise ValueError(f"Could not compile improvement component {entry}: {exc}") from exc
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
                    raise ValueError(
                        f"Improvement component {entry} has an unresolved child {terminal['entry']}"
                    )
                try:
                    child_id = ensure_asset(resolution["package_path"], terminal["entry"])
                except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                    if point.get("selection", {}).get("cull") != "optional":
                        visiting.discard(key)
                        raise
                    point["binding_status"] = "component_compile_unresolved"
                    rejected_optional_dependencies.append(
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

    mine_package = strategy["mine"]["source_package"]
    mine_eras = []
    for era in strategy["mine"]["eras"]:
        variants = [ensure_asset(mine_package, entry) for entry in era["source_entries"]]
        mine_eras.append(
            {"id": era["id"], "civ3_eras": era["civ3_eras"], "variants": variants}
        )

    base_farm_package = strategy["farm"]["packages"]["base"]
    farm_eras = []
    for era in strategy["farm"]["eras"]:
        farm_eras.append(
            {
                "id": era["id"],
                "civ3_eras": era["civ3_eras"],
                "building_pieces": [
                    ensure_asset(base_farm_package, entry) for entry in era["building_entries"]
                ],
                "tile_bases": [
                    ensure_asset(base_farm_package, entry) for entry in era["tile_entries"]
                ],
            }
        )
    crop_styles = []
    for crop in strategy["farm"]["crop_styles"]:
        source_package = strategy["farm"]["packages"][crop["package"]]
        pieces = []
        for entry in crop["source_entries"]:
            try:
                pieces.append(ensure_asset(source_package, entry))
            except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                if crop["runtime_policy"] == "default":
                    raise
                rejected_optional_roots.append(
                    {
                        "style": crop["id"],
                        "source_package": source_package,
                        "source_entry": entry,
                        "reason": str(exc),
                    }
                )
        crop_styles.append(
            {
                "id": crop["id"],
                "runtime_policy": crop["runtime_policy"],
                "pieces": pieces,
            }
        )

    discovered_library = {"mine": [], "farm": []}
    if compile_discovered_library:
        for kind, graph_id in (("mine", "improvement/mine"), ("farm", "improvement/farm")):
            graph = next(item for item in graph_report["graphs"] if item["graph_id"] == graph_id)
            terminals = sorted(
                {
                    (item["package_path"], item["entry"])
                    for item in graph["terminals"]
                    if item["scope"] == "map_visual"
                }
            )
            for package_relative, entry in terminals:
                try:
                    discovered_library[kind].append(
                        ensure_asset(package_relative, entry)
                    )
                except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                    rejected_discovered_roots.append(
                        {
                            "kind": kind,
                            "source_package": package_relative,
                            "source_entry": entry,
                            "reason": str(exc),
                        }
                    )

    catalog_path = "improvement_catalog.json"
    _write_json(
        pack / catalog_path,
        {
            "schema": "c3x.improvement_catalog.v0",
            "composition_status": "representative_intake_only",
            "mine": {"eras": mine_eras, "runtime": strategy["mine"]["runtime"]},
            "farm": {
                "eras": farm_eras,
                "crop_styles": crop_styles,
                "runtime": strategy["farm"]["runtime"],
            },
            "intake_library": {
                "status": (
                    "complete_discovered_sweep"
                    if compile_discovered_library
                    else "representative_only"
                ),
                "mine_assets": sorted(set(discovered_library["mine"])),
                "farm_assets": sorted(set(discovered_library["farm"])),
                "runtime_selection": "not_enabled",
            },
            "provenance": {
                "kind": "local_normalized_import",
                "adapter": "c3x.improvement_component.v0",
                "source_format_dependency": None,
            },
        },
    )
    _write_json(
        pack / "manifest.json",
        {
            "schema": "c3x.asset_pack.v0",
            "name": "ImprovementsNormalized",
            "display_name": "Normalized Mine And Farm Components",
            "source_policy": "Local licensed-source import; derived art is not redistributable.",
            "assets": dict(sorted(assets.items())),
            "improvement_catalog": catalog_path,
        },
    )
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime improvement pack is source-dependent: " + "; ".join(independence_errors))

    evidence = list(evidence_by_key.values())
    materials = [material for item in evidence for material in item["materials"]]
    attachments = [point for item in evidence for point in item["attachments"]["points"]]
    report = {
        "schema": "c3x.source_improvement_component_build.v0",
        "strategy": {"path": str(strategy_path), "sha256": _sha256(strategy_path.read_bytes())},
        "source_graph": {
            "mine": _graph_summary(graph_report, "improvement/mine"),
            "farm": _graph_summary(graph_report, "improvement/farm"),
            "unresolved_visual_edges": graph_report["summary"]["unresolved_visual_edges"],
            "unresolved_visual_terminals": graph_report["summary"]["unresolved_visual_terminals"],
        },
        "packages": [package_reports[key] for key in sorted(package_reports)],
        "assets": [
            {"source_package": key[0], **value}
            for key, value in sorted(evidence_by_key.items())
        ],
        "skipped_source_conditions": skipped_source_conditions,
        "rejected_optional_dependencies": rejected_optional_dependencies,
        "rejected_optional_roots": rejected_optional_roots,
        "rejected_discovered_roots": rejected_discovered_roots,
        "outputs": {
            "pack": str(pack),
            "mine_root_variants": sum(len(item["variants"]) for item in mine_eras),
            "farm_root_pieces": sum(
                len(item["building_pieces"]) + len(item["tile_bases"]) for item in farm_eras
            ) + sum(len(item["pieces"]) for item in crop_styles),
            "compiled_components_with_dependencies": len(assets),
            "geometry_parts": sum(len(item["geometry"]) for item in evidence),
            "materials": len(materials),
            "emissive_materials": sum(
                material.get("texture_slots", {}).get("emissive", {}).get("status") == "accepted"
                for material in materials
            ),
            "decal_descriptors": sum(item["decal"].get("count", 0) for item in evidence),
            "attachment_points": len(attachments),
            "resolved_component_attachments": sum(
                item["binding_status"] == "component_unresolved" for item in attachments
            ),
            "unresolved_effect_resources": sum(
                item["binding_status"] == "resource_unresolved" for item in attachments
            ),
            "excluded_source_resource_conditions": len(skipped_source_conditions),
            "unresolved_component_transforms": sum(
                item["binding_status"] == "component_transform_unresolved" for item in attachments
            ),
            "omitted_source_placeholders": sum(
                item["binding_status"] == "source_asset_absent" for item in attachments
            ),
            "preserved_unresolved_terrain_edits": sum(
                item["terrain_edit"]["status"] == "preserved_unresolved" for item in evidence
            ),
            "rejected_optional_dependencies": len(rejected_optional_dependencies),
            "rejected_optional_roots": len(rejected_optional_roots),
            "discovered_mine_roots": len(set(discovered_library["mine"])),
            "discovered_farm_roots": len(set(discovered_library["farm"])),
            "rejected_discovered_roots": len(rejected_discovered_roots),
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
    parser.add_argument(
        "--compile-discovered-library",
        action="store_true",
        help="normalize every unique mine/farm terminal that passes strict conversion",
    )
    args = parser.parse_args(argv)
    try:
        report = compile_improvements(
            args.assets_root,
            args.strategy,
            args.pack,
            args.report,
            args.compile_discovered_library,
        )
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
