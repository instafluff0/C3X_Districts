#!/usr/bin/env python3
"""Compile goody-hut and colony-stand-in source graphs into a generic local pack."""

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
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.improvement_asset_importer import _content_root, _shared_roots


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY = Path(__file__).with_name("tile_object_render_strategy.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "TileObjectsNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "tile_objects" / "build.json"
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
        and value.endswith(".blp")
    )


def load_strategy(path: Path = DEFAULT_STRATEGY) -> dict[str, Any]:
    strategy = json.loads(path.read_text(encoding="utf-8"))
    if strategy.get("schema") != "c3x.source_tile_object_strategy.v0":
        raise ValueError("Unsupported tile-object strategy schema")
    scale = strategy.get("source_units_per_tile")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Tile-object source_units_per_tile must be positive and finite")
    if not _safe_source_path(strategy.get("source_package")):
        raise ValueError("Tile-object strategy needs a safe source package")

    hut = strategy.get("goody_hut")
    variants = hut.get("variants") if isinstance(hut, dict) else None
    if not isinstance(variants, list) or len(variants) < 2:
        raise ValueError("Goody-hut strategy needs multiple source variants")
    ids: set[str] = set()
    entries: set[str] = set()
    for variant in variants:
        asset_id = variant.get("asset_id")
        entry = variant.get("source_entry")
        if not isinstance(asset_id, str) or not SAFE_ID.fullmatch(asset_id):
            raise ValueError("Goody-hut strategy has an invalid asset ID")
        if not isinstance(entry, str) or not entry or asset_id in ids or entry in entries:
            raise ValueError("Goody-hut strategy contains an invalid or duplicate source variant")
        ids.add(asset_id)
        entries.add(entry)
    hut_runtime = hut.get("runtime", {})
    bucket_count = hut_runtime.get("civ3_reference_buckets")
    bucket_map = hut_runtime.get("bucket_to_variant")
    if bucket_count != 8 or not isinstance(bucket_map, list) or len(bucket_map) != bucket_count:
        raise ValueError("Goody-hut strategy must map all eight Civ III reference buckets")
    if any(not isinstance(index, int) or index < 0 or index >= len(variants) for index in bucket_map):
        raise ValueError("Goody-hut bucket mapping refers to an unavailable variant")
    if hut_runtime.get("culture_policy") != "neutral_tribal_site" or hut_runtime.get("era_policy") != "none":
        raise ValueError("Goody huts must remain culture- and era-neutral")

    colony = strategy.get("colony")
    eras = colony.get("eras") if isinstance(colony, dict) else None
    if not isinstance(eras, list) or not eras:
        raise ValueError("Colony strategy has no era profiles")
    era_coverage: list[int] = []
    for era in eras:
        era_id = era.get("id")
        source_entries = era.get("source_entries")
        if not isinstance(era_id, str) or not SAFE_ID.fullmatch(era_id):
            raise ValueError("Colony strategy has an invalid era ID")
        if not isinstance(source_entries, list) or not source_entries or not all(
            isinstance(entry, str) and entry for entry in source_entries
        ):
            raise ValueError("Colony era has no source entries")
        era_coverage.extend(era.get("civ3_eras", []))
    if sorted(era_coverage) != list(range(4)):
        raise ValueError("Colony strategy must cover Civ III eras 0 through 3 exactly once")
    industrial = colony.get("industrial_source_resolution")
    if not isinstance(industrial, list) or len(industrial) != 3 or not all(
        isinstance(item.get("source_entry"), str)
        and item.get("status") == "passed_strict_bind_pose_after_matrix_order_correction"
        and isinstance(item.get("maximum_error"), (int, float))
        and 0 <= item["maximum_error"] <= 2.0e-5
        for item in industrial
    ):
        raise ValueError("Colony strategy must preserve strict industrial-camp resolution evidence")
    colony_runtime = colony.get("runtime", {})
    if colony_runtime.get("owner_source") != "Colony_Body.OwnerID":
        raise ValueError("Colony owner color must come from the colony body")
    if colony_runtime.get("territory_owner_is_not_colony_owner") is not True:
        raise ValueError("Colony strategy must preserve extraterritorial owner semantics")
    if colony_runtime.get("resource_policy") != "preserve_independently_rendered_resource_body":
        raise ValueError("Colony strategy must preserve the resource body")
    body_scale = colony_runtime.get("body_scale")
    if not isinstance(body_scale, (int, float)) or not 0 < body_scale < 1:
        raise ValueError("Colony stand-in must be smaller than one tile")
    owner_marker = colony_runtime.get("owner_marker", {})
    colored_fraction = owner_marker.get("maximum_colored_surface_fraction")
    if not isinstance(colored_fraction, (int, float)) or not 0 < colored_fraction <= 0.2:
        raise ValueError("Colony owner-color surface fraction is outside the restrained range")
    if strategy.get("runtime_integration") != "not_enabled":
        raise ValueError("Tile-object intake must remain offline-only")

    infrastructure = strategy.get("infrastructure")
    source_assets = infrastructure.get("source_assets") if isinstance(infrastructure, dict) else None
    if not isinstance(source_assets, list) or not source_assets:
        raise ValueError("Infrastructure strategy has no accepted source assets")
    infrastructure_ids: set[str] = set()
    infrastructure_entries: set[str] = set()
    for item in source_assets:
        asset_id = item.get("asset_id")
        source_entry = item.get("source_entry")
        if (
            not isinstance(asset_id, str)
            or not SAFE_ID.fullmatch(asset_id)
            or not asset_id.startswith("infrastructure/")
            or not isinstance(source_entry, str)
            or not source_entry
            or asset_id in infrastructure_ids
            or source_entry in infrastructure_entries
        ):
            raise ValueError("Infrastructure strategy has an invalid or duplicate source asset")
        infrastructure_ids.add(asset_id)
        infrastructure_entries.add(source_entry)
    families = infrastructure.get("families")
    if not isinstance(families, dict) or set(families) != {
        "fortress", "barricade", "airfield", "outpost"
    }:
        raise ValueError("Infrastructure strategy must cover the four accepted families")
    for family_id, family in families.items():
        era_assets = family.get("civ3_era_assets")
        if (
            not isinstance(era_assets, list)
            or len(era_assets) != 4
            or any(asset_id not in infrastructure_ids for asset_id in era_assets)
        ):
            raise ValueError(f"Infrastructure family {family_id} must map all four Civ III eras")
    deferred = infrastructure.get("deferred_families")
    if not isinstance(deferred, dict) or set(deferred) != {
        "radar_tower", "pollution", "crater", "victory_location"
    }:
        raise ValueError("Infrastructure strategy must retain every deferred family")
    rejected = infrastructure.get("rejected_source_candidates")
    if not isinstance(rejected, list) or len(rejected) < 2:
        raise ValueError("Infrastructure strategy must preserve rejected source evidence")
    return strategy


def goody_variant_for_bucket(strategy: dict[str, Any], bucket: int) -> str:
    mapping = strategy["goody_hut"]["runtime"]["bucket_to_variant"]
    if not isinstance(bucket, int) or bucket < 0 or bucket >= len(mapping):
        raise ValueError("Goody-hut bucket is outside the Civ III range")
    return strategy["goody_hut"]["variants"][mapping[bucket]]["asset_id"]


def _dependency_asset_id(package_relative: str, entry: str) -> str:
    digest = _sha256((package_relative + "\0" + entry).encode("utf-8"))[:16]
    return f"tile_object/component/{digest}"


def compile_tile_objects(
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
        raise ValueError("Tile-object source report must be outside the runtime pack")

    source_package = strategy["source_package"]
    root_ids = {
        (source_package, item["source_entry"]): item["asset_id"]
        for item in strategy["goody_hut"]["variants"]
    }
    for era in strategy["colony"]["eras"]:
        for entry in era["source_entries"]:
            root_ids.setdefault(
                (source_package, entry),
                "tile_object/colony/body/" + re.sub(r"[^a-z0-9]+", "_", entry.lower()).strip("_"),
            )
    for item in strategy["infrastructure"]["source_assets"]:
        root_ids[(source_package, item["source_entry"])] = item["asset_id"]

    packages = _package_index(assets_root)
    package_bytes: dict[str, bytes] = {}
    package_cache: dict[str, IndexedStaticPackage] = {}
    package_reports: dict[str, dict[str, Any]] = {}
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    assets: dict[str, Any] = {}
    evidence_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    rejected_optional_dependencies = []
    skipped_source_conditions = []
    visiting: set[tuple[str, str]] = set()

    def ensure_asset(package_relative: str, entry: str) -> str:
        key = (package_relative, entry)
        asset_id = root_ids.get(key, _dependency_asset_id(*key))
        if key in evidence_by_key:
            return asset_id
        if key in visiting:
            raise ValueError(f"Tile-object component dependency cycle at {entry}")
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
            raise ValueError(f"Could not compile tile-object component {entry}: {exc}") from exc
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
                        f"Tile-object component {entry} has an unresolved child {terminal['entry']}"
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
                    {"source_package": package_relative, "source_entry": entry, "attachment": source_point}
                )
        _write_json(document_path, document)
        assets[asset_id] = manifest_asset
        evidence_by_key[key] = evidence
        visiting.remove(key)
        return asset_id

    hut_variants = [
        ensure_asset(source_package, item["source_entry"])
        for item in strategy["goody_hut"]["variants"]
    ]
    colony_eras = []
    for era in strategy["colony"]["eras"]:
        colony_eras.append(
            {
                "id": era["id"],
                "civ3_eras": era["civ3_eras"],
                "source_stage": era["source_stage"],
                "variants": [ensure_asset(source_package, entry) for entry in era["source_entries"]],
            }
        )
    infrastructure_assets = [
        ensure_asset(source_package, item["source_entry"])
        for item in strategy["infrastructure"]["source_assets"]
    ]

    catalog_path = "tile_object_catalog.json"
    _write_json(
        pack / catalog_path,
        {
            "schema": "c3x.tile_object_catalog.v0",
            "goody_hut": {
                "variants": hut_variants,
                "bucket_to_variant": strategy["goody_hut"]["runtime"]["bucket_to_variant"],
                "runtime": strategy["goody_hut"]["runtime"],
            },
            "colony": {
                "stand_in": strategy["colony"]["stand_in"],
                "eras": colony_eras,
                "runtime": strategy["colony"]["runtime"],
                "industrial_source_resolution": strategy["colony"]["industrial_source_resolution"],
                "promotion_status": "offline_intake_complete_visual_approval_pending_l19a",
            },
            "infrastructure": {
                "available_assets": infrastructure_assets,
                "families": strategy["infrastructure"]["families"],
                "runtime": strategy["infrastructure"]["runtime"],
                "promotion_status": "offline_intake_only",
            },
            "provenance": {
                "kind": "local_normalized_import",
                "adapter": "c3x.tile_object_component.v0",
                "source_format_dependency": None,
            },
            "runtime_integration": "not_enabled",
        },
    )
    _write_json(
        pack / "manifest.json",
        {
            "schema": "c3x.asset_pack.v0",
            "name": "TileObjectsNormalized",
            "display_name": "Normalized Goody Huts, Colonies, And Tile Infrastructure",
            "source_policy": "Local licensed-source import; derived art is not redistributable.",
            "assets": dict(sorted(assets.items())),
            "tile_object_catalog": catalog_path,
            "runtime_integration": "not_enabled",
        },
    )
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime tile-object pack is source-dependent: " + "; ".join(independence_errors))

    evidence = list(evidence_by_key.values())
    materials = [material for item in evidence for material in item["materials"]]
    attachments = [point for item in evidence for point in item["attachments"]["points"]]
    report = {
        "schema": "c3x.source_tile_object_build.v0",
        "strategy": {"path": str(strategy_path), "sha256": _sha256(strategy_path.read_bytes())},
        "packages": [package_reports[key] for key in sorted(package_reports)],
        "assets": [
            {"source_package": key[0], **value}
            for key, value in sorted(evidence_by_key.items())
        ],
        "skipped_source_conditions": skipped_source_conditions,
        "rejected_optional_dependencies": rejected_optional_dependencies,
        "resolved_industrial_colony_source_candidates": strategy["colony"]["industrial_source_resolution"],
        "rejected_infrastructure_source_candidates": strategy["infrastructure"]["rejected_source_candidates"],
        "deferred_infrastructure_families": strategy["infrastructure"]["deferred_families"],
        "outputs": {
            "pack": str(pack),
            "hut_root_variants": len(hut_variants),
            "colony_root_variants": len({value for era in colony_eras for value in era["variants"]}),
            "infrastructure_root_assets": len(infrastructure_assets),
            "compiled_components_with_dependencies": len(assets),
            "geometry_parts": sum(len(item["geometry"]) for item in evidence),
            "materials": len(materials),
            "emissive_materials": sum(
                material.get("texture_slots", {}).get("emissive", {}).get("status") == "accepted"
                for material in materials
            ),
            "attachment_points": len(attachments),
            "unresolved_effect_resources": sum(
                item["binding_status"] == "resource_unresolved" for item in attachments
            ),
            "rejected_optional_dependencies": len(rejected_optional_dependencies),
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
        report = compile_tile_objects(args.assets_root, args.strategy, args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, struct.error, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
