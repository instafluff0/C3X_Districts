#!/usr/bin/env python3
"""Normalize source-backed city-wall kits and audit the capital marker offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT
from Renderer.tools.asset_compiler.compound_landmark_importer import _compile_asset
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = Path(__file__).with_name("city_adjunct_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "CityAdjunctsNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "cities" / "adjunct_build.json"
SAFE_ID = re.compile(r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]?[a-z0-9]+)*)*$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_mapping(path: Path = DEFAULT_MAPPING) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "c3x.source_city_adjunct_mapping.v0":
        raise ValueError("Unsupported city-adjunct mapping schema")
    scale = document.get("source_units_per_tile")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("City-adjunct source scale must be positive")
    assets = document.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("City-adjunct mapping contains no assets")
    ids: set[str] = set()
    sources: set[tuple[str, str]] = set()
    allowed_kinds = {"wall_piece"}
    allowed_roles = {"half", "segment", "gate", "tower"}
    allowed_eras = {"ancient", "medieval", "industrial"}
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("City-adjunct asset mapping must be an object")
        asset_id = asset.get("asset_id")
        package = asset.get("source_package")
        entry = asset.get("source_entry")
        if not isinstance(asset_id, str) or not SAFE_ID.fullmatch(asset_id):
            raise ValueError("City-adjunct mapping has an invalid asset ID")
        if not isinstance(package, str) or not package.endswith(".blp") or ".." in Path(package).parts:
            raise ValueError(f"City-adjunct mapping has an invalid package: {asset_id}")
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"City-adjunct mapping has an invalid source entry: {asset_id}")
        if asset.get("kind") not in allowed_kinds or asset.get("role") not in allowed_roles:
            raise ValueError(f"City-adjunct mapping has an invalid kind or role: {asset_id}")
        if asset.get("era") not in allowed_eras:
            raise ValueError(f"City-adjunct mapping has an invalid era: {asset_id}")
        if asset_id in ids or (package, entry) in sources:
            raise ValueError("City-adjunct mapping contains a duplicate")
        ids.add(asset_id)
        sources.add((package, entry))
    era_selection = document.get("era_selection")
    if era_selection != {"0": "ancient", "1": "medieval", "2": "industrial", "3": "industrial"}:
        raise ValueError("City-adjunct era selection must cover all four Civ III eras")
    capital_probe = document.get("capital_probe", {})
    if (
        capital_probe.get("status") != "composition_marker_not_terminal_asset"
        or capital_probe.get("source_building") != "BUILDING_PALACE"
        or not capital_probe.get("attachment_point")
    ):
        raise ValueError("City-adjunct capital probe is invalid")
    return document


def _shared_roots(assets_root: Path, package_relative: str) -> list[Path]:
    parts = Path(package_relative).parts
    lowered = [part.lower() for part in parts]
    index = lowered.index("blps")
    local = assets_root / Path(*parts[: index + 1]) / "SHARED_DATA"
    base = assets_root / "Base" / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA"
    return [local] if local == base else [local, base]


def compile_city_adjuncts(
    assets_root: Path,
    mapping_path: Path = DEFAULT_MAPPING,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    mapping = load_mapping(mapping_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in mapping["assets"]:
        grouped[asset["source_package"]].append(asset)
    assets: dict[str, Any] = {}
    evidence = []
    package_reports = []
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for package_relative, mapped_assets in sorted(grouped.items()):
        source_path = assets_root / package_relative
        package = IndexedStaticPackage(source_path, mapped_assets[0]["source_entry"])
        package_reports.append(
            {
                "source": str(source_path),
                "sha256": _sha256(source_path),
                "allocation_count": len(package.allocations),
            }
        )
        for mapped in mapped_assets:
            try:
                manifest_asset, asset_evidence = _compile_asset(
                    package,
                    _shared_roots(assets_root, package_relative),
                    pack,
                    mapped["source_entry"],
                    mapped["asset_id"],
                    float(mapping["source_units_per_tile"]),
                    texture_cache,
                )
            except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                raise ValueError(
                    f"Failed city adjunct {mapped['asset_id']} from {mapped['source_entry']}: {exc}"
                ) from exc
            assets[mapped["asset_id"]] = manifest_asset
            evidence.append({"mapping": mapped, "evidence": asset_evidence})
    wall_kits = {}
    for era in ("ancient", "medieval", "industrial"):
        pieces: dict[str, list[str]] = defaultdict(list)
        for mapped in mapping["assets"]:
            if mapped["kind"] == "wall_piece" and mapped["era"] == era:
                pieces[mapped["role"]].append(mapped["asset_id"])
        if set(pieces) != {"half", "segment", "gate", "tower"}:
            raise ValueError(f"City wall kit is incomplete: {era}")
        wall_kits[era] = {role: sorted(values) for role, values in sorted(pieces.items())}
    catalog = {
        "schema": "c3x.city_adjunct_catalog.v0",
        "capital": {
            "asset": None,
            "composition": "source_generator_state_requires_resolution",
            "binding_status": "no_terminal_asset_confirmed",
            "missing_policy": "retain_native_capital_indicator",
        },
        "walls": {
            "era_selection": mapping["era_selection"],
            "kits": wall_kits,
            "composition": "perimeter_from_authoritative_city_footprint",
            "topology_status": "lab_mapping_required",
            "missing_policy": "retain_native_city_walls",
        },
        "runtime_integration": "not_enabled",
    }
    _write_json(pack / "city_adjunct_catalog.json", catalog)
    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "CityAdjunctsNormalized",
        "display_name": "Normalized City Wall Kits And Capital Evidence",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "assets": dict(sorted(assets.items())),
        "city_adjunct_catalog": "city_adjunct_catalog.json",
        "runtime_integration": "not_enabled",
    }
    _write_json(pack / "manifest.json", manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime city-adjunct pack is source-dependent: " + "; ".join(independence_errors))
    report = {
        "schema": "c3x.source_city_adjunct_build.v0",
        "mapping": {"path": str(mapping_path), "sha256": _sha256(mapping_path)},
        "capital_probe": mapping["capital_probe"],
        "packages": package_reports,
        "assets": evidence,
        "outputs": {
            "pack": str(pack),
            "assets": len(assets),
            "capital_accents": 0,
            "wall_pieces": sum(item["mapping"]["kind"] == "wall_piece" for item in evidence),
            "geometry_parts": sum(len(item["evidence"]["geometry"]) for item in evidence),
            "materials": sum(len(item["evidence"]["materials"]) for item in evidence),
            "textures": len(texture_cache),
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
        report = compile_city_adjuncts(args.assets_root, args.mapping, args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, struct.error, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
