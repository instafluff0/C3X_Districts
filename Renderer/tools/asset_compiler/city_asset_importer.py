#!/usr/bin/env python3
"""Compile representative generated-city components into a generic local pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
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


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY = Path(__file__).with_name("city_render_strategy.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "CityComponentsNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "cities" / "build.json"
SAFE_ID = re.compile(
    r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]?[a-z0-9]+)*)*$"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attribute(element: ET.Element | None, name: str = "text") -> str:
    return "" if element is None else element.attrib.get(name, "")


def _field_payload(value: ET.Element) -> dict[str, str]:
    parameter = _attribute(value.find("m_ParamName"))
    if not parameter:
        raise ValueError("City block field has no parameter name")
    if parameter in ("Tag_Culture", "Tag_Era"):
        payload = _attribute(value.find("m_ElementName"))
    elif parameter == "Asset_CityBlock":
        if _attribute(value.find("m_XLPClass")) != "CityBuildings":
            raise ValueError("City block terminal is not a CityBuildings entry")
        payload = _attribute(value.find("m_EntryName"))
    else:
        raise ValueError(f"Unexpected city block field {parameter}")
    if not payload:
        raise ValueError(f"City block field {parameter} is empty")
    result = {"parameter": parameter, "value": payload}
    if parameter == "Asset_CityBlock":
        package = _attribute(value.find("m_BLPPackage"))
        if not package:
            raise ValueError("City block terminal has no package")
        result["package"] = package
    return result


def parse_city_generator_blocks(path: Path, relative: Path) -> list[dict[str, Any]]:
    document = ET.parse(path).getroot()
    collections = [
        collection
        for collection in document.findall("./m_RootCollections/Element")
        if _attribute(collection.find("m_CollectionName")) == "GeneratorBlockList"
    ]
    records = []
    for collection in collections:
        for block_list in collection.findall("Element"):
            list_name = _attribute(block_list.find("m_Name"))
            child_collections = block_list.findall("./m_ChildCollections/Element")
            if len(child_collections) != 1 or _attribute(
                child_collections[0].find("m_CollectionName")
            ) != "Block":
                raise ValueError(f"Unsupported GeneratorBlockList layout in {relative}")
            for block in child_collections[0].findall("Element"):
                fields = [_field_payload(value) for value in block.findall("./m_Fields/m_Values/Element")]
                by_name = {field["parameter"]: field for field in fields}
                required = {"Tag_Culture", "Tag_Era", "Asset_CityBlock"}
                if set(by_name) != required or len(fields) != len(required):
                    raise ValueError(f"City block has unexpected fields in {relative}")
                records.append(
                    {
                        "source_artdef": relative.as_posix(),
                        "list_name": list_name,
                        "block_name": _attribute(block.find("m_Name")),
                        "source_culture": by_name["Tag_Culture"]["value"],
                        "source_art_era": by_name["Tag_Era"]["value"],
                        "entry": by_name["Asset_CityBlock"]["value"],
                        "package": by_name["Asset_CityBlock"]["package"],
                    }
                )
    return records


def _content_root(relative: Path) -> str:
    parts = relative.parts
    if "ArtDefs" not in parts:
        raise ValueError(f"City ArtDef path has no ArtDefs component: {relative}")
    return Path(*parts[: parts.index("ArtDefs")]).as_posix()


def read_city_blocks(assets_root: Path) -> list[dict[str, Any]]:
    packages = _package_index(assets_root)
    package_bytes: dict[str, bytes] = {}
    records = []
    for path in sorted(assets_root.rglob("CityGenerators*.artdef")):
        relative = path.relative_to(assets_root)
        for record in parse_city_generator_blocks(path, relative):
            resolution = _resolve_package(
                packages,
                record["package"],
                _content_root(relative),
                record["entry"],
                package_bytes,
            )
            records.append({**record, **resolution})
    if not records:
        raise ValueError("Installed city generator graph contains no city blocks")
    unresolved = [record for record in records if record["status"] != "resolved"]
    if unresolved:
        raise ValueError(f"City generator graph has {len(unresolved)} unresolved block bindings")
    return records


def load_strategy(path: Path = DEFAULT_STRATEGY) -> dict[str, Any]:
    strategy = json.loads(path.read_text(encoding="utf-8"))
    if strategy.get("schema") != "c3x.source_city_strategy.v0":
        raise ValueError("Unsupported city strategy schema")
    count = strategy.get("proof_components_per_pool")
    if not isinstance(count, int) or count < 1:
        raise ValueError("City strategy needs a positive proof component count")
    eras = strategy.get("eras")
    if not isinstance(eras, list) or {item.get("civ3_era") for item in eras} != set(range(4)):
        raise ValueError("City strategy must define Civ III eras 0 through 3")
    era_ids = [item.get("id") for item in eras]
    if len(set(era_ids)) != 4 or not all(isinstance(value, str) and SAFE_ID.fullmatch(value) for value in era_ids):
        raise ValueError("City strategy era IDs must be unique generic IDs")
    styles = strategy.get("styles")
    if not isinstance(styles, list) or {item.get("civ3_culture_group") for item in styles} != set(range(5)):
        raise ValueError("City strategy must define Civ III culture groups 0 through 4")
    style_ids = [item.get("id") for item in styles]
    if len(set(style_ids)) != 5 or not all(isinstance(value, str) and SAFE_ID.fullmatch(value) for value in style_ids):
        raise ValueError("City strategy style IDs must be unique generic IDs")
    for style in styles:
        mapping = style.get("source_culture_by_era")
        if not isinstance(mapping, dict) or set(mapping) != set(era_ids) or not all(
            isinstance(value, str) and value for value in mapping.values()
        ):
            raise ValueError(f"City style {style.get('id')} has an incomplete source culture map")
    recipes = strategy.get("runtime", {}).get("size_recipes")
    if not isinstance(recipes, list) or len(recipes) != 3:
        raise ValueError("City strategy must define town, city, and metropolis recipes")
    expected = [("town", 1, 6), ("city", 7, 12), ("metropolis", 13, None)]
    actual = [(item.get("id"), item.get("population_min"), item.get("population_max")) for item in recipes]
    if actual != expected:
        raise ValueError("City size recipes must cover populations 1-6, 7-12, and 13+")
    for recipe in recipes:
        if not isinstance(recipe.get("component_count"), int) or recipe["component_count"] < 1:
            raise ValueError("City size recipes need positive component counts")
        for field in ("footprint_radius_tiles", "height_scale"):
            value = recipe.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"City size recipe has an invalid {field}")
    return strategy


def build_candidate_pools(
    blocks: list[dict[str, Any]], strategy: dict[str, Any]
) -> list[dict[str, Any]]:
    by_tags: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for block in blocks:
        key = (block["package_path"], block["entry"])
        by_tags[(block["source_culture"], block["source_art_era"])].setdefault(key, block)
    eras = {item["id"]: item for item in strategy["eras"]}
    pools = []
    minimum = strategy["proof_components_per_pool"]
    for style in sorted(strategy["styles"], key=lambda item: item["civ3_culture_group"]):
        for era_id in [item["id"] for item in sorted(strategy["eras"], key=lambda item: item["civ3_era"])]:
            culture = style["source_culture_by_era"][era_id]
            candidates = sorted(
                by_tags.get((culture, eras[era_id]["source_art_era"]), {}).values(),
                key=lambda item: (item["package_path"], item["entry"]),
            )
            if len(candidates) < minimum:
                raise ValueError(
                    f"City pool {style['id']}/{era_id} has {len(candidates)} candidates; needs {minimum}"
                )
            pools.append(
                {
                    "id": f"city/pool/{style['id']}/{era_id}",
                    "style": style,
                    "era": eras[era_id],
                    "source_culture": culture,
                    "candidates": candidates,
                }
            )
    return pools


def _component_id(package_path: str, entry: str) -> str:
    digest = _sha256((package_path + "\0" + entry).encode("utf-8"))[:16]
    return f"city/component/{digest}"


def _shared_roots(assets_root: Path, package_relative: str) -> list[Path]:
    relative = Path(package_relative)
    lowered = [part.lower() for part in relative.parts]
    if "blps" not in lowered:
        raise ValueError(f"City package has no BLPs path component: {package_relative}")
    index = lowered.index("blps")
    local = assets_root / Path(*relative.parts[: index + 1]) / "SHARED_DATA"
    base = assets_root / "Base" / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA"
    roots = [local]
    if base != local:
        roots.append(base)
    missing = [path for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(missing[0])
    return roots


def compile_city_assets(
    assets_root: Path,
    strategy_path: Path = DEFAULT_STRATEGY,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    strategy = load_strategy(strategy_path)
    blocks = read_city_blocks(assets_root)
    candidate_pools = build_candidate_pools(blocks, strategy)
    try:
        report_path.resolve().relative_to(pack.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("City source report must be outside the runtime pack")

    assets: dict[str, Any] = {}
    compiled: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    package_cache: dict[str, IndexedStaticPackage] = {}
    package_reports: dict[str, dict[str, Any]] = {}
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    runtime_pools: dict[str, dict[str, Any]] = {}
    pool_reports = []
    minimum = strategy["proof_components_per_pool"]

    for pool in candidate_pools:
        selected = []
        selected_sources = []
        for candidate in pool["candidates"]:
            source_key = (candidate["package_path"], candidate["entry"])
            if source_key not in compiled:
                try:
                    package = package_cache.get(candidate["package_path"])
                    if package is None:
                        source_path = assets_root / candidate["package_path"]
                        package = IndexedStaticPackage(source_path, candidate["entry"])
                        package_cache[candidate["package_path"]] = package
                        package_reports[candidate["package_path"]] = {
                            "source": str(source_path),
                            "source_sha256": _sha256(package.data),
                            "allocation_count": len(package.allocations),
                            "header": package.header,
                        }
                    asset_id = _component_id(*source_key)
                    manifest_asset, evidence = _compile_asset(
                        package,
                        _shared_roots(assets_root, candidate["package_path"]),
                        pack,
                        candidate["entry"],
                        asset_id,
                        100.0,
                        texture_cache,
                    )
                    assets[asset_id] = manifest_asset
                    compiled[source_key] = (asset_id, evidence)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    rejected.append(
                        {
                            "pool": pool["id"],
                            "source_package": candidate["package_path"],
                            "source_entry": candidate["entry"],
                            "reason": str(exc),
                        }
                    )
                    continue
            asset_id, _evidence = compiled[source_key]
            selected.append(asset_id)
            selected_sources.append(
                {"package": candidate["package_path"], "entry": candidate["entry"], "asset_id": asset_id}
            )
            if len(selected) == minimum:
                break
        if len(selected) != minimum:
            raise ValueError(f"City pool {pool['id']} compiled only {len(selected)} components")
        runtime_pools[pool["id"]] = {"components": selected}
        pool_reports.append(
            {
                "pool": pool["id"],
                "source_culture": pool["source_culture"],
                "source_art_era": pool["era"]["source_art_era"],
                "candidate_count": len(pool["candidates"]),
                "selected": selected_sources,
            }
        )

    catalog_path = "city_catalog.json"
    _write_json(
        pack / catalog_path,
        {
            "schema": "c3x.city_catalog.v0",
            "composition_status": "representative_intake_only",
            "eras": [
                {"civ3_era": item["civ3_era"], "id": item["id"]}
                for item in sorted(strategy["eras"], key=lambda item: item["civ3_era"])
            ],
            "styles": [
                {
                    "civ3_culture_group": style["civ3_culture_group"],
                    "id": style["id"],
                    "era_pools": {
                        era["id"]: f"city/pool/{style['id']}/{era['id']}"
                        for era in sorted(strategy["eras"], key=lambda item: item["civ3_era"])
                    },
                }
                for style in sorted(strategy["styles"], key=lambda item: item["civ3_culture_group"])
            ],
            "pools": runtime_pools,
            "runtime": strategy["runtime"],
            "provenance": {
                "kind": "local_normalized_import",
                "adapter": "c3x.city_component.v0",
                "source_format_dependency": None,
            },
        },
    )
    _write_json(
        pack / "manifest.json",
        {
            "schema": "c3x.asset_pack.v0",
            "name": "CityComponentsNormalized",
            "display_name": "Normalized City Components",
            "source_policy": "Local licensed-source import; derived art is not redistributable.",
            "assets": dict(sorted(assets.items())),
            "city_catalog": catalog_path,
        },
    )
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime city pack is source-dependent: " + "; ".join(independence_errors))

    evidence = [item[1] for item in compiled.values()]
    material_records = [material for item in evidence for material in item["materials"]]
    attachment_records = [point for item in evidence for point in item["attachments"]["points"]]
    unique_bindings = {(item["package_path"], item["entry"]) for item in blocks}
    report = {
        "schema": "c3x.source_city_component_build.v0",
        "strategy": {"path": str(strategy_path), "sha256": _sha256(strategy_path.read_bytes())},
        "source_graph": {
            "bindings": len(blocks),
            "unique_components": len(unique_bindings),
            "source_pools": len({(item["source_culture"], item["source_art_era"]) for item in blocks}),
            "unresolved_bindings": 0,
        },
        "packages": [package_reports[key] for key in sorted(package_reports)],
        "pools": pool_reports,
        "assets": evidence,
        "rejected_candidates": rejected,
        "outputs": {
            "pack": str(pack),
            "proof_pools": len(runtime_pools),
            "compiled_components": len(compiled),
            "geometry_parts": sum(len(item["geometry"]) for item in evidence),
            "materials": len(material_records),
            "emissive_materials": sum(
                material.get("texture_slots", {}).get("emissive", {}).get("status") == "accepted"
                for material in material_records
            ),
            "attachment_points": len(attachment_records),
            "attachment_semantics": dict(sorted(Counter(item["semantic"] for item in attachment_records).items())),
            "unresolved_attachment_resources": sum(
                item["binding_status"] == "resource_unresolved" for item in attachment_records
            ),
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
        report = compile_city_assets(args.assets_root, args.strategy, args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
