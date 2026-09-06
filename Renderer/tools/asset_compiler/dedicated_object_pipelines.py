#!/usr/bin/env python3
"""Validate and inventory the independent city and unit source pipelines."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, deque
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.inventory.unit_mapping_inventory import validate_mapping as validate_unit_mapping
from Renderer.tools.asset_compiler.artdef_graph_resolver import (
    DEFAULT_ASSETS_ROOT,
    _package_index,
    _resolve_package,
    index_artdefs,
    resolve_reference,
)


RENDERER_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RENDERER_ROOT.parent
DEFAULT_CITY_CONTRACT = Path(__file__).with_name("city_pipeline.json")
DEFAULT_UNIT_CONTRACT = Path(__file__).with_name("unit_pipeline.json")
DEFAULT_REPORT = (
    RENDERER_ROOT / "preview" / "out" / "object_pipelines" / "city_unit_inventory.json"
)
SCHEMA = "c3x.source_dedicated_pipeline.v0"
REQUIRED_CITY_AXES = {"civilization", "culture_group", "era", "size", "owner"}
REQUIRED_UNIT_AXES = {"unit_type", "civilization", "owner", "action", "direction"}
UNIT_GRAPH_ROOTS = {
    "Units",
    "UnitMemberTypes",
    "UnitFormationTypes",
    "UnitFormationLayoutTypes",
    "UnitMovementTypes",
    "MemberCombat",
    "UnitCombat",
    "CombatAttack",
    "CombatFormation",
    "UnitDomainTypes",
    "UnitTintTypes",
    "UnitAttachmentBins",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_namespace(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("/"):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in value


def load_contract(path: Path, expected_id: str) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != SCHEMA:
        raise ValueError(f"{path.name} has an unsupported pipeline schema")
    expected_category = {"city": "cities", "unit": "units"}[expected_id]
    if contract.get("pipeline_id") != expected_id or contract.get("category") != expected_category:
        raise ValueError(f"{path.name} is not the dedicated {expected_id} pipeline")
    if contract.get("runtime_integration") != "not_enabled":
        raise ValueError(f"{expected_id} pipeline must remain lab/offline-only")
    if not _safe_namespace(contract.get("runtime_namespace")):
        raise ValueError(f"{expected_id} pipeline has an unsafe runtime namespace")
    roots = contract.get("graph_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError(f"{expected_id} pipeline has no ArtDef graph roots")
    root_keys = []
    for root in roots:
        if not isinstance(root, dict) or not all(
            isinstance(root.get(field), str) and root[field]
            for field in ("document_prefix", "root_collection")
        ):
            raise ValueError(f"{expected_id} pipeline contains an invalid graph root")
        root_keys.append((root["document_prefix"].lower(), root["root_collection"]))
    if len(root_keys) != len(set(root_keys)):
        raise ValueError(f"{expected_id} pipeline repeats an ArtDef graph root")
    axes = contract.get("selector_axes")
    if not isinstance(axes, list) or len(axes) != len(set(axes)):
        raise ValueError(f"{expected_id} pipeline selector axes are invalid or duplicated")
    required_axes = REQUIRED_CITY_AXES if expected_id == "city" else REQUIRED_UNIT_AXES
    if not required_axes.issubset(axes):
        raise ValueError(f"{expected_id} pipeline is missing required selector axes")
    stages = contract.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"{expected_id} pipeline has no compiler stages")
    stage_ids = []
    for stage in stages:
        if not isinstance(stage, dict) or not all(
            isinstance(stage.get(field), str) and stage[field] for field in ("id", "adapter")
        ):
            raise ValueError(f"{expected_id} pipeline contains an invalid compiler stage")
        stage_ids.append(stage["id"])
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError(f"{expected_id} pipeline repeats a compiler stage")
    return contract


def validate_separation(city: dict[str, Any], unit: dict[str, Any]) -> None:
    if city["runtime_namespace"] == unit["runtime_namespace"]:
        raise ValueError("City and unit pipelines must use separate normalized namespaces")
    city_roots = {item["root_collection"] for item in city["graph_roots"]}
    unit_roots = {item["root_collection"] for item in unit["graph_roots"]}
    if city_roots & set(city.get("forbidden_source_roots", [])):
        raise ValueError("City pipeline claims a forbidden unit graph root")
    if unit_roots & set(unit.get("forbidden_source_roots", [])):
        raise ValueError("Unit pipeline claims a forbidden city graph root")
    if city["composition_mode"] == unit["composition_mode"]:
        raise ValueError("City and unit pipelines must not share a composition mode")
    city_outputs = set(city.get("normalized_outputs", []))
    unit_outputs = set(unit.get("normalized_outputs", []))
    if not city_outputs or not unit_outputs:
        raise ValueError("City and unit pipelines require non-empty output contracts")
    if "composition_recipes" not in city_outputs or "formation_recipes" not in unit_outputs:
        raise ValueError("City composition and unit formation outputs must remain distinct")


def _matching_nodes(index: dict[str, Any], root: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = root["document_prefix"].lower()
    collection = root["root_collection"]
    return sorted(
        (
            node
            for node in index["nodes"].values()
            if node["root_collection"] == collection
            and node["document"].lower().startswith(prefix)
        ),
        key=lambda node: node["id"],
    )


def _root_inventory(index: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    inventory = []
    for root in contract["graph_roots"]:
        nodes = _matching_nodes(index, root)
        inventory.append(
            {
                **root,
                "nodes": len(nodes),
                "documents": sorted({node["path"] for node in nodes}),
                "terminals": sum(len(node["terminals"]) for node in nodes),
            }
        )
    return inventory


def _city_axis_evidence(assets_root: Path) -> dict[str, Any]:
    populations: set[int] = set()
    art_eras: set[str] = set()
    civilizations: set[str] = set()
    culture_profiles = 0
    for path in sorted(assets_root.rglob("*.artdef")):
        lower = path.name.lower()
        if not (lower.startswith("citygenerators") or lower.startswith("cultures")):
            continue
        document = ET.parse(path).getroot()
        roots = document.find("m_RootCollections")
        if roots is None:
            continue
        for collection in roots.findall("Element"):
            collection_node = collection.find("m_CollectionName")
            collection_name = "" if collection_node is None else collection_node.get("text", "")
            if collection_name == "Culture":
                culture_profiles += len(collection.findall("Element"))
            for item in collection.findall("Element"):
                for value in item.findall(".//Element"):
                    parameter = value.find("m_ParamName")
                    parameter_name = "" if parameter is None else parameter.get("text", "")
                    if parameter_name == "Var_Population":
                        raw = value.find("m_nValue")
                        if raw is not None and raw.text is not None:
                            populations.add(int(raw.text))
                    element = value.find("m_ElementName")
                    target_root = value.find("m_RootCollectionName")
                    if element is None or target_root is None:
                        continue
                    target_name = element.get("text", "")
                    if target_root.get("text") == "ArtEra" and target_name:
                        art_eras.add(target_name)
                    if target_root.get("text") == "Civilization" and target_name:
                        civilizations.add(target_name)
    return {
        "growth_populations": sorted(populations),
        "art_eras": sorted(art_eras),
        "culture_profiles": culture_profiles,
        "civilizations": sorted(civilizations),
    }


def build_city_inventory(
    assets_root: Path, index: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    packages = _package_index(assets_root)
    package_bytes: dict[str, bytes] = {}
    bindings = []
    for root in contract["graph_roots"]:
        for node in _matching_nodes(index, root):
            for terminal in node["terminals"]:
                if terminal["class"] != "CityBuildings":
                    continue
                resolution = _resolve_package(
                    packages,
                    terminal["package"],
                    node["content_root"],
                    terminal["entry"],
                    package_bytes,
                )
                bindings.append(
                    {
                        "source": node["id"],
                        "collection_path": terminal["collection_path"],
                        "entry": terminal["entry"],
                        "package": terminal["package"],
                        **resolution,
                    }
                )
    bindings.sort(key=lambda item: (item["source"], item["entry"]))
    unresolved = [item for item in bindings if item["status"] != "resolved"]
    axes = _city_axis_evidence(assets_root)
    return {
        "pipeline_id": "city",
        "contract": contract,
        "root_inventory": _root_inventory(index, contract),
        "axis_evidence": axes,
        "component_bindings": bindings,
        "summary": {
            "city_building_bindings": len(bindings),
            "unique_city_building_assets": len(
                {(item.get("package_path"), item["entry"]) for item in bindings if item["status"] == "resolved"}
            ),
            "unresolved_component_bindings": len(unresolved),
            "growth_stages": len(axes["growth_populations"]),
            "art_eras": len(axes["art_eras"]),
            "culture_profiles": axes["culture_profiles"],
            "civilizations": len(axes["civilizations"]),
        },
        "unresolved_component_bindings": unresolved,
    }


def _unit_definitions(index: dict[str, Any], target: str) -> list[str]:
    return sorted(
        identifier
        for identifier in index["by_root_name"].get(("Units", target), [])
        if index["nodes"][identifier]["document"].lower().startswith("unit")
    )


def _unit_target_graph(index: dict[str, Any], target: str) -> dict[str, Any]:
    definitions = _unit_definitions(index, target)
    queue = deque(definitions)
    visited: set[str] = set()
    edges = []
    terminals = []
    while queue:
        identifier = queue.popleft()
        if identifier in visited:
            continue
        visited.add(identifier)
        node = index["nodes"][identifier]
        for terminal in node["terminals"]:
            terminals.append({"source": identifier, **terminal})
        for reference in node["references"]:
            if reference["target_root"] not in UNIT_GRAPH_ROOTS:
                continue
            resolution = resolve_reference(index, node, reference)
            edges.append({"source": identifier, **reference, **resolution})
            if resolution["status"] == "resolved":
                queue.append(resolution["target"])
    edges.sort(
        key=lambda item: (
            item["source"],
            "/".join(item["collection_path"]),
            item["parameter"],
            item["target_name"],
        )
    )
    return {
        "target": target,
        "definitions": definitions,
        "nodes": sorted(visited),
        "root_counts": dict(
            sorted(Counter(index["nodes"][identifier]["root_collection"] for identifier in visited).items())
        ),
        "edges": edges,
        "terminals": sorted(terminals, key=lambda item: (item["source"], item["entry"])),
        "unresolved_internal_edges": [edge for edge in edges if edge["status"] != "resolved"],
    }


def build_unit_inventory(
    assets_root: Path, index: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    mapping_path = PROJECT_ROOT / contract["mapping"]
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_errors = validate_unit_mapping(mapping)
    if mapping_errors:
        raise ValueError("Invalid unit mapping: " + "; ".join(mapping_errors))
    targets = sorted({item["civ6_artdef"] for item in mapping["mappings"] if item["civ6_artdef"]})
    target_graphs = [_unit_target_graph(index, target) for target in targets]
    graph_by_target = {graph["target"]: graph for graph in target_graphs}
    mappings = []
    for item in mapping["mappings"]:
        target = item["civ6_artdef"]
        mappings.append(
            {
                "civ3_id": item["civ3_id"],
                "civ3_name": item["civ3_name"],
                "target": target,
                "status": (
                    "deferred_effect"
                    if target is None
                    else "resolved"
                    if graph_by_target[target]["definitions"]
                    else "unavailable"
                ),
            }
        )
    unresolved_graphs = [
        graph for graph in target_graphs if not graph["definitions"] or graph["unresolved_internal_edges"]
    ]
    packages = _package_index(assets_root)
    package_names = {item["logical"] for item in packages}
    bin_terminals = [
        terminal
        for root in contract["graph_roots"]
        if root["root_collection"] == "UnitAttachmentBins"
        for node in _matching_nodes(index, root)
        for terminal in node["terminals"]
    ]
    missing_bin_packages = sorted(
        {
            terminal["package"]
            for terminal in bin_terminals
            if terminal["package"].removesuffix(".blp").lower() not in package_names
        }
    )
    return {
        "pipeline_id": "unit",
        "contract": contract,
        "root_inventory": _root_inventory(index, contract),
        "mappings": mappings,
        "target_graphs": target_graphs,
        "summary": {
            "mapping_records": len(mappings),
            "resolved_mapping_records": sum(item["status"] == "resolved" for item in mappings),
            "deferred_effect_records": sum(item["status"] == "deferred_effect" for item in mappings),
            "unavailable_mapping_records": sum(item["status"] == "unavailable" for item in mappings),
            "unique_art_targets": len(target_graphs),
            "visited_graph_nodes": sum(len(graph["nodes"]) for graph in target_graphs),
            "internal_edges": sum(len(graph["edges"]) for graph in target_graphs),
            "unresolved_target_graphs": len(unresolved_graphs),
            "unit_bin_terminals": len(bin_terminals),
            "missing_unit_bin_packages": len(missing_bin_packages),
        },
        "unresolved_target_graphs": [graph["target"] for graph in unresolved_graphs],
        "missing_unit_bin_packages": missing_bin_packages,
    }


def build_inventory(
    assets_root: Path, city_contract_path: Path, unit_contract_path: Path
) -> dict[str, Any]:
    city = load_contract(city_contract_path, "city")
    unit = load_contract(unit_contract_path, "unit")
    validate_separation(city, unit)
    index = index_artdefs(assets_root)
    if index["parse_errors"]:
        raise ValueError(f"Installed ArtDef parse failures: {len(index['parse_errors'])}")
    city_inventory = build_city_inventory(assets_root, index, city)
    unit_inventory = build_unit_inventory(assets_root, index, unit)
    return {
        "schema": "c3x.source_dedicated_object_pipeline_inventory.v0",
        "source": {
            "assets_root": str(assets_root),
            "artdef_documents": len(index["paths"]),
            "artdef_nodes": len(index["nodes"]),
        },
        "pipelines": {"city": city_inventory, "unit": unit_inventory},
        "separation": {
            "independent_namespaces": [city["runtime_namespace"], unit["runtime_namespace"]],
            "independent_composition_modes": [city["composition_mode"], unit["composition_mode"]],
            "shared_layer": "typed low-level model/material/animation primitives only",
            "runtime_integration": "not_enabled",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--city-contract", type=Path, default=DEFAULT_CITY_CONTRACT)
    parser.add_argument("--unit-contract", type=Path, default=DEFAULT_UNIT_CONTRACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_inventory(args.assets_root, args.city_contract, args.unit_contract)
        city = report["pipelines"]["city"]["summary"]
        unit = report["pipelines"]["unit"]["summary"]
        if args.require_closed and (
            city["unresolved_component_bindings"]
            or unit["unavailable_mapping_records"]
            or unit["unresolved_target_graphs"]
            or unit["missing_unit_bin_packages"]
        ):
            raise ValueError("Dedicated city/unit source pipeline inventory is not closed")
        _write_json(args.report, report)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        "City pipeline: "
        f"{city['unique_city_building_assets']} component assets, "
        f"{city['growth_stages']} growth thresholds, "
        f"{city['art_eras']} art eras"
    )
    print(
        "Unit pipeline: "
        f"{unit['resolved_mapping_records']} resolved mappings, "
        f"{unit['deferred_effect_records']} deferred effects, "
        f"{unit['unique_art_targets']} unique target graphs"
    )
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
