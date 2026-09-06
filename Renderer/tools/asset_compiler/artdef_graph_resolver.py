#!/usr/bin/env python3
"""Resolve resource and improvement ArtDef graphs to cooked package terminals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[2]
MAC_ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
WINDOWS_ASSETS_ROOT = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets"
)
DEFAULT_ASSETS_ROOT = next(
    (path for path in (MAC_ASSETS_ROOT, WINDOWS_ASSETS_ROOT) if path.exists()),
    MAC_ASSETS_ROOT,
)
DEFAULT_RESOURCE_MAPPING = (
    RENDERER_ROOT / "inventory" / "vanilla_conquests_to_civ6_resources.json"
)
DEFAULT_REPORT = (
    RENDERER_ROOT
    / "preview"
    / "out"
    / "artdef_graphs"
    / "resource_improvement_graphs.json"
)
TARGET_COLLECTIONS = {
    "feature": "Feature",
    "resource": "Resource",
    "improvement": "Improvement",
}
TARGET_DOCUMENT_PREFIXES = {
    "feature": "Features",
    "resource": "Resources",
    "improvement": "Improvements",
}
VISUAL_ROOTS = {
    "ClutterSets",
    "Landmarks",
    "Farms",
    "PlotSets",
    "TileSets",
    "GreatWall",
    "Improvement",
    "Resource",
}
SPECIALIZED_VISUAL_ROOTS = {"Farms", "PlotSets", "TileSets", "GreatWall"}
CONDITION_ROOTS = {
    "BuildStates",
    "Civilizations",
    "Districts",
    "Feature",
    "ResourceTags",
    "Terrain",
    "TerrainTags",
}
CONDITION_PARAMETERS = {
    "BuildState",
    "Civilization",
    "Destination",
    "Feature",
    "Improvement",
    "Origin",
    "Resource",
    "State",
    "Terrain",
    "Type",
}
IGNORED_CONTEXTS = {"Audio", "StrategicView"}
IMPLICIT_XREFS = {
    "Clutter": ("Clutter.artdef", "ClutterSets"),
    "ClutterVariants": ("Clutter.artdef", "ClutterSets"),
    "Landmark": ("Landmarks.artdef", "Landmarks"),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attribute(element: ET.Element | None, name: str = "text") -> str:
    return "" if element is None else element.attrib.get(name, "")


def _content_root(relative: Path) -> str:
    parts = relative.parts
    if "ArtDefs" not in parts:
        raise ValueError(f"ArtDef path has no ArtDefs component: {relative}")
    return Path(*parts[: parts.index("ArtDefs")]).as_posix()


def _node_id(relative: Path, collection: str, name: str) -> str:
    return f"{relative.as_posix()}#{collection}/{name}"


def _direct_parameters(item: ET.Element) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for value in item.findall("./m_Fields/m_Values/Element"):
        parameter = _attribute(value.find("m_ParamName"))
        if not parameter:
            continue
        if parameter in result:
            raise ValueError(f"ArtDef item contains duplicate parameter {parameter}")
        result[parameter] = value
    return result


def _scalar(value: ET.Element | None) -> str:
    if value is None:
        return ""
    for child in value:
        if child.tag == "m_ParamName":
            continue
        return child.attrib.get("text", child.text or "")
    return ""


def _walk_item(
    item: ET.Element,
    collection_path: tuple[str, ...],
    references: list[dict[str, Any]],
    terminals: list[dict[str, Any]],
) -> None:
    parameters = _direct_parameters(item)
    xref_name = _scalar(parameters.get("XrefName"))
    saw_xref = False
    for parameter, value in parameters.items():
        entry = _attribute(value.find("m_EntryName"))
        package = _attribute(value.find("m_BLPPackage"))
        # Blank BLP entries are intentional inherited/disabled asset slots in the
        # shipped ArtDefs.  Only a complete pair is a concrete package terminal.
        if entry and package:
            terminals.append(
                {
                    "collection_path": list(collection_path),
                    "parameter": parameter,
                    "entry": entry,
                    "package": package,
                    "class": _attribute(value.find("m_XLPClass")),
                    "library": _attribute(value.find("m_LibraryName")),
                }
            )
        element_node = value.find("m_ElementName")
        if element_node is None:
            continue
        target_name = _attribute(element_node)
        target_root = _attribute(value.find("m_RootCollectionName"))
        target_path = _attribute(value.find("m_ArtDefPath"))
        if parameter == "Xref":
            saw_xref = True
            target_name = target_name or xref_name
            if collection_path and collection_path[-1] in IMPLICIT_XREFS:
                inferred_path, inferred_root = IMPLICIT_XREFS[collection_path[-1]]
                target_path = target_path or inferred_path
                target_root = target_root or inferred_root
        if target_name and target_name != "NONE":
            references.append(
                {
                    "collection_path": list(collection_path),
                    "parameter": parameter,
                    "target_name": target_name,
                    "target_root": target_root,
                    "target_path": target_path,
                    "template": _attribute(value.find("m_TemplateName")),
                }
            )
    if xref_name and not saw_xref and collection_path and collection_path[-1] in IMPLICIT_XREFS:
        target_path, target_root = IMPLICIT_XREFS[collection_path[-1]]
        references.append(
            {
                "collection_path": list(collection_path),
                "parameter": "XrefName",
                "target_name": xref_name,
                "target_root": target_root,
                "target_path": target_path,
                "template": "",
            }
        )
    children = item.find("m_ChildCollections")
    for collection in [] if children is None else children.findall("Element"):
        name = _attribute(collection.find("m_CollectionName"))
        if not name:
            raise ValueError("Nested ArtDef collection has no name")
        for child in collection.findall("Element"):
            _walk_item(child, collection_path + (name,), references, terminals)


def index_artdefs(assets_root: Path) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    by_root_name: dict[tuple[str, str], list[str]] = defaultdict(list)
    documents: dict[str, list[str]] = defaultdict(list)
    parse_errors = []
    paths = sorted(assets_root.rglob("*.artdef"))
    for path in paths:
        relative = path.relative_to(assets_root)
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            parse_errors.append({"path": relative.as_posix(), "error": str(exc)})
            continue
        roots = root.find("m_RootCollections")
        if roots is None:
            continue
        documents[path.name.lower()].append(relative.as_posix())
        for collection in roots.findall("Element"):
            collection_name = _attribute(collection.find("m_CollectionName"))
            if not collection_name:
                raise ValueError(f"Unnamed root collection in {relative}")
            for item in collection.findall("Element"):
                name = _attribute(item.find("m_Name"))
                if not name:
                    raise ValueError(f"Unnamed {collection_name} item in {relative}")
                identifier = _node_id(relative, collection_name, name)
                if identifier in nodes:
                    raise ValueError(f"Duplicate ArtDef node {identifier}")
                references: list[dict[str, Any]] = []
                terminals: list[dict[str, Any]] = []
                try:
                    _walk_item(item, (), references, terminals)
                except ValueError as exc:
                    raise ValueError(
                        f"{relative}#{collection_name}/{name}: {exc}"
                    ) from exc
                node = {
                    "id": identifier,
                    "path": relative.as_posix(),
                    "document": path.name,
                    "content_root": _content_root(relative),
                    "root_collection": collection_name,
                    "name": name,
                    "references": references,
                    "terminals": terminals,
                }
                nodes[identifier] = node
                by_root_name[(collection_name, name)].append(identifier)
    incoming: dict[tuple[str, str], list[str]] = defaultdict(list)
    for identifier, node in nodes.items():
        for reference in node["references"]:
            if reference["target_root"] and reference["target_name"]:
                incoming[(reference["target_root"], reference["target_name"])].append(identifier)
    return {
        "nodes": nodes,
        "by_root_name": dict(by_root_name),
        "incoming": dict(incoming),
        "documents": dict(documents),
        "paths": paths,
        "parse_errors": parse_errors,
    }


def _edge_scope(reference: dict[str, Any]) -> str:
    if any(part in IGNORED_CONTEXTS for part in reference["collection_path"]):
        return "retained_non_map"
    if (
        reference["target_root"] in CONDITION_ROOTS
        or reference["parameter"] in CONDITION_PARAMETERS
    ):
        return "selector_condition"
    if reference["target_root"] in VISUAL_ROOTS or reference["parameter"].lower().startswith("xref"):
        return "visual_dependency"
    return "metadata_dependency"


def resolve_reference(
    index: dict[str, Any], source: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    target_name = reference["target_name"]
    target_root = reference["target_root"]
    if target_root:
        candidates = list(index["by_root_name"].get((target_root, target_name), []))
    else:
        candidates = [
            identifier
            for (root, name), identifiers in index["by_root_name"].items()
            if name == target_name
            for identifier in identifiers
        ]
    target_path = reference["target_path"]
    if target_path:
        basename = Path(target_path).name.lower()
        candidates = [
            identifier
            for identifier in candidates
            if index["nodes"][identifier]["document"].lower() == basename
        ]
    if not candidates:
        return {"status": "unresolved", "candidates": []}
    expected_local = (
        Path(source["path"]).parent / target_path
        if target_path
        else Path(source["path"])
    ).as_posix().lower()
    expected_base = (
        Path("Base/ArtDefs") / Path(target_path).name
        if target_path
        else Path("Base/ArtDefs") / source["document"]
    ).as_posix().lower()

    def score(identifier: str) -> int:
        path = index["nodes"][identifier]["path"].lower()
        if path == expected_local:
            return 0
        if path == expected_base:
            return 1
        return 2

    best = min(score(identifier) for identifier in candidates)
    winners = sorted(identifier for identifier in candidates if score(identifier) == best)
    if len(winners) != 1:
        return {"status": "ambiguous", "candidates": winners}
    return {"status": "resolved", "target": winners[0], "candidates": winners}


def _target_definitions(index: dict[str, Any], kind: str, name: str) -> list[str]:
    root = TARGET_COLLECTIONS[kind]
    prefix = TARGET_DOCUMENT_PREFIXES[kind].lower()
    return sorted(
        identifier
        for identifier in index["by_root_name"].get((root, name), [])
        if index["nodes"][identifier]["document"].lower().startswith(prefix)
    )


def resolve_target_graph(
    index: dict[str, Any], kind: str, name: str, graph_id: str
) -> dict[str, Any]:
    definitions = _target_definitions(index, kind, name)
    if not definitions:
        raise ValueError(f"No installed {kind} ArtDef definition for {name}")
    associated = []
    if kind == "improvement":
        for identifier in index["incoming"].get(("Improvement", name), []):
            node = index["nodes"][identifier]
            if node["root_collection"] in SPECIALIZED_VISUAL_ROOTS:
                associated.append(identifier)
    queue = deque(sorted(set(definitions + associated)))
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
            context = terminal["collection_path"]
            terminals.append(
                {
                    **terminal,
                    "source": identifier,
                    "scope": (
                        "retained_non_map"
                        if any(part in IGNORED_CONTEXTS for part in context)
                        else "map_visual"
                    ),
                }
            )
        for reference in node["references"]:
            scope = _edge_scope(reference)
            resolution = resolve_reference(index, node, reference)
            edge = {**reference, "source": identifier, "scope": scope, **resolution}
            edges.append(edge)
            if (
                resolution["status"] == "resolved"
                and scope == "visual_dependency"
                and index["nodes"][resolution["target"]]["root_collection"] in VISUAL_ROOTS
            ):
                queue.append(resolution["target"])
    nodes = [
        {
            key: index["nodes"][identifier][key]
            for key in ("id", "path", "document", "content_root", "root_collection", "name")
        }
        for identifier in sorted(visited)
    ]
    edges.sort(
        key=lambda item: (
            item["source"],
            "/".join(item["collection_path"]),
            item["parameter"],
            item["target_name"],
        )
    )
    terminals.sort(key=lambda item: (item["source"], item["package"], item["entry"]))
    return {
        "graph_id": graph_id,
        "target_kind": kind,
        "target_name": name,
        "definitions": definitions,
        "associated_visual_roots": sorted(set(associated)),
        "nodes": nodes,
        "edges": edges,
        "terminals": terminals,
    }


def _package_index(assets_root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(assets_root.rglob("*.blp")):
        relative = path.relative_to(assets_root)
        parts = relative.parts
        lowered = [part.lower() for part in parts]
        if "blps" not in lowered or "platforms" not in lowered:
            continue
        blp_index = lowered.index("blps")
        platforms_index = lowered.index("platforms")
        logical = Path(*parts[blp_index + 1 :]).with_suffix("").as_posix().lower()
        content_root = Path(*parts[:platforms_index]).as_posix()
        result.append(
            {
                "path": path,
                "relative": relative.as_posix(),
                "logical": logical,
                "content_root": content_root,
            }
        )
    return result


def _resolve_package(
    packages: list[dict[str, Any]],
    logical: str,
    source_content_root: str,
    entry: str,
    package_bytes: dict[str, bytes],
) -> dict[str, Any]:
    normalized = logical.removesuffix(".blp").lower()
    candidates = [item for item in packages if item["logical"] == normalized]
    if not candidates:
        return {"status": "missing_package", "candidates": []}
    local = [item for item in candidates if item["content_root"] == source_content_root]
    base = [item for item in candidates if item["content_root"] == "Base"]
    encoded = entry.encode("ascii") + b"\0"
    tried: set[str] = set()
    for priority, group in (("same_content", local), ("base_fallback", base), ("other_content", candidates)):
        untried = [item for item in group if item["relative"] not in tried]
        tried.update(item["relative"] for item in untried)
        matching = []
        for item in untried:
            relative = item["relative"]
            if relative not in package_bytes:
                package_bytes[relative] = item["path"].read_bytes()
            count = package_bytes[relative].count(encoded)
            if count:
                matching.append((item, count))
        if len(matching) == 1:
            item, count = matching[0]
            return {
                "status": "resolved",
                "package_path": item["relative"],
                "package_resolution": priority,
                "entry_occurrences": count,
                "entry_name_is_unique": count == 1,
            }
        if len(matching) > 1:
            return {
                "status": "ambiguous_package",
                "package_resolution": priority,
                "candidates": [item["relative"] for item, _count in matching],
            }
    return {
        "status": "missing_entry",
        "candidates": [item["relative"] for item in candidates],
    }


def resolve_terminal_packages(
    graphs: list[dict[str, Any]], index: dict[str, Any], assets_root: Path
) -> None:
    packages = _package_index(assets_root)
    package_bytes: dict[str, bytes] = {}
    for graph in graphs:
        for terminal in graph["terminals"]:
            source = index["nodes"][terminal["source"]]
            resolution = _resolve_package(
                packages,
                terminal["package"],
                source["content_root"],
                terminal["entry"],
                package_bytes,
            )
            terminal.update(resolution)


def _improvement_targets(index: dict[str, Any]) -> list[str]:
    base_path = "Base/ArtDefs/Improvements.artdef"
    return sorted(
        node["name"]
        for node in index["nodes"].values()
        if node["path"] == base_path and node["root_collection"] == "Improvement"
    )


def build_resource_improvement_graphs(
    assets_root: Path, resource_mapping: Path
) -> dict[str, Any]:
    index = index_artdefs(assets_root)
    if index["parse_errors"]:
        raise ValueError(f"Installed ArtDef parse failures: {len(index['parse_errors'])}")
    mapping = json.loads(resource_mapping.read_text(encoding="utf-8"))
    if mapping.get("schema") != "c3x.civ3_to_civ6_resource_mapping.v0":
        raise ValueError("Unsupported resource mapping schema for graph resolution")
    graphs = []
    for item in mapping["mappings"]:
        graphs.append(
            resolve_target_graph(
                index,
                item["target_kind"],
                item["civ6_artdef"],
                "resource/" + item["civ3_id"].removeprefix("GOOD_").lower(),
            )
        )
    for name in _improvement_targets(index):
        graphs.append(
            resolve_target_graph(
                index,
                "improvement",
                name,
                "improvement/" + name.removeprefix("IMPROVEMENT_").lower(),
            )
        )
    resolve_terminal_packages(graphs, index, assets_root)
    unresolved_visual_edges = [
        (graph["graph_id"], edge)
        for graph in graphs
        for edge in graph["edges"]
        if edge["scope"] == "visual_dependency" and edge["status"] != "resolved"
    ]
    unresolved_visual_terminals = [
        (graph["graph_id"], terminal)
        for graph in graphs
        for terminal in graph["terminals"]
        if terminal["scope"] == "map_visual" and terminal["status"] != "resolved"
    ]
    return {
        "schema": "c3x.source_artdef_graph_resolution.v0",
        "source": {
            "assets_root": str(assets_root),
            "resource_mapping": str(resource_mapping),
            "resource_mapping_sha256": _sha256(resource_mapping.read_bytes()),
            "artdef_documents": len(index["paths"]),
            "artdef_nodes": len(index["nodes"]),
        },
        "graphs": graphs,
        "summary": {
            "resource_graphs": sum(graph["target_kind"] in ("resource", "feature") for graph in graphs),
            "improvement_graphs": sum(graph["target_kind"] == "improvement" for graph in graphs),
            "nodes": sum(len(graph["nodes"]) for graph in graphs),
            "edges": sum(len(graph["edges"]) for graph in graphs),
            "visual_terminals": sum(
                terminal["scope"] == "map_visual"
                for graph in graphs
                for terminal in graph["terminals"]
            ),
            "unique_visual_assets": len(
                {
                    (terminal.get("package_path"), terminal["entry"])
                    for graph in graphs
                    for terminal in graph["terminals"]
                    if terminal["scope"] == "map_visual" and terminal["status"] == "resolved"
                }
            ),
            "unresolved_visual_edges": len(unresolved_visual_edges),
            "unresolved_visual_terminals": len(unresolved_visual_terminals),
        },
        "unresolved_visual_edges": [
            {"graph_id": graph_id, **edge} for graph_id, edge in unresolved_visual_edges
        ],
        "unresolved_visual_terminals": [
            {"graph_id": graph_id, **terminal}
            for graph_id, terminal in unresolved_visual_terminals
        ],
        "runtime_integration": "not_enabled",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--resource-mapping", type=Path, default=DEFAULT_RESOURCE_MAPPING)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_resource_improvement_graphs(args.assets_root, args.resource_mapping)
        if args.require_closed and (
            report["summary"]["unresolved_visual_edges"]
            or report["summary"]["unresolved_visual_terminals"]
        ):
            raise ValueError("Resource/improvement ArtDef graph has unresolved visual dependencies")
        _write_json(args.report, report)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = report["summary"]
    print(
        f"Resolved {summary['resource_graphs']} resource and "
        f"{summary['improvement_graphs']} improvement graphs: "
        f"{summary['unique_visual_assets']} unique visual assets"
    )
    print(
        f"Unresolved visual edges: {summary['unresolved_visual_edges']}; "
        f"terminals: {summary['unresolved_visual_terminals']}"
    )
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
