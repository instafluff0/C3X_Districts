#!/usr/bin/env python3
"""Inventory every installed Civ VI BUILDING_PALACE map-asset binding."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Renderer.tools.asset_compiler.artdef_graph_resolver import (
    DEFAULT_ASSETS_ROOT,
    _package_index,
    _resolve_package,
)


DEFAULT_REPORT = (
    REPOSITORY_ROOT
    / "Renderer"
    / "preview"
    / "out"
    / "cities"
    / "palace_asset_probe.json"
)


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return element.attrib.get("text", (element.text or "").strip())


def _parameters(element: ET.Element) -> dict[str, ET.Element]:
    result = {}
    for value in element.findall("./m_Fields/m_Values/Element"):
        name = _text(value.find("m_ParamName"))
        if name:
            result[name] = value
    return result


def _reference(value: ET.Element | None) -> tuple[str, str]:
    if value is None:
        return "", ""
    return _text(value.find("m_RootCollectionName")), _text(value.find("m_ElementName"))


def _content_root(relative: Path) -> str:
    parts = relative.parts
    return Path(*parts[: parts.index("ArtDefs")]).as_posix()


def scan(assets_root: Path) -> dict[str, Any]:
    package_index = _package_index(assets_root)
    package_bytes: dict[str, bytes] = {}
    bindings = []
    parse_errors = []
    for path in sorted(assets_root.rglob("*.artdef")):
        relative = path.relative_to(assets_root)
        if path.name.lower() != "landmarks.artdef":
            continue
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            parse_errors.append({"artdef": relative.as_posix(), "error": str(exc)})
            continue
        for candidate in root.findall(".//Element"):
            parameters = _parameters(candidate)
            hero_root, hero_name = _reference(parameters.get("Tag_HeroBuilding"))
            if hero_root != "Building" or hero_name != "BUILDING_PALACE":
                continue
            asset = parameters.get("Asset")
            if asset is None:
                continue
            entry = _text(asset.find("m_EntryName"))
            package = _text(asset.find("m_BLPPackage"))
            if not entry or not package:
                continue
            culture_root, culture = _reference(parameters.get("Tag_Culture"))
            era_root, era = _reference(parameters.get("Tag_Era"))
            resolution = _resolve_package(
                package_index,
                package,
                _content_root(relative),
                entry,
                package_bytes,
            )
            scenario_only = any("scenario" in part.lower() for part in relative.parts)
            bindings.append(
                {
                    "artdef": relative.as_posix(),
                    "binding_name": _text(candidate.find("m_Name")),
                    "selector": {
                        "culture_root": culture_root,
                        "culture": culture,
                        "era_root": era_root,
                        "era": era,
                    },
                    "source_entry": entry,
                    "logical_package": package,
                    "source_content": _content_root(relative),
                    "scenario_only": scenario_only,
                    **resolution,
                }
            )

    bindings.sort(
        key=lambda item: (
            item["scenario_only"],
            item["selector"]["culture"],
            item["source_entry"],
            item["artdef"],
        )
    )
    unresolved = [item for item in bindings if item.get("status") != "resolved"]
    accepted_by_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        if not binding["scenario_only"] and binding.get("status") == "resolved":
            accepted_by_source[(binding["package_path"], binding["source_entry"])].append(binding)
    accepted = []
    for (package_path, source_entry), rows in sorted(accepted_by_source.items()):
        selectors = sorted(
            {
                (
                    row["selector"]["culture_root"],
                    row["selector"]["culture"],
                    row["selector"]["era_root"],
                    row["selector"]["era"],
                )
                for row in rows
            }
        )
        accepted.append(
            {
                "source_package": package_path,
                "source_entry": source_entry,
                "selectors": [
                    {
                        "culture_root": selector[0],
                        "culture": selector[1],
                        "era_root": selector[2],
                        "era": selector[3],
                    }
                    for selector in selectors
                ],
                "artdefs": sorted({row["artdef"] for row in rows}),
            }
        )
    return {
        "schema": "c3x.source_palace_asset_probe.v0",
        "status": "passed" if not parse_errors and not unresolved else "incomplete",
        "summary": {
            "bindings": len(bindings),
            "standard_bindings": sum(not item["scenario_only"] for item in bindings),
            "scenario_bindings": sum(item["scenario_only"] for item in bindings),
            "unique_standard_assets": len(accepted),
            "unresolved_bindings": len(unresolved),
            "parse_errors": len(parse_errors),
        },
        "accepted_standard_assets": accepted,
        "all_bindings": bindings,
        "unresolved_bindings": unresolved,
        "parse_errors": parse_errors,
        "runtime_integration": "not_enabled",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        report = scan(args.assets_root)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Report: {args.report}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
