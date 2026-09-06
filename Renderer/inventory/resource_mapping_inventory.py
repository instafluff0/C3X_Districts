"""Validate the editable vanilla Civ III to Civ VI map-resource seed map."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


SCHEMA = "c3x.civ3_to_civ6_resource_mapping.v0"
DEFAULT_MAPPING = Path(__file__).with_name("vanilla_conquests_to_civ6_resources.json")
DEFAULT_BIQ_SEMANTICS = Path(__file__).with_name("vanilla_conquests_biq_semantics.json")
MATCH_KINDS = {"direct", "close", "stand_in"}
CONFIDENCE_LEVELS = {"high", "medium", "low", "none"}
RESOURCE_CLASSES = {0: "bonus", 1: "luxury", 2: "strategic"}
TARGET_PREFIXES = {"resource": "RESOURCE_", "feature": "FEATURE_"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_mapping(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    policy = document.get("policy", {})
    if "map resource body" not in policy.get("replacement_ownership", ""):
        errors.append("policy replacement_ownership must be limited to the map resource body")
    if "Civilopedia" not in policy.get("retained_ownership", ""):
        errors.append("policy retained_ownership must preserve native non-map resource icons")

    mappings = document.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        return errors + ["mappings must be a non-empty list"]
    expected_count = document.get("source_roster", {}).get("resource_count")
    if expected_count != len(mappings):
        errors.append(
            f"source_roster.resource_count is {expected_count!r}, but {len(mappings)} mappings exist"
        )

    required = {
        "civ3_biq_index", "civ3_id", "civ3_name", "resource_class",
        "civ3_resource_class", "civ3_icon_index", "civ6_artdef", "target_kind",
        "match", "confidence", "alternatives", "fallback", "reason",
    }
    seen_ids: set[str] = set()
    seen_biq_indices: set[int] = set()
    for index, entry in enumerate(mappings):
        label = f"mappings[{index}]"
        missing = sorted(required - set(entry))
        if missing:
            errors.append(f"{label} is missing fields: {', '.join(missing)}")
            continue
        source_id = entry["civ3_id"]
        if not isinstance(source_id, str) or not source_id.startswith("GOOD_"):
            errors.append(f"{label}.civ3_id must start with GOOD_")
        elif source_id in seen_ids:
            errors.append(f"duplicate Civ III ID: {source_id}")
        else:
            seen_ids.add(source_id)
        biq_index = entry["civ3_biq_index"]
        if not isinstance(biq_index, int) or biq_index < 0:
            errors.append(f"{label}.civ3_biq_index must be a nonnegative integer")
        elif biq_index in seen_biq_indices:
            errors.append(f"duplicate Civ III BIQ index: {biq_index}")
        else:
            seen_biq_indices.add(biq_index)
        source_class = entry["civ3_resource_class"]
        if RESOURCE_CLASSES.get(source_class) != entry["resource_class"]:
            errors.append(f"{label} has inconsistent numeric and named resource classes")
        target_kind = entry["target_kind"]
        prefix = TARGET_PREFIXES.get(target_kind)
        if prefix is None:
            errors.append(f"{label}.target_kind has unknown value {target_kind!r}")
        elif not isinstance(entry["civ6_artdef"], str) or not entry["civ6_artdef"].startswith(prefix):
            errors.append(f"{label}.civ6_artdef must start with {prefix!r}")
        if entry["match"] not in MATCH_KINDS:
            errors.append(f"{label}.match has unknown value {entry['match']!r}")
        if entry["confidence"] not in CONFIDENCE_LEVELS:
            errors.append(f"{label}.confidence has unknown value {entry['confidence']!r}")
        if entry["fallback"] != "vanilla_map_resource":
            errors.append(f"{label}.fallback must be 'vanilla_map_resource'")
        if not isinstance(entry["alternatives"], list):
            errors.append(f"{label}.alternatives must be a list")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            errors.append(f"{label}.reason must be non-empty")
    return errors


def validate_against_biq(document: dict[str, Any], semantics: dict[str, Any]) -> list[str]:
    expected = {
        (
            item["biq_index"], item["civilopedia_entry"], item["name"],
            item["resource_class"], item["icon_index"],
        )
        for item in semantics.get("resources", [])
    }
    actual = {
        (
            item["civ3_biq_index"], item["civ3_id"], item["civ3_name"],
            item["civ3_resource_class"], item["civ3_icon_index"],
        )
        for item in document.get("mappings", [])
    }
    errors: list[str] = []
    for missing in sorted(expected - actual):
        errors.append(f"missing or mismatched BIQ resource: {missing}")
    for extra in sorted(actual - expected):
        errors.append(f"mapping absent from BIQ semantics: {extra}")
    return errors


def discover_civ6_targets(assets_root: Path) -> tuple[dict[str, list[str]], list[str]]:
    discovered: dict[str, set[str]] = {}
    parse_errors: list[str] = []
    for path in sorted(assets_root.rglob("*.artdef")):
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as exc:
            parse_errors.append(f"{path}: {exc}")
            continue
        for element in root.iter("m_Name"):
            name = element.get("text", "")
            if name.startswith(("RESOURCE_", "FEATURE_")):
                relative = path.relative_to(assets_root).as_posix()
                discovered.setdefault(name, set()).add(relative)
    return {name: sorted(paths) for name, paths in sorted(discovered.items())}, parse_errors


def build_resolution_report(document: dict[str, Any], assets_root: Path) -> dict[str, Any]:
    discovered, parse_errors = discover_civ6_targets(assets_root)
    resolved: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for entry in document["mappings"]:
        record = {
            "civ3_id": entry["civ3_id"],
            "civ3_name": entry["civ3_name"],
            "civ6_artdef": entry["civ6_artdef"],
            "match": entry["match"],
            "confidence": entry["confidence"],
        }
        paths = discovered.get(entry["civ6_artdef"])
        if paths:
            record["artdef_paths"] = paths
            resolved.append(record)
        else:
            unavailable.append(record)
    return {
        "schema": "c3x.civ3_to_civ6_resource_resolution.v0",
        "mapping_schema": document["schema"],
        "discovered_resource_and_feature_artdefs": len(discovered),
        "resolved": resolved,
        "unavailable": unavailable,
        "parse_errors": parse_errors,
        "summary": {
            "mapping_count": len(document["mappings"]),
            "resolved_count": len(resolved),
            "unavailable_count": len(unavailable),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--biq-semantics", type=Path, default=DEFAULT_BIQ_SEMANTICS)
    parser.add_argument("--civ6-assets-root", type=Path)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--require-all-targets", action="store_true")
    args = parser.parse_args(argv)

    document = load_json(args.mapping)
    errors = validate_mapping(document)
    errors.extend(validate_against_biq(document, load_json(args.biq_semantics)))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Valid map-resource mapping seed: {len(document['mappings'])} Civ III resources")
    if args.civ6_assets_root is None:
        return 0

    report = build_resolution_report(document, args.civ6_assets_root)
    summary = report["summary"]
    print(
        f"Civ VI ArtDefs: {summary['resolved_count']} resolved, "
        f"{summary['unavailable_count']} unavailable"
    )
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.require_all_targets and report["unavailable"]:
        for item in report["unavailable"]:
            print(f"UNAVAILABLE: {item['civ3_id']} -> {item['civ6_artdef']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
