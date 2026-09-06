"""Validate the editable vanilla Civ III to Civ VI unit-art seed map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


SCHEMA = "c3x.civ3_to_civ6_unit_mapping.v0"
DEFAULT_MAPPING = Path(__file__).with_name("vanilla_conquests_to_civ6_units.json")
MATCH_KINDS = {"direct", "close", "stand_in", "deferred_effect"}
CONFIDENCE_LEVELS = {"high", "medium", "low", "none"}


def load_mapping(path: Path = DEFAULT_MAPPING) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_mapping(document: dict) -> list[str]:
    errors: list[str] = []
    if document.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")

    mappings = document.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        return errors + ["mappings must be a non-empty list"]

    expected_count = document.get("source_roster", {}).get("unit_count")
    if expected_count != len(mappings):
        errors.append(
            f"source_roster.unit_count is {expected_count!r}, but {len(mappings)} mappings exist"
        )

    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    required = {
        "civ3_id",
        "civ3_name",
        "role",
        "civ6_artdef",
        "match",
        "confidence",
        "fallback",
        "reason",
    }
    for index, entry in enumerate(mappings):
        label = f"mappings[{index}]"
        missing = sorted(required - set(entry))
        if missing:
            errors.append(f"{label} is missing fields: {', '.join(missing)}")
            continue

        civ3_id = entry["civ3_id"]
        civ3_name = entry["civ3_name"]
        if not isinstance(civ3_id, str) or not civ3_id.startswith("PRTO_"):
            errors.append(f"{label}.civ3_id must start with PRTO_")
        elif civ3_id in seen_ids:
            errors.append(f"duplicate Civ III ID: {civ3_id}")
        else:
            seen_ids.add(civ3_id)

        if not isinstance(civ3_name, str) or not civ3_name:
            errors.append(f"{label}.civ3_name must be non-empty")
        elif civ3_name in seen_names:
            errors.append(f"duplicate Civ III name: {civ3_name}")
        else:
            seen_names.add(civ3_name)

        match = entry["match"]
        if match not in MATCH_KINDS:
            errors.append(f"{label}.match has unknown value {match!r}")
        if entry["confidence"] not in CONFIDENCE_LEVELS:
            errors.append(f"{label}.confidence has unknown value {entry['confidence']!r}")

        target = entry["civ6_artdef"]
        if match == "deferred_effect":
            if target is not None:
                errors.append(f"{label}.civ6_artdef must be null for deferred effects")
            if entry["fallback"] != "vanilla":
                errors.append(f"{label}.fallback must be vanilla for deferred effects")
        elif not isinstance(target, str) or not target.startswith("UNIT_"):
            errors.append(f"{label}.civ6_artdef must be a UNIT_ ArtDef identifier")

        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            errors.append(f"{label}.reason must be non-empty")

    return errors


def discover_civ6_unit_artdefs(assets_root: Path) -> tuple[dict[str, list[str]], list[str]]:
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
            if name.startswith("UNIT_"):
                relative = path.relative_to(assets_root).as_posix()
                discovered.setdefault(name, set()).add(relative)
    return (
        {name: sorted(paths) for name, paths in sorted(discovered.items())},
        parse_errors,
    )


def build_resolution_report(document: dict, assets_root: Path) -> dict:
    discovered, parse_errors = discover_civ6_unit_artdefs(assets_root)
    resolved: list[dict] = []
    unavailable: list[dict] = []
    deferred: list[dict] = []
    for entry in document["mappings"]:
        target = entry["civ6_artdef"]
        record = {
            "civ3_id": entry["civ3_id"],
            "civ3_name": entry["civ3_name"],
            "civ6_artdef": target,
        }
        if target is None:
            deferred.append(record)
        elif target in discovered:
            record["artdef_paths"] = discovered[target]
            resolved.append(record)
        else:
            unavailable.append(record)

    return {
        "schema": "c3x.civ3_to_civ6_unit_resolution.v0",
        "mapping_schema": document["schema"],
        "discovered_unit_artdefs": len(discovered),
        "resolved": resolved,
        "unavailable": unavailable,
        "deferred": deferred,
        "parse_errors": parse_errors,
        "summary": {
            "mapping_count": len(document["mappings"]),
            "resolved_count": len(resolved),
            "unavailable_count": len(unavailable),
            "deferred_count": len(deferred),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--civ6-assets-root", type=Path)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--require-all-targets", action="store_true")
    args = parser.parse_args(argv)

    document = load_mapping(args.mapping)
    errors = validate_mapping(document)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"Valid unit mapping seed: {len(document['mappings'])} Civ III units")
    if args.civ6_assets_root is None:
        return 0

    report = build_resolution_report(document, args.civ6_assets_root)
    summary = report["summary"]
    print(
        "Civ VI ArtDefs: "
        f"{summary['resolved_count']} resolved, "
        f"{summary['unavailable_count']} unavailable, "
        f"{summary['deferred_count']} intentionally deferred"
    )
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.require_all_targets and report["unavailable"]:
        for item in report["unavailable"]:
            print(
                f"UNAVAILABLE: {item['civ3_id']} -> {item['civ6_artdef']}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
