"""Validate and resolve the editable C3X-to-Civ-VI natural-wonder seed map.

Sources come from the installed base game, a named Steam Workshop item, or an
authored Lab kit; all of them are conversion input only."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


SCHEMA = "c3x.c3x_to_civ6_natural_wonder_mapping.v1"
DEFAULT_MAPPING = Path(__file__).with_name("vanilla_c3x_to_civ6_natural_wonders.json")
DEFAULT_CONFIG = Path(__file__).parents[2] / "default.districts_natural_wonders_config.txt"
MAPPING_STATUSES = {"exact", "approximate", "authored", "authored_required"}
SOURCE_KINDS = {"base", "workshop", "authored"}
CONFIDENCE_LEVELS = {"high", "medium", "low", "none"}
INTEGER_FIELDS = {
    "img_row", "img_column", "culture_bonus", "science_bonus", "food_bonus",
    "gold_bonus", "shield_bonus", "happiness_bonus", "impassable",
    "impassable_to_wheeled",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_natural_wonder_config(path: Path) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line == "#Wonder":
            current = {"animations": []}
            definitions.append(current)
            continue
        if current is None or not line or line.startswith((";", "[")) or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key == "animation":
            current["animations"].append(value)
        elif key in INTEGER_FIELDS:
            current[key] = int(value)
        else:
            current[key] = value
    return definitions


def validate_mapping(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    policy = document.get("policy", {})
    if "natural-wonder map body" not in policy.get("replacement_ownership", ""):
        errors.append("policy replacement_ownership must be limited to the natural-wonder map body")
    if "fog and shroud" not in policy.get("retained_ownership", ""):
        errors.append("policy retained_ownership must preserve native fog and shroud")
    if "under the fog" not in policy.get("label", ""):
        errors.append("policy label must keep the name label under the fog and unexplored overlay")

    mappings = document.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        return errors + ["mappings must be a non-empty list"]
    expected_count = document.get("source_roster", {}).get("definition_count")
    if expected_count != len(mappings):
        errors.append(
            f"source_roster.definition_count is {expected_count!r}, but {len(mappings)} mappings exist"
        )

    required = {
        "c3x_default_index", "c3x_key", "c3x_name", "terrain_type", "adjacent_to",
        "adjacency_dir", "native_sprite", "configured_animation_count", "civ6_artdef",
        "mapping_status", "confidence", "alternatives", "source", "kit_parts",
        "season_parts", "orientation_basis", "fallback", "reason",
    }
    seen_names: set[str] = set()
    seen_keys: set[str] = set()
    seen_indices: set[int] = set()
    seen_cells: set[tuple[str, int, int]] = set()
    for index, entry in enumerate(mappings):
        label = f"mappings[{index}]"
        missing = sorted(required - set(entry))
        if missing:
            errors.append(f"{label} is missing fields: {', '.join(missing)}")
            continue
        for field, seen in (
            ("c3x_name", seen_names), ("c3x_key", seen_keys),
            ("c3x_default_index", seen_indices),
        ):
            value = entry[field]
            if value in seen:
                errors.append(f"duplicate {field}: {value}")
            seen.add(value)
        if entry["c3x_default_index"] != index:
            errors.append(f"{label}.c3x_default_index must preserve default config order")
        sprite = entry["native_sprite"]
        if not isinstance(sprite, dict) or set(sprite) != {"img_path", "row", "column"}:
            errors.append(f"{label}.native_sprite must contain only img_path, row, and column")
        else:
            cell = (sprite["img_path"], sprite["row"], sprite["column"])
            if cell in seen_cells:
                errors.append(f"duplicate native sprite cell: {cell}")
            seen_cells.add(cell)
        status = entry["mapping_status"]
        if status not in MAPPING_STATUSES:
            errors.append(f"{label}.mapping_status has unknown value {status!r}")
        confidence = entry["confidence"]
        if confidence not in CONFIDENCE_LEVELS:
            errors.append(f"{label}.confidence has unknown value {confidence!r}")
        target = entry["civ6_artdef"]
        source = entry["source"]
        kind = source.get("kind") if isinstance(source, dict) else None
        if kind not in SOURCE_KINDS:
            errors.append(f"{label}.source.kind must be one of {sorted(SOURCE_KINDS)}")
            continue
        features = source.get("features")
        if not isinstance(features, list) or any(
            not isinstance(item, str) or not item.startswith("FEATURE_") for item in features
        ):
            errors.append(f"{label}.source.features must contain only FEATURE_ identifiers")
            features = []
        workshop = source.get("workshop_item")
        if workshop is not None and not (isinstance(workshop, str) and workshop.isdigit()
                                         and source.get("workshop_title")):
            errors.append(f"{label}.source workshop_item must be a numeric id with a workshop_title")
        if status in ("authored", "authored_required"):
            if target is not None or kind != "authored":
                errors.append(f"{label} {status} mappings need a null target and an authored source")
            if status == "authored_required" and confidence != "none":
                errors.append(f"{label} authored_required mappings need none confidence")
            if status == "authored" and (confidence == "none" or not source.get("parts")):
                errors.append(f"{label} authored mappings need a confidence and named source parts")
        elif not isinstance(target, str) or not target.startswith("FEATURE_"):
            errors.append(f"{label}.civ6_artdef must be a FEATURE_ identifier")
        elif kind == "authored" or target not in features:
            errors.append(f"{label} needs a base or workshop source listing {target}")
        if kind == "workshop" and workshop is None:
            errors.append(f"{label} workshop sources need a workshop_item")
        if status == "exact" and confidence != "high":
            errors.append(f"{label} exact mappings must have high confidence")
        if entry["fallback"] != "native_natural_wonder":
            errors.append(f"{label}.fallback must be 'native_natural_wonder'")
        if not isinstance(entry["alternatives"], list) or any(
            not isinstance(item, str) or not item.startswith("FEATURE_")
            for item in entry["alternatives"]
        ):
            errors.append(f"{label}.alternatives must contain only FEATURE_ identifiers")
        if not isinstance(entry["kit_parts"], list) or not entry["kit_parts"]:
            errors.append(f"{label}.kit_parts must be a non-empty list")
        if not isinstance(entry["season_parts"], list):
            errors.append(f"{label}.season_parts must be a list")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            errors.append(f"{label}.reason must be non-empty")
    return errors


def validate_against_default_config(
    document: dict[str, Any], definitions: list[dict[str, Any]]
) -> list[str]:
    expected = []
    for index, item in enumerate(definitions):
        expected.append({
            "c3x_default_index": index,
            "c3x_name": item.get("name"),
            "terrain_type": item.get("terrain_type"),
            "adjacent_to": item.get("adjacent_to"),
            "adjacency_dir": item.get("adjacency_dir"),
            "native_sprite": {
                "img_path": item.get("img_path"),
                "row": item.get("img_row"),
                "column": item.get("img_column"),
            },
            "configured_animation_count": len(item.get("animations", [])),
        })
    actual = [
        {field: item[field] for field in expected[index]}
        for index, item in enumerate(document.get("mappings", []))
        if index < len(expected)
    ]
    errors: list[str] = []
    if len(expected) != len(document.get("mappings", [])):
        errors.append(
            f"default config has {len(expected)} definitions but mapping has "
            f"{len(document.get('mappings', []))}"
        )
    for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
        if expected_item != actual_item:
            errors.append(
                f"mapping {index} does not match default config: "
                f"expected {expected_item!r}, got {actual_item!r}"
            )
    return errors


def discover_civ6_features(assets_root: Path) -> tuple[dict[str, list[str]], list[str]]:
    discovered: dict[str, set[str]] = {}
    parse_errors: list[str] = []
    for path in sorted(assets_root.rglob("Features.artdef")):
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as exc:
            parse_errors.append(f"{path}: {exc}")
            continue
        for element in root.iter("m_Name"):
            name = element.get("text", "")
            if name.startswith("FEATURE_"):
                relative = path.relative_to(assets_root).as_posix()
                discovered.setdefault(name, set()).add(relative)
    return {name: sorted(paths) for name, paths in sorted(discovered.items())}, parse_errors


def build_resolution_report(
    document: dict[str, Any], assets_root: Path, workshop_root: Path | None = None
) -> dict[str, Any]:
    """Resolve each source feature where it comes from: base features in the
    installed game, Workshop features in their own item folder."""
    discovered, parse_errors = discover_civ6_features(assets_root)
    workshop: dict[str, dict[str, list[str]]] = {}
    resolved: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    authored_required: list[dict[str, Any]] = []
    for entry in document["mappings"]:
        record = {
            "c3x_key": entry["c3x_key"],
            "c3x_name": entry["c3x_name"],
            "civ6_artdef": entry["civ6_artdef"],
            "mapping_status": entry["mapping_status"],
            "confidence": entry["confidence"],
        }
        source = entry["source"]
        if entry["mapping_status"] == "authored_required":
            authored_required.append(record)
            continue
        item = source.get("workshop_item")
        if source["kind"] == "workshop":
            record["workshop_item"] = item
            if item not in workshop:
                folder = None if workshop_root is None else workshop_root / item
                found, errors = ({}, []) if folder is None or not folder.is_dir() else discover_civ6_features(folder)
                workshop[item] = found
                parse_errors.extend(errors)
            table = workshop[item]
        else:
            table = discovered
        paths = {name: table.get(name) for name in source["features"]}
        missing = sorted(name for name, found in paths.items() if not found)
        if missing:
            record["missing_features"] = missing
            unavailable.append(record)
        else:
            record["artdef_paths"] = {name: found for name, found in paths.items()}
            resolved.append(record)
    return {
        "schema": "c3x.c3x_to_civ6_natural_wonder_resolution.v0",
        "mapping_schema": document["schema"],
        "discovered_feature_artdefs": len(discovered),
        "resolved": resolved,
        "unavailable": unavailable,
        "authored_required": authored_required,
        "parse_errors": parse_errors,
        "summary": {
            "mapping_count": len(document["mappings"]),
            "resolved_count": len(resolved),
            "unavailable_count": len(unavailable),
            "authored_required_count": len(authored_required),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--default-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--civ6-assets-root", type=Path)
    parser.add_argument("--workshop-root", type=Path,
                        help="Steam workshop/content/289070 folder holding the Workshop items the ledger names")
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--require-all-targets", action="store_true")
    args = parser.parse_args(argv)

    document = load_json(args.mapping)
    errors = validate_mapping(document)
    errors.extend(validate_against_default_config(
        document, parse_natural_wonder_config(args.default_config)
    ))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Valid natural-wonder mapping seed: {len(document['mappings'])} C3X definitions")
    if args.civ6_assets_root is None:
        return 0

    report = build_resolution_report(document, args.civ6_assets_root, args.workshop_root)
    summary = report["summary"]
    print(
        f"Civ VI Features: {summary['resolved_count']} resolved, "
        f"{summary['unavailable_count']} unavailable, "
        f"{summary['authored_required_count']} authored"
    )
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.require_all_targets and report["unavailable"]:
        for item in report["unavailable"]:
            print(f"UNAVAILABLE: {item['c3x_name']} -> {', '.join(item['missing_features'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
