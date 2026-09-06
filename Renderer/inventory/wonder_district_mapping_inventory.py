"""Validate and resolve the default C3X constructed-wonder and district seed maps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[2]
WONDER_SCHEMA = "c3x.c3x_to_civ6_constructed_wonder_mapping.v0"
DISTRICT_SCHEMA = "c3x.c3x_to_civ6_district_mapping.v0"
WONDER_ROSTER_SCHEMA = "c3x.vanilla_conquests_wonder_roster.v0"
DEFAULT_WONDER_MAPPING = Path(__file__).with_name("vanilla_c3x_to_civ6_constructed_wonders.json")
DEFAULT_DISTRICT_MAPPING = Path(__file__).with_name("vanilla_c3x_to_civ6_districts.json")
DEFAULT_WONDER_ROSTER = Path(__file__).with_name("vanilla_conquests_wonder_roster.json")
DEFAULT_WONDER_CONFIG = ROOT / "default.districts_wonders_config.txt"
DEFAULT_DISTRICT_CONFIG = ROOT / "default.districts_config.txt"
DEFAULT_BIQ_SEMANTICS = Path(__file__).with_name("vanilla_conquests_biq_semantics.json")
STATUSES = {"exact", "approximate", "authored_required"}
CONFIDENCE = {"high", "medium", "low", "none"}
TARGET_PREFIXES = {
    "building": "BUILDING_",
    "district": "DISTRICT_",
    "improvement": "IMPROVEMENT_",
    "authored": None,
}
LIST_FIELDS = {
    "buildable_on", "buildable_on_overlays", "img_paths", "dependent_improvs",
}
INTEGER_FIELDS = {
    "img_construct_row", "img_construct_column", "img_row", "img_column",
    "enable_img_alt_dir", "img_alt_dir_construct_row", "img_alt_dir_construct_column",
    "img_alt_dir_row", "img_alt_dir_column", "img_column_count", "vary_img_by_era",
    "vary_img_by_culture", "buildable_on_rivers",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in next(csv.reader([value], skipinitialspace=True)) if item.strip()]


def parse_blocks(path: Path, marker: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line == marker:
            current = {}
            records.append(current)
            continue
        if current is None or not line or line.startswith((";", "[")) or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in LIST_FIELDS:
            current[key] = parse_list(value)
        elif key in INTEGER_FIELDS:
            current[key] = int(value)
        else:
            current[key] = value
    return records


def validate_target(
    label: str, kind: str, target: Any, status: str, confidence: str
) -> list[str]:
    errors: list[str] = []
    if status not in STATUSES:
        errors.append(f"{label}.mapping_status has unknown value {status!r}")
    if confidence not in CONFIDENCE:
        errors.append(f"{label}.confidence has unknown value {confidence!r}")
    prefix = TARGET_PREFIXES.get(kind, "missing")
    if prefix == "missing":
        errors.append(f"{label}.target_kind has unknown value {kind!r}")
    elif status == "authored_required":
        if kind != "authored" or target is not None or confidence != "none":
            errors.append(
                f"{label} authored_required targets need authored kind, null ID, and none confidence"
            )
    elif prefix is None or not isinstance(target, str) or not target.startswith(prefix):
        errors.append(f"{label}.civ6_artdef must start with {prefix!r}")
    if status == "exact" and confidence != "high":
        errors.append(f"{label} exact mappings must have high confidence")
    return errors


def validate_wonder_mapping(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema") != WONDER_SCHEMA:
        errors.append(f"schema must be {WONDER_SCHEMA!r}")
    policy = document.get("policy", {})
    if "not_map_rendered" not in policy.get("placement", ""):
        errors.append("wonder placement policy must preserve mapless BIQ wonders")
    if "fog and shroud" not in policy.get("retained_ownership", ""):
        errors.append("wonder retained ownership must preserve fog and shroud")
    mappings = document.get("mappings", [])
    if document.get("source_roster", {}).get("definition_count") != len(mappings):
        errors.append("wonder definition_count does not match mappings")
    required = {
        "c3x_default_index", "c3x_key", "c3x_name", "biq_index", "civ3_id",
        "wonder_class", "buildable_on", "buildable_on_rivers", "native_art",
        "target_kind", "civ6_artdef", "mapping_status", "confidence", "alternatives",
        "kit_parts", "fallback", "reason",
    }
    seen_names: set[str] = set()
    seen_indices: set[int] = set()
    for index, entry in enumerate(mappings):
        label = f"mappings[{index}]"
        missing = sorted(required - set(entry))
        if missing:
            errors.append(f"{label} is missing fields: {', '.join(missing)}")
            continue
        if entry["c3x_default_index"] != index:
            errors.append(f"{label}.c3x_default_index must preserve config order")
        if entry["c3x_name"] in seen_names:
            errors.append(f"duplicate wonder name: {entry['c3x_name']}")
        seen_names.add(entry["c3x_name"])
        if entry["biq_index"] in seen_indices:
            errors.append(f"duplicate wonder BIQ index: {entry['biq_index']}")
        seen_indices.add(entry["biq_index"])
        if entry["wonder_class"] not in {"great", "small"}:
            errors.append(f"{label}.wonder_class must be great or small")
        errors.extend(validate_target(
            label, entry["target_kind"], entry["civ6_artdef"],
            entry["mapping_status"], entry["confidence"],
        ))
        for alt_index, alternative in enumerate(entry["alternatives"]):
            alt_label = f"{label}.alternatives[{alt_index}]"
            if not isinstance(alternative, dict) or set(alternative) != {"kind", "id"}:
                errors.append(f"{alt_label} must contain kind and id")
            else:
                errors.extend(validate_target(
                    alt_label, alternative["kind"], alternative["id"], "approximate", "low"
                ))
        if entry["fallback"] != "native_constructed_wonder":
            errors.append(f"{label}.fallback must be native_constructed_wonder")
        if not entry["kit_parts"] or not entry["reason"].strip():
            errors.append(f"{label} needs kit_parts and reason")
    return errors


def expected_wonder_config_record(index: int, item: dict[str, Any]) -> dict[str, Any]:
    alternate = None
    if item.get("enable_img_alt_dir", 0):
        alternate = {
            "construct": [item.get("img_alt_dir_construct_row"), item.get("img_alt_dir_construct_column")],
            "complete": [item.get("img_alt_dir_row"), item.get("img_alt_dir_column")],
        }
    return {
        "c3x_default_index": index,
        "c3x_name": item.get("name"),
        "wonder_class": "small" if item.get("img_path", "Wonders.pcx").startswith("SmallWonders") else "great",
        "buildable_on": item.get("buildable_on", []),
        "buildable_on_rivers": bool(item.get("buildable_on_rivers", 0)),
        "native_art": {
            "img_path": item.get("img_path", "Wonders.pcx"),
            "construct": [item.get("img_construct_row"), item.get("img_construct_column")],
            "complete": [item.get("img_row"), item.get("img_column")],
            "alternate": alternate,
        },
    }


def validate_wonders_against_sources(
    document: dict[str, Any], config: list[dict[str, Any]], semantics: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    biq = {
        item["name"]: (item["biq_index"], item["civilopedia_entry"], item.get("wonder_class"))
        for item in semantics.get("improvements_and_wonders", [])
    }
    if len(config) != len(document.get("mappings", [])):
        errors.append(f"wonder config has {len(config)} blocks but mapping has {len(document.get('mappings', []))}")
    for index, (entry, config_item) in enumerate(zip(document.get("mappings", []), config)):
        expected = expected_wonder_config_record(index, config_item)
        actual = {key: entry[key] for key in expected}
        if actual != expected:
            errors.append(f"wonder mapping {index} does not match config: expected {expected!r}, got {actual!r}")
        expected_biq = biq.get(entry["c3x_name"])
        if expected_biq != (entry["biq_index"], entry["civ3_id"], entry["wonder_class"]):
            errors.append(f"wonder {entry['c3x_name']!r} does not match BIQ semantics")
    return errors


def validate_wonder_roster(
    roster: dict[str, Any], semantics: dict[str, Any], mapping: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if roster.get("schema") != WONDER_ROSTER_SCHEMA:
        errors.append(f"wonder roster schema must be {WONDER_ROSTER_SCHEMA!r}")
    records = roster.get("wonders", [])
    counts = roster.get("counts", {})
    actual_counts = {
        "total": len(records),
        "great": sum(item.get("wonder_class") == "great" for item in records),
        "small": sum(item.get("wonder_class") == "small" for item in records),
        "c3x_configured": sum(item.get("map_status") == "c3x_configured_candidate" for item in records),
        "not_map_rendered": sum(item.get("map_status") == "not_map_rendered" for item in records),
    }
    if counts != actual_counts:
        errors.append(f"wonder roster counts are {counts!r}, expected {actual_counts!r}")
    expected_biq = {
        (item["biq_index"], item["civilopedia_entry"], item["name"], item["wonder_class"])
        for item in semantics.get("improvements_and_wonders", [])
        if item.get("wonder_class") in {"great", "small"}
    }
    actual_biq = {
        (item["biq_index"], item["civ3_id"], item["civ3_name"], item["wonder_class"])
        for item in records
    }
    if expected_biq != actual_biq:
        errors.append("wonder roster does not exactly match BIQ Great/Small Wonder identities")
    configured = {item["c3x_key"]: item for item in mapping.get("mappings", [])}
    linked_keys: set[str] = set()
    for index, entry in enumerate(records):
        label = f"wonder_roster[{index}]"
        if entry["map_status"] == "c3x_configured_candidate":
            key = entry.get("c3x_mapping_key")
            linked = configured.get(key)
            if linked is None:
                errors.append(f"{label} references missing C3X mapping key {key!r}")
            elif (linked["biq_index"], linked["civ3_id"], linked["c3x_name"], linked["wonder_class"]) != (
                entry["biq_index"], entry["civ3_id"], entry["civ3_name"], entry["wonder_class"]
            ):
                errors.append(f"{label} does not match its configured C3X mapping")
            linked_keys.add(key)
            if entry.get("source_seed") is not None:
                errors.append(f"{label} must inherit its source from the configured mapping")
        elif entry["map_status"] == "not_map_rendered":
            if entry.get("c3x_mapping_key") is not None:
                errors.append(f"{label} mapless wonder must not reference a C3X mapping")
            source = entry.get("source_seed")
            if not isinstance(source, dict) or not source.get("reason", "").strip():
                errors.append(f"{label} mapless wonder needs a documented source seed")
            else:
                errors.extend(validate_target(
                    f"{label}.source_seed", source.get("target_kind"), source.get("civ6_artdef"),
                    source.get("mapping_status"), source.get("confidence"),
                ))
        else:
            errors.append(f"{label}.map_status is invalid")
    if linked_keys != set(configured):
        errors.append("wonder roster does not link every configured C3X wonder exactly once")
    return errors


def validate_district_mapping(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema") != DISTRICT_SCHEMA:
        errors.append(f"schema must be {DISTRICT_SCHEMA!r}")
    policy = document.get("policy", {})
    if "effective stage" not in policy.get("replacement_ownership", ""):
        errors.append("district ownership must cover the complete effective stage")
    if "fog and shroud" not in policy.get("retained_ownership", ""):
        errors.append("district retained ownership must preserve fog and shroud")
    mappings = document.get("mappings", [])
    if document.get("source_roster", {}).get("definition_count") != len(mappings):
        errors.append("district definition_count does not match mappings")
    required = {
        "c3x_default_index", "c3x_key", "c3x_name", "render_strategy", "district_type",
        "img_paths", "native_column_count", "vary_by_era", "vary_by_culture",
        "buildable_on", "buildable_on_overlays", "target_kind", "civ6_artdef",
        "mapping_status", "confidence", "special_family", "attachments", "fallback", "reason",
    }
    seen_names: set[str] = set()
    for index, entry in enumerate(mappings):
        label = f"mappings[{index}]"
        missing = sorted(required - set(entry))
        if missing:
            errors.append(f"{label} is missing fields: {', '.join(missing)}")
            continue
        if entry["c3x_default_index"] != index:
            errors.append(f"{label}.c3x_default_index must preserve config order")
        if entry["c3x_name"] in seen_names:
            errors.append(f"duplicate district name: {entry['c3x_name']}")
        seen_names.add(entry["c3x_name"])
        if entry["render_strategy"] not in {"by-count", "by-building"}:
            errors.append(f"{label}.render_strategy is invalid")
        errors.extend(validate_target(
            label, entry["target_kind"], entry["civ6_artdef"],
            entry["mapping_status"], entry["confidence"],
        ))
        attachment_names: set[str] = set()
        for attachment_index, attachment in enumerate(entry["attachments"]):
            attachment_label = f"{label}.attachments[{attachment_index}]"
            attachment_names.add(attachment.get("civ3_name", ""))
            errors.extend(validate_target(
                attachment_label, attachment.get("target_kind"), attachment.get("civ6_artdef"),
                attachment.get("mapping_status"), attachment.get("confidence"),
            ))
        if len(attachment_names) != len(entry["attachments"]):
            errors.append(f"{label} has duplicate attachment names")
        if entry["fallback"] != "native_district":
            errors.append(f"{label}.fallback must be native_district")
        if not entry["reason"].strip():
            errors.append(f"{label}.reason must be non-empty")
    return errors


def expected_district_config_record(index: int, item: dict[str, Any]) -> dict[str, Any]:
    dependent = item.get("dependent_improvs", [])
    return {
        "c3x_default_index": index,
        "c3x_name": item.get("name"),
        "render_strategy": item.get("render_strategy", "by-count"),
        "district_type": item.get("type", "district"),
        "img_paths": item.get("img_paths", []),
        "native_column_count": item.get("img_column_count", len(dependent) + 1),
        "vary_by_era": bool(item.get("vary_img_by_era", 0)),
        "vary_by_culture": bool(item.get("vary_img_by_culture", 0)),
        "buildable_on": item.get("buildable_on", []),
        "buildable_on_overlays": item.get("buildable_on_overlays", []),
        "dependent_improvs": dependent,
    }


def validate_districts_against_config(
    document: dict[str, Any], config: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    if len(config) != len(document.get("mappings", [])):
        errors.append(f"district config has {len(config)} blocks but mapping has {len(document.get('mappings', []))}")
    for index, (entry, config_item) in enumerate(zip(document.get("mappings", []), config)):
        expected = expected_district_config_record(index, config_item)
        actual = {key: entry[key] for key in expected if key != "dependent_improvs"}
        if actual != {key: value for key, value in expected.items() if key != "dependent_improvs"}:
            errors.append(f"district mapping {index} does not match config metadata")
        if [item["civ3_name"] for item in entry["attachments"]] != expected["dependent_improvs"]:
            errors.append(f"district {entry['c3x_name']!r} attachment order does not match dependent_improvs")
    return errors


def discover_civ6_targets(assets_root: Path) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    discovered: dict[str, dict[str, set[str]]] = {kind: {} for kind in TARGET_PREFIXES if kind != "authored"}
    parse_errors: list[str] = []
    file_stems = {
        "building": ("Buildings",),
        "district": ("Districts",),
        "improvement": ("Improvements",),
    }
    for path in sorted(assets_root.rglob("*.artdef")):
        kinds = [kind for kind, stems in file_stems.items() if path.stem.startswith(stems)]
        if not kinds:
            continue
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as exc:
            parse_errors.append(f"{path}: {exc}")
            continue
        relative = path.relative_to(assets_root).as_posix()
        for element in root.iter("m_Name"):
            name = element.get("text", "")
            for kind in kinds:
                if name.startswith(TARGET_PREFIXES[kind]):
                    discovered[kind].setdefault(name, set()).add(relative)
    normalized = {
        kind: {name: sorted(paths) for name, paths in sorted(entries.items())}
        for kind, entries in discovered.items()
    }
    return normalized, parse_errors


def iter_targets(document: dict[str, Any], category: str):
    for entry in document["mappings"]:
        yield category, entry["c3x_name"], "base", entry
        for attachment in entry.get("attachments", []):
            yield category, entry["c3x_name"], attachment["civ3_name"], attachment


def build_resolution_report(
    wonders: dict[str, Any], districts: dict[str, Any], assets_root: Path,
    wonder_roster: dict[str, Any] | None = None,
) -> dict[str, Any]:
    discovered, parse_errors = discover_civ6_targets(assets_root)
    resolved: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    authored_required: list[dict[str, Any]] = []
    targets = (
        list(iter_targets(wonders, "constructed_wonder")) +
        list(iter_targets(districts, "district"))
    )
    if wonder_roster is not None:
        targets.extend(
            ("mapless_biq_wonder", item["civ3_name"], "dormant_source_seed", item["source_seed"])
            for item in wonder_roster["wonders"]
            if item["map_status"] == "not_map_rendered"
        )
    for category, owner, role, target in targets:
        record = {
            "category": category, "owner": owner, "role": role,
            "target_kind": target["target_kind"], "civ6_artdef": target["civ6_artdef"],
            "mapping_status": target["mapping_status"], "confidence": target["confidence"],
        }
        if target["mapping_status"] == "authored_required":
            authored_required.append(record)
            continue
        paths = discovered[target["target_kind"]].get(target["civ6_artdef"])
        if paths:
            record["artdef_paths"] = paths
            resolved.append(record)
        else:
            unavailable.append(record)
    return {
        "schema": "c3x.c3x_to_civ6_wonder_district_resolution.v0",
        "resolved": resolved,
        "unavailable": unavailable,
        "authored_required": authored_required,
        "parse_errors": parse_errors,
        "summary": {
            "target_count": len(resolved) + len(unavailable) + len(authored_required),
            "resolved_count": len(resolved),
            "unavailable_count": len(unavailable),
            "authored_required_count": len(authored_required),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wonder-mapping", type=Path, default=DEFAULT_WONDER_MAPPING)
    parser.add_argument("--district-mapping", type=Path, default=DEFAULT_DISTRICT_MAPPING)
    parser.add_argument("--wonder-roster", type=Path, default=DEFAULT_WONDER_ROSTER)
    parser.add_argument("--wonder-config", type=Path, default=DEFAULT_WONDER_CONFIG)
    parser.add_argument("--district-config", type=Path, default=DEFAULT_DISTRICT_CONFIG)
    parser.add_argument("--biq-semantics", type=Path, default=DEFAULT_BIQ_SEMANTICS)
    parser.add_argument("--civ6-assets-root", type=Path)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--require-all-targets", action="store_true")
    args = parser.parse_args(argv)

    wonders = load_json(args.wonder_mapping)
    districts = load_json(args.district_mapping)
    wonder_roster = load_json(args.wonder_roster)
    semantics = load_json(args.biq_semantics)
    errors = validate_wonder_mapping(wonders)
    errors.extend(validate_wonders_against_sources(
        wonders, parse_blocks(args.wonder_config, "#Wonder"), semantics
    ))
    errors.extend(validate_wonder_roster(wonder_roster, semantics, wonders))
    errors.extend(validate_district_mapping(districts))
    errors.extend(validate_districts_against_config(
        districts, parse_blocks(args.district_config, "#District")
    ))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Valid constructed-wonder mapping seed: {len(wonders['mappings'])} definitions")
    print(f"Valid BIQ wonder roster: {len(wonder_roster['wonders'])} Great/Small Wonders")
    print(f"Valid district mapping seed: {len(districts['mappings'])} definitions")
    if args.civ6_assets_root is None:
        return 0
    report = build_resolution_report(wonders, districts, args.civ6_assets_root, wonder_roster)
    summary = report["summary"]
    print(
        f"Civ VI targets: {summary['resolved_count']} resolved, "
        f"{summary['unavailable_count']} unavailable, "
        f"{summary['authored_required_count']} authored"
    )
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.require_all_targets and report["unavailable"]:
        for item in report["unavailable"]:
            print(f"UNAVAILABLE: {item['owner']} / {item['role']} -> {item['civ6_artdef']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
