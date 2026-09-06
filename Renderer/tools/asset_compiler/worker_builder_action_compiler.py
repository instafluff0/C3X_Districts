#!/usr/bin/env python3
"""Validate installed Worker/Builder evidence and emit a generic runtime map."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RENDERER_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
from Renderer.tools.asset_compiler.unit_member_resolver import resolve_unit

DEFAULT_STRATEGY = Path(__file__).with_name("worker_builder_action_strategy.json")
DEFAULT_CIV3_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_CIV6_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
DEFAULT_OUTPUT = RENDERER_ROOT / "preview/out/units/worker_builder_runtime_mapping.json"
DEFAULT_REPORT = RENDERER_ROOT / "preview/out/units/worker_builder_source_probe.json"

WORKER_INIS = (
    "Art/Units/Worker/Worker.INI",
    "Art/Units/Worker Industrial Ages/Worker Industrial Ages.INI",
    "Art/Units/Worker Modern Times/Worker Modern Times.INI",
)
REQUIRED_WORKER_SLOTS = (
    "DEFAULT", "RUN", "DEATH", "ROAD", "MINE", "IRRIGATE", "FORTRESS",
    "CAPTURE", "JUNGLE", "FOREST", "PLANT",
)
SOURCE_FREE_FORBIDDEN = (
    "ANIMATION_", "ACTIVITY_", "UNIT_BUILDER", "Civ6.app", "SHARED_DATA", ".blp",
)
WORKER_JOB_ENUMS = (
    "WJ_Build_Mines", "WJ_Irrigate", "WJ_Build_Fortress", "WJ_Build_Road",
    "WJ_Build_Railroad", "WJ_Plant_Forest", "WJ_Clear_Forest", "WJ_Clear_Swamp",
    "WJ_Clean_Pollution", "WJ_Build_Airfield", "WJ_Build_Radar", "WJ_Build_Outpost",
    "WJ_Build_Barricade",
)
WORKER_STATE_ENUMS = (
    "UnitState_Build_Mines", "UnitState_Irrigate", "UnitState_Build_Fortress",
    "UnitState_Build_Road", "UnitState_Build_Railroad", "UnitState_Plant_Forest",
    "UnitState_Clear_Forest", "UnitState_Clear_Wetlands", "UnitState_Clear_Damage",
    "UnitState_Build_Airfield", "UnitState_Build_Radar_Tower", "UnitState_Build_Outpost",
    "UnitState_Build_Barricade",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _element_name(node: ET.Element) -> str:
    value = node.find("m_Name")
    return "" if value is None else value.get("text", value.text or "")


def _field_text(node: ET.Element, param_name: str) -> str:
    for value in node.findall("m_Fields/m_Values/Element"):
        param = value.find("m_ParamName")
        name = "" if param is None else param.get("text", param.text or "")
        if name != param_name:
            continue
        for tag in ("m_Value", "m_ElementName"):
            child = value.find(tag)
            if child is not None:
                return child.get("text", child.text or "")
    return ""


def _named_elements(root: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in root.iter("Element") if _element_name(node) == name]


def _one_named(root: ET.Element, name: str) -> ET.Element:
    matches = _named_elements(root, name)
    if len(matches) != 1:
        raise ValueError(f"expected one ArtDef element {name!r}, found {len(matches)}")
    return matches[0]


def _read_worker_ini(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str.upper
    parser.read_string(path.read_text(encoding="latin-1"))
    if not parser.has_section("Animations"):
        raise ValueError(f"{path} has no [Animations] section")
    result = {key.upper(): value.strip() for key, value in parser.items("Animations")}
    missing = [slot for slot in REQUIRED_WORKER_SLOTS if not result.get(slot)]
    if missing:
        raise ValueError(f"{path} has empty required worker slots: {', '.join(missing)}")
    if result.get("BUILD"):
        raise ValueError(f"{path} unexpectedly uses BUILD; persistent work must remain job-driven")
    return result


def _enum_value(source: str, name: str) -> int:
    match = re.search(rf"\b{re.escape(name)}=(0x[0-9a-fA-F]+|\d+)", source)
    if match is None:
        raise ValueError(f"native source has no enum value for {name}")
    return int(match.group(1), 0)


def _probe_native_worker_mapping(strategy: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / "ref/Civ3Conquests_master.exe.c"
    source = path.read_text(encoding="utf-8", errors="replace")
    start = source.find("void __thiscall FUN_004068e0(void *this,Unit *unit,int job_id)")
    end = source.find("void FUN_004069c0", start)
    if start < 0 or end < 0:
        raise ValueError("native worker job-to-animation function is unavailable")
    body = source[start:end]
    expected_sequences = (
        r"case 0:\s*anim_type = AT_MINE;",
        r"case 1:\s*case 8:\s*anim_type = AT_IRRIGATE;",
        r"case 2:\s*case 0xc:\s*anim_type = AT_FORTRESS;",
        r"case 3:\s*case 4:\s*anim_type = AT_ROAD;",
        r"case 5:\s*anim_type = AT_PLANT;",
        r"case 6:\s*anim_type = AT_FOREST;",
        r"case 7:\s*anim_type = AT_JUNGLE;",
        r"default:\s*anim_type = AT_DEFAULT;",
    )
    for pattern in expected_sequences:
        if re.search(pattern, body) is None:
            raise ValueError(f"native worker job mapping changed near pattern {pattern!r}")
    job_values = [_enum_value(source, name) for name in WORKER_JOB_ENUMS]
    state_values = [_enum_value(source, name) for name in WORKER_STATE_ENUMS]
    if job_values != [job["id"] for job in strategy["worker_jobs"]]:
        raise ValueError("strategy job IDs disagree with the native Worker_Jobs enum")
    if state_values != [job["unit_state"] for job in strategy["worker_jobs"]]:
        raise ValueError("strategy states disagree with the native UnitStateType enum")
    return {
        "function": "FUN_004068e0",
        "worker_job_enum_values": dict(zip(WORKER_JOB_ENUMS, job_values)),
        "unit_state_enum_values": dict(zip(WORKER_STATE_ENUMS, state_values)),
        "job_to_flc_aliases": {
            "0": "MINE", "1,8": "IRRIGATE", "2,12": "FORTRESS",
            "3,4": "ROAD", "5": "PLANT", "6": "FOREST", "7": "JUNGLE",
            "9,10,11": "DEFAULT",
        },
    }


def validate_strategy(strategy: dict[str, Any]) -> None:
    if strategy.get("schema") != "c3x.worker_builder_action_strategy.v0":
        raise ValueError("unsupported worker-builder strategy schema")
    if strategy["authority"]["primary"] != "civ3_worker_job_id":
        raise ValueError("worker job id must remain the primary presentation authority")
    if strategy["applicability"]["unit_name_detection"]:
        raise ValueError("worker specialty actions must not depend on a unit name")
    jobs = strategy["worker_jobs"]
    if [job["id"] for job in jobs] != list(range(13)):
        raise ValueError("worker jobs must cover the exact Civ III IDs 0 through 12")
    if len({job["unit_state"] for job in jobs}) != 13:
        raise ValueError("worker job unit-state fallbacks must be unique")
    clips = strategy["clips"]
    for job in jobs:
        if job["action"] not in clips:
            raise ValueError(f"job {job['id']} references unknown action {job['action']!r}")
    attachment = strategy["attachment_selection"]
    if attachment["mode"] != "exclusive_attachment_group":
        raise ValueError("worker tools must use an exclusive attachment group")
    if not attachment["select_exactly_one_for_work_action"]:
        raise ValueError("a work action must select exactly one tool")
    if set(attachment["source_candidates"]) != set(attachment["source_tool_assets"]):
        raise ValueError("every source tool bin must have one expected asset")
    for name, clip in clips.items():
        if clip["playback"] not in ("loop", "clamp"):
            raise ValueError(f"clip {name!r} has invalid playback")
        tool = clip.get("tool")
        if tool is not None and tool not in attachment["runtime_candidates"]:
            raise ValueError(f"clip {name!r} references a tool outside the exclusive group")
    for name in ("CAPTURE", "REPAIR"):
        actions = strategy["special_actions"][name]["actions"]
        if len(actions) != 4 or any(action not in clips for action in actions):
            raise ValueError(f"{name} must select four existing deterministic variants")


def probe_sources(strategy: dict[str, Any], civ3_root: Path, civ6_root: Path) -> dict[str, Any]:
    native_report = _probe_native_worker_mapping(strategy)
    ini_reports = []
    for relative in WORKER_INIS:
        path = civ3_root / relative
        animations = _read_worker_ini(path)
        ini_reports.append({
            "path": relative,
            "sha256": _sha256(path),
            "slots": {slot: animations.get(slot, "") for slot in REQUIRED_WORKER_SLOTS + ("BUILD",)},
        })

    activities_path = civ6_root / "Base/ArtDefs/UnitActivities.artdef"
    operations_path = civ6_root / "Base/ArtDefs/UnitOperations.artdef"
    improvements_path = civ6_root / "Base/ArtDefs/Improvements.artdef"
    units_path = civ6_root / "Base/ArtDefs/Units.artdef"
    activities = ET.parse(activities_path).getroot()
    operations = ET.parse(operations_path).getroot()
    improvements = ET.parse(improvements_path).getroot()
    units = ET.parse(units_path).getroot()

    activity_report = {}
    for name, expected in strategy["source_activities"].items():
        actual = _field_text(_one_named(activities, name), "ExpendAnimation")
        if actual != expected["expend_animation"]:
            raise ValueError(f"{name} expend animation is {actual!r}, expected {expected['expend_animation']!r}")
        activity_report[name] = actual

    operation_report = {}
    for name, expected in strategy["source_operation_evidence"].items():
        actual = _field_text(_one_named(operations, name), "DefaultTargetActivity")
        if actual != expected:
            raise ValueError(f"{name} default activity is {actual!r}, expected {expected!r}")
        operation_report[name] = actual

    improvement_report = {}
    for name, expected in strategy["source_improvement_evidence"].items():
        improvement = _one_named(improvements, name)
        build_nodes = [node for node in improvement.iter("Element") if _element_name(node) == "BUILD"]
        actual = [_field_text(node, "XRef") for node in build_nodes]
        if expected not in actual:
            raise ValueError(f"{name} does not bind BUILD to {expected}")
        improvement_report[name] = expected

    body_report = {}
    for profile in strategy["source_body_profiles"]:
        unit = _one_named(units, profile["source_unit"])
        members = [
            node for node in unit.iter("Element")
            if _field_text(node, "Type") and _field_text(node, "Type").startswith("Builder_")
        ]
        if len(members) != 4:
            raise ValueError(f"{profile['source_unit']} exposes {len(members)} Builder recipes, expected 4")
        body_report[profile["source_unit"]] = [_field_text(node, "Type") for node in members]

    builder_recipe = resolve_unit(civ6_root, "UNIT_BUILDER", member_index=0)
    tool_components = [
        component for component in builder_recipe["selected_components"]
        if component["role"] == "Tool"
    ]
    actual_tools = {component["bin"]: component["source_entry"] for component in tool_components}
    expected_tools = strategy["attachment_selection"]["source_tool_assets"]
    if actual_tools != expected_tools:
        raise ValueError(f"Builder tool bins changed: {actual_tools!r}")

    shared_data = civ6_root / "Base/Platforms/Windows/BLPs/SHARED_DATA"
    clip_report = {}
    for logical, clip in strategy["clips"].items():
        path = shared_data / clip["source_entry"]
        if not path.is_file():
            raise ValueError(f"missing installed Builder animation {clip['source_entry']}")
        clip_report[logical] = {
            "source_entry": clip["source_entry"],
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    return {
        "schema": "c3x.worker_builder_source_probe.v0",
        "civ3_native_mapping": native_report,
        "civ3_worker_inis": ini_reports,
        "civ6_activities": activity_report,
        "civ6_operations": operation_report,
        "civ6_improvements": improvement_report,
        "civ6_body_profiles": body_report,
        "civ6_tool_bins": actual_tools,
        "tool_bin_interpretation": "exclusive_action_selected_alternatives_not_additive_components",
        "source_clips": clip_report,
        "finding": "worker_job_id_is_required_because_native_flc_slots_are_many_to_one",
        "runtime_integration": "not_enabled",
    }


def compile_runtime(strategy: dict[str, Any]) -> dict[str, Any]:
    clips = {
        name: {
            "clip": record["runtime_clip"],
            "path": record["output"],
            "playback": record["playback"],
            "tool": record.get("tool"),
        }
        for name, record in strategy["clips"].items()
    }
    result = {
        "schema": "c3x.worker_action_mapping.v0",
        "authority": {
            "primary": "worker_job_id",
            "fallback": "unit_state",
            "animation_slot": "diagnostic_only",
        },
        "applicability": strategy["applicability"],
        "default_bodies_by_era": {
            str(profile["era"]): profile["runtime_body"]
            for profile in strategy["source_body_profiles"]
        },
        "clips": clips,
        "jobs": {
            str(job["id"]): {
                "name": job["name"],
                "unit_state": job["unit_state"],
                "action": job["action"],
                "tool": clips[job["action"]]["tool"],
                "vfx": job["vfx"],
            }
            for job in strategy["worker_jobs"]
        },
        "capture": {
            "selection": "stable_body_variation_modulo_4",
            "actions": strategy["special_actions"]["CAPTURE"]["actions"],
        },
        "generic_build": {
            "action": strategy["special_actions"]["BUILD"]["action"],
            "persistent_jobs_must_not_use_this": True,
        },
        "optional_repair": {
            "selection": strategy["special_actions"]["REPAIR"]["selection"],
            "actions": strategy["special_actions"]["REPAIR"]["actions"],
            "ordinary_worker_job": None,
        },
        "attachments": {
            "mode": strategy["attachment_selection"]["mode"],
            "socket": strategy["attachment_selection"]["socket"],
            "group": strategy["attachment_selection"]["group"],
            "hide_for_non_work_action": strategy["attachment_selection"]["hide_for_non_work_action"],
        },
        "timing": strategy["timing"],
        "effects": strategy["effects"],
        "runtime_integration": "not_enabled",
    }
    encoded = json.dumps(result, sort_keys=True)
    forbidden = [needle for needle in SOURCE_FREE_FORBIDDEN if needle in encoded]
    if forbidden:
        raise ValueError(f"runtime mapping leaked source identifiers: {', '.join(forbidden)}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--civ3-root", type=Path, default=DEFAULT_CIV3_ROOT)
    parser.add_argument("--civ6-root", type=Path, default=DEFAULT_CIV6_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-source-probe", action="store_true")
    args = parser.parse_args(argv)
    try:
        strategy = json.loads(args.strategy.read_text(encoding="utf-8"))
        validate_strategy(strategy)
        runtime = compile_runtime(strategy)
        report = (
            {"schema": "c3x.worker_builder_source_probe.v0", "status": "skipped"}
            if args.skip_source_probe
            else probe_sources(strategy, args.civ3_root, args.civ6_root)
        )
        _write_json(args.output, runtime)
        _write_json(args.report, report)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Compiled {len(runtime['jobs'])} worker jobs and {len(runtime['clips'])} specialty clips")
    print(f"Runtime map: {args.output}")
    print(f"Source probe: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
