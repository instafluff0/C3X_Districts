#!/usr/bin/env python3
"""Validate the generic eight-facing/two-zoom unit and formation contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = TOOLS / "unit_visual_calibration.json"
DEFAULT_FORMATIONS = TOOLS / "unit_formation_strategy.json"
DEFAULT_FAMILIES = TOOLS / "unit_family_strategy.json"
DEFAULT_FAMILY_PACK = Path(__file__).resolve().parents[2] / "packs/UnitFamilyLab"
DEFAULT_COMPOUND_PACK = Path(__file__).resolve().parents[2] / "packs/CompoundUnitLab"
BASIC_ACTIONS = {"idle", "fidget", "move", "fortify", "attack", "defend", "death", "victory"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_visual_contract(
    calibration_path: Path = DEFAULT_CALIBRATION,
    formations_path: Path = DEFAULT_FORMATIONS,
    families_path: Path = DEFAULT_FAMILIES,
    family_pack: Path | None = DEFAULT_FAMILY_PACK,
    compound_pack: Path | None = DEFAULT_COMPOUND_PACK,
) -> dict[str, Any]:
    calibration = _load(calibration_path)
    formations = _load(formations_path)
    families = _load(families_path)
    if calibration.get("schema") != "c3x.unit_visual_calibration.v0":
        raise ValueError("unsupported unit visual calibration schema")
    facings = calibration.get("facing", {}).get("facings", [])
    if (
        [item.get("name") for item in facings] != ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        or {item.get("civ3_direction") for item in facings} != set(range(1, 9))
        or [item.get("yaw_degrees") for item in facings] != list(range(0, 360, 45))
    ):
        raise ValueError("unit facing table does not exactly cover Civ III's eight visible directions")
    zooms = calibration.get("zooms", [])
    if len(zooms) != 2 or [item.get("id") for item in zooms] != ["normal", "reduced"]:
        raise ValueError("unit calibration must declare normal and reduced zoom")
    if zooms[0]["tile_pixels"] != [128, 64] or zooms[1]["tile_pixels"] != [64, 32]:
        raise ValueError("unit zoom tiles do not match Civ III's 2:1 pixel basis")
    if not math.isclose(zooms[1]["projection_scale"] / zooms[0]["projection_scale"], 0.5):
        raise ValueError("reduced unit projection must be exactly half normal scale")
    if set(calibration.get("actions", [])) != BASIC_ACTIONS:
        raise ValueError("unit calibration must cover exactly the first eight basic actions")

    if formations.get("schema") != "c3x.unit_formation_strategy.v0":
        raise ValueError("unsupported unit formation schema")
    profiles = formations.get("profiles", {})
    if profiles.get("single", {}).get("offsets_tile") != [[0.0, 0.0]]:
        raise ValueError("ordinary single-unit formation must remain centered")
    triad = profiles.get("restrained_triad", {})
    if triad.get("semantic_bodies") != 3 or not triad.get("pack_opt_in"):
        raise ValueError("the only ordinary multi-body profile must be a pack-opt-in restrained triad")
    army = profiles.get("army_commander_and_member", {})
    if army.get("normal_member_offset_pixels") != 40 or army.get("reduced_member_offset_pixels") != 20:
        raise ValueError("Army offsets have drifted from the audited native two-zoom placement")
    forbidden = set(formations.get("policy", {}).get("forbidden_drivers", []))
    if forbidden != {"hit_points", "loaded_unit_count", "stack_size", "attack_value"}:
        raise ValueError("formation choice must not encode gameplay strength or roster size")

    units = {}
    for unit in families.get("units", []):
        actions = set(unit.get("actions", {})) | set(unit.get("additional_actions", {}))
        gaps = sorted(BASIC_ACTIONS - actions)
        units[unit["slug"]] = {
            "kind": "simple",
            "formation_profile": "single",
            "basic_action_gaps": gaps,
            "calibration_cells": len(actions & BASIC_ACTIONS) * len(facings) * len(zooms),
        }

    family_cache_gaps = {}
    if family_pack is not None and (family_pack / "manifest.json").is_file():
        family_manifest = _load(family_pack / "manifest.json")
        for unit_id, entry in family_manifest.get("units", {}).items():
            recipe = _load(family_pack / entry["recipe"])
            gaps = sorted(
                action
                for action, animation_id in recipe.get("actions", {}).items()
                if family_manifest["animations"][animation_id].get("pose_cache_status")
                != "validated_model_aware_world_matrices"
                or not family_manifest["animations"][animation_id].get("pose_caches")
            )
            if gaps:
                family_cache_gaps[unit_id.removeprefix("unit/")] = gaps

    if compound_pack is not None and (compound_pack / "manifest.json").is_file():
        manifest = _load(compound_pack / "manifest.json")
        for unit_id, entry in manifest.get("units", {}).items():
            recipe = _load(compound_pack / entry["recipe"])
            actions = set(recipe.get("actions", {}))
            missing_caches = sorted(
                action
                for action, binding in recipe.get("actions", {}).items()
                if set(binding.get("node_pose_caches", {})) != set(recipe.get("nodes", {}))
            )
            units[unit_id.removeprefix("unit/")] = {
                "kind": "compound",
                "formation_profile": "single",
                "semantic_bodies": 1,
                "composition_nodes": len(recipe.get("nodes", {})),
                "basic_action_gaps": sorted(BASIC_ACTIONS - actions),
                "actions_missing_pose_caches": missing_caches,
                "calibration_cells": len(actions & BASIC_ACTIONS) * len(facings) * len(zooms),
            }

    unresolved = {
        unit: record["basic_action_gaps"]
        for unit, record in units.items()
        if record["basic_action_gaps"]
    }
    cache_gaps = {
        unit: record.get("actions_missing_pose_caches", [])
        for unit, record in units.items()
        if record.get("actions_missing_pose_caches")
    }
    return {
        "schema": "c3x.unit_visual_calibration_validation.v0",
        "facing_count": len(facings),
        "zoom_count": len(zooms),
        "cells_per_basic_complete_unit": len(BASIC_ACTIONS) * len(facings) * len(zooms),
        "units": units,
        "unresolved_basic_action_gaps": unresolved,
        "family_pose_cache_gaps": family_cache_gaps,
        "pose_cache_gaps": cache_gaps,
        "formation_decision": "single_by_default_restrained_triad_pack_opt_in_army_two_body",
        "visual_measurement_status": "pending_authorized_l20_render",
        "runtime_integration": "not_enabled",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--formations", type=Path, default=DEFAULT_FORMATIONS)
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--family-pack", type=Path, default=DEFAULT_FAMILY_PACK)
    parser.add_argument("--compound-pack", type=Path, default=DEFAULT_COMPOUND_PACK)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_visual_contract(
            args.calibration, args.formations, args.families, args.family_pack, args.compound_pack
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Validated {report['facing_count']} facings x {report['zoom_count']} zooms; "
        f"{len(report['units'])} unit profiles; gaps={report['unresolved_basic_action_gaps']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
