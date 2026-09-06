#!/usr/bin/env python3
"""Validate and exercise the source-independent Civ III army composite contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_STRATEGY = Path(__file__).with_name("army_render_strategy.json")
REQUIRED_ERAS = {"ancient", "middle_ages", "industrial", "modern"}
REQUIRED_ACTIONS = {"idle", "fidget", "move", "fortify", "attack", "defend", "victory", "death"}


def load_strategy(path: Path = DEFAULT_STRATEGY) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "c3x.army_render_strategy.v0":
        raise ValueError("Army render strategy has an invalid schema")
    if document.get("runtime_status") != "not_enabled":
        raise ValueError("Offline army investigation must not enable runtime integration")
    authority = document.get("authority", {})
    if authority.get("displayed_member") != "civ3_army_displayed_member_id":
        raise ValueError("Civ III must remain authoritative for the displayed army member")
    composition = document.get("composition", {})
    if composition.get("children") != ["displayed_member", "commander"]:
        raise ValueError("The basic army composite must contain member then commander")
    if composition.get("additional_loaded_members") != "not_rendered":
        raise ValueError("The basic army composite must not crowd the tile with the full roster")
    if composition.get("hud_instances") != 1:
        raise ValueError("An army must retain exactly one parent HUD instance")
    if set(document.get("era_profiles", {})) != REQUIRED_ERAS:
        raise ValueError("Army commander profiles must cover the four Civ III eras")
    for era, profile in document["era_profiles"].items():
        if not profile.get("asset_id", "").startswith("unit/army/commander/"):
            raise ValueError(f"Army era {era} has a non-generic asset ID")
        if not profile.get("source_unit", "").startswith("UNIT_GREAT_GENERAL"):
            raise ValueError(f"Army era {era} must use dedicated Great General source art")
    native_pixels = document.get("placement", {}).get("native_reference_pixels", {})
    if native_pixels != {"normal_zoom_horizontal": 40, "reduced_zoom_horizontal": 20}:
        raise ValueError("Army placement must preserve both audited native offset references")
    if set(document.get("lab_matrix", {}).get("actions", [])) != REQUIRED_ACTIONS:
        raise ValueError("Army lab matrix must cover the basic unit action set")
    fallback = document.get("fallback", {})
    if fallback.get("forbid_mixed_native_custom_composite") is not True:
        raise ValueError("Army fallback must not mix native and custom bodies")
    if fallback.get("forbid_baked_army_member_combinations") is not True:
        raise ValueError("Army assets must remain composable for arbitrary scenario units")
    return document


def compose_army(snapshot: dict[str, Any], strategy: dict[str, Any] | None = None) -> dict[str, Any]:
    strategy = strategy or load_strategy()
    required = {"army_id", "era", "army_anchor", "army_action", "army_direction"}
    missing = required - set(snapshot)
    if missing:
        raise ValueError(f"Army snapshot is missing fields: {sorted(missing)}")
    era = snapshot["era"]
    if era not in strategy["era_profiles"]:
        raise ValueError(f"Unsupported Civ III era: {era!r}")
    army_id = snapshot["army_id"]
    children = []
    displayed_member = snapshot.get("displayed_member")
    if displayed_member is not None:
        member_fields = {"unit_id", "unit_type", "anchor", "action", "direction"}
        member_missing = member_fields - set(displayed_member)
        if member_missing:
            raise ValueError(f"Displayed army member is missing fields: {sorted(member_missing)}")
        children.append(
            {
                "instance_id": f"army/{army_id}/member/{displayed_member['unit_id']}",
                "role": "displayed_member",
                "asset_selector": {"unit_type": displayed_member["unit_type"]},
                "anchor": displayed_member["anchor"],
                "action": displayed_member["action"],
                "direction": displayed_member["direction"],
            }
        )
    profile = strategy["era_profiles"][era]
    children.append(
        {
            "instance_id": f"army/{army_id}/commander",
            "role": "commander",
            "asset_id": profile["asset_id"],
            "anchor": snapshot["army_anchor"],
            "action": snapshot["army_action"],
            "direction": snapshot["army_direction"],
        }
    )
    return {
        "kind": "army",
        "instance_id": f"army/{army_id}",
        "children": children,
        "retained_parent_hud_instances": 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    args = parser.parse_args(argv)
    try:
        strategy = load_strategy(args.strategy)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps({
        "schema": strategy["schema"],
        "era_profiles": sorted(strategy["era_profiles"]),
        "runtime_status": strategy["runtime_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
