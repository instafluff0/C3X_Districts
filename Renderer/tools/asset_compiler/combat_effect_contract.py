#!/usr/bin/env python3
"""Validate the combat-effect handoff and sample completed event traces deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT = Path(__file__).with_name("combat_effect_contract.json")
REQUIRED_PROFILES = {"ballistic_shell", "dropped_bomb", "guided_missile", "nuclear_detonation"}
REQUIRED_IMPACT_CODES = {str(value) for value in range(3, 9)}


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "c3x.combat_effect_contract.v0":
        raise ValueError("Combat-effect contract has an invalid schema")
    if document.get("runtime_status") != "not_enabled":
        raise ValueError("Preimplementation combat-effect contract must not enable runtime integration")
    authority = document.get("authority", {})
    if authority.get("gameplay") != "civ3" or authority.get("audio") != "civ3":
        raise ValueError("Civ III must retain gameplay and audio authority")
    impact_codes = document.get("native_bridge", {}).get("native_impact_effects", {})
    if set(impact_codes) != REQUIRED_IMPACT_CODES:
        raise ValueError("Native impact mapping must cover AE_Hit through AE_WaterMiss exactly")
    native_bridge = document.get("native_bridge", {})
    suppression = native_bridge.get("pixel_suppression", {})
    required_suppression_fields = {
        "ordinary_effect_loader", "standalone_animation_loader", "native_draw_rule",
        "field_access", "ownership_gate",
    }
    if set(suppression) != required_suppression_fields:
        raise ValueError("Pixel suppression must define the complete audio-preserving native bridge")
    if "low byte" not in suppression["field_access"] or "0x184" not in suppression["field_access"]:
        raise ValueError("Pixel suppression must target only the FLC draw-enable byte at 0x184")
    nuclear = native_bridge.get("nuclear_policy", {})
    required_nuclear_fields = {
        "delivery", "predicate", "detonated_boundary", "intercepted_boundary",
        "local_visuals", "rule",
    }
    if set(nuclear) != required_nuclear_fields:
        raise ValueError("Nuclear policy must define both authoritative outcome branches")
    if "Unit::do_nuke_tile" not in nuclear["detonated_boundary"]:
        raise ValueError("Nuclear detonation must bind to Unit::do_nuke_tile")
    if "Unit::get_intercepted_as_nuke" not in nuclear["intercepted_boundary"]:
        raise ValueError("Nuclear interception must bind to Unit::get_intercepted_as_nuke")
    profiles = document.get("profiles", {})
    if set(profiles) != REQUIRED_PROFILES:
        raise ValueError("Combat-effect profiles do not match the basic supported family set")
    required_profile_fields = {
        "projectile", "trajectory", "muzzle", "land_hit", "land_miss",
        "water_hit_or_miss", "impact_duration_ms", "night_light",
    }
    for profile_id, profile in profiles.items():
        if set(profile) != required_profile_fields:
            raise ValueError(f"Combat-effect profile {profile_id} has invalid fields")
        duration = profile["impact_duration_ms"]
        if not isinstance(duration, int) or not 1 <= duration <= 10000:
            raise ValueError(f"Combat-effect profile {profile_id} has invalid duration")
        for value in profile.values():
            if isinstance(value, str) and "civ6" in value.lower():
                raise ValueError("Runtime combat-effect profiles must be source-independent")
    ownership = document.get("ownership", {})
    if ownership.get("double_effects_forbidden") is not True:
        raise ValueError("Combat-effect ownership must forbid double effects")
    if document.get("bindings", {}).get("default_when_unmapped") != "native_unit_and_native_effect":
        raise ValueError("Unmapped arbitrary units must remain atomically native")
    unresolved = document.get("source_evidence", {}).get("unresolved", [])
    if any("suppression boundary" in item or "nuclear event" in item for item in unresolved):
        raise ValueError("Completed M7.5 boundary investigations must not remain unresolved")
    return document


def sample_event(event: dict[str, Any], now_ms: int) -> dict[str, Any]:
    """Sample a finalized trace; runtime may learn impact/cleanup timestamps incrementally."""
    required = {"event_id", "profile_id", "spawn_ms", "release_ms", "impact_ms", "cleanup_ms"}
    missing = required - set(event)
    if missing:
        raise ValueError(f"Combat event is missing fields: {sorted(missing)}")
    times = [event[key] for key in ("spawn_ms", "release_ms", "impact_ms", "cleanup_ms")]
    if not all(isinstance(value, int) for value in times) or times != sorted(times):
        raise ValueError("Combat event timestamps must be ordered integers")
    interrupted_ms = event.get("interrupted_ms")
    if interrupted_ms is not None and (not isinstance(interrupted_ms, int) or interrupted_ms < event["spawn_ms"]):
        raise ValueError("Combat event interruption timestamp is invalid")
    if interrupted_ms is not None and now_ms >= interrupted_ms:
        return {"event_id": event["event_id"], "state": "interrupted", "active": False, "phase": 1.0}
    if now_ms < event["release_ms"]:
        span = max(1, event["release_ms"] - event["spawn_ms"])
        phase = (now_ms - event["spawn_ms"]) / span
        state = "staged"
    elif now_ms < event["impact_ms"]:
        span = max(1, event["impact_ms"] - event["release_ms"])
        phase = (now_ms - event["release_ms"]) / span
        state = "flight"
    elif now_ms < event["cleanup_ms"]:
        span = max(1, event["cleanup_ms"] - event["impact_ms"])
        phase = (now_ms - event["impact_ms"]) / span
        state = "impact"
    else:
        return {"event_id": event["event_id"], "state": "complete", "active": False, "phase": 1.0}
    return {
        "event_id": event["event_id"],
        "state": state,
        "active": True,
        "phase": max(0.0, min(1.0, phase)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args(argv)
    try:
        contract = load_contract(args.contract)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps({"schema": contract["schema"], "profiles": sorted(contract["profiles"]), "runtime_status": contract["runtime_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
