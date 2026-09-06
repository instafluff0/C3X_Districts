"""Deterministic renderer-definition rule matching and diagnostics.

This module operates only on the validated catalog produced by
``definition_parser.merge_layers``.  It deliberately does not open asset files:
availability is supplied as asset identifiers so selection can be tested before
the asset compiler or renderer exists.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


SCHEMA = "c3x.renderer_rule_resolution.v0"

CATEGORY_PROFILE_KEYS = {
    "terrain": "terrain",
    "feature": "features",
    "road": "roads",
    "river": "rivers",
    "improvement": "improvements",
    "resource": "resources",
    "city": "cities",
    "unit": "units",
    "effect": "effects",
}

RESULT_FIELDS = {
    "asset",
    "animation",
    "category",
    "disabled",
    "priority",
    "replacement",
    "variant_selection",
}

SEASON_ALIASES = {"autumn": "fall"}


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        value = value.casefold()
        return SEASON_ALIASES.get(value, value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return sorted(value, key=str)
    return value


def _match_hours(expected: list[Any], actual: Any) -> tuple[bool, str]:
    if actual is None:
        return False, "metadata has no hour"
    if isinstance(actual, bool) or not isinstance(actual, int) or not 0 <= actual <= 23:
        return False, "metadata hour is not an integer from 0 through 23"

    for entry in expected:
        if isinstance(entry, int) and actual == entry:
            return True, "hour is listed"
        if isinstance(entry, Mapping):
            start = entry["start"]
            end = entry["end"]
            if start <= end and start <= actual <= end:
                return True, "hour is inside range"
            if start > end and (actual >= start or actual <= end):
                return True, "hour is inside wrapped range"
    return False, "hour is outside every listed hour or range"


def _match_selector(key: str, expected: Any, metadata: Mapping[str, Any]) -> tuple[bool, Any, str]:
    if key == "show_in_day_night_hours":
        actual = metadata.get("hour")
        matched, reason = _match_hours(expected, actual)
        return matched, actual, reason

    if key == "show_in_seasons":
        actual = metadata.get("season")
        if actual is None:
            return False, actual, "metadata has no season"
        matched = _normalized(actual) in {_normalized(value) for value in expected}
        return matched, actual, "season is listed" if matched else "season is not listed"

    if key not in metadata:
        return False, None, f"metadata has no {key}"

    actual = metadata[key]
    if key == "adjacent_to" and isinstance(actual, (list, tuple, set, frozenset)):
        matched = _normalized(expected) in {_normalized(value) for value in actual}
        return matched, _json_value(actual), "adjacency is present" if matched else "adjacency is absent"

    matched = _normalized(actual) == _normalized(expected)
    return matched, _json_value(actual), "values match" if matched else "values differ"


def _rank(rule: Mapping[str, Any]) -> dict[str, int]:
    values = rule["values"]
    selector_count = sum(key not in RESULT_FIELDS for key in values)
    source = rule["source"]
    return {
        "priority": values.get("priority", 0),
        "specificity": selector_count,
        "layer": source["layer_index"],
        "declaration": source["declaration_index"],
    }


def _rank_tuple(rank: Mapping[str, int]) -> tuple[int, int, int, int]:
    return rank["priority"], rank["specificity"], rank["layer"], rank["declaration"]


def _loser_reason(loser: Mapping[str, int], winner: Mapping[str, int]) -> str:
    labels = (
        ("priority", "lower_priority"),
        ("specificity", "lower_specificity"),
        ("layer", "lower_layer_precedence"),
        ("declaration", "earlier_declaration"),
    )
    for field, reason in labels:
        if loser[field] != winner[field]:
            return reason
    return "equal_rank"


def coordinate_variant_seed(rule_id: str, map_x: int, map_y: int, world_seed: Any) -> int:
    """Return a stable 64-bit variant seed independent of frame/enumeration order."""

    material = f"{rule_id}\0{map_x}\0{map_y}\0{world_seed}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big")


def _fallback(
    category: Any,
    reason: str,
    *,
    candidates: list[dict[str, Any]] | None = None,
    policy: str | None = None,
    winner: dict[str, Any] | None = None,
    asset_checks: int = 0,
) -> dict[str, Any]:
    fallback = {"action": "civ3", "reason": reason}
    if policy is not None:
        fallback["policy"] = policy
    return {
        "schema": SCHEMA,
        "status": "fallback",
        "category": category,
        "winner": winner,
        "candidates": candidates or [],
        "fallback": fallback,
        "asset_payload_loads": 0,
        "asset_availability_checks": asset_checks,
    }


def resolve_rule(
    catalog: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    profile_id: str = "default",
    enabled: bool = True,
    world_seed: Any = 0,
    available_assets: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Resolve one renderable item and return a complete selection explanation.

    ``available_assets`` is an identifier inventory, not a payload loader.  When
    omitted, every declared asset is considered available.
    """

    category = _normalized(metadata.get("category"))
    if not enabled:
        return _fallback(category, "config_off")

    profiles = {entry["id"]: entry for entry in catalog.get("profiles", [])}
    profile = profiles.get(profile_id)
    if profile is None:
        return _fallback(category, "profile_not_found")

    profile_values = profile["values"]
    profile_key = CATEGORY_PROFILE_KEYS.get(category)
    if profile_key is None:
        return _fallback(category, "unknown_category")

    ownership = profile_values.get(profile_key, "civ3")
    if ownership == "civ3":
        return _fallback(category, "category_owned_by_civ3")
    if ownership == "capture-only":
        return _fallback(category, "category_capture_only")

    candidates: list[dict[str, Any]] = []
    matched: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for rule in catalog.get("rules", []):
        values = rule["values"]
        candidate: dict[str, Any] = {
            "rule_id": rule["id"],
            "source": rule["source"],
            "rank": _rank(rule),
            "matched_selectors": [],
            "failed_selectors": [],
        }

        if _normalized(values.get("category")) != category:
            candidate["status"] = "rejected"
            candidate["failed_selectors"].append(
                {
                    "key": "category",
                    "expected": values.get("category"),
                    "actual": metadata.get("category"),
                    "reason": "category differs",
                }
            )
            candidates.append(candidate)
            continue

        for key, expected in values.items():
            if key in RESULT_FIELDS:
                continue
            selector_matched, actual, reason = _match_selector(key, expected, metadata)
            detail = {"key": key, "expected": expected, "actual": actual, "reason": reason}
            if selector_matched:
                candidate["matched_selectors"].append(detail)
            else:
                candidate["failed_selectors"].append(detail)

        if candidate["failed_selectors"]:
            candidate["status"] = "rejected"
        else:
            candidate["status"] = "matched"
            matched.append((candidate, rule))
        candidates.append(candidate)

    if not matched:
        return _fallback(category, "no_matching_rule", candidates=candidates)

    matched.sort(key=lambda pair: _rank_tuple(pair[0]["rank"]))
    winning_candidate, winning_rule = matched[-1]
    winning_candidate["status"] = "winner"
    for candidate, _rule in matched[:-1]:
        candidate["status"] = "matched_loser"
        candidate["loser_reason"] = _loser_reason(candidate["rank"], winning_candidate["rank"])

    values = winning_rule["values"]
    winner: dict[str, Any] = {
        "rule_id": winning_rule["id"],
        "asset_id": values["asset"],
        "animation": values.get("animation"),
        "replacement": values.get("replacement", ownership),
        "rank": winning_candidate["rank"],
        "source": winning_rule["source"],
        "variant": None,
    }

    if values.get("variant_selection") == "coordinate-hash":
        map_x = metadata.get("map_x")
        map_y = metadata.get("map_y")
        if isinstance(map_x, bool) or not isinstance(map_x, int) or isinstance(map_y, bool) or not isinstance(map_y, int):
            return _fallback(
                category,
                "missing_variant_coordinates",
                candidates=candidates,
                winner=winner,
            )
        winner["variant"] = {
            "method": "coordinate-hash",
            "seed": coordinate_variant_seed(winning_rule["id"], map_x, map_y, world_seed),
        }

    declared_assets = {entry["id"] for entry in catalog.get("assets", [])}
    available = declared_assets if available_assets is None else set(available_assets)
    if winner["asset_id"] not in available:
        return _fallback(
            category,
            "missing_asset",
            candidates=candidates,
            policy=profile_values.get("missing_asset", "fallback"),
            winner=winner,
            asset_checks=1,
        )

    return {
        "schema": SCHEMA,
        "status": "matched",
        "category": category,
        "winner": winner,
        "candidates": candidates,
        "fallback": None,
        "asset_payload_loads": 0,
        "asset_availability_checks": 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain one C3X renderer rule selection")
    parser.add_argument("catalog", type=Path, help="Merged definition catalog JSON")
    parser.add_argument("metadata", type=Path, help="Captured item metadata JSON")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--world-seed", default="0")
    parser.add_argument("--config-off", action="store_true")
    parser.add_argument("--available-assets", type=Path, help="Optional JSON list of available asset IDs")
    args = parser.parse_args(argv)
    try:
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        available = None
        if args.available_assets:
            available = json.loads(args.available_assets.read_text(encoding="utf-8"))
        result = resolve_rule(
            catalog,
            metadata,
            profile_id=args.profile,
            enabled=not args.config_off,
            world_seed=args.world_seed,
            available_assets=available,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"rule resolution failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
