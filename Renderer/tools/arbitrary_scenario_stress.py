#!/usr/bin/env python3
"""Validate arbitrary scenario IDs against future category ownership contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.tools.integration_contract_compiler import compile_contracts


RENDERER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = RENDERER_ROOT / "samples/contracts/arbitrary_scenario_stress.json"
DEFAULT_OUTPUT = RENDERER_ROOT / "preview/out/contracts/arbitrary_scenario_stress_result.json"


def _resolve(enabled: set[str], rules: dict[tuple[str, str], str], case: dict[str, Any]) -> dict[str, Any]:
    category = case["category"]
    if category not in enabled:
        return {"outcome": "native", "asset": None}
    if case["visibility"] == "hidden":
        return {"outcome": "omitted", "asset": None}
    asset = rules.get((category, case["selector"]))
    if asset is None:
        return {"outcome": "hard_failure_no_native_replay", "asset": None}
    return {"outcome": "custom", "asset": asset}


def run_stress(fixture_path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    raw = fixture_path.read_bytes()
    fixture = json.loads(raw)
    if fixture.get("schema") != "c3x.arbitrary_scenario_stress.v0":
        raise ValueError("unsupported arbitrary-scenario stress schema")
    contracts = compile_contracts()["categories"]
    enabled = set(fixture.get("enabled_categories", []))
    if not enabled <= set(contracts):
        raise ValueError("stress fixture enables an unknown category")
    palette_rows = fixture.get("owner_palette_rows", {})
    if "31" not in palette_rows or len(palette_rows["31"]) != 4:
        raise ValueError("stress fixture does not exercise an arbitrary owner palette row")
    rules = {}
    for rule in fixture.get("rules", []):
        key = (rule["category"], rule["selector"])
        if key in rules or rule["category"] not in contracts:
            raise ValueError("stress fixture contains a duplicate or unknown rule")
        if not rule["selector"].startswith(("resource/scenario_", "city_style/scenario_", "mine/scenario_", "farm/scenario_", "tile_object/scenario_", "unit/scenario_", "infrastructure/scenario_")):
            raise ValueError("stress fixture selector is not demonstrably scenario-defined")
        rules[key] = rule["asset"]

    results = []
    seen = set()
    for case in fixture.get("cases", []):
        if case["case_id"] in seen:
            raise ValueError("stress fixture repeats a case id")
        seen.add(case["case_id"])
        if case["owner_palette_id"] != 31:
            raise ValueError("stress fixture bypasses the arbitrary palette proof")
        resolution = _resolve(enabled, rules, case)
        if resolution["outcome"] != case["expected"]:
            raise ValueError(f"{case['case_id']} resolved to {resolution['outcome']}, expected {case['expected']}")
        results.append({**case, **resolution})

    override = fixture["partial_override"]
    key = (override["category"], override["selector"])
    before = dict(rules)
    after = dict(rules)
    after[key] = override["replacement_asset"]
    changed = sorted(rule for rule in after if before.get(rule) != after.get(rule))
    if changed != [key]:
        raise ValueError("partial scenario override changed more than its selected rule")
    outcomes = {result["outcome"] for result in results}
    required = {"custom", "omitted", "hard_failure_no_native_replay", "native"}
    if outcomes != required:
        raise ValueError("stress fixture does not cover every ownership outcome")
    return {
        "schema": "c3x.arbitrary_scenario_stress_result.v0",
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "scenario_id": fixture["scenario_id"],
        "cases": results,
        "partial_override_changed_rules": [list(value) for value in changed],
        "summary": {
            "cases": len(results),
            "custom": sum(result["outcome"] == "custom" for result in results),
            "omitted": sum(result["outcome"] == "omitted" for result in results),
            "hard_failures": sum(result["outcome"] == "hard_failure_no_native_replay" for result in results),
            "native_disabled": sum(result["outcome"] == "native" for result in results),
            "arbitrary_palette_row": 31,
        },
        "runtime_activation": "none",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = run_stress(args.fixture)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Passed {result['summary']['cases']} arbitrary-scenario ownership cases at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
