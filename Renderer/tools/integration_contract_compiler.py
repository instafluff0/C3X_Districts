#!/usr/bin/env python3
"""Compile future renderer-category capture/cache/ownership preflight contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = RENDERER_ROOT / "integration/category_contracts.json"
DEFAULT_OUTPUT = RENDERER_ROOT / "preview/out/contracts/category_preflight.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def cache_key(contract: dict[str, Any], scene_record: dict[str, Any]) -> str:
    missing = [field for field in contract["cache_inputs"] if field not in scene_record]
    if missing:
        raise ValueError(f"scene record is missing cache inputs: {missing}")
    payload = [[field, scene_record[field]] for field in contract["cache_inputs"]]
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _different(value: Any) -> Any:
    if value is None:
        return "changed"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return value + "/changed"
    raise ValueError(f"cannot synthesize a changed fixture value for {value!r}")


def compile_contracts(source_path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema") != "c3x.category_contract_sources.v0":
        raise ValueError("unsupported category-contract source schema")
    global_policy = source.get("global", {})
    if global_policy.get("enabled_failure") != "custom_category_hard_failure_no_native_replay":
        raise ValueError("enabled custom-category failure must not replay native art")
    if global_policy.get("disabled_behavior") != "native_category_owns_complete_plane":
        raise ValueError("disabled categories must remain wholly native-owned")
    compiled = {}
    for category, contract in sorted(source.get("categories", {}).items()):
        fields = contract.get("scene_fields", [])
        cache_inputs = contract.get("cache_inputs", [])
        invalidation = contract.get("invalidation", [])
        sample = contract.get("sample", {})
        if len(fields) != len(set(fields)) or set(sample) != set(fields):
            raise ValueError(f"{category} scene fields and sample do not match exactly")
        if not set(cache_inputs) <= set(fields) or not set(invalidation) <= set(cache_inputs):
            raise ValueError(f"{category} cache/invalidation fields escape the scene contract")
        baseline = cache_key(contract, sample)
        invalidation_proofs = {}
        for field in invalidation:
            changed = deepcopy(sample)
            changed[field] = _different(changed[field])
            changed_key = cache_key(contract, changed)
            if changed_key == baseline:
                raise ValueError(f"{category}.{field} does not invalidate its cache key")
            invalidation_proofs[field] = changed_key
        compiled[category] = {
            "gate": contract["gate"],
            "plane": contract["plane"],
            "scene_fields": fields,
            "cache_key_recipe": [[index, field] for index, field in enumerate(cache_inputs)],
            "invalidation": invalidation,
            "fixture": sample,
            "fixture_cache_key": baseline,
            "invalidation_fixture_keys": invalidation_proofs,
            "ownership": {
                "enabled": "renderer_owns_every_visible_instance_atomically",
                "enabled_failure": global_policy["enabled_failure"],
                "disabled": global_policy["disabled_behavior"],
            },
            "retained_layers": global_policy["retained_layers"],
            "activation": "not_enabled",
        }
    return {
        "schema": "c3x.category_integration_preflight.v0",
        "categories": compiled,
        "runtime_activation": "none",
        "injected_code_changes": 0,
        "required_user_action": [],
    }


def write_contracts(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    value = compile_contracts(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = write_contracts(args.source, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Compiled {len(result['categories'])} inactive category contracts at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
