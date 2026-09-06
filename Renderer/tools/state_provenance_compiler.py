#!/usr/bin/env python3
"""Cross-check category scene fields against authoritative Civ III/C3X provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = RENDERER_ROOT / "integration/category_contracts.json"
DEFAULT_PROVENANCE = RENDERER_ROOT / "integration/state_provenance.json"
DEFAULT_OUTPUT = RENDERER_ROOT / "preview/out/contracts/state_provenance_audit.json"
STATUSES = {"confirmed_existing_capture", "confirmed_existing_accessor", "derived_from_confirmed_fields", "gate_audit_required"}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_state_provenance(
    contracts_path: Path = DEFAULT_CONTRACTS,
    provenance_path: Path = DEFAULT_PROVENANCE,
) -> dict[str, Any]:
    contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if contracts.get("schema") != "c3x.category_contract_sources.v0":
        raise ValueError("unsupported category contract schema")
    if provenance.get("schema") != "c3x.category_state_provenance_sources.v0":
        raise ValueError("unsupported state provenance schema")
    if provenance.get("required_user_action") != []:
        raise ValueError("offline provenance must not contain an unproved patch request")
    common = provenance.get("common", {})
    category_sources = provenance.get("categories", {})
    if set(category_sources) != set(contracts.get("categories", {})):
        raise ValueError("provenance categories do not exactly match integration categories")
    compiled = {}
    audits = []
    symbols = set()
    for category, contract in sorted(contracts["categories"].items()):
        overrides = category_sources[category]
        unknown = set(overrides) - set(contract["scene_fields"])
        if unknown:
            raise ValueError(f"{category} provenance contains unknown fields: {sorted(unknown)}")
        fields = {}
        for field in contract["scene_fields"]:
            record = overrides.get(field, common.get(field))
            if not isinstance(record, dict):
                raise ValueError(f"{category}.{field} has no provenance")
            if record.get("status") not in STATUSES or not isinstance(record.get("source"), str):
                raise ValueError(f"{category}.{field} has invalid provenance")
            field_symbols = record.get("symbols", [])
            if not isinstance(field_symbols, list) or not all(isinstance(value, str) and value for value in field_symbols):
                raise ValueError(f"{category}.{field} has invalid symbol evidence")
            symbols.update(field_symbols)
            fields[field] = record
            if record["status"] == "gate_audit_required":
                if not isinstance(record.get("audit"), str) or not record["audit"]:
                    raise ValueError(f"{category}.{field} audit has no bounded question")
                audits.append({"gate": contract["gate"], "category": category, "field": field, "question": record["audit"], "candidate_symbols": field_symbols})
        compiled[category] = {
            "gate": contract["gate"],
            "fields": fields,
            "coverage": {"declared": len(contract["scene_fields"]), "proven": len(fields)},
            "unresolved_audits": sum(record["status"] == "gate_audit_required" for record in fields.values()),
        }
    return {
        "schema": "c3x.category_state_provenance_audit.v0",
        "inputs": {"contracts_sha256": _hash(contracts_path), "provenance_sha256": _hash(provenance_path)},
        "categories": compiled,
        "audit_queue": audits,
        "existing_or_candidate_symbols": sorted(symbols),
        "summary": {
            "categories": len(compiled),
            "scene_fields": sum(value["coverage"]["declared"] for value in compiled.values()),
            "covered_fields": sum(value["coverage"]["proven"] for value in compiled.values()),
            "gate_audits": len(audits),
            "new_patch_requests": 0,
        },
        "required_user_action": [],
        "runtime_activation": "none",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = compile_state_provenance(args.contracts, args.provenance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        f"Covered {result['summary']['covered_fields']}/{result['summary']['scene_fields']} scene fields; "
        f"queued {result['summary']['gate_audits']} bounded gate audits; no patch request"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
