#!/usr/bin/env python3
"""Generate deterministic future-gate Lab fixture matrices and draft handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = Path(__file__).with_name("lab_promotion_profiles.json")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def load_profiles(path: Path = DEFAULT_PROFILES) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "c3x.lab_promotion_profiles.v0":
        raise ValueError("unsupported Lab promotion-profile schema")
    fixture = value.get("fixture", {})
    if fixture.get("width_tiles", 0) * fixture.get("height_tiles", 0) != fixture.get("tile_count"):
        raise ValueError("Lab promotion fixture dimensions do not match its tile count")
    if fixture.get("tile_count") != 192:
        raise ValueError("future Lab promotion profiles require the canonical 192-tile fixture")
    zooms = value.get("zooms")
    if [zoom.get("id") for zoom in zooms or []] != ["normal", "reduced"]:
        raise ValueError("future Lab promotion profiles require normal and reduced zoom")
    for gate, profile in value.get("profiles", {}).items():
        if gate == "L15" or not gate.startswith("L"):
            raise ValueError("promotion-kit profiles must not replace the active L15 workflow")
        if not profile.get("category") or not profile.get("selectors") or not profile.get("states"):
            raise ValueError(f"{gate} promotion profile is incomplete")
    return value


def build_promotion_kit(gate: str, profiles_path: Path = DEFAULT_PROFILES) -> dict[str, Any]:
    source = load_profiles(profiles_path)
    if gate not in source["profiles"]:
        raise ValueError(f"unknown future Lab gate {gate}")
    profile = source["profiles"][gate]
    cases = [
        {
            "case_id": f"{gate.lower()}_{zoom['id']}_{selector}_{state}",
            "zoom": zoom["id"],
            "scale_percent": zoom["scale_percent"],
            "selector": selector,
            "state": state,
        }
        for zoom in source["zooms"]
        for selector in profile["selectors"]
        for state in profile["states"]
    ]
    fixture = source["fixture"]
    tiles = []
    for index in range(fixture["tile_count"]):
        case = cases[index % len(cases)]
        tiles.append(
            {
                "tile_index": index,
                "map_x": index % fixture["width_tiles"],
                "map_y": index // fixture["width_tiles"],
                "case_id": case["case_id"],
                "stable_instance_seed": index,
            }
        )
    kit = {
        "schema": "c3x.lab_promotion_kit.v0",
        "gate": gate,
        "category": profile["category"],
        "fixture": {**fixture, "tiles": tiles},
        "cases": cases,
        "render_variants": ["complete", "category_only", "without_category", "thumbnail"],
        "required_evidence": [
            "standalone_render",
            "normal_and_reduced_zoom_contact_sheet",
            "category_isolation_render",
            "without_category_comparison",
            "automated_visual_metrics",
            "mapping_coverage_report",
        ],
        "handoff_status": "draft_not_approved",
        "promotion_authority": "renderer_lab_only",
        "integration_authority": "not_transferred",
    }
    kit["matrix_sha256"] = hashlib.sha256(_canonical(kit)).hexdigest()
    return kit


def write_promotion_kit(gate: str, output: Path, profiles_path: Path = DEFAULT_PROFILES) -> dict[str, Any]:
    kit = build_promotion_kit(gate, profiles_path)
    matrix = output / "promotion_matrix.json"
    handoff = output / "handoff_draft.json"
    _write(matrix, kit)
    _write(
        handoff,
        {
            "schema": "c3x.lab_handoff_draft.v0",
            "gate": gate,
            "category": kit["category"],
            "status": "draft_not_approved",
            "matrix": matrix.name,
            "matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
            "artifact_hashes": {},
            "known_limitations": [],
            "approval": None,
        },
    )
    return kit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    args = parser.parse_args()
    output = args.output or RENDERER_ROOT / f"preview/out/promotion_kits/{args.gate.lower()}"
    try:
        kit = write_promotion_kit(args.gate, output, args.profiles)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Generated {args.gate} {kit['category']} Lab kit with {len(kit['cases'])} cases at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
