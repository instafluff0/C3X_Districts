#!/usr/bin/env python3
"""Catalog exact normalized model sockets and their unresolved Light/VFX identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS = [
    RENDERER_ROOT / "preview/out/cities/build.json",
    RENDERER_ROOT / "preview/out/improvements/build.json",
    RENDERER_ROOT / "preview/out/tile_objects/build.json",
    RENDERER_ROOT / "preview/out/future_gate_candidates/build.json",
]
DEFAULT_OUTPUT = RENDERER_ROOT / "preview/out/ambient_effects/attachment_identities.json"


def _identity_kind(name: str, semantic: str) -> str:
    lowered = name.lower()
    if lowered.startswith("light") or semantic == "night_light":
        return "analytic_light_candidate"
    if lowered.startswith("fx") or semantic in {"flame", "smoke"}:
        return "vfx_resource_candidate"
    return "component_or_unknown_resource"


def compile_attachment_identities(reports: list[Path]) -> dict[str, Any]:
    identities: dict[str, dict[str, Any]] = {}
    input_hashes = []
    for path in reports:
        data = path.read_bytes()
        report = json.loads(data)
        input_hashes.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        for asset in report.get("assets", []):
            asset_id = asset.get("asset_id") or asset.get("normalized_asset_id") or "unknown"
            for point in asset.get("attachments", {}).get("points", []):
                name = point.get("source_name")
                if not isinstance(name, str) or not name:
                    continue
                identity = identities.setdefault(
                    name,
                    {
                        "source_identity": name,
                        "kind": _identity_kind(name, point.get("semantic", "")),
                        "decoder_status": "resource_graph_pending",
                        "bindings": [],
                    },
                )
                identity["bindings"].append(
                    {
                        "asset": asset_id,
                        "socket": point.get("id"),
                        "bone": point.get("bone"),
                        "skeleton": point.get("skeleton"),
                        "semantic": point.get("semantic"),
                        "state_hint": point.get("state_hint"),
                        "transform": point.get("bone_local_transform"),
                    }
                )
    for identity in identities.values():
        identity["bindings"].sort(key=lambda value: (value["asset"], value["socket"] or ""))
    counts = {
        kind: sum(value["kind"] == kind for value in identities.values())
        for kind in ("analytic_light_candidate", "vfx_resource_candidate", "component_or_unknown_resource")
    }
    return {
        "schema": "c3x.attachment_identity_catalog.v0",
        "inputs": input_hashes,
        "identities": [identities[key] for key in sorted(identities)],
        "summary": {"identities": len(identities), **counts},
        "runtime_binding": "not_enabled",
        "interpretation": "socket/name/transform identity is confirmed; resource script behavior remains pending",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reports = args.report or DEFAULT_REPORTS
    try:
        result = compile_attachment_identities(reports)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Cataloged {result['summary']['identities']} exact attachment identities at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
