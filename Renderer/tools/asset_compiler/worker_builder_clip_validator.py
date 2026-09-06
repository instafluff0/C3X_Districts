#!/usr/bin/env python3
"""Validate converted Worker/Builder specialty clips without source formats."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation
from Renderer.tools.asset_compiler.worker_builder_action_compiler import DEFAULT_STRATEGY


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK = RENDERER_ROOT / "packs/WorkerBuilderLab"
DEFAULT_REPORT = RENDERER_ROOT / "preview/out/units/worker_builder_clip_validation.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_clips(strategy_path: Path, pack: Path) -> dict[str, Any]:
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    records = {}
    for logical, definition in strategy["clips"].items():
        path = pack / definition["output"]
        clip = normalized_animation.load_clip(path)
        if clip.frame_count < 2 or clip.duration <= 0.0:
            raise ValueError(f"{logical} is not a usable motion clip")
        if len(clip.groups) != 1 or clip.groups[0].name != "Root":
            raise ValueError(f"{logical} does not have one Root track group")
        if len(clip.groups[0].tracks) < 30:
            raise ValueError(f"{logical} has too few humanoid tracks")
        records[logical] = {
            "path": definition["output"],
            "duration": clip.duration,
            "sample_rate": clip.sample_rate,
            "frame_count": clip.frame_count,
            "track_count": len(clip.groups[0].tracks),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema": "c3x.worker_builder_clip_validation.v0",
        "clips": records,
        "status": "validated_normalized_animation_payloads",
        "boundary": "model-aware skeleton binding and rendered tool/socket calibration remain L20 work",
        "runtime_integration": "not_enabled",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = validate_clips(args.strategy, args.pack)
        _write_json(args.report, report)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Validated {len(report['clips'])} normalized Worker/Builder specialty clips")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
