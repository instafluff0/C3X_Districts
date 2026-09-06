#!/usr/bin/env python3
"""Convert extracted resource FGX clips into source-independent C3X animations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACT_REPORT = RENDERER_ROOT / "preview" / "out" / "resources" / "resource_animation_extract.json"
DEFAULT_RAW_ROOT = RENDERER_ROOT / "preview" / "out" / "resources" / "raw_animations"
DEFAULT_PACK = RENDERER_ROOT / "packs" / "ResourceNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "resources" / "resource_animation_conversion.json"
DEFAULT_CONVERTER = Path(__file__).with_name("CONVERT_CIV6_ANIMATION.bat")


def _safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe resource-animation path: {value!r}")
    return Path(*path.parts)


def load_extract_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "c3x.resource_animation_extract.v0":
        raise ValueError("unsupported resource-animation extraction schema")
    clips = report.get("unique_clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("resource-animation extraction report contains no clips")
    seen = set()
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("resource-animation clip record is invalid")
        key = (clip.get("source_package"), clip.get("table_index"))
        if key in seen:
            raise ValueError("resource-animation extraction report repeats a package table entry")
        seen.add(key)
        _safe_relative(clip.get("raw_fgx", ""))
        _safe_relative(clip.get("normalized_clip", ""))
        scale = clip.get("translation_scale")
        if not isinstance(scale, (int, float)) or not 0.0 < scale <= 1.0:
            raise ValueError("resource-animation translation scale is invalid")
    return report


def convert_resource_animations(
    extract_report: Path,
    raw_root: Path,
    pack: Path,
    report_path: Path,
    converter: Path,
    force: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    if os.name != "nt" and not validate_only:
        raise ValueError("resource FGX conversion must run on Windows with CivNexus6")
    source = load_extract_report(extract_report)
    outputs = []
    for clip in source["unique_clips"]:
        raw = raw_root / _safe_relative(clip["raw_fgx"])
        target = pack / _safe_relative(clip["normalized_clip"])
        if not raw.is_file():
            raise FileNotFoundError(raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        converted = not validate_only and (force or not target.is_file())
        if not validate_only and converted:
            subprocess.run(
                [
                    "cmd.exe",
                    "/d",
                    "/c",
                    str(converter),
                    str(raw),
                    str(target),
                    format(clip["translation_scale"], ".12g"),
                ],
                check=True,
            )
        loaded = normalized_animation.load_clip(target)
        outputs.append(
            {
                "source_package": clip["source_package"],
                "source_name": clip["name"],
                "table_index": clip["table_index"],
                "clip": clip["normalized_clip"],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "duration": loaded.duration,
                "sample_rate": loaded.sample_rate,
                "frame_count": loaded.frame_count,
                "track_groups": len(loaded.groups),
                "tracks": sum(len(group.tracks) for group in loaded.groups),
                "converted_this_run": converted,
                "binding_status": (
                    "model_aware_pose_cache_required"
                    if clip["source_package"] == "environment/clutter"
                    else "normalized_clip"
                ),
            }
        )
    report = {
        "schema": "c3x.resource_animation_conversion.v0",
        "clips": outputs,
        "summary": {
            "clips": len(outputs),
            "converted_this_run": sum(item["converted_this_run"] for item in outputs),
            "frames": sum(item["frame_count"] for item in outputs),
            "tracks": sum(item["tracks"] for item in outputs),
            "body_profiles_pending": sum(
                item["binding_status"] == "model_aware_pose_cache_required" for item in outputs
            ),
        },
        "runtime_integration": "not_enabled",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-report", type=Path, default=DEFAULT_EXTRACT_REPORT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate already-converted clips and write the conversion report",
    )
    args = parser.parse_args(argv)
    try:
        report = convert_resource_animations(
            args.extract_report,
            args.raw_root,
            args.pack,
            args.report,
            args.converter,
            args.force,
            args.validate_only,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
