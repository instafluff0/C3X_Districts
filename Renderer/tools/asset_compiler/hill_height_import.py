#!/usr/bin/env python3
"""Import a loose authored R8 hill heightmap into a generic C3X DDS.

This is an offline source adapter. The renderer consumes only the normalized
DDS written below and never depends on the source game, mod, or DDS dialect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.terrain_relief_builder import make_r8_dds


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/workshop/content/289070/2361535448"
    / "Textures/Hills_Bump_Test.dds"
)
DEFAULT_OUTPUT = ROOT / "Renderer/packs/HillierHillsSource/height.dds"


def decode_r8_dds(raw: bytes) -> tuple[int, int, bytes, str]:
    if len(raw) < 128 or raw[:4] != b"DDS " or struct.unpack_from("<I", raw, 4)[0] != 124:
        raise ValueError("Source is not a supported DDS")
    height, width = struct.unpack_from("<2I", raw, 12)
    if width < 4 or height < 4 or width > 4096 or height > 4096:
        raise ValueError("R8 dimensions are invalid")
    if struct.unpack_from("<I", raw, 76)[0] != 32:
        raise ValueError("DDS pixel-format header is invalid")
    fourcc = raw[84:88]
    if fourcc == b"DX10":
        if len(raw) < 148 or struct.unpack_from("<I", raw, 128)[0] != 61:
            raise ValueError("DX10 source must use R8_UNORM")
        offset = 148
        dialect = "dx10-r8-unorm"
    else:
        bits = struct.unpack_from("<I", raw, 88)[0]
        masks = struct.unpack_from("<4I", raw, 92)
        if fourcc != b"\0\0\0\0" or bits != 8 or masks != (0xff, 0, 0, 0):
            raise ValueError("Legacy source must be an unpacked 8-bit single-channel DDS")
        offset = 128
        dialect = "legacy-r8"
    size = width * height
    if len(raw) != offset + size:
        raise ValueError("R8 DDS payload size does not match its dimensions")
    return width, height, raw[offset:], dialect


def import_height(source: Path, output: Path) -> dict[str, Any]:
    raw = source.read_bytes()
    width, height, pixels, dialect = decode_r8_dds(raw)
    normalized = make_r8_dds(width, height, pixels)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(normalized)
    report = {
        "schema": "c3x.local_hill_height_import.v0",
        "source_kind": "loose-authored-r8-heightmap",
        "source_dialect": dialect,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": hashlib.sha256(normalized).hexdigest(),
        "width": width,
        "height": height,
        "minimum": min(pixels),
        "maximum": max(pixels),
        "redistribution": "local-only",
        "runtime_source_independent": True,
    }
    report_path = output.with_name("import.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = import_height(args.source, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(
        f"Imported {report['width']}x{report['height']} local hill heightmap: "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
