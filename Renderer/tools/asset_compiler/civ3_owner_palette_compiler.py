#!/usr/bin/env python3
"""Compile Civ III ntp00..ntp31 PCX palettes into a generic owner-color LUT."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Iterable


TABLE_COUNT = 32
COLORS_PER_TABLE = 64
PRIMARY_RAMP_START = 0
PRIMARY_RAMP_COUNT = 16
DISPLAY_COLOR_INDEX = 6


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def read_pcx_palette(path: Path) -> list[list[int]]:
    """Read the 256-entry RGB palette from an 8-bit, single-plane PCX."""
    data = path.read_bytes()
    if len(data) < 128 + 1 + 769 or data[0] != 0x0A:
        raise ValueError(f"not a complete PCX file: {path}")
    if data[3] != 8 or data[65] != 1:
        raise ValueError(f"owner palette PCX must be 8-bit and single-plane: {path}")
    palette_marker = len(data) - 769
    if data[palette_marker] != 12:
        raise ValueError(f"PCX has no trailing 256-color palette marker: {path}")
    palette = data[palette_marker + 1 :]
    if len(palette) != 768:
        raise ValueError(f"PCX palette is truncated: {path}")
    return [list(palette[index : index + 3]) for index in range(0, 768, 3)]


def _casefold_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"palette root is not a directory: {root}")
    result: dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        folded = path.name.casefold()
        if folded in result:
            raise ValueError(f"palette root contains case-insensitive duplicate {path.name}: {root}")
        result[folded] = path
    return result


def resolve_palette_sources(roots: Iterable[Path]) -> list[dict[str, Any]]:
    """Resolve low-to-high precedence roots with per-file scenario fallback."""
    indexed_roots = [(root, _casefold_files(root)) for root in roots]
    if not indexed_roots:
        raise ValueError("at least one --palette-root is required")
    resolved = []
    for table_id in range(TABLE_COUNT):
        filename = f"ntp{table_id:02d}.pcx"
        candidates = [
            (priority, root, files[filename])
            for priority, (root, files) in enumerate(indexed_roots)
            if filename in files
        ]
        if not candidates:
            raise ValueError(f"no palette root supplies {filename}")
        priority, root, path = candidates[-1]
        resolved.append(
            {
                "color_table_id": table_id,
                "filename": filename,
                "source_root": str(root),
                "source_path": str(path),
                "source_priority": priority,
                "overrode_lower_priority": len(candidates) > 1,
            }
        )
    return resolved


def compile_owner_palettes(roots: Iterable[Path], output: Path, report_path: Path | None = None) -> dict[str, Any]:
    sources = resolve_palette_sources(roots)
    tables = []
    rgba = bytearray()
    provenance = []
    for source in sources:
        path = Path(source["source_path"])
        source_bytes = path.read_bytes()
        colors = read_pcx_palette(path)[:COLORS_PER_TABLE]
        for color in colors:
            rgba.extend((*color, 255))
        table_id = source["color_table_id"]
        tables.append(
            {
                "color_table_id": table_id,
                "logical_id": f"owner-color/civ3/{table_id:02d}",
                "colors": colors,
                "display_color": colors[DISPLAY_COLOR_INDEX],
                "primary_ramp": colors[PRIMARY_RAMP_START : PRIMARY_RAMP_START + PRIMARY_RAMP_COUNT],
            }
        )
        provenance.append(
            {
                **source,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    lut_path = output / "owner_colors.rgba8"
    table_path = output / "owner_colors.json"
    manifest_path = output / "manifest.json"
    lut_path.write_bytes(rgba)
    table_document = {
        "schema": "c3x.owner_color_tables.v0",
        "table_count": TABLE_COUNT,
        "colors_per_table": COLORS_PER_TABLE,
        "primary_ramp": {"start": PRIMARY_RAMP_START, "count": PRIMARY_RAMP_COUNT},
        "display_color_index": DISPLAY_COLOR_INDEX,
        "tables": tables,
    }
    table_path.write_text(_canonical_json(table_document), encoding="utf-8")
    manifest = {
        "schema": "c3x.owner_color_pack.v0",
        "source_independent": True,
        "color_tables": "owner_colors.json",
        "gpu_lut": {
            "path": "owner_colors.rgba8",
            "format": "rgba8_unorm_srgb",
            "width": COLORS_PER_TABLE,
            "height": TABLE_COUNT,
            "row_semantic": "color_table_id",
            "column_semantic": "civ3_palette_index_0_to_63",
            "sha256": hashlib.sha256(rgba).hexdigest(),
        },
    }
    manifest_path.write_text(_canonical_json(manifest), encoding="utf-8")
    report = {
        "schema": "c3x.owner_color_compile_report.v0",
        "pack": str(output),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "resolved_sources": provenance,
        "scenario_override_count": sum(item["overrode_lower_priority"] for item in provenance),
        "evidence": {
            "civ3_runtime": "Units_Image_Data loads Art/Units/Palettes/ntp00.pcx through ntp31.pcx; FLC recoloring copies entries 0..63 selected by Leader.Color_Table_ID",
            "primary_ramp": "Civ3FlcEdit identifies palette index 6 as CIV_COLOR; entries 0..15 form the coherent light-to-dark owner ramp",
        },
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_canonical_json(report), encoding="utf-8")
    return report


def load_owner_color_table(pack: Path, color_table_id: int) -> dict[str, Any]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.owner_color_pack.v0":
        raise ValueError("owner-color pack has the wrong schema")
    lut = manifest.get("gpu_lut", {})
    if (
        lut.get("format") != "rgba8_unorm_srgb"
        or lut.get("width") != COLORS_PER_TABLE
        or lut.get("height") != TABLE_COUNT
    ):
        raise ValueError("owner-color GPU LUT contract is invalid")
    lut_bytes = (pack / lut["path"]).read_bytes()
    if len(lut_bytes) != TABLE_COUNT * COLORS_PER_TABLE * 4:
        raise ValueError("owner-color GPU LUT byte size is invalid")
    if hashlib.sha256(lut_bytes).hexdigest() != lut.get("sha256"):
        raise ValueError("owner-color GPU LUT hash does not match its manifest")
    document = json.loads((pack / manifest["color_tables"]).read_text(encoding="utf-8"))
    if document.get("schema") != "c3x.owner_color_tables.v0" or len(document.get("tables", [])) != TABLE_COUNT:
        raise ValueError("owner-color table document is incomplete")
    if not 0 <= color_table_id < TABLE_COUNT:
        raise ValueError(f"color table ID must be 0..{TABLE_COUNT - 1}")
    table = document["tables"][color_table_id]
    if table.get("color_table_id") != color_table_id or len(table.get("colors", [])) != COLORS_PER_TABLE:
        raise ValueError("owner-color table ordering or size is invalid")
    ramp = document.get("primary_ramp", {})
    expected_ramp = table["colors"][ramp.get("start", -1) : ramp.get("start", -1) + ramp.get("count", -1)]
    if ramp != {"start": PRIMARY_RAMP_START, "count": PRIMARY_RAMP_COUNT} or table.get("primary_ramp") != expected_ramp:
        raise ValueError("owner-color primary ramp contract is invalid")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--palette-root",
        type=Path,
        action="append",
        required=True,
        help="directory containing ntp*.pcx; repeat in low-to-high precedence order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compile_owner_palettes(args.palette_root, args.output, args.report)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Compiled {TABLE_COUNT} Civ III owner-color tables to {args.output} "
        f"({report['scenario_override_count']} higher-priority overrides)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
