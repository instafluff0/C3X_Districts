#!/usr/bin/env python3
"""Extract verified Civ VI cliff-rock and polar-ice geometry for terrain labs.

The cooked-package decoding lives in ``clutter_blp_extractor`` because these
shore features use the same reflected static-mesh records.  This adapter keeps
the runtime pack source-agnostic and records source names only in the ignored
provenance report.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.clutter_blp_extractor import (
    StaticPackage,
    build_feature,
    default_blp_root,
    sha256_bytes,
)
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK = RENDERER_ROOT / "packs" / "ShoreNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "shore" / "shore_build.json"


def feature_spec(
    source_name: str, asset_id: str, manifest_key: str, stem: str, group: str
) -> dict[str, str]:
    return {
        "source_name": source_name,
        "asset_id": asset_id,
        "manifest_key": manifest_key,
        "stem": stem,
        "group": group,
    }


SHORE_SPECS = tuple(
    [
        feature_spec(
            f"TER_Cliffs_Rock{index:02d}",
            f"terrain.coast.cliff_large.{index:02d}",
            f"terrain/coast/cliff_large/{index:02d}",
            f"cliff_large_{index:02d}",
            "cliff_large",
        )
        for index in range(1, 5)
    ]
    + [
        feature_spec(
            f"TER_Cliffs_RockSmall{index:02d}",
            f"terrain.coast.cliff_small.{index:02d}",
            f"terrain/coast/cliff_small/{index:02d}",
            f"cliff_small_{index:02d}",
            "cliff_small",
        )
        for index in range(1, 3)
    ]
    + [
        feature_spec(
            f"TER_Ice_Chunk_{index:02d}",
            f"terrain.polar_ice.chunk.{index:02d}",
            f"terrain/polar_ice/chunk/{index:02d}",
            f"ice_chunk_{index:02d}",
            "polar_ice",
        )
        for index in range(1, 17)
    ]
    + [
        feature_spec(
            f"TER_RiverRock{index:02d}",
            f"terrain.river.rock.{index:02d}",
            f"terrain/river/rock/{index:02d}",
            f"river_rock_{index:02d}",
            "river_rock",
        )
        for index in (1, 2, 3, 5, 6)
    ]
)


# These two source candidates reach the proven mesh decoder, but contain
# zero-area indexed triangles.  Keep the strict validator and record the
# bounded omission instead of silently repairing licensed source geometry.
SOURCE_EXCLUSIONS = (
    {
        "source_name": "TER_Cliffs_RockSmall03",
        "reason": "strict normalization rejected a degenerate indexed triangle",
    },
    {
        "source_name": "TER_Cliffs_RockSmall04",
        "reason": "strict normalization rejected a degenerate indexed triangle",
    },
    *(
        {
            "source_name": f"TER_Coast_Decal{index:02d}",
            "reason": (
                "decal entry has no static feature Model container; normalized coast/ocean "
                "decal texture channels are already supplied by the terrain water pack"
            ),
        }
        for index in range(1, 5)
    ),
    {
        "source_name": "TER_RiverRock04",
        "reason": "strict normalization rejected a degenerate indexed triangle",
    },
    *(
        {
            "source_name": f"TER_RiverRock_Decal{index:02d}",
            "reason": (
                "decal entry has no static feature Model container; normalized river-clutter "
                "decal texture channels are already supplied by the terrain water pack"
            ),
        }
        for index in range(1, 7)
    ),
    *(
        {
            "source_name": f"TER_RiverSand_Decal{index:02d}",
            "reason": (
                "decal entry has no static feature Model container; the connected river-bank "
                "surface remains topology-generated"
            ),
        }
        for index in range(1, 5)
    ),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_shore_pack(
    package_path: Path, shared_data: Path, pack: Path, report_path: Path
) -> dict[str, Any]:
    package = StaticPackage(package_path, SHORE_SPECS[0]["source_name"])
    assets: dict[str, dict[str, Any]] = {}
    reports = []
    feature_groups: dict[str, list[str]] = {}
    for spec in SHORE_SPECS:
        manifest_asset, report = build_feature(package, shared_data, pack, spec)
        assets[spec["manifest_key"]] = manifest_asset
        feature_groups.setdefault(spec["group"], []).append(spec["manifest_key"])
        reports.append(report)

    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "ShoreNormalized",
        "display_name": "Normalized Shore Features",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "projection": {
            "tile_width_px": 128,
            "tile_height_px": 64,
            "height_scale_px": 96,
            "basis": {"x": [64, 32], "y": [-64, 32], "z": [0, -96]},
        },
        "assets": assets,
        "feature_sets": {
            group: {
                "variants": variants,
                "status": (
                    "complete_verified_set"
                    if group in {"cliff_large", "polar_ice"}
                    else "verified_subset"
                ),
            }
            for group, variants in feature_groups.items()
        },
    }
    write_json(pack / "manifest.json", manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime pack is source-dependent: " + "; ".join(independence_errors))

    report = {
        "schema": "c3x.civ6_shore_extract.v0",
        "source": str(package_path),
        "source_sha256": sha256_bytes(package.data),
        "allocation_table": {
            "package_offset": package.table_offset,
            "allocation_count": len(package.allocations),
            "stripe_bases": package.stripe_bases,
        },
        "assets": reports,
        "excluded_source_candidates": SOURCE_EXCLUSIONS,
        "pack": str(pack),
        "runtime_independence": "passed",
    }
    write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    root = default_blp_root()
    parser = argparse.ArgumentParser(
        description="Extract verified cliff-rock and polar-ice features into a C3X pack"
    )
    parser.add_argument("--package", type=Path, default=root / "environment" / "clutter.blp")
    parser.add_argument("--shared-data", type=Path, default=root / "SHARED_DATA")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = build_shore_pack(args.package, args.shared_data, args.pack, args.report)
    except (OSError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Extracted {len(report['assets'])} verified shore features")
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
