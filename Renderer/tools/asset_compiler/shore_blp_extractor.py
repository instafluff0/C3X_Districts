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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

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
CLIFF_ARTDEF_GROUPS = {
    "cliff_large": "CLUTTER_CLIFF",
    "cliff_small": "CLUTTER_CLIFF_SMALL",
}


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
        for index in range(1, 5)
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


SOURCE_EXCLUSIONS = (
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


def _artdef_value(value: ET.Element) -> Optional[str]:
    child = next((child for child in value if child.tag != "m_ParamName"), None)
    return None if child is None else child.attrib.get("text", child.text)


def read_cliff_artdef(path: Path) -> dict[str, dict[str, Any]]:
    """Preserve the source cliff clutter controls under generic asset ids."""
    root = ET.parse(path).getroot()
    source_to_asset = {spec["source_name"]: spec["manifest_key"] for spec in SHORE_SPECS}
    result: dict[str, dict[str, Any]] = {}
    for group, set_name in CLIFF_ARTDEF_GROUPS.items():
        matches = [
            element
            for element in root.iter("Element")
            if element.find("m_Name") is not None
            and element.find("m_Name").attrib.get("text") == set_name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one ArtDef clutter set {set_name}, found {len(matches)}")
        owner = matches[0]
        values = {}
        for value in owner.findall("./m_Fields/m_Values/Element"):
            parameter = value.find("m_ParamName")
            if parameter is not None:
                values[parameter.attrib["text"]] = _artdef_value(value)
        plants = [
            collection
            for collection in owner.findall("./m_ChildCollections/Element")
            if collection.find("m_CollectionName") is not None
            and collection.find("m_CollectionName").attrib.get("text") == "Plants"
        ]
        if len(plants) != 1:
            raise ValueError(f"Expected one Plants collection in {set_name}")
        placements = []
        for item in plants[0].findall("Element"):
            item_values = {}
            for value in item.findall("./m_Fields/m_Values/Element"):
                parameter = value.find("m_ParamName")
                if parameter is not None:
                    item_values[parameter.attrib["text"]] = _artdef_value(value)
            source_asset = item_values.get("Asset")
            if source_asset not in source_to_asset:
                continue
            placements.append(
                {
                    "asset": source_to_asset[source_asset],
                    "scale": float(item_values["Scale"]),
                    "count": int(item_values["Count"]),
                    "scale_variation": float(item_values["ScaleVariation"]),
                    "low_end_reduction": float(item_values.get("LowendReduction", 0.0)),
                    "show_decal": item_values.get("ShowDecal", "false").lower() == "true",
                    "priority": int(item_values.get("Priority", 0)),
                    "width": float(item_values.get("Width", 0.0)),
                    "rotate_mode": item_values.get("RotateMode", "RotateZ"),
                    "is_center_model": item_values.get("IsCenterModel", "false").lower() == "true",
                    "allow_overlap": item_values.get("AllowOverlap", "false").lower() == "true",
                    "min_count": int(item_values.get("MinCount", 0)),
                }
            )
        result[group] = {
            "controls": {
                "edge_falloff": float(values["EdgeFalloff"]),
                "clip_low": float(values["ClipLow"]),
                "clip_high": float(values["ClipHigh"]),
                "fixed_height": float(values["FixedHeight"]),
                "clip_river": values["ClipRiver"].lower() == "true",
                "clip_buildings": values["ClipBuildings"].lower() == "true",
                "clip_coastline": values["ClipCoastline"].lower() == "true",
                "terrain_height": values["TerrainHeight"].lower() == "true",
                "clip_sloped": values["ClipSloped"].lower() == "true",
                "density": float(values["Density"]),
                "mode": values["Mode"],
            },
            "placements": placements,
        }
    return result


def build_shore_pack(
    package_path: Path,
    shared_data: Path,
    pack: Path,
    report_path: Path,
    artdef_path: Optional[Path] = None,
) -> dict[str, Any]:
    package = StaticPackage(package_path, SHORE_SPECS[0]["source_name"])
    assets: dict[str, dict[str, Any]] = {}
    reports = []
    feature_groups: dict[str, list[str]] = {}
    for spec in SHORE_SPECS:
        manifest_asset, report = build_feature(
            package,
            shared_data,
            pack,
            spec,
            use_authored_normals=spec["group"] in {"cliff_large", "cliff_small"},
            # The two final small cliff bodies contain source-authored
            # zero-area triangles. D3D discards them; remove only those
            # triangles while retaining every usable source vertex and UV.
            drop_degenerate_triangles=spec["source_name"] in {
                "TER_Cliffs_RockSmall03",
                "TER_Cliffs_RockSmall04",
            },
        )
        assets[spec["manifest_key"]] = manifest_asset
        feature_groups.setdefault(spec["group"], []).append(spec["manifest_key"])
        reports.append(report)

    artdef_groups = read_cliff_artdef(artdef_path) if artdef_path is not None else {}
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
                    if group in {"cliff_large", "cliff_small", "polar_ice"}
                    else "verified_subset"
                ),
                **artdef_groups.get(group, {}),
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
    parser.add_argument(
        "--artdef",
        type=Path,
        default=root.parents[2] / "ArtDefs" / "Clutter.artdef",
    )
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = build_shore_pack(
            args.package, args.shared_data, args.pack, args.report, args.artdef
        )
    except (OSError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Extracted {len(report['assets'])} verified shore features")
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
