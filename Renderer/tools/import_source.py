#!/usr/bin/env python3
"""Build a source-independent C3X terrain pack from Civ VI source folders."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.tools.asset_compiler import grassland_pack_builder
from Renderer.tools.asset_compiler import pack_equivalence
from Renderer.tools.asset_compiler import terrain_geometry_resolver
from Renderer.tools.asset_compiler import terrain_pack_builder


PERMISSION_SCHEMA = "c3x.source_conversion_permission.v0"
REQUIRED_OVERLAY_PERMISSIONS = {"conversion", "cross-game-use"}
BASE_TERRAIN_PACKAGE = Path("Base/Platforms/Windows/BLPs/terrain/TerrainMaterialSet_Base.blp")
BASE_RELIEF_PACKAGE = Path("Base/Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp")
OVERLAY_TERRAIN_PACKAGE = Path("Platforms/Windows/BLPs/terrain/TerrainMaterialSet_Base.blp")
OVERLAY_RELIEF_PACKAGE = Path("Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp")
OVERLAY_MATERIAL_OCCURRENCES = {"desert": 0, "mountains": 0}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_permission_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError(
            "An overlay source requires --permission-record with documented conversion and cross-game-use permission"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read permission record {path}: {exc}") from exc
    permissions = record.get("permissions")
    if record.get("schema") != PERMISSION_SCHEMA or not isinstance(permissions, list):
        raise ValueError("Unsupported or incomplete source-conversion permission record")
    missing = sorted(REQUIRED_OVERLAY_PERMISSIONS - set(permissions))
    if missing:
        raise ValueError(f"Permission record is missing: {', '.join(missing)}")
    for field in ("source_name", "rights_holder", "grant_reference"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Permission record is missing {field}")
    if record.get("redistribution") not in {"local-only", "allowed"}:
        raise ValueError("Permission record redistribution must be local-only or allowed")
    return record


def resolve_source_plan(source_root: Path, overlay_root: Path | None) -> dict[str, Any]:
    source_root = source_root.resolve()
    package = source_root / BASE_TERRAIN_PACKAGE
    relief = source_root / BASE_RELIEF_PACKAGE
    overlay_relief = None
    occurrences = dict(terrain_pack_builder.MATERIAL_OCCURRENCES)
    source_kind = "baseline"
    if overlay_root is not None:
        overlay_root = overlay_root.resolve()
        package = overlay_root / OVERLAY_TERRAIN_PACKAGE
        overlay_relief = overlay_root / OVERLAY_RELIEF_PACKAGE
        occurrences = dict(OVERLAY_MATERIAL_OCCURRENCES)
        source_kind = "overlay"
    for label, path in (("terrain material package", package), ("terrain relief package", relief)):
        if not path.is_file():
            raise ValueError(f"Missing {label}: {path}")
    return {
        "source_root": source_root,
        "overlay_root": overlay_root,
        "package": package,
        "relief_package": relief,
        "overlay_relief_package": overlay_relief,
        "relief_source_kind": "baseline",
        "material_occurrences": occurrences,
        "source_kind": source_kind,
    }


def build_variant(
    source_root: Path,
    output: Path,
    variant: str,
    overlay_root: Path | None = None,
    baseline_pack: Path | None = None,
    permission_record: Path | None = None,
    local_testing_only: bool = False,
    mesh: Path = terrain_geometry_resolver.DEFAULT_MESH,
    replace: bool = False,
) -> dict[str, Any]:
    if not variant.strip():
        raise ValueError("Variant name must not be empty")
    permission = None
    if overlay_root is not None:
        if permission_record is not None:
            permission = validate_permission_record(permission_record)
        elif local_testing_only:
            permission = {
                "basis": "explicit-user-local-testing-only",
                "redistribution": "prohibited",
                "notice": "Generated output is an ignored local test artifact and must not be distributed.",
            }
        else:
            raise ValueError(
                "An overlay source requires --permission-record or an explicit --local-testing-only acknowledgement"
            )
    if overlay_root is not None and baseline_pack is None:
        raise ValueError("An overlay source requires --baseline-pack for deterministic equivalence reporting")
    plan = resolve_source_plan(source_root, overlay_root)
    output = output.resolve()
    if output.exists() and not replace:
        raise ValueError(f"Output already exists; pass --replace explicitly: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        source_report_path = stage / "provenance" / "source_report.json"
        terrain_pack_builder.build_local_terrain_pack(
            plan["package"],
            mesh,
            stage,
            source_report_path,
            relief_package=plan["relief_package"],
            assets_root=plan["source_root"],
            material_occurrences=plan["material_occurrences"],
        )
        source_report = {
            "schema": "c3x.source_import.v0",
            "adapter": "civ6-terrain",
            "variant": variant,
            "source_kind": plan["source_kind"],
            "source_root": str(plan["source_root"]),
            "overlay_root": None if plan["overlay_root"] is None else str(plan["overlay_root"]),
            "terrain_package": str(plan["package"]),
            "relief_package": str(plan["relief_package"]),
            "overlay_relief_package": (
                None if plan["overlay_relief_package"] is None
                else str(plan["overlay_relief_package"])
            ),
            "relief_source_kind": plan["relief_source_kind"],
            "permission": permission,
            "inherited_components": ["relief", "water"] if overlay_root is not None else [],
            "water_source_kind": "baseline",
            "runtime_source_independent": True,
        }
        _write_json(stage / "provenance" / "import.json", source_report)
        equivalence = None
        if baseline_pack is not None:
            equivalence = pack_equivalence.compare_packs(baseline_pack.resolve(), stage)
            pack_equivalence.write_report(stage / "provenance" / "equivalence_report.json", equivalence)
        independence_errors = grassland_pack_builder.validate_runtime_independence(stage)
        if independence_errors:
            raise ValueError("Runtime pack is not source-independent: " + "; ".join(independence_errors))
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
        return {
            "output": output,
            "source": source_report,
            "equivalence": equivalence,
        }
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("civ6-terrain",), default="civ6-terrain")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--baseline-pack", type=Path)
    parser.add_argument("--permission-record", type=Path)
    parser.add_argument(
        "--local-testing-only",
        action="store_true",
        help="Acknowledge that overlay output is an ignored, non-distributed local test artifact",
    )
    parser.add_argument("--mesh", type=Path, default=terrain_geometry_resolver.DEFAULT_MESH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_variant(
            args.source_root,
            args.output,
            args.variant,
            overlay_root=args.overlay_root,
            baseline_pack=args.baseline_pack,
            permission_record=args.permission_record,
            local_testing_only=args.local_testing_only,
            mesh=args.mesh,
            replace=args.replace,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote source-independent C3X pack: {result['output']}")
    if result["equivalence"] is not None:
        counts = result["equivalence"]["counts"]
        print(
            f"Equivalence: {counts['replaced']} replaced, {counts['inherited']} inherited, "
            f"{counts['missing']} missing logical IDs"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
