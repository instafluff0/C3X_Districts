#!/usr/bin/env python3
"""Validate and batch-render game-assisted visible-scene exports.

This is the offline half of M5.2.  It deliberately starts at the versioned
visible-scene boundary and has no dependency on injected capture structures.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.definitions.definition_parser import RULE_CATEGORIES
from Renderer.scenes import scene_contract
from Renderer.standalone.whole_viewport_renderer import PackAssetLoader, load_catalog
from Renderer.tools import render_fixture_matrix as matrix


PLAN_SCHEMA = "c3x.scene_export_batch_plan.v0"
REPORT_SCHEMA = "c3x.scene_export_batch_report.v0"
SOURCE_KINDS = {"save", "biq", "save_from_biq", "synthetic"}
LAYER_CHECKS = ("fog", "borders", "labels", "highlights", "hud", "ui")
REVIEW_STATES = {"pending", "pass", "fail"}
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _expect_keys(value: Any, required: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    missing = required - set(value)
    extra = set(value) - required
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{path} has unknown fields: {', '.join(sorted(extra))}")
    return value


def _slug(value: Any, path: str) -> str:
    if not isinstance(value, str) or not SLUG_RE.fullmatch(value):
        raise ValueError(f"{path} must match {SLUG_RE.pattern}")
    return value


def _relative_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{path} must be relative to the batch plan")
    return value


def validate_plan(value: Any) -> dict[str, Any]:
    plan = _expect_keys(value, {"schema", "batch_id", "source", "fixtures"}, "$")
    if plan["schema"] != PLAN_SCHEMA:
        raise ValueError(f"$.schema must be {PLAN_SCHEMA!r}")
    _slug(plan["batch_id"], "$.batch_id")

    source = _expect_keys(plan["source"], {"id", "kind", "label", "artifact"}, "$.source")
    _slug(source["id"], "$.source.id")
    if source["kind"] not in SOURCE_KINDS:
        raise ValueError(f"$.source.kind must be one of {', '.join(sorted(SOURCE_KINDS))}")
    if not isinstance(source["label"], str) or not source["label"].strip():
        raise ValueError("$.source.label must be a nonempty string")
    if source["artifact"] is not None:
        _relative_path(source["artifact"], "$.source.artifact")

    fixtures = plan["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("$.fixtures must be a nonempty array")
    seen = set()
    for index, raw_fixture in enumerate(fixtures):
        path = f"$.fixtures[{index}]"
        fixture = _expect_keys(
            raw_fixture,
            {"id", "scene", "required_categories", "in_game_evidence"},
            path,
        )
        fixture_id = _slug(fixture["id"], f"{path}.id")
        if fixture_id in seen:
            raise ValueError(f"{path}.id duplicates {fixture_id!r}")
        seen.add(fixture_id)
        scene_path = _relative_path(fixture["scene"], f"{path}.scene")
        if Path(scene_path).name != f"{fixture_id}.scene.json":
            raise ValueError(
                f"{path}.scene must use the canonical filename {fixture_id}.scene.json"
            )
        categories = fixture["required_categories"]
        if (
            not isinstance(categories, list)
            or not categories
            or any(category not in RULE_CATEGORIES for category in categories)
            or len(categories) != len(set(categories))
        ):
            raise ValueError(f"{path}.required_categories must be unique renderer categories")

        evidence = _expect_keys(
            fixture["in_game_evidence"], {"screenshot", "layer_checks"}, f"{path}.in_game_evidence"
        )
        if evidence["screenshot"] is not None:
            screenshot = _relative_path(evidence["screenshot"], f"{path}.in_game_evidence.screenshot")
            if Path(screenshot).name != f"{fixture_id}__ingame.png":
                raise ValueError(
                    f"{path}.in_game_evidence.screenshot must use the canonical filename "
                    f"{fixture_id}__ingame.png"
                )
        checks = _expect_keys(
            evidence["layer_checks"], set(LAYER_CHECKS), f"{path}.in_game_evidence.layer_checks"
        )
        for check in LAYER_CHECKS:
            if checks[check] not in REVIEW_STATES:
                raise ValueError(
                    f"{path}.in_game_evidence.layer_checks.{check} must be pending, pass, or fail"
                )
        if evidence["screenshot"] is None and any(state != "pending" for state in checks.values()):
            raise ValueError(f"{path}.in_game_evidence checks must stay pending without a screenshot")
    return dict(plan)


def load_plan(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read batch plan {path}: {exc}") from exc
    return validate_plan(value)


def _resolved(plan_path: Path, relative: str) -> Path:
    return (plan_path.parent / relative).resolve()


def _input_record(path: Path, mod_root: Path) -> dict[str, str]:
    return {
        **matrix.stable_input_path(path, mod_root),
        "sha256": matrix.sha256_file(path),
    }


def _scene_inventory(scene: Mapping[str, Any]) -> dict[str, Any]:
    counts = {category: 0 for category in sorted(RULE_CATEGORIES)}
    counts["terrain"] = len(scene["tiles"])
    for instance in scene["instances"]:
        counts[instance["category"]] += 1
    return {
        "tile_count": len(scene["tiles"]),
        "instance_count": len(scene["instances"]),
        "category_counts": counts,
    }


def _artifact_record(plan_path: Path, relative: str | None, mod_root: Path) -> dict[str, Any]:
    if relative is None:
        return {"status": "not_recorded"}
    path = _resolved(plan_path, relative)
    if not path.is_file():
        raise ValueError(f"Source artifact does not exist: {relative}")
    return {"status": "recorded", "input": _input_record(path, mod_root)}


def _evidence_record(
    plan_path: Path, fixture_id: str, evidence: Mapping[str, Any], mod_root: Path
) -> dict[str, Any]:
    screenshot_relative = evidence["screenshot"]
    checks = {name: evidence["layer_checks"][name] for name in LAYER_CHECKS}
    if screenshot_relative is None:
        return {
            "status": "pending",
            "screenshot": None,
            "layer_checks": checks,
            "passed": False,
        }
    screenshot = _resolved(plan_path, screenshot_relative)
    if not screenshot.is_file():
        raise ValueError(f"In-game screenshot does not exist for {fixture_id}: {screenshot_relative}")
    if screenshot.read_bytes()[:8] != PNG_SIGNATURE:
        raise ValueError(f"In-game screenshot is not a PNG for {fixture_id}: {screenshot_relative}")
    passed = all(state == "pass" for state in checks.values())
    return {
        "status": "reviewed" if passed else "needs_review",
        "screenshot": _input_record(screenshot, mod_root),
        "layer_checks": checks,
        "passed": passed,
    }


def _write_html(report: Mapping[str, Any], path: Path) -> None:
    rows = []
    for fixture in report["fixtures"]:
        fixture_id = html.escape(fixture["id"])
        missing = ", ".join(fixture["scene_validation"]["missing_required_categories"]) or "none"
        matrix_link = f"fixtures/{fixture_id}/contact_sheet.png"
        evidence = fixture["in_game_evidence"]
        rows.append(
            "<section>"
            f"<h2>{fixture_id}</h2>"
            f"<p>Offline: {'PASS' if fixture['offline_passed'] else 'FAIL'}; "
            f"matched in-game evidence: {html.escape(evidence['status'])}; "
            f"missing categories: {html.escape(missing)}</p>"
            f"<a href=\"{matrix_link}\"><img src=\"{matrix_link}\" alt=\"{fixture_id} matrix\"></a>"
            "</section>"
        )
    body = "".join(rows)
    document = (
        "<!doctype html><meta charset=\"utf-8\"><title>C3X scene export batch</title>"
        "<style>body{background:#181c1f;color:#dee2e5;font-family:sans-serif;margin:24px}"
        "section{border-top:1px solid #566;padding:16px 0}img{max-width:100%;height:auto}</style>"
        f"<h1>{html.escape(report['batch_id'])}</h1>"
        f"<p>Offline batch: {'PASS' if report['summary']['offline_passed'] else 'FAIL'}; "
        f"full M5.2 evidence: {'PASS' if report['summary']['full_m5_2_evidence_passed'] else 'PENDING'}</p>"
        f"{body}\n"
    )
    path.write_text(document, encoding="utf-8")


def process_export_batch(
    plan_path: Path,
    catalog: Mapping[str, Any],
    assets: PackAssetLoader,
    references: Mapping[str, Any],
    output_dir: Path,
    *,
    mod_root: Path,
    definition_records: list[dict[str, Any]],
    reference_record: dict[str, Any],
    viewports: tuple[tuple[int, int], ...] = matrix.DEFAULT_VIEWPORTS,
    hours: tuple[int, ...] = matrix.DEFAULT_HOURS,
    seasons: tuple[str, ...] = matrix.DEFAULT_SEASONS,
    thumbnail_size: tuple[int, int] = (200, 150),
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    matrix.validate_reference_catalog(references)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = output_dir / "canonical_scenes"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    fixture_reports = []
    for fixture in sorted(plan["fixtures"], key=lambda item: item["id"]):
        fixture_id = fixture["id"]
        scene_path = _resolved(plan_path, fixture["scene"])
        scene = scene_contract.load_scene(scene_path)
        canonical_path = canonical_dir / f"{fixture_id}.scene.json"
        canonical_path.write_text(scene_contract.canonical_json(scene), encoding="utf-8")
        inventory = _scene_inventory(scene)
        missing = sorted(
            category
            for category in fixture["required_categories"]
            if inventory["category_counts"][category] == 0
        )
        input_records = {
            "scene": _input_record(scene_path, mod_root),
            "definitions": definition_records,
            "reference_catalog": reference_record,
        }
        matrix_dir = output_dir / "fixtures" / fixture_id
        matrix_report = matrix.render_fixture_matrix(
            scene,
            catalog,
            assets,
            matrix_dir,
            scene_label=fixture_id,
            input_records=input_records,
            references=references,
            viewports=viewports,
            hours=hours,
            seasons=seasons,
            thumbnail_size=thumbnail_size,
        )
        evidence = _evidence_record(plan_path, fixture_id, fixture["in_game_evidence"], mod_root)
        scene_validation = {
            "passed": not missing,
            "required_categories": fixture["required_categories"],
            "missing_required_categories": missing,
            **inventory,
        }
        offline_passed = scene_validation["passed"] and matrix_report["summary"]["passed"]
        fixture_reports.append(
            {
                "id": fixture_id,
                "source_scene": _input_record(scene_path, mod_root),
                "canonical_scene": {
                    "path": f"canonical_scenes/{fixture_id}.scene.json",
                    "sha256": matrix.sha256_file(canonical_path),
                },
                "scene_validation": scene_validation,
                "fixture_matrix": {
                    "path": f"fixtures/{fixture_id}/manifest.json",
                    "sha256": matrix.sha256_file(matrix_dir / "manifest.json"),
                    "contact_sheet": f"fixtures/{fixture_id}/contact_sheet.png",
                    "summary": matrix_report["summary"],
                },
                "in_game_evidence": evidence,
                "offline_passed": offline_passed,
            }
        )

    offline_passed = all(fixture["offline_passed"] for fixture in fixture_reports)
    full_evidence = offline_passed and all(
        fixture["in_game_evidence"]["passed"] for fixture in fixture_reports
    )
    report = {
        "schema": REPORT_SCHEMA,
        "batch_id": plan["batch_id"],
        "plan": _input_record(plan_path, mod_root),
        "source": {
            "id": plan["source"]["id"],
            "kind": plan["source"]["kind"],
            "label": plan["source"]["label"],
            "artifact": _artifact_record(plan_path, plan["source"]["artifact"], mod_root),
        },
        "fixtures": fixture_reports,
        "summary": {
            "fixture_count": len(fixture_reports),
            "offline_passed": offline_passed,
            "matched_in_game_evidence_count": sum(
                fixture["in_game_evidence"]["passed"] for fixture in fixture_reports
            ),
            "full_m5_2_evidence_passed": full_evidence,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_bytes(matrix.canonical_bytes(report))
    _write_html(report, output_dir / "contact_sheet.html")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and batch-render game-assisted C3X visible-scene exports"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--custom", type=Path)
    parser.add_argument("--mod-root", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("Renderer/samples/validation/reference_metadata.json"),
    )
    parser.add_argument("--viewports", default="640x480,1024x768")
    parser.add_argument("--hours", default="0,6,12,18")
    parser.add_argument("--seasons", default="summer,fall,winter,spring")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(
            args.default,
            mod_root=args.mod_root,
            scenario_path=args.scenario,
            custom_path=args.custom,
            scenario_root=args.scenario_root,
        )
        assets = PackAssetLoader(
            catalog, mod_root=args.mod_root, scenario_root=args.scenario_root
        )
        references = matrix.validate_reference_catalog(
            json.loads(args.references.read_text(encoding="utf-8"))
        )
        definition_paths = [
            ("default", args.default),
            ("scenario", args.scenario),
            ("custom", args.custom),
        ]
        definitions = [
            {"layer": layer, **_input_record(path, args.mod_root)}
            for layer, path in definition_paths
            if path is not None
        ]
        report = process_export_batch(
            args.plan,
            catalog,
            assets,
            references,
            args.output,
            mod_root=args.mod_root,
            definition_records=definitions,
            reference_record=_input_record(args.references, args.mod_root),
            viewports=matrix.parse_viewports(args.viewports),
            hours=matrix.parse_hours(args.hours),
            seasons=matrix.parse_seasons(args.seasons),
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), **report["summary"]}, sort_keys=True))
    return 0 if report["summary"]["offline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
