#!/usr/bin/env python3
"""Render deterministic C3X scene matrices, metrics, and contact sheets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import BACKGROUND, write_png
from Renderer.scenes import scene_contract
from Renderer.standalone.whole_viewport_renderer import (
    RENDERER_ID,
    RENDERER_VERSION,
    PackAssetLoader,
    WholeViewportRenderer,
    load_catalog,
)


MATRIX_SCHEMA = "c3x.fixture_matrix.v0"
REFERENCE_SCHEMA = "c3x.visual_reference_catalog.v0"
DEFAULT_HOURS = (0, 6, 12, 18)
DEFAULT_SEASONS = ("summer", "fall", "winter", "spring")
DEFAULT_VIEWPORTS = ((640, 480), (1024, 768))
EXCLUDED_FALLBACKS = {"category_owned_by_civ3", "category_capture_only"}
THRESHOLDS = {
    "non_background_minimum_basis_points": 100,
    "required_mapping_minimum_basis_points": 10000,
    "map_bounds_violations_maximum": 0,
    "anchor_misses_maximum": 0,
    "invalid_depth_values_maximum": 0,
    "noon_over_midnight_luminance_delta_minimum_x1000": 1000,
    "environment_output_must_differ": True,
}


GLYPHS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "P": ("110", "101", "110", "100", "100"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "-": ("000", "000", "111", "000", "000"),
    " ": ("000", "000", "000", "000", "000"),
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_input_path(path: Path, mod_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    root = mod_root.resolve()
    if resolved == root or root in resolved.parents:
        return {"scope": "mod", "path": resolved.relative_to(root).as_posix()}
    return {"scope": "external", "path": path.name}


def parse_viewports(raw: str) -> tuple[tuple[int, int], ...]:
    result = []
    for item in raw.split(","):
        match = re.fullmatch(r"\s*(\d+)x(\d+)\s*", item, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid viewport {item!r}; expected WIDTHxHEIGHT")
        width, height = int(match.group(1)), int(match.group(2))
        if width < 1 or height < 1:
            raise ValueError("Viewport dimensions must be positive")
        pair = (width, height)
        if pair not in result:
            result.append(pair)
    if not result:
        raise ValueError("At least one viewport is required")
    return tuple(result)


def parse_hours(raw: str) -> tuple[int, ...]:
    result = []
    for item in raw.split(","):
        try:
            hour = int(item.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid hour {item!r}") from exc
        if not 0 <= hour <= 23:
            raise ValueError(f"Hour is outside 0..23: {hour}")
        if hour not in result:
            result.append(hour)
    if not result:
        raise ValueError("At least one hour is required")
    return tuple(result)


def parse_seasons(raw: str) -> tuple[str, ...]:
    result = []
    for item in raw.split(","):
        season = item.strip().lower()
        if season == "autumn":
            season = "fall"
        if season not in DEFAULT_SEASONS:
            raise ValueError(f"Unknown season {item!r}")
        if season not in result:
            result.append(season)
    if not result:
        raise ValueError("At least one season is required")
    return tuple(result)


def validate_reference_catalog(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "references"}:
        raise ValueError("Reference catalog must contain only schema and references")
    if value.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("Unsupported visual-reference catalog schema")
    references = value.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("Reference catalog must contain a nonempty references array")
    seen = set()
    required = {"id", "kind", "source", "availability", "comparison_mode", "time_basis", "purpose"}
    for index, reference in enumerate(references):
        if not isinstance(reference, dict) or not required <= set(reference):
            raise ValueError(f"Reference {index} is missing required metadata")
        identifier = reference["id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"Reference {index} has a missing or duplicate ID")
        seen.add(identifier)
        kind = reference["kind"]
        if kind == "structural_regression":
            if reference["comparison_mode"] != "exact_hash" or reference["time_basis"] != "exact_render_state":
                raise ValueError(f"Structural reference {identifier!r} must use exact render-state hashes")
        elif kind == "art_direction":
            if reference["comparison_mode"] != "qualitative":
                raise ValueError(f"Art-direction reference {identifier!r} cannot be a pixel-equality gate")
            if reference["time_basis"] != "lighting_phase_only" or "exact_hour" in reference:
                raise ValueError(f"Art-direction reference {identifier!r} must identify lighting phases, not an inferred exact hour")
            phases = reference.get("phases")
            if not isinstance(phases, list) or not phases:
                raise ValueError(f"Art-direction reference {identifier!r} has no lighting phases")
        else:
            raise ValueError(f"Reference {identifier!r} has unknown kind {kind!r}")
    return value


def resize_scene(scene: Mapping[str, Any], width: int, height: int) -> dict[str, Any]:
    resized = copy.deepcopy(scene)
    viewport = resized["viewport"]
    old_width, old_height = viewport["width_px"], viewport["height_px"]
    rect = viewport["map_rect_px"]
    right_margin = old_width - rect["x"] - rect["width"]
    bottom_margin = old_height - rect["y"] - rect["height"]
    new_rect = {
        "x": rect["x"],
        "y": rect["y"],
        "width": width - rect["x"] - right_margin,
        "height": height - rect["y"] - bottom_margin,
    }
    if new_rect["width"] < 1 or new_rect["height"] < 1:
        raise ValueError("Viewport override cannot preserve the scene's map-rectangle margins")
    old_center = (rect["x"] + rect["width"] // 2, rect["y"] + rect["height"] // 2)
    new_center = (new_rect["x"] + new_rect["width"] // 2, new_rect["y"] + new_rect["height"] // 2)
    shift_x, shift_y = new_center[0] - old_center[0], new_center[1] - old_center[1]
    viewport["width_px"], viewport["height_px"] = width, height
    viewport["map_rect_px"] = new_rect
    resized["projection"]["origin_px"]["x"] += shift_x
    resized["projection"]["origin_px"]["y"] += shift_y
    for tile in resized["tiles"]:
        tile["anchor_px"]["x"] += shift_x
        tile["anchor_px"]["y"] += shift_y
    for instance in resized["instances"]:
        instance["anchor_px"]["x"] += shift_x
        instance["anchor_px"]["y"] += shift_y
    resized["scene_id"] = scene_contract.scene_identifier(resized)
    return scene_contract.validate_scene(resized)


def _luminance_x1000(color: tuple[int, int, int]) -> int:
    return (2126 * color[0] + 7152 * color[1] + 722 * color[2]) // 10


def frame_metrics(frame, scene: Mapping[str, Any]) -> dict[str, Any]:
    drawn = [pixel for pixel, owner in zip(frame.canvas.pixels, frame.owner_buffer) if owner is not None]
    if not drawn:
        luminances = []
        mean_rgb = [0, 0, 0]
    else:
        luminances = [_luminance_x1000(pixel) for pixel in drawn]
        mean_rgb = [sum(pixel[index] for pixel in drawn) * 1000 // len(drawn) for index in range(3)]
    histogram = [0] * 16
    for luminance in luminances:
        histogram[min(15, luminance * 16 // 256000)] += 1

    rect = scene["viewport"]["map_rect_px"]
    right, bottom = rect["x"] + rect["width"], rect["y"] + rect["height"]
    outside = 0
    invalid_depth = 0
    for index, owner in enumerate(frame.owner_buffer):
        if owner is None:
            continue
        x, y = index % frame.canvas.width, index // frame.canvas.width
        if not (rect["x"] <= x < right and rect["y"] <= y < bottom):
            outside += 1
        if not math.isfinite(frame.depth_buffer[index]):
            invalid_depth += 1

    rendered = set(frame.stats["rendered_ids"])
    required = []
    excluded = []
    by_category: dict[str, dict[str, int]] = {}
    for item in frame.inspection["items"]:
        category = item["resolution"]["category"]
        category_metrics = by_category.setdefault(category, {"total": 0, "required": 0, "rendered": 0, "fallback": 0, "excluded": 0})
        category_metrics["total"] += 1
        if item["instance_id"] in rendered:
            required.append(item["instance_id"])
            category_metrics["required"] += 1
            category_metrics["rendered"] += 1
        else:
            reason = item["resolution"].get("fallback", {}).get("reason")
            category_metrics["fallback"] += 1
            if reason in EXCLUDED_FALLBACKS:
                excluded.append(item["instance_id"])
                category_metrics["excluded"] += 1
            else:
                required.append(item["instance_id"])
                category_metrics["required"] += 1
    mapped_required = sum(identifier in rendered for identifier in required)
    mapping_bp = 10000 if not required else mapped_required * 10000 // len(required)
    anchor_misses = sum(
        frame.stats["anchor_owners"].get(identifier) != identifier for identifier in rendered
    )
    map_pixels = rect["width"] * rect["height"]
    nonblank_minimum = max(1, map_pixels * THRESHOLDS["non_background_minimum_basis_points"] // 10000)
    checks = {
        "nonblank": {"actual": len(drawn), "minimum": nonblank_minimum, "passed": len(drawn) >= nonblank_minimum},
        "required_mapping": {"actual_basis_points": mapping_bp, "minimum_basis_points": THRESHOLDS["required_mapping_minimum_basis_points"], "passed": mapping_bp >= THRESHOLDS["required_mapping_minimum_basis_points"]},
        "map_bounds": {"violations": outside, "maximum": THRESHOLDS["map_bounds_violations_maximum"], "passed": outside <= THRESHOLDS["map_bounds_violations_maximum"]},
        "anchor_alignment": {"misses": anchor_misses, "maximum": THRESHOLDS["anchor_misses_maximum"], "passed": anchor_misses <= THRESHOLDS["anchor_misses_maximum"]},
        "depth_values": {"invalid": invalid_depth, "maximum": THRESHOLDS["invalid_depth_values_maximum"], "passed": invalid_depth <= THRESHOLDS["invalid_depth_values_maximum"]},
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "mapping": {
            "total_instances": len(frame.inspection["items"]),
            "required_instances": len(required),
            "rendered_required_instances": mapped_required,
            "excluded_civ3_owned_instances": len(excluded),
            "coverage_basis_points": mapping_bp,
            "by_category": {key: by_category[key] for key in sorted(by_category)},
        },
        "bounds": {"outside_map_rect_pixels": outside},
        "anchors": {"rendered": len(rendered), "misses": anchor_misses},
        "depth": {
            "hidden_fragments_rejected": frame.stats["pixels_depth_rejected"],
            "fragments_passed": frame.stats["pixels_depth_passed"],
            "invalid_values": invalid_depth,
        },
        "color": {
            "drawn_pixels": len(drawn),
            "unique_colors": len(set(drawn)),
            "mean_rgb_x1000": mean_rgb,
            "luminance_minimum_x1000": min(luminances) if luminances else 0,
            "luminance_maximum_x1000": max(luminances) if luminances else 0,
            "luminance_mean_x1000": sum(luminances) // len(luminances) if luminances else 0,
            "luminance_histogram_16": histogram,
        },
    }


def draw_text(canvas: Canvas, x: int, y: int, label: str, color=(222, 226, 229), scale: int = 1) -> None:
    cursor = x
    for character in label.upper():
        glyph = GLYPHS.get(character, GLYPHS["-"])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    for dy in range(scale):
                        for dx in range(scale):
                            canvas.set_pixel(cursor + column * scale + dx, y + row * scale + dy, color)
        cursor += 4 * scale


def blit_thumbnail(source: Canvas, target: Canvas, x: int, y: int, width: int, height: int) -> None:
    scale = min(width / source.width, height / source.height)
    drawn_width = max(1, int(source.width * scale))
    drawn_height = max(1, int(source.height * scale))
    offset_x, offset_y = x + (width - drawn_width) // 2, y + (height - drawn_height) // 2
    for target_y in range(drawn_height):
        source_y = min(source.height - 1, target_y * source.height // drawn_height)
        source_row = source_y * source.width
        for target_x in range(drawn_width):
            source_x = min(source.width - 1, target_x * source.width // drawn_width)
            target.set_pixel(offset_x + target_x, offset_y + target_y, source.pixels[source_row + source_x])


def _cell_name(scene_label: str, width: int, height: int, hour: int, season: str) -> str:
    return f"{scene_label}__{width}x{height}__h{hour:02d}__{season}.png"


def _comparisons(cells: list[dict[str, Any]], viewports, hours, seasons) -> list[dict[str, Any]]:
    by_key = {(cell["viewport"]["width"], cell["viewport"]["height"], cell["hour"], cell["season"]): cell for cell in cells}
    comparisons = []
    if 0 in hours and 12 in hours:
        for width, height in viewports:
            for season in seasons:
                midnight = by_key[(width, height, 0, season)]
                noon = by_key[(width, height, 12, season)]
                delta = noon["metrics"]["color"]["luminance_mean_x1000"] - midnight["metrics"]["color"]["luminance_mean_x1000"]
                differs = noon["image_sha256"] != midnight["image_sha256"]
                passed = differs and delta >= THRESHOLDS["noon_over_midnight_luminance_delta_minimum_x1000"]
                comparisons.append({
                    "id": f"time:{width}x{height}:{season}:noon-vs-midnight",
                    "kind": "time_response",
                    "outputs_differ": differs,
                    "luminance_delta_x1000": delta,
                    "minimum_delta_x1000": THRESHOLDS["noon_over_midnight_luminance_delta_minimum_x1000"],
                    "passed": passed,
                })
    if "summer" in seasons and "winter" in seasons:
        for width, height in viewports:
            for hour in hours:
                summer = by_key[(width, height, hour, "summer")]
                winter = by_key[(width, height, hour, "winter")]
                summer_rgb = summer["metrics"]["color"]["mean_rgb_x1000"]
                winter_rgb = winter["metrics"]["color"]["mean_rgb_x1000"]
                differs = summer["image_sha256"] != winter["image_sha256"]
                passed = differs and summer_rgb != winter_rgb
                comparisons.append({
                    "id": f"season:{width}x{height}:h{hour:02d}:summer-vs-winter",
                    "kind": "season_response",
                    "outputs_differ": differs,
                    "mean_rgb_delta_x1000": [summer_rgb[index] - winter_rgb[index] for index in range(3)],
                    "passed": passed,
                })
    return comparisons


def render_fixture_matrix(
    scene: Mapping[str, Any],
    catalog: Mapping[str, Any],
    assets: PackAssetLoader,
    output_dir: Path,
    *,
    scene_label: str,
    input_records: Mapping[str, Any],
    references: Mapping[str, Any],
    viewports: tuple[tuple[int, int], ...] = DEFAULT_VIEWPORTS,
    hours: tuple[int, ...] = DEFAULT_HOURS,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    thumbnail_size: tuple[int, int] = (200, 150),
) -> dict[str, Any]:
    scene_contract.validate_scene(scene)
    validate_reference_catalog(references)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    thumb_width, thumb_height = thumbnail_size
    gap, label_width, header_height = 4, 104, 18
    rows = len(viewports) * len(hours)
    contact_width = label_width + gap + len(seasons) * (thumb_width + gap)
    contact_height = header_height + gap + rows * (thumb_height + gap)
    contact = Canvas(contact_width, contact_height, (24, 28, 31))
    for column, season in enumerate(seasons):
        draw_text(contact, label_width + gap + column * (thumb_width + gap) + 3, 6, season)

    device = WholeViewportRenderer(*viewports[0])
    cells = []
    row = 0
    try:
        for width, height in viewports:
            viewport_scene = resize_scene(scene, width, height)
            for hour in hours:
                draw_text(contact, 4, header_height + gap + row * (thumb_height + gap) + 4, f"{width}X{height} H{hour:02d}")
                for column, season in enumerate(seasons):
                    cell_scene = copy.deepcopy(viewport_scene)
                    cell_scene["environment"]["hour"] = hour
                    cell_scene["environment"]["season"] = season
                    cell_scene["scene_id"] = scene_contract.scene_identifier(cell_scene)
                    frame = device.render(cell_scene, catalog, assets)
                    filename = _cell_name(scene_label, width, height, hour, season)
                    image_path = images_dir / filename
                    write_png(frame.canvas, image_path)
                    metrics = frame_metrics(frame, cell_scene)
                    cells.append({
                        "id": f"{width}x{height}:h{hour:02d}:{season}",
                        "scene_id": cell_scene["scene_id"],
                        "viewport": {"width": width, "height": height},
                        "hour": hour,
                        "season": season,
                        "image": f"images/{filename}",
                        "image_sha256": sha256_file(image_path),
                        "renderer_generation": frame.stats["renderer_generation"],
                        "metrics": metrics,
                    })
                    blit_thumbnail(
                        frame.canvas,
                        contact,
                        label_width + gap + column * (thumb_width + gap),
                        header_height + gap + row * (thumb_height + gap),
                        thumb_width,
                        thumb_height,
                    )
                row += 1
    finally:
        device.close()

    contact_path = output_dir / "contact_sheet.png"
    write_png(contact, contact_path)
    comparisons = _comparisons(cells, viewports, hours, seasons)
    manifest = {
        "schema": MATRIX_SCHEMA,
        "renderer": {"id": RENDERER_ID, "version": RENDERER_VERSION},
        "inputs": {
            **input_records,
            "catalog_sha256": sha256_bytes(canonical_bytes(catalog)),
            "scene_canonical_sha256": sha256_bytes(canonical_bytes(scene)),
            "loaded_assets": assets.loaded_input_records(),
            "references": references,
            "references_sha256": sha256_bytes(canonical_bytes(references)),
        },
        "matrix": {
            "viewports": [{"width": width, "height": height} for width, height in viewports],
            "hours": list(hours),
            "seasons": list(seasons),
            "cell_order": "viewport, hour, season",
        },
        "thresholds": THRESHOLDS,
        "cells": cells,
        "comparisons": comparisons,
        "contact_sheet": {
            "path": "contact_sheet.png",
            "sha256": sha256_file(contact_path),
            "width": contact.width,
            "height": contact.height,
            "columns": list(seasons),
            "rows": [f"{width}x{height}:h{hour:02d}" for width, height in viewports for hour in hours],
        },
        "summary": {
            "cell_count": len(cells),
            "comparison_count": len(comparisons),
            "all_cells_passed": all(cell["metrics"]["passed"] for cell in cells),
            "all_comparisons_passed": all(comparison["passed"] for comparison in comparisons),
            "passed": all(cell["metrics"]["passed"] for cell in cells) and all(comparison["passed"] for comparison in comparisons),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render deterministic C3X viewport/hour/season fixture matrices")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--custom", type=Path)
    parser.add_argument("--mod-root", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument("--references", type=Path, default=Path("Renderer/samples/validation/reference_metadata.json"))
    parser.add_argument("--viewports", default="640x480,1024x768")
    parser.add_argument("--hours", default="0,6,12,18")
    parser.add_argument("--seasons", default="summer,fall,winter,spring")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        scene = scene_contract.load_scene(args.scene)
        catalog = load_catalog(
            args.default,
            mod_root=args.mod_root,
            scenario_path=args.scenario,
            custom_path=args.custom,
            scenario_root=args.scenario_root,
        )
        assets = PackAssetLoader(catalog, mod_root=args.mod_root, scenario_root=args.scenario_root)
        references = validate_reference_catalog(json.loads(args.references.read_text(encoding="utf-8")))
        definition_paths = [("default", args.default), ("scenario", args.scenario), ("custom", args.custom)]
        input_records = {
            "scene": {**stable_input_path(args.scene, args.mod_root), "sha256": sha256_file(args.scene)},
            "definitions": [
                {"layer": layer, **stable_input_path(path, args.mod_root), "sha256": sha256_file(path)}
                for layer, path in definition_paths
                if path is not None
            ],
            "reference_catalog": {**stable_input_path(args.references, args.mod_root), "sha256": sha256_file(args.references)},
        }
        label = re.sub(r"[^A-Za-z0-9_-]+", "_", args.scene.name.removesuffix(".scene.json")).strip("_") or "scene"
        manifest = render_fixture_matrix(
            scene,
            catalog,
            assets,
            args.output,
            scene_label=label,
            input_records=input_records,
            references=references,
            viewports=parse_viewports(args.viewports),
            hours=parse_hours(args.hours),
            seasons=parse_seasons(args.seasons),
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), **manifest["summary"]}, sort_keys=True))
    return 0 if manifest["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

