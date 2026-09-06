#!/usr/bin/env python3
"""Dependency-free visual QA metrics for Lab renders, zoom pairs, and animation frames."""

from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable


def _png(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG")
    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise ValueError("visual QA supports non-interlaced RGB8 PNG only")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if width < 1 or height < 1:
        raise ValueError("PNG has no valid dimensions")
    raw = zlib.decompress(bytes(compressed))
    stride = width * 3
    if len(raw) != height * (stride + 1):
        raise ValueError("PNG decompressed length is inconsistent")
    rows: list[bytearray] = []
    cursor = 0
    for _y in range(height):
        filter_type = raw[cursor]
        source = raw[cursor + 1 : cursor + 1 + stride]
        cursor += stride + 1
        if filter_type not in range(5):
            raise ValueError("PNG uses an unsupported row filter")
        row = bytearray(stride)
        prior = rows[-1] if rows else bytearray(stride)
        for index, value in enumerate(source):
            left = row[index - 3] if index >= 3 else 0
            up = prior[index]
            upper_left = prior[index - 3] if index >= 3 else 0
            if filter_type == 0:
                prediction = 0
            elif filter_type == 1:
                prediction = left
            elif filter_type == 2:
                prediction = up
            elif filter_type == 3:
                prediction = (left + up) // 2
            else:
                p = left + up - upper_left
                distances = (abs(p - left), abs(p - up), abs(p - upper_left))
                prediction = (left, up, upper_left)[distances.index(min(distances))]
            row[index] = (value + prediction) & 0xFF
        rows.append(row)
    return width, height, [tuple(row[index : index + 3]) for row in rows for index in range(0, stride, 3)]


def _bmp(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise ValueError("not a BMP")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    header_size = struct.unpack_from("<I", data, 14)[0]
    width, signed_height = struct.unpack_from("<ii", data, 18)
    planes, bits, compression = struct.unpack_from("<HHI", data, 26)
    if header_size < 40 or width < 1 or signed_height == 0 or planes != 1 or bits != 24 or compression != 0:
        raise ValueError("visual QA supports uncompressed 24-bit BMP only")
    height = abs(signed_height)
    stride = (width * 3 + 3) & ~3
    if pixel_offset + stride * height > len(data):
        raise ValueError("BMP pixels extend past the file")
    rows = []
    for file_row in range(height):
        start = pixel_offset + file_row * stride
        row = [tuple(reversed(data[start + x * 3 : start + x * 3 + 3])) for x in range(width)]
        rows.append(row)
    if signed_height > 0:
        rows.reverse()
    return width, height, [pixel for row in rows for pixel in row]


def read_image(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    signature = path.read_bytes()[:8]
    if signature.startswith(b"\x89PNG"):
        return _png(path)
    if signature.startswith(b"BM"):
        return _bmp(path)
    raise ValueError("visual QA supports RGB8 PNG and uncompressed 24-bit BMP")


def _distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return max(abs(a[index] - b[index]) for index in range(3))


def _luminance(pixel: tuple[int, int, int]) -> float:
    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]


def subject_mask(pixels: list[tuple[int, int, int]], background: tuple[int, int, int], tolerance: int = 0) -> list[bool]:
    return [_distance(pixel, background) > tolerance for pixel in pixels]


def image_metrics(
    width: int,
    height: int,
    pixels: list[tuple[int, int, int]],
    background: tuple[int, int, int],
    *,
    background_tolerance: int = 0,
    allowed_bounds: list[int] | None = None,
    ground_y: int | None = None,
    civ_colors: list[tuple[int, int, int]] | None = None,
    civ_color_tolerance: int = 20,
) -> dict[str, Any]:
    if len(pixels) != width * height:
        raise ValueError("visual QA pixel count does not match dimensions")
    mask = subject_mask(pixels, background, background_tolerance)
    indices = [index for index, present in enumerate(mask) if present]
    if not indices:
        raise ValueError("visual QA image contains no subject pixels")
    xs = [index % width for index in indices]
    ys = [index // width for index in indices]
    bounds = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
    edge_touch = sum(1 for index in indices if index % width in {0, width - 1} or index // width in {0, height - 1})
    subject_luminance = sorted(_luminance(pixels[index]) for index in indices)
    bg_luminance = _luminance(background)
    spill = 0
    if allowed_bounds is not None:
        if len(allowed_bounds) != 4:
            raise ValueError("allowed bounds must be [left, top, right, bottom]")
        spill = sum(
            1
            for index in indices
            if not (allowed_bounds[0] <= index % width < allowed_bounds[2] and allowed_bounds[1] <= index // width < allowed_bounds[3])
        )
    civ_pixels = 0
    if civ_colors:
        civ_pixels = sum(any(_distance(pixels[index], color) <= civ_color_tolerance for color in civ_colors) for index in indices)
    return {
        "width": width,
        "height": height,
        "subject_pixels": len(indices),
        "coverage_basis_points": len(indices) * 10000 // (width * height),
        "silhouette_bounds_px": bounds,
        "silhouette_aspect_x1000": (bounds[2] - bounds[0]) * 1000 // max(1, bounds[3] - bounds[1]),
        "edge_touch_pixels": edge_touch,
        "neighbor_spill_pixels": spill,
        "grounding_gap_px": None if ground_y is None else ground_y - bounds[3],
        "mean_subject_luminance_x1000": round(sum(subject_luminance) * 1000 / len(subject_luminance)),
        "background_contrast_x1000": round(abs(sum(subject_luminance) / len(subject_luminance) - bg_luminance) * 1000),
        "luminance_span_x1000": round((subject_luminance[int(0.9 * (len(subject_luminance) - 1))] - subject_luminance[int(0.1 * (len(subject_luminance) - 1))]) * 1000),
        "civ_color_pixels": civ_pixels,
        "civ_color_fraction_basis_points": civ_pixels * 10000 // len(indices),
    }


def compare_day_night(
    day: tuple[int, int, list[tuple[int, int, int]]],
    night: tuple[int, int, list[tuple[int, int, int]]],
    emissive_delta: float = 12.0,
) -> dict[str, Any]:
    if day[:2] != night[:2]:
        raise ValueError("day/night images have different dimensions")
    deltas = [_luminance(b) - _luminance(a) for a, b in zip(day[2], night[2])]
    emissive = [value for value in deltas if value >= emissive_delta]
    return {
        "emissive_pixels": len(emissive),
        "emissive_fraction_basis_points": len(emissive) * 10000 // len(deltas),
        "mean_positive_emissive_delta_x1000": 0 if not emissive else round(sum(emissive) * 1000 / len(emissive)),
        "whole_frame_luminance_delta_x1000": round(sum(deltas) * 1000 / len(deltas)),
    }


def compare_zoom(normal: dict[str, Any], reduced: dict[str, Any]) -> dict[str, Any]:
    coverage_delta = abs(normal["coverage_basis_points"] - reduced["coverage_basis_points"])
    aspect_delta = abs(normal["silhouette_aspect_x1000"] - reduced["silhouette_aspect_x1000"])
    return {
        "normalized_coverage_delta_basis_points": coverage_delta,
        "silhouette_aspect_delta_x1000": aspect_delta,
        "consistent": coverage_delta <= 1800 and aspect_delta <= 350,
    }


def animation_metrics(images: list[tuple[int, int, list[tuple[int, int, int]]]], background: tuple[int, int, int]) -> dict[str, Any]:
    if len(images) < 2 or len({image[:2] for image in images}) != 1:
        raise ValueError("animation QA needs at least two equal-sized frames")
    masks = [subject_mask(image[2], background) for image in images]
    union = sum(any(mask[index] for mask in masks) for index in range(len(masks[0])))
    intersection = sum(all(mask[index] for mask in masks) for index in range(len(masks[0])))
    changed = sum(len({image[2][index] for image in images}) > 1 for index in range(len(masks[0])))
    return {
        "frames": len(images),
        "temporal_union_pixels": union,
        "temporal_intersection_pixels": intersection,
        "changed_pixels": changed,
        "motion_occupancy_basis_points": 0 if union == 0 else (union - intersection) * 10000 // union,
        "animated": changed > 0,
    }


def analyze_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "c3x.visual_qa_plan.v0":
        raise ValueError("unsupported visual QA plan")
    root = path.parent
    results = {}
    for case in plan.get("cases", []):
        case_id = case["id"]
        kind = case["kind"]
        background = tuple(case.get("background", [37, 43, 46]))
        if kind == "image":
            image = read_image(root / case["image"])
            results[case_id] = image_metrics(*image, background, allowed_bounds=case.get("allowed_bounds"), ground_y=case.get("ground_y"), civ_colors=[tuple(value) for value in case.get("civ_colors", [])])
        elif kind == "day_night":
            results[case_id] = compare_day_night(read_image(root / case["day"]), read_image(root / case["night"]))
        elif kind == "zoom_pair":
            normal_image = read_image(root / case["normal"])
            reduced_image = read_image(root / case["reduced"])
            results[case_id] = compare_zoom(image_metrics(*normal_image, background), image_metrics(*reduced_image, background))
        elif kind == "animation":
            results[case_id] = animation_metrics([read_image(root / value) for value in case["frames"]], background)
        else:
            raise ValueError(f"unknown visual QA case kind: {kind}")
    return {"schema": "c3x.visual_qa_report.v0", "cases": results, "runtime_activation": "none"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--background", default="37,43,46")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (args.plan is None) == (args.image is None):
            raise ValueError("choose exactly one of --plan or --image")
        if args.plan:
            report = analyze_plan(args.plan)
        else:
            background = tuple(int(value) for value in args.background.split(","))
            if len(background) != 3:
                raise ValueError("background must be R,G,B")
            report = {"schema": "c3x.visual_qa_report.v0", "cases": {"image": image_metrics(*read_image(args.image), background)}, "runtime_activation": "none"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, zlib.error) as exc:
        parser.error(str(exc))
    print(f"Measured {len(report['cases'])} visual QA cases at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
