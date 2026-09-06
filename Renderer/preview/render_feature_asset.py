#!/usr/bin/env python3
"""Dependency-free two-view renderer for one normalized feature asset."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview.render_iso import Canvas
from preview.render_textured_patch import BACKGROUND, safe_pack_path, write_png


SUPPORTED_DXGI = {71: "BC1_UNORM", 72: "BC1_UNORM_SRGB"}


def rgb565(value: int) -> tuple[int, int, int]:
    return (
        (((value >> 11) & 31) * 255 + 15) // 31,
        (((value >> 5) & 63) * 255 + 31) // 63,
        ((value & 31) * 255 + 15) // 31,
    )


def decode_bc1(block: bytes, pixel_x: int, pixel_y: int) -> tuple[int, int, int, int]:
    if len(block) != 8:
        raise ValueError("BC1 block must contain exactly 8 bytes")
    color0, color1, selectors = struct.unpack("<HHI", block)
    first = rgb565(color0)
    second = rgb565(color1)
    if color0 > color1:
        palette = (
            (*first, 255),
            (*second, 255),
            (*(int((2 * first[channel] + second[channel]) / 3) for channel in range(3)), 255),
            (*(int((first[channel] + 2 * second[channel]) / 3) for channel in range(3)), 255),
        )
    else:
        palette = (
            (*first, 255),
            (*second, 255),
            (*(int((first[channel] + second[channel]) / 2) for channel in range(3)), 255),
            (0, 0, 0, 0),
        )
    selector = (selectors >> (2 * (pixel_y * 4 + pixel_x))) & 3
    return palette[selector]


class DdsBc1Texture:
    HEADER_BYTES = 148

    def __init__(
        self, data: bytes, address_mode_u: str = "clamp", address_mode_v: str = "clamp"
    ) -> None:
        if (
            len(data) < self.HEADER_BYTES
            or data[:4] != b"DDS "
            or data[84:88] != b"DX10"
        ):
            raise ValueError("Texture is not a DDS file with a DX10 header")
        self.height = struct.unpack_from("<I", data, 12)[0]
        self.width = struct.unpack_from("<I", data, 16)[0]
        self.mip_count = struct.unpack_from("<I", data, 28)[0]
        self.dxgi_format = struct.unpack_from("<I", data, 128)[0]
        if self.dxgi_format not in SUPPORTED_DXGI:
            raise ValueError(f"Feature preview supports BC1 DDS only, found DXGI {self.dxgi_format}")
        if not self.width or not self.height or not self.mip_count:
            raise ValueError("DDS has a zero dimension or mip count")
        if address_mode_u not in ("clamp", "wrap") or address_mode_v not in ("clamp", "wrap"):
            raise ValueError("Feature preview supports clamp or wrap texture addressing")
        self.address_mode_u = address_mode_u
        self.address_mode_v = address_mode_v
        self.blocks_wide = max(1, (self.width + 3) // 4)
        top_level_bytes = self.blocks_wide * max(1, (self.height + 3) // 4) * 8
        if self.HEADER_BYTES + top_level_bytes > len(data):
            raise ValueError("DDS top-level BC1 image extends past the file")
        self.data = data

    @classmethod
    def from_file(
        cls, path: Path, address_mode_u: str = "clamp", address_mode_v: str = "clamp"
    ) -> "DdsBc1Texture":
        return cls(path.read_bytes(), address_mode_u, address_mode_v)

    def sample(self, u: float, v: float) -> tuple[int, int, int, int]:
        u = u % 1.0 if self.address_mode_u == "wrap" else max(0.0, min(1.0, u))
        v = v % 1.0 if self.address_mode_v == "wrap" else max(0.0, min(1.0, v))
        x = min(self.width - 1, int(u * self.width))
        y = min(self.height - 1, int(v * self.height))
        block_offset = self.HEADER_BYTES + ((y // 4) * self.blocks_wide + (x // 4)) * 8
        return decode_bc1(self.data[block_offset : block_offset + 8], x & 3, y & 3)


def edge(
    a: tuple[float, float], b: tuple[float, float], point: tuple[float, float]
) -> float:
    return (point[0] - a[0]) * (b[1] - a[1]) - (point[1] - a[1]) * (b[0] - a[0])


def normalize(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if length <= 1.0e-12:
        raise ValueError("Cannot normalize a zero vector")
    return tuple(component / length for component in value)


def rotate_z(value: list[float], angle: float) -> tuple[float, float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        value[0] * cosine - value[1] * sine,
        value[0] * sine + value[1] * cosine,
        value[2],
    )


def draw_ground(canvas: Canvas, center: tuple[int, int], scale: float) -> None:
    cx, cy = center
    half_width = int(scale * 0.55)
    half_height = int(scale * 0.25)
    canvas.fill_polygon(
        [(cx, cy - half_height), (cx + half_width, cy), (cx, cy + half_height), (cx - half_width, cy)],
        (55, 68, 56),
    )


def draw_mesh(
    canvas: Canvas,
    depth_buffer: list[float],
    mesh: dict[str, Any],
    texture: DdsBc1Texture,
    center: tuple[int, int],
    scale: float,
    rotation: float,
    model_scale: float = 1.0,
) -> None:
    transformed = []
    for vertex in mesh["vertices"]:
        position = tuple(
            component * model_scale for component in rotate_z(vertex["position"], rotation)
        )
        normal = normalize(rotate_z(vertex["normal"], rotation))
        transformed.append(
            {
                "screen": (
                    center[0] + (position[0] - position[1]) * scale * 0.72,
                    center[1] + (position[0] + position[1]) * scale * 0.36 - position[2] * scale,
                ),
                "depth": position[0] + position[1] + position[2] * 0.65,
                "normal": normal,
                "uv": vertex["uv0"],
            }
        )
    light = normalize((-0.45, -0.60, 1.0))
    indices = mesh["topology"]["indices"]
    for start in range(0, len(indices), 3):
        vertices = [transformed[index] for index in indices[start : start + 3]]
        points = [vertex["screen"] for vertex in vertices]
        area = edge(points[0], points[1], points[2])
        if abs(area) <= 1.0e-9:
            continue
        min_x = max(0, int(math.floor(min(point[0] for point in points))))
        max_x = min(canvas.width - 1, int(math.ceil(max(point[0] for point in points))))
        min_y = max(0, int(math.floor(min(point[1] for point in points))))
        max_y = min(canvas.height - 1, int(math.ceil(max(point[1] for point in points))))
        for pixel_y in range(min_y, max_y + 1):
            for pixel_x in range(min_x, max_x + 1):
                sample = (pixel_x + 0.5, pixel_y + 0.5)
                weights = (
                    edge(points[1], points[2], sample) / area,
                    edge(points[2], points[0], sample) / area,
                )
                weights = (weights[0], weights[1], 1.0 - weights[0] - weights[1])
                if min(weights) < -1.0e-9:
                    continue
                depth = sum(weights[index] * vertices[index]["depth"] for index in range(3))
                pixel_index = pixel_y * canvas.width + pixel_x
                if depth <= depth_buffer[pixel_index]:
                    continue
                u = sum(weights[index] * vertices[index]["uv"][0] for index in range(3))
                v = sum(weights[index] * vertices[index]["uv"][1] for index in range(3))
                color = texture.sample(u, v)
                if color[3] < 128:
                    continue
                normal = normalize(
                    tuple(
                        sum(weights[index] * vertices[index]["normal"][axis] for index in range(3))
                        for axis in range(3)
                    )
                )
                diffuse = abs(sum(normal[axis] * light[axis] for axis in range(3)))
                shade = 0.48 + diffuse * 0.62
                canvas.set_pixel(
                    pixel_x,
                    pixel_y,
                    tuple(max(0, min(255, int(round(channel * shade)))) for channel in color[:3]),
                )
                depth_buffer[pixel_index] = depth


def load_feature(
    manifest_path: Path, asset_id: str
) -> tuple[dict[str, Any], dict[str, Any], DdsBc1Texture, float]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in ("c3x.asset_pack.v0", "c3x.resource_pack.v0"):
        raise ValueError("Unsupported asset-pack schema")
    asset = manifest.get("assets", {}).get(asset_id)
    if not isinstance(asset, dict) or asset.get("type") != "feature":
        raise ValueError(f"Pack has no feature asset {asset_id}")
    root = manifest_path.parent
    mesh = json.loads(safe_pack_path(root, asset["mesh"]).read_text(encoding="utf-8"))
    material = json.loads(safe_pack_path(root, asset["material"]).read_text(encoding="utf-8"))
    if mesh.get("schema") != "c3x.normalized_mesh.v0":
        raise ValueError("Unsupported normalized-mesh schema")
    if material.get("schema") != "c3x.material.v0":
        raise ValueError("Unsupported material schema")
    base_color = material["base_color"]
    texture = DdsBc1Texture.from_file(
        safe_pack_path(root, base_color["texture"]),
        base_color.get("address_mode_u", "clamp"),
        base_color.get("address_mode_v", "clamp"),
    )
    preview_scale = float(asset.get("preview_scale", 1.0))
    if preview_scale <= 0.0:
        raise ValueError("Feature preview scale must be positive")
    return mesh, material, texture, preview_scale


def render_feature(
    manifest_path: Path,
    asset_id: str,
    width: int,
    height: int,
    model_scale_multiplier: float = 1.0,
) -> Canvas:
    if width < 256 or height < 256:
        raise ValueError("Feature contact sheet must be at least 256x256")
    mesh, _material, texture, model_scale = load_feature(manifest_path, asset_id)
    if model_scale_multiplier <= 0.0:
        raise ValueError("Feature model-scale multiplier must be positive")
    model_scale *= model_scale_multiplier
    canvas = Canvas(width, height, BACKGROUND)
    depth_buffer = [-math.inf] * (width * height)
    scale = min(width * 0.36, height * 0.68)
    centers = ((width // 4, int(height * 0.82)), (width * 3 // 4, int(height * 0.82)))
    for center in centers:
        draw_ground(canvas, center, scale)
    draw_mesh(
        canvas, depth_buffer, mesh, texture, centers[0], scale, math.radians(25.0), model_scale
    )
    draw_mesh(
        canvas, depth_buffer, mesh, texture, centers[1], scale, math.radians(115.0), model_scale
    )
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a normalized feature contact sheet")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--asset", default="feature/jungle/palm_01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args(argv)
    try:
        canvas = render_feature(args.pack, args.asset, args.width, args.height)
        write_png(canvas, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output} ({canvas.non_background_pixels()} drawn pixels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
