#!/usr/bin/env python3
"""Dependency-free textured preview for normalized C3X terrain packs."""

from __future__ import annotations

import argparse
import binascii
import json
import math
import struct
import sys
import zlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview.render_iso import Canvas, screen_center


BACKGROUND = (37, 43, 46)
DDS_HEADER_BYTES = 148
SUPPORTED_DXGI = {77: "BC3_UNORM", 78: "BC3_UNORM_SRGB"}


def rgb565(value: int) -> tuple[int, int, int]:
    r = (value >> 11) & 31
    g = (value >> 5) & 63
    b = value & 31
    return ((r * 255 + 15) // 31, (g * 255 + 31) // 63, (b * 255 + 15) // 31)


def decode_bc3_color(block: bytes, pixel_x: int, pixel_y: int) -> tuple[int, int, int]:
    if len(block) != 16:
        raise ValueError("BC3 block must contain exactly 16 bytes")
    color0, color1, color_indices = struct.unpack_from("<HHI", block, 8)
    first = rgb565(color0)
    second = rgb565(color1)
    palette = (
        first,
        second,
        tuple((2 * first[index] + second[index]) // 3 for index in range(3)),
        tuple((first[index] + 2 * second[index]) // 3 for index in range(3)),
    )
    selector = (color_indices >> (2 * (pixel_y * 4 + pixel_x))) & 3
    return palette[selector]


def decode_bc3_alpha(block: bytes, pixel_x: int, pixel_y: int) -> int:
    if len(block) != 16:
        raise ValueError("BC3 block must contain exactly 16 bytes")
    first, second = block[0], block[1]
    if first > second:
        palette = (
            first,
            second,
            *(int(((8 - index) * first + (index - 1) * second) / 7) for index in range(2, 8)),
        )
    else:
        palette = (
            first,
            second,
            *(int(((6 - index) * first + (index - 1) * second) / 5) for index in range(2, 6)),
            0,
            255,
        )
    selectors = int.from_bytes(block[2:8], "little")
    selector = (selectors >> (3 * (pixel_y * 4 + pixel_x))) & 7
    return palette[selector]


class DdsBc3Texture:
    def __init__(self, data: bytes) -> None:
        if len(data) < DDS_HEADER_BYTES or data[:4] != b"DDS " or data[84:88] != b"DX10":
            raise ValueError("Texture is not a DDS file with a DX10 header")
        self.height = struct.unpack_from("<I", data, 12)[0]
        self.width = struct.unpack_from("<I", data, 16)[0]
        self.mip_count = struct.unpack_from("<I", data, 28)[0]
        self.dxgi_format = struct.unpack_from("<I", data, 128)[0]
        if self.dxgi_format not in SUPPORTED_DXGI:
            raise ValueError(f"Textured preview supports BC3 DDS only, found DXGI {self.dxgi_format}")
        if not self.width or not self.height or not self.mip_count:
            raise ValueError("DDS has a zero dimension or mip count")
        self.blocks_wide = max(1, (self.width + 3) // 4)
        top_level_bytes = self.blocks_wide * max(1, (self.height + 3) // 4) * 16
        if DDS_HEADER_BYTES + top_level_bytes > len(data):
            raise ValueError("DDS top-level BC3 image extends past the file")
        self.data = data

    @classmethod
    def from_file(cls, path: Path) -> "DdsBc3Texture":
        return cls(path.read_bytes())

    def sample(self, u: float, v: float) -> tuple[int, int, int]:
        u = u - math.floor(u)
        v = v - math.floor(v)
        x = min(self.width - 1, int(u * self.width))
        y = min(self.height - 1, int(v * self.height))
        block_offset = DDS_HEADER_BYTES + ((y // 4) * self.blocks_wide + (x // 4)) * 16
        return decode_bc3_color(self.data[block_offset : block_offset + 16], x & 3, y & 3)

    def sample_rgba(self, u: float, v: float) -> tuple[int, int, int, int]:
        u = u - math.floor(u)
        v = v - math.floor(v)
        x = min(self.width - 1, int(u * self.width))
        y = min(self.height - 1, int(v * self.height))
        block_offset = DDS_HEADER_BYTES + ((y // 4) * self.blocks_wide + (x // 4)) * 16
        block = self.data[block_offset : block_offset + 16]
        return (*decode_bc3_color(block, x & 3, y & 3), decode_bc3_alpha(block, x & 3, y & 3))


def safe_pack_path(pack_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or PureWindowsPath(relative).is_absolute()
        or PurePosixPath(relative).is_absolute()
    ):
        raise ValueError(f"Pack asset path must be relative: {relative}")
    resolved_root = pack_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Pack asset escapes its root: {relative}")
    return resolved


def load_runtime_pack(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], DdsBc3Texture]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.asset_pack.v0":
        raise ValueError("Unsupported asset-pack schema")
    terrain = manifest.get("terrains", {}).get("grassland")
    if not isinstance(terrain, dict):
        raise ValueError("Pack has no grassland terrain")
    root = manifest_path.parent
    mesh = json.loads(safe_pack_path(root, terrain["mesh"]).read_text(encoding="utf-8"))
    material = json.loads(safe_pack_path(root, terrain["material"]).read_text(encoding="utf-8"))
    if mesh.get("schema") != "c3x.normalized_mesh.v0":
        raise ValueError("Unsupported normalized-mesh schema")
    if material.get("schema") != "c3x.material.v0":
        raise ValueError("Unsupported material schema")
    texture_path = safe_pack_path(root, material["base_color"]["texture"])
    texture = DdsBc3Texture.from_file(texture_path)
    return manifest, mesh, material, texture


def edge(a: tuple[float, float], b: tuple[float, float], point: tuple[float, float]) -> float:
    return (point[0] - a[0]) * (b[1] - a[1]) - (point[1] - a[1]) * (b[0] - a[0])


def draw_textured_mesh(
    canvas: Canvas,
    center: tuple[int, int],
    tile_width: int,
    tile_height: int,
    height_scale: int,
    mesh: dict[str, Any],
    texture: DdsBc3Texture,
) -> None:
    vertices = mesh["vertices"]
    projected = []
    for vertex in vertices:
        x, y, z = vertex["position"]
        projected.append(
            (
                center[0] + (x - y) * tile_width / 2.0,
                center[1] + (x + y) * tile_height / 2.0 - z * height_scale,
            )
        )
    indices = mesh["topology"]["indices"]
    for start in range(0, len(indices), 3):
        vertex_indices = indices[start : start + 3]
        points = [projected[index] for index in vertex_indices]
        uvs = [vertices[index]["uv0"] for index in vertex_indices]
        area = edge(points[0], points[1], points[2])
        if area == 0:
            raise ValueError("Projected mesh contains a degenerate triangle")
        min_x = max(0, int(math.floor(min(point[0] for point in points))))
        max_x = min(canvas.width - 1, int(math.ceil(max(point[0] for point in points))))
        min_y = max(0, int(math.floor(min(point[1] for point in points))))
        max_y = min(canvas.height - 1, int(math.ceil(max(point[1] for point in points))))
        for pixel_y in range(min_y, max_y + 1):
            for pixel_x in range(min_x, max_x + 1):
                sample_point = (pixel_x + 0.5, pixel_y + 0.5)
                w0 = edge(points[1], points[2], sample_point) / area
                w1 = edge(points[2], points[0], sample_point) / area
                w2 = 1.0 - w0 - w1
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                    continue
                u = w0 * uvs[0][0] + w1 * uvs[1][0] + w2 * uvs[2][0]
                v = w0 * uvs[0][1] + w1 * uvs[1][1] + w2 * uvs[2][1]
                canvas.set_pixel(pixel_x, pixel_y, texture.sample(u, v))


def render_pack(manifest_path: Path, width: int, height: int, grid: int) -> Canvas:
    if width < 1 or height < 1 or grid < 1:
        raise ValueError("Width, height, and grid must be positive")
    manifest, mesh, _material, texture = load_runtime_pack(manifest_path)
    projection = manifest["projection"]
    tile_width = int(projection["tile_width_px"])
    tile_height = int(projection["tile_height_px"])
    height_scale = int(projection.get("height_scale_px", 54))
    origin = (width // 2, max(80, height // 8))
    canvas = Canvas(width, height, BACKGROUND)
    for diagonal in range(grid * 2 - 1):
        for tile_x in range(grid):
            tile_y = diagonal - tile_x
            if 0 <= tile_y < grid:
                center = screen_center(tile_x, tile_y, origin, tile_width, tile_height)
                draw_textured_mesh(
                    canvas, center, tile_width, tile_height, height_scale, mesh, texture
                )
    return canvas


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)


def write_png(canvas: Canvas, path: Path) -> None:
    raw = bytearray()
    for row in range(canvas.height):
        raw.append(0)
        for color in canvas.pixels[row * canvas.width : (row + 1) * canvas.width]:
            raw.extend(color)
    header = struct.pack(">IIBBBBB", canvas.width, canvas.height, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header)
    data += png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a normalized textured terrain pack")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--grid", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        canvas = render_pack(args.pack, args.width, args.height, args.grid)
        write_png(canvas, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output} ({canvas.non_background_pixels()} drawn pixels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
