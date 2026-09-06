#!/usr/bin/env python3
"""Pure-Python C3X isometric terrain preview renderer.

This is not the final D3D renderer. It is a deterministic, dependency-free
preview harness for validating pack shape and Civ III-style projection math.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


Color = tuple[int, int, int]
Point = tuple[int, int]


TERRAIN_ORDER = ("grassland", "plains", "desert", "tundra")


def clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def shade(color: Iterable[int], factor: float) -> Color:
    r, g, b = color
    return (clamp_channel(r * factor), clamp_channel(g * factor), clamp_channel(b * factor))


def stable_hash(x: int, y: int, seed: int) -> int:
    value = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)
    value ^= value >> 13
    value *= 1274126177
    return value & 0xFFFFFFFF


def terrain_at(x: int, y: int) -> str:
    band = (x * 2 + y * 3 + (x // 3) - (y // 4)) % 11
    if band in (0, 1, 2):
        return "grassland"
    if band in (3, 4, 5):
        return "plains"
    if band in (6, 7):
        return "desert"
    return "tundra"


def has_mountain(x: int, y: int) -> bool:
    return stable_hash(x, y, 17) % 13 in (0, 5)


class Canvas:
    def __init__(self, width: int, height: int, background: Color = (37, 43, 46)) -> None:
        self.width = width
        self.height = height
        self.pixels = [background for _ in range(width * height)]

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y * self.width + x] = color

    def fill_polygon(self, points: list[Point], color: Color) -> None:
        if len(points) < 3:
            return
        min_y = max(0, min(p[1] for p in points))
        max_y = min(self.height - 1, max(p[1] for p in points))
        for y in range(min_y, max_y + 1):
            nodes: list[float] = []
            prev = points[-1]
            for cur in points:
                if (cur[1] < y <= prev[1]) or (prev[1] < y <= cur[1]):
                    denom = prev[1] - cur[1]
                    if denom != 0:
                        nodes.append(cur[0] + (y - cur[1]) * (prev[0] - cur[0]) / denom)
                prev = cur
            nodes.sort()
            for i in range(0, len(nodes) - 1, 2):
                start = max(0, int(math.ceil(nodes[i])))
                end = min(self.width - 1, int(math.floor(nodes[i + 1])))
                for x in range(start, end + 1):
                    self.set_pixel(x, y, color)

    def draw_line(self, a: Point, b: Point, color: Color) -> None:
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def write_bmp(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        row_stride = (self.width * 3 + 3) & ~3
        pixel_bytes = row_stride * self.height
        file_size = 54 + pixel_bytes
        with path.open("wb") as f:
            f.write(b"BM")
            f.write(file_size.to_bytes(4, "little"))
            f.write((0).to_bytes(4, "little"))
            f.write((54).to_bytes(4, "little"))
            f.write((40).to_bytes(4, "little"))
            f.write(self.width.to_bytes(4, "little", signed=True))
            f.write(self.height.to_bytes(4, "little", signed=True))
            f.write((1).to_bytes(2, "little"))
            f.write((24).to_bytes(2, "little"))
            f.write((0).to_bytes(4, "little"))
            f.write(pixel_bytes.to_bytes(4, "little"))
            f.write((2835).to_bytes(4, "little", signed=True))
            f.write((2835).to_bytes(4, "little", signed=True))
            f.write((0).to_bytes(4, "little"))
            f.write((0).to_bytes(4, "little"))
            pad = b"\x00" * (row_stride - self.width * 3)
            for y in range(self.height - 1, -1, -1):
                offset = y * self.width
                for r, g, b in self.pixels[offset : offset + self.width]:
                    f.write(bytes((b, g, r)))
                f.write(pad)

    def non_background_pixels(self, background: Color = (37, 43, 46)) -> int:
        return sum(1 for pixel in self.pixels if pixel != background)


def screen_center(x: int, y: int, origin: Point, tile_w: int, tile_h: int) -> Point:
    return (
        origin[0] + (x - y) * tile_w // 2,
        origin[1] + (x + y) * tile_h // 2,
    )


def draw_tile(canvas: Canvas, center: Point, tile_w: int, tile_h: int, color: list[int], lift: int) -> None:
    half_w = tile_w // 2
    half_h = tile_h // 2
    top = (center[0], center[1] - half_h - lift)
    right = (center[0] + half_w, center[1] - lift // 2)
    bottom = (center[0], center[1] + half_h)
    left = (center[0] - half_w, center[1] - lift // 2)
    canvas.fill_polygon([top, right, bottom, left], shade(color, 0.86 + lift / 500.0))
    canvas.fill_polygon([top, left, bottom], shade(color, 0.72 + lift / 700.0))
    canvas.fill_polygon([top, right, bottom], shade(color, 0.96 + lift / 800.0))
    edge = shade(color, 0.48)
    canvas.draw_line(top, right, edge)
    canvas.draw_line(right, bottom, edge)
    canvas.draw_line(bottom, left, edge)
    canvas.draw_line(left, top, edge)


def draw_mountain(canvas: Canvas, center: Point, tile_w: int, tile_h: int, height_scale: int, variant: dict) -> None:
    half_w = int(tile_w * 0.34)
    base_y = center[1] + tile_h // 9
    height = int(height_scale * float(variant.get("preview_height", 0.9)) * 1.45)
    peak = (center[0] + (stable_hash(center[0], center[1], 3) % 15) - 7, base_y - height)
    left = (center[0] - half_w, base_y)
    right = (center[0] + half_w, base_y)
    front = (center[0], base_y + tile_h // 5)
    color = tuple(variant.get("preview_color", [118, 118, 108]))
    canvas.fill_polygon([left, peak, front], shade(color, 0.84))
    canvas.fill_polygon([peak, right, front], shade(color, 0.62))
    canvas.fill_polygon([left, peak, right], shade(color, 1.08))
    snow = (226, 230, 221)
    canvas.fill_polygon(
        [
            peak,
            (peak[0] - half_w // 5, peak[1] + height // 4),
            (peak[0] + half_w // 6, peak[1] + height // 5),
        ],
        snow,
    )


def render(pack: dict, width: int, height: int, grid: int, seed: int, force_terrain: str | None = None) -> Canvas:
    projection = pack.get("projection", {})
    tile_w = int(projection.get("tile_width_px", 128))
    tile_h = int(projection.get("tile_height_px", 64))
    height_scale = int(projection.get("height_scale_px", 54))
    origin = (width // 2, max(80, height // 8))

    canvas = Canvas(width, height)
    terrains = pack["terrains"]
    mountain_variants = pack.get("relief", {}).get("mountains", {}).get("variants", [])

    for diagonal in range(grid * 2 - 1):
        for x in range(grid):
            y = diagonal - x
            if y < 0 or y >= grid:
                continue
            terrain_name = force_terrain or terrain_at(x, y)
            terrain = terrains.get(terrain_name, terrains[TERRAIN_ORDER[0]])
            center = screen_center(x, y, origin, tile_w, tile_h)
            lift = 0
            if mountain_variants and has_mountain(x, y):
                lift = height_scale // 4
            elif terrain_name == "tundra" and stable_hash(x, y, seed) % 4 == 0:
                lift = height_scale // 10
            draw_tile(canvas, center, tile_w, tile_h, terrain["preview_color"], lift)
            if mountain_variants and has_mountain(x, y):
                variant = mountain_variants[stable_hash(x, y, seed) % len(mountain_variants)]
                draw_mountain(canvas, center, tile_w, tile_h, height_scale, variant)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a C3X terrain pack preview")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--force-terrain", choices=TERRAIN_ORDER)
    args = parser.parse_args()

    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    canvas = render(pack, args.width, args.height, args.grid, args.seed, args.force_terrain)
    canvas.write_bmp(args.output)
    print(f"Wrote {args.output} ({canvas.non_background_pixels()} drawn pixels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
