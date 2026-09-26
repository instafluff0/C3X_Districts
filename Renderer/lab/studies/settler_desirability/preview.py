"""Lab-only city-site color study over a current production terrain bitmap."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

STEPS = 11
WHITE = (247, 250, 244)
GREEN = (19, 105, 52)
CENTER = (16, 16)
CUSTOM_TILE_WIDTHS = (128, 160, 192)


def c_div(numerator: int, denominator: int) -> int:
    """C integer division truncates toward zero, unlike Python's // operator."""
    return numerator // denominator if numerator >= 0 else -((-numerator) // denominator)


def grade(evaluation: int) -> int | None:
    """Mirror C3X's existing eleven TileHighlights.pcx selection."""
    if evaluation <= 0:
        return None
    midpoint = 1_000_000 - 10 // 2
    delta = evaluation - midpoint
    offset = c_div(delta, 10) if evaluation >= midpoint else c_div(delta, 10) - 1
    return max(0, min(STEPS - 1, STEPS // 2 + offset))


def color(level: int) -> tuple[int, int, int]:
    if not 0 <= level < STEPS:
        raise ValueError("desirability grade outside the eleven-step palette")
    t = level / (STEPS - 1)
    return tuple(round(a + (b - a) * t) for a, b in zip(WHITE, GREEN))


def invented_evaluation(x: int, y: int, case: str) -> int:
    """A bounded, deterministic visual fixture; zero stands for no highlight."""
    if case not in ("wash", "outlined", "coastal"):
        raise ValueError("unknown city-site preview case")
    dx, dy = x - CENTER[0], y - CENTER[1]
    if abs(dx) > 6 or abs(dy) > 6 or (x + y) % 2:
        return 0
    if case == "coastal" and x >= 19:
        return 0
    # Open gaps show the unpainted terrain and stand in for forbidden sites.
    if (dx, dy) in ((-4, 0), (-3, 1), (0, -2), (2, 4), (4, -2)):
        return 0
    best = max(0, round(10 - math.hypot(dx + 2, dy) * 1.35))
    second = max(0, round(9 - math.hypot(dx - 3, dy + 1) * 1.35))
    level = min(10, max(best, second) + (1 if dx == 0 and dy == 0 else 0))
    return 999_950 + level * 10


def diamond(x: int, y: int, tile_width: int, image_size: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Match the integer anchor and 2:1 tile basis in biq_preview.cpp."""
    width, height = image_size
    tile_height = tile_width // 2
    left = x * tile_width // 2 + width // 2 - tile_width // 2 - CENTER[0] * tile_width // 2
    top = y * tile_height // 2 + height // 2 - tile_height // 2 - CENTER[1] * tile_height // 2
    return ((left + tile_width // 2, top), (left + tile_width, top + tile_height // 2),
            (left + tile_width // 2, top + tile_height), (left, top + tile_height // 2))


def compose(source: Image.Image, tile_width: int, case: str) -> Image.Image:
    if tile_width not in CUSTOM_TILE_WIDTHS:
        raise ValueError("unsupported study zoom")
    if case not in ("wash", "outlined", "coastal"):
        raise ValueError("unknown city-site preview case")
    base = source.convert("RGBA")
    scale = 3
    layer = Image.new("RGBA", (base.width * scale, base.height * scale))
    pen = ImageDraw.Draw(layer)
    for y in range(9, 24):
        for x in range(8 + y % 2, 24, 2):
            level = grade(invented_evaluation(x, y, case))
            if level is None:
                continue
            points = [(round(px * scale), round(py * scale)) for px, py in diamond(x, y, tile_width, base.size)]
            tint = color(level)
            pen.polygon(points, fill=(*tint, 132 if case == "wash" else 106))
            if case != "wash":
                pen.line(points + [points[0]], fill=(*tint, 210), width=2 * scale, joint="curve")
    layer = layer.resize(base.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(base, layer).convert("RGB")


def render_bitmap(path: Path, tile_width: int, case: str) -> None:
    with Image.open(path) as image:
        result = compose(image, tile_width, case)
    result.save(path)
