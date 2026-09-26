"""Lab-only continuous civ-color border over a production-rendered city scene."""
from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFilter

CITY = (16, 16)
NEIGHBORS = ((1, -1), (1, 1), (-1, 1), (-1, -1))
CASES = {
    "crimson-fine": ((196, 35, 50), 2.8),
    "crimson-brush": ((196, 35, 50), 4.2),
    "azure-brush": ((32, 110, 194), 4.2),
}


def city_territory(center: tuple[int, int] = CITY) -> set[tuple[int, int]]:
    """An uneven one-city territory on Civ III's parity-constrained tile lattice."""
    owned = set()
    for dc in range(-2, 3):
        for dr in range(-2, 3):
            if abs(dc) + abs(dr) <= 3 and (dc, dr) not in ((2, -1), (-2, 1)):
                owned.add((center[0] + dc + dr, center[1] + dc - dr))
    owned.update(((center[0] + 3, center[1] + 3), (center[0] - 3, center[1] - 3)))
    return owned


def read_biq_terrain(terrain_csv: Path) -> dict[tuple[int, int], tuple[int, int]]:
    """Read base and real terrain from the existing test.biq capture format."""
    rows = terrain_csv.read_text().splitlines()
    if not rows or not rows[0].startswith("C3X_BIQ_TERRAIN_V3,"):
        raise ValueError("expected a captured BIQ terrain CSV")
    terrain = {}
    for row in rows[1:]:
        x, y, base, real, *_ = map(int, row.split(","))
        terrain[(x, y)] = (base, real)
    return terrain


def biq_city_territory(center: tuple[int, int], terrain_csv: Path) -> set[tuple[int, int]]:
    """Select a small Lab-owned city region from real land tiles in a BIQ capture."""
    terrain = read_biq_terrain(terrain_csv)
    # All coordinates are genuine map tiles. Ownership is a deliberately
    # invented visual fixture because terrain CSV has no culture ownership.
    offsets = {(dc, dr) for dc in range(-1, 2) for dr in range(-1, 2)} | {(2, 0), (2, -1)}
    selected = {(center[0] + dc + dr, center[1] + dc - dr) for dc, dr in offsets}
    if not all(tile in terrain and terrain[tile][0] < 11 for tile in selected):
        raise ValueError("selected city territory is not all BIQ land tiles")
    return selected


class ReliefStudy:
    """Continuous BIQ-type relief proxy for the visual draft, not mesh heights."""

    def __init__(self, terrain: dict[tuple[int, int], tuple[int, int]],
                 tile_width: int, image_size: tuple[int, int], center: tuple[int, int]):
        self.width = tile_width
        self.image_size = image_size
        self.center = center
        self.landforms = []
        for (x, y), (base, real) in terrain.items():
            if abs(x-center[0]) > 9 or abs(y-center[1]) > 9:
                continue
            c, r = (x+y)/2, (x-y)/2
            if real in (5, 6, 10):
                amplitude = {5: 0.19, 6: 0.37, 10: 0.34}[real] * tile_width
                self.landforms.append((c, r, amplitude))

    def world_at(self, px: float, py: float) -> tuple[float, float]:
        dx = (px-self.image_size[0]/2)/(self.width/2)
        dy = (py-self.image_size[1]/2)/(self.width/4)
        return ((self.center[0]+self.center[1])/2+(dx+dy)/2,
                (self.center[0]-self.center[1])/2+(dx-dy)/2)

    def height(self, px: float, py: float) -> float:
        c, r = self.world_at(px, py)
        lift = sum(amplitude * math.exp(-((c-nc)**2+(r-nr)**2)/0.66)
                   for nc, nr, amplitude in self.landforms)
        return min(lift, self.width*0.40)

def diamond(x: int, y: int, tile_width: int, image_size: tuple[int, int],
            center: tuple[int, int] = CITY):
    """Match the synthetic fixture anchors in biq_preview.cpp."""
    width, height = image_size
    tile_height = tile_width // 2
    left = x * tile_width / 2 + width / 2 - tile_width / 2 - center[0] * tile_width / 2
    top = y * tile_height / 2 + height / 2 - tile_height / 2 - center[1] * tile_height / 2
    return ((left + tile_width / 2, top), (left + tile_width, top + tile_height / 2),
            (left + tile_width / 2, top + tile_height), (left, top + tile_height / 2))


def exposed_edges(owned: set[tuple[int, int]]):
    for x, y in sorted(owned):
        for side, (dx, dy) in enumerate(NEIGHBORS):
            if (x + dx, y + dy) not in owned:
                yield x, y, side


def perimeter_loops(owned: set[tuple[int, int]], tile_width: int,
                    image_size: tuple[int, int], center: tuple[int, int] = CITY) -> list[list[tuple[float, float]]]:
    """Join exposed diamond edges into complete, ordered territory outlines."""
    following = {}
    for x, y, side in exposed_edges(owned):
        corners = diamond(x, y, tile_width, image_size, center)
        start, end = corners[side], corners[(side + 1) % 4]
        if start in following:
            raise ValueError("branching territory perimeter")
        following[start] = end
    loops = []
    while following:
        start = min(following)
        current = start
        loop = []
        while current in following:
            loop.append(current)
            current = following.pop(current)
            if current == start:
                break
        if current != start:
            raise ValueError("open territory perimeter")
        loops.append(loop)
    return loops


def round_tile_corners(corners: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep straight strokes on tile edges; curve only near their shared corner."""
    rounded = []
    inset = 0.12
    for index, corner in enumerate(corners):
        before, after = corners[index - 1], corners[(index + 1) % len(corners)]
        approach = (corner[0] * (1 - inset) + before[0] * inset,
                    corner[1] * (1 - inset) + before[1] * inset)
        departure = (corner[0] * (1 - inset) + after[0] * inset,
                     corner[1] * (1 - inset) + after[1] * inset)
        rounded.append(approach)
        for step in range(1, 9):
            t = step / 8
            rounded.append(((1-t)**2 * approach[0] + 2*(1-t)*t * corner[0] + t*t * departure[0],
                            (1-t)**2 * approach[1] + 2*(1-t)*t * corner[1] + t*t * departure[1]))
    return rounded


def arc_lengths(points: list[tuple[float, float]]) -> list[float]:
    distances = [0.0]
    for a, b in zip(points, points[1:] + points[:1]):
        distances.append(distances[-1] + math.dist(a, b))
    return distances


def point_at(points: list[tuple[float, float]], distances: list[float], distance: float):
    index = min(bisect.bisect_right(distances, distance) - 1, len(points) - 1)
    a, b = points[index], points[(index + 1) % len(points)]
    t = (distance - distances[index]) / (distances[index + 1] - distances[index])
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def draped_paths(owned: set[tuple[int, int]], image_size: tuple[int, int],
                 tile_width: int, center: tuple[int, int],
                 relief: ReliefStudy | None = None) -> list[list[tuple[float, float]]]:
    """Sample the tile perimeter densely before projecting each point uphill."""
    paths = []
    for corners in perimeter_loops(owned, tile_width, image_size, center):
        points = round_tile_corners(corners)
        distances = arc_lengths(points)
        total = distances[-1]
        samples = max(2, math.ceil(total / 2.0))
        path = []
        for step in range(samples + 1):
            x, y = point_at(points, distances, total * step / samples)
            path.append((x, y - (relief.height(x, y) if relief is not None else 0)))
        paths.append(path)
    return paths


def painted_mask(paths: list[list[tuple[float, float]]], image_size: tuple[int, int],
                 stroke_width: float, scale: int, texture: bool = False) -> Image.Image:
    mask = Image.new("L", (image_size[0]*scale, image_size[1]*scale))
    pen = ImageDraw.Draw(mask)
    width = max(2, round(stroke_width*scale))
    radius = width/2
    for ordinal, path in enumerate(paths):
        pixels = [(x*scale, y*scale) for x, y in path]
        if texture:
            # Broad, continuous opacity changes avoid the stamped/dashed look
            # of the first draft while letting the ground texture read through.
            for step, (a, b) in enumerate(zip(pixels, pixels[1:])):
                coverage = round(235 + 13*math.sin(step*.073+ordinal*.9) +
                                 7*math.sin(step*.19+ordinal*.4))
                pen.line((a, b), fill=coverage, width=width, joint="curve")
        else:
            pen.line(pixels, fill=255, width=width, joint="curve")
        for x, y in (pixels[0], pixels[-1]):
            pen.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
    return mask.resize(image_size, Image.Resampling.LANCZOS)


def overlay(image_size: tuple[int, int], tile_width: int, color: tuple[int, int, int],
            stroke_width: float, owned: set[tuple[int, int]] | None = None,
            center: tuple[int, int] = CITY,
            terrain: dict[tuple[int, int], tuple[int, int]] | None = None) -> Image.Image:
    """Drape a soft, translucent civ-color ribbon across tile-edge relief."""
    scale = 3
    relief = ReliefStudy(terrain, tile_width, image_size, center) if terrain else None
    paths = draped_paths(owned if owned is not None else city_territory(center),
                         image_size, tile_width, center, relief)
    width = stroke_width * (tile_width / 128) ** 0.65
    soft = painted_mask(paths, image_size, width*1.5, scale).filter(
        ImageFilter.GaussianBlur(radius=max(0.8, width*0.48)))
    main = painted_mask(paths, image_size, width, scale, texture=True).filter(
        ImageFilter.GaussianBlur(radius=max(0.35, width*0.12)))
    centerline = painted_mask(paths, image_size, max(1.0, width*0.35), scale).filter(
        ImageFilter.GaussianBlur(radius=max(0.35, width*0.12)))
    result = Image.new("RGBA", image_size)
    for mask, strength in (
        (soft, 105),
        (main, 175),
        (centerline, 45),
    ):
        layer = Image.new("RGBA", image_size, (*color, 0))
        layer.putalpha(mask.point(lambda value: value*strength//255))
        result = Image.alpha_composite(result, layer)
    return result


def compose(source: Image.Image, tile_width: int, case: str,
            civ_color: tuple[int, int, int] | None = None,
            owned: set[tuple[int, int]] | None = None,
            center: tuple[int, int] = CITY,
            terrain: dict[tuple[int, int], tuple[int, int]] | None = None) -> Image.Image:
    if case not in CASES:
        raise ValueError("unknown border preview case")
    if tile_width not in (64, 128, 256):
        raise ValueError("unsupported border study zoom")
    base = source.convert("RGBA")
    color, stroke_width = CASES[case]
    if civ_color is not None:
        if len(civ_color) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in civ_color):
            raise ValueError("civ color must be three RGB bytes")
        color = civ_color
    return Image.alpha_composite(base, overlay(base.size, tile_width, color, stroke_width,
                                               owned, center, terrain)).convert("RGB")


def render_bitmap(path: Path, tile_width: int, case: str,
                  terrain_csv: Path | None = None) -> None:
    with Image.open(path) as source:
        result = compose(source, tile_width, case,
                         terrain=read_biq_terrain(terrain_csv) if terrain_csv else None)
    result.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a border draft over an existing city bitmap")
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--case", choices=tuple(CASES), default="crimson-brush")
    parser.add_argument("--color", help="Override the sample civilization color, as #RRGGBB")
    parser.add_argument("--site", help="BIQ city tile as x,y, matching the background's view center")
    parser.add_argument("--terrain-csv", type=Path, help="BIQ terrain capture for checked city territory tiles")
    args = parser.parse_args()
    civ_color = None
    if args.color is not None:
        if len(args.color) != 7 or not args.color.startswith("#"):
            parser.error("--color must use #RRGGBB")
        try:
            civ_color = ImageColor.getrgb(args.color)
        except ValueError:
            parser.error("--color must use #RRGGBB")
    if bool(args.site) != bool(args.terrain_csv):
        parser.error("--site and --terrain-csv must be supplied together")
    center = CITY
    owned = None
    terrain = None
    if args.site:
        try:
            center = tuple(map(int, args.site.split(",")))
        except ValueError:
            parser.error("--site must be x,y")
        if len(center) != 2:
            parser.error("--site must be x,y")
        terrain = read_biq_terrain(args.terrain_csv)
        owned = biq_city_territory(center, args.terrain_csv)
    with Image.open(args.background) as source:
        result = compose(source, args.tile_width, args.case, civ_color, owned, center, terrain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == "__main__":
    main()
