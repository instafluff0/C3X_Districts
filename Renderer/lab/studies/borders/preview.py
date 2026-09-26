"""Lab-only continuous civ-color border over a production-rendered city scene."""
from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter
from Renderer.lab.studies.borders.mesh_surface import GroundSurface, ProjectedTerrain, RenderDepth

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
    offsets = {(dc, dr) for dc in range(-1, 2) for dr in range(-1, 2)} | {(2, 0)}
    selected = {(center[0] + dc + dr, center[1] + dc - dr) for dc, dr in offsets}
    if not all(tile in terrain and terrain[tile][0] < 11 for tile in selected):
        raise ValueError("selected city territory is not all BIQ land tiles")
    return selected


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
                 surface: GroundSurface | None = None,
                 depths: list[list[float]] | None = None,
                 flat_paths: list[list[tuple[float, float]]] | None = None) -> list[list[tuple[float, float]]]:
    """Sample the tile perimeter densely and project onto renderer triangles."""
    paths = []
    for corners in perimeter_loops(owned, tile_width, image_size, center):
        points = round_tile_corners(corners)
        distances = arc_lengths(points)
        total = distances[-1]
        samples = max(2, math.ceil(total / 2.0))
        path = []
        path_depths = []
        flat_path = []
        for step in range(samples + 1):
            x, y = point_at(points, distances, total * step / samples)
            flat_path.append((x, y))
            if surface is not None:
                projected_x, projected_y, depth = surface.project_with_depth(
                    x, y, image_size, tile_width, center)
                path.append((projected_x, projected_y))
                path_depths.append(depth)
            else:
                path.append((x, y))
        paths.append(path)
        if depths is not None:
            depths.append(path_depths)
        if flat_paths is not None:
            flat_paths.append(flat_path)
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


def coherent_occlusion(flags: list[bool],
                       strong: list[bool] | None = None) -> list[bool]:
    """Keep coherent crossings with convincing foreground depth."""
    if not flags:
        return flags
    count = len(flags)
    joined = [value or (flags[(index-1) % count] and flags[(index+1) % count])
              for index, value in enumerate(flags)]
    clean = [value and sum(joined[(index+delta) % count]
                           for delta in (-2, -1, 0, 1, 2)) >= 3
             for index, value in enumerate(joined)]
    if strong is None or not any(clean):
        return clean
    if all(clean):
        return clean if sum(strong) >= 4 else [False]*count
    result = clean[:]
    start = next(index for index, value in enumerate(clean) if not value)
    position = 1
    while position <= count:
        index = (start+position) % count
        if not clean[index]:
            position += 1
            continue
        run = []
        while position <= count and clean[(start+position) % count]:
            run.append((start+position) % count)
            position += 1
        if sum(strong[item] for item in run) < 4:
            for item in run:
                result[item] = False
    return result


def inward_fade_mask(flat_paths: list[list[tuple[float, float]]],
                     image_size: tuple[int, int], stroke_width: float,
                     scale: int, surface: GroundSurface | None = None,
                     tile_width: int = 128, center: tuple[int, int] = CITY,
                     occluders: ProjectedTerrain | RenderDepth | None = None) -> Image.Image:
    """Build a varied, fading inward band from ground-draped parallel paths."""
    mask = Image.new("L", (image_size[0]*scale, image_size[1]*scale))
    pen = ImageDraw.Draw(mask)
    reach = stroke_width*5.0
    steps = max(2, math.ceil(reach/1.6))
    line_width = max(2, round(2.9*scale))
    for loop_index, flat_path in enumerate(flat_paths):
        ring = flat_path[:-1] if flat_path[0] == flat_path[-1] else flat_path
        signed_area = sum(a[0]*b[1]-b[0]*a[1]
                          for a, b in zip(ring, ring[1:]+ring[:1]))
        direction = 1 if signed_area > 0 else -1
        normals = []
        for index in range(len(ring)):
            before, after = ring[index-1], ring[(index+1) % len(ring)]
            dx, dy = after[0]-before[0], after[1]-before[1]
            length = math.hypot(dx, dy)
            normals.append((-dy/length*direction, dx/length*direction))
        # Draw from the transparent inner edge toward the stripe, so nearby
        # stronger samples cover any overlap between parallel contours.
        for radial_step in range(steps, -1, -1):
            distance = reach*radial_step/steps
            projected = []
            coverage = []
            hidden = []
            strong = []
            for index, ((x, y), (nx, ny)) in enumerate(zip(ring, normals)):
                flat_x, flat_y = x+nx*distance, y+ny*distance
                if surface is None:
                    px, py, depth = flat_x, flat_y, None
                else:
                    px, py, depth = surface.project_with_depth(
                        flat_x, flat_y, image_size, tile_width, center)
                edge_variation = 0.93+0.05*math.sin(index*.051+loop_index*1.9)+\
                                 0.025*math.sin(index*.17+loop_index*.8)
                fade = max(0.0, 1-distance/(reach*edge_variation))**1.25
                texture = 0.87+0.09*math.sin(index*.073+loop_index*.9)+\
                          0.07*math.sin(index*.21+distance*.14)
                opacity = round(145*fade*texture)
                projected.append((px*scale, py*scale))
                coverage.append(opacity)
                hidden.append(occluders is not None and depth is not None and
                              occluders.is_occluded(px, py, depth, clearance=20.0))
                if isinstance(occluders, RenderDepth) and depth is not None:
                    strong.append(occluders.is_occluded(
                        px, py, depth, clearance=85.0*image_size[1]/800))
            for index, occluded in enumerate(coherent_occlusion(
                    hidden, strong if strong else None)):
                if occluded:
                    coverage[index] = round(coverage[index]*.34)
            projected.append(projected[0])
            coverage.append(coverage[0])
            for (a, b), opacity in zip(zip(projected, projected[1:]), coverage):
                if opacity:
                    pen.line((a, b), fill=opacity, width=line_width, joint="curve")
    band = mask.resize(image_size, Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(radius=0.45))
    # Smooth, fixed-value noise breaks up a perfectly even translucent ribbon
    # without turning the border into dashes or changing its color.
    values = bytearray(band.tobytes())
    noise_grid = {}
    def noise(gx: int, gy: int) -> float:
        key = gx, gy
        if key not in noise_grid:
            value = (gx*374761393+gy*668265263) & 0xffffffff
            value = ((value ^ (value >> 13))*1274126177) & 0xffffffff
            noise_grid[key] = (value & 255)/255
        return noise_grid[key]
    for index, alpha in enumerate(values):
        if not alpha:
            continue
        x, y = index % image_size[0], index // image_size[0]
        gx, gy = x//11, y//11
        u, v = (x%11)/11, (y%11)/11
        u, v = u*u*(3-2*u), v*v*(3-2*v)
        variation = ((noise(gx, gy)*(1-u)+noise(gx+1, gy)*u)*(1-v)+
                     (noise(gx, gy+1)*(1-u)+noise(gx+1, gy+1)*u)*v)
        grain = ((x*73856093 ^ y*19349663) & 31)/31-0.5
        values[index] = min(255, round(alpha*(0.80+0.40*variation+0.06*grain)))
    return Image.frombytes("L", image_size, bytes(values))


def terrain_visibility(paths: list[list[tuple[float, float]]],
                       depths: list[list[float]], terrain: ProjectedTerrain | RenderDepth,
                       image_size: tuple[int, int], stroke_width: float) -> Image.Image:
    """Lower border opacity only where projected terrain is nearer to camera."""
    hidden = Image.new("L", image_size)
    pen = ImageDraw.Draw(hidden)
    for path, path_depths in zip(paths, depths):
        occluded = coherent_occlusion(
            [terrain.is_occluded(x, y, depth)
             for (x, y), depth in zip(path, path_depths)],
            [terrain.is_occluded(x, y, depth, clearance=85.0*image_size[1]/800)
             for (x, y), depth in zip(path, path_depths)]
            if isinstance(terrain, RenderDepth) else None)
        for i, (a, b) in enumerate(zip(path, path[1:])):
            if occluded[i] or occluded[i+1]:
                pen.line((a, b), fill=255, width=max(3, round(stroke_width*2.2)))
        # The last and first samples coincide on a closed territory perimeter.
        if occluded[-1] or occluded[0]:
            pen.line((path[-1], path[0]), fill=255,
                     width=max(3, round(stroke_width*2.2)))
    hidden = hidden.filter(ImageFilter.GaussianBlur(radius=1.5))
    return hidden.point(lambda value: 255-round(value*0.66))


def overlay(image_size: tuple[int, int], tile_width: int, color: tuple[int, int, int],
            stroke_width: float, owned: set[tuple[int, int]] | None = None,
            center: tuple[int, int] = CITY,
            surface: GroundSurface | None = None) -> Image.Image:
    """Drape a soft, translucent civ-color ribbon across tile-edge relief."""
    scale = 3
    depths = []
    flat_paths = []
    paths = draped_paths(owned if owned is not None else city_territory(center),
                         image_size, tile_width, center, surface, depths, flat_paths)
    occluders = None
    if surface is not None:
        depth_files = list(surface.prefix.parent.glob(surface.prefix.name + ".depth.*_*.bin"))
        occluders = (RenderDepth(surface.prefix, image_size) if depth_files else
                     ProjectedTerrain(surface, image_size, tile_width, center))
    width = stroke_width * (tile_width / 128) ** 0.65
    soft = painted_mask(paths, image_size, width*1.25, scale).filter(
        ImageFilter.GaussianBlur(radius=max(0.65, width*0.32)))
    main = painted_mask(paths, image_size, width, scale, texture=True).filter(
        ImageFilter.GaussianBlur(radius=max(0.25, width*0.08)))
    centerline = painted_mask(paths, image_size, max(1.0, width*0.35), scale).filter(
        ImageFilter.GaussianBlur(radius=max(0.25, width*0.08)))
    result = Image.new("RGBA", image_size, (*color, 0))
    result.putalpha(inward_fade_mask(flat_paths, image_size, width, scale,
                                     surface, tile_width, center, occluders))
    for mask, strength in (
        (soft, 75),
        (main, 198),
        (centerline, 42),
    ):
        layer = Image.new("RGBA", image_size, (*color, 0))
        layer.putalpha(mask.point(lambda value: value*strength//255))
        result = Image.alpha_composite(result, layer)
    if surface is not None:
        visibility = terrain_visibility(paths, depths, occluders, image_size, width)
        result.putalpha(ImageChops.multiply(result.getchannel("A"), visibility))
    return result


def compose(source: Image.Image, tile_width: int, case: str,
            civ_color: tuple[int, int, int] | None = None,
            owned: set[tuple[int, int]] | None = None,
            center: tuple[int, int] = CITY,
            surface: GroundSurface | None = None) -> Image.Image:
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
                                               owned, center, surface)).convert("RGB")


def render_bitmap(path: Path, tile_width: int, case: str,
                  mesh_prefix: Path) -> None:
    with Image.open(path) as source:
        result = compose(source, tile_width, case,
                         surface=GroundSurface(mesh_prefix, tile_width))
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
    parser.add_argument("--mesh-prefix", type=Path,
                        help="Native renderer ground mesh export for the same city view")
    args = parser.parse_args()
    civ_color = None
    if args.color is not None:
        if len(args.color) != 7 or not args.color.startswith("#"):
            parser.error("--color must use #RRGGBB")
        try:
            civ_color = ImageColor.getrgb(args.color)
        except ValueError:
            parser.error("--color must use #RRGGBB")
    if any((args.site, args.terrain_csv, args.mesh_prefix)) and not all(
            (args.site, args.terrain_csv, args.mesh_prefix)):
        parser.error("--site, --terrain-csv and --mesh-prefix are required together")
    center = CITY
    owned = None
    surface = None
    if args.site:
        try:
            center = tuple(map(int, args.site.split(",")))
        except ValueError:
            parser.error("--site must be x,y")
        if len(center) != 2:
            parser.error("--site must be x,y")
        owned = biq_city_territory(center, args.terrain_csv)
        surface = GroundSurface(args.mesh_prefix, args.tile_width)
    with Image.open(args.background) as source:
        result = compose(source, args.tile_width, args.case, civ_color, owned, center, surface)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == "__main__":
    main()
