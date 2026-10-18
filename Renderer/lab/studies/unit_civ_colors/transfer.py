#!/usr/bin/env python3
"""Lab: place each 3D unit's civ colour where its Civ III sprite has it.

Civ III marks civ colour in unit sprites with palette indices 0..63 (replaced
from ntpNN.pcx); how much an index is civ colour is measured from how much it
varies across the 32 civ palettes. For each of the eight directions the
unit's DEFAULT sprite is fitted to the paired 3D idle pose, rendered in the
same direction at the live projection (128-pixel tiles): the frame centre is
the ground point, then scale (per axis) and shift are searched for the best
silhouette overlap, since the shapes differ.

Every texel of the model's own textures is a point on its surface. Each
direction that sees a texel (depth test) samples the sprite's civ colour
where that point projects, weighted by how squarely the surface faces that
view, so the frontal views decide and grazing ones barely count. The texel
field is smoothed within its UV island (a stripe stays on its part), then
thresholded so the model shows as much civ colour as the Civ III sprite does.

    python3 Renderer/lab/studies/unit_civ_colors/transfer.py [--units PRTO_Infantry,PRTO_Tank]

Writes a review gallery, summary and masks to Renderer/lab/out/unit-civ-colors/.
Mac-only Lab tool: no pack, staged or game file changes. No unit names
select behaviour; any unit with Civ III art and a 3D binding works.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "unit_readability"))
from image_io import write_png  # noqa: E402
from Renderer.tools.asset_compiler import civ3_flc  # noqa: E402
from Renderer.tools.asset_compiler import unit_owner_coverage as coverage  # noqa: E402
from Renderer.tools.asset_compiler.civ3_unit_sprites import default_flc, game_roots, pedia_names  # noqa: E402

PACK = ROOT / "Renderer/packs/UnitAnimationFidelity"
SOURCE_PACK = ROOT / "Renderer/packs/UnitAnimationRuntime"   # authored (pre-heuristic) owner masks
OUT = ROOT / "Renderer/lab/out/unit-civ-colors"
CIV3_ROOT = Path(os.environ.get("C3X_CIV3_ROOT", ROOT.parents[1]))
FLC_DIRECTIONS = (5, 4, 3, 2, 1, 8, 7, 6)   # FLC blocks SW S SE E NE N NW W -> Civ III direction
SHOWN = (5, 3, 1, 7)                        # gallery: SW, SE, NE, NW
DEFAULT_UNITS = ("PRTO_Infantry", "PRTO_Tank", "PRTO_Rifleman", "PRTO_Knight", "PRTO_Battleship", "PRTO_Bomber")
SUPERSAMPLE = 3
LIGHT = np.array([-.45, -.55, .7]) / np.linalg.norm([-.45, -.55, .7])
TOWARD_CAMERA = np.array([1.0, 1.0, 64 / coverage.Z_PIXELS]) / np.linalg.norm([1.0, 1.0, 64 / coverage.Z_PIXELS])
EDGE = .07          # soft threshold half-width
FACING_POWER = 8    # the most frontal view decides a texel
SEGMENT = .3        # share of a texel's evidence taken from its segment
SMOOTH = .02        # surface smoothing cell, a share of the model's size
MATERIAL = .06      # chromaticity/lightness distance under which colour clusters are one material
CLUSTERS = 6        # albedo colour clusters (materials) per texture
FLATTEN = .65       # owner areas: luminance pulled this far toward the local mean
BAND = .8           # how far evidence moves to its island's 1D profile (rings, bands)
GROW = .7           # hysteresis: connected areas join down to this share of the threshold
PRIOR = .05         # authored paintable surfaces (cloth, panels) win close calls


# ---- Civ III sprites -------------------------------------------------------------

def civness() -> np.ndarray:
    """Weight of each palette index as civ colour: how much it varies across
    the 32 ntpNN civ palettes (0..15 the civ ramp; even 16..63 civ blends;
    odd 16..63 fixed neutrals)."""
    palettes = np.stack([civ3_flc.read_pcx_palette(p).astype(float)
                         for p in sorted((CIV3_ROOT / "Art/Units/Palettes").glob("ntp*.pcx"))])
    weight = np.zeros(256)
    weight[:64] = np.clip(palettes[:, :64].std(0).mean(1) / 55, 0, 1)
    return weight


CIVNESS = None


def civ3_art(keys) -> Path | None:
    """DEFAULT animation of the first of a binding's Civ III keys that has
    art: its #ANIMNAME or folder, else the BIQ unit's art folder (matched by
    civilopedia entry), else its first era variant (Leader_ERAS_...)."""
    roots = game_roots(CIV3_ROOT)
    names = pedia_names(roots)
    biq_path = ROOT / "Renderer/inventory/vanilla_conquests_biq_semantics.json"
    folders = {}
    if biq_path.is_file():
        folders = {u["civilopedia_entry"].upper(): u.get("direct_art_folder")
                   for u in json.loads(biq_path.read_text())["unit_types"]}
    for key in keys:
        for name in (names.get(key), folders.get(key.upper()),
                     *[names[k] for k in sorted(names) if k.startswith(key + "_ERAS_")]):
            path = default_flc(roots, name) if name else None
            if path is not None:
                return path
    return None


def sprite_views(key: str, binding: dict | None = None):
    """Full-frame (unit-centred) palette indices per Civ III direction, and
    the animation's palette."""
    keys = [binding[f"key{i}"] for i in range(binding["key_count"])] if binding else [key]
    path = civ3_art(keys)
    if path is None:
        raise ValueError(f"no Civ III DEFAULT animation for {key}")
    flc = civ3_flc.read(path)
    views = {}
    for block in range(min(8, flc.directions)):
        frame = flc.direction_frame(block)
        full = np.full((flc.full_height, flc.full_width), 255, np.uint8)
        h, w = frame.shape
        full[flc.y_offset:flc.y_offset + h, flc.x_offset:flc.x_offset + w] = frame
        views[FLC_DIRECTIONS[block]] = full
    return views, flc.palette


# ---- 3D raster -------------------------------------------------------------------

def rotate(binding: dict, direction: int, points: np.ndarray, normals: np.ndarray):
    """Object space -> the live projection's tile space for a direction."""
    scale, offset = binding["scale"], binding["offset_z"]
    angle = math.radians(binding.get("yaw_offset", 225.0) + (direction % 8) * 45)
    c, s = math.cos(angle), math.sin(angle)
    x = (points[:, 0] * c - points[:, 1] * s) * scale
    y = (points[:, 0] * s + points[:, 1] * c) * scale
    z = (points[:, 2] + offset) * scale
    n = np.stack([normals[:, 0] * c - normals[:, 1] * s, normals[:, 0] * s + normals[:, 1] * c, normals[:, 2]], 1)
    return x, y, z, n


def screen(x, y, z):
    """Tile space -> pixels from the ground point, and depth (larger is nearer)."""
    return (x - y) * 64, (x + y) * 32 - z * coverage.Z_PIXELS, (x + y) * 5 + z * .1


def raster(parts: list, binding: dict, direction: int, supersample: int = SUPERSAMPLE) -> dict:
    """Idle pose at the live projection: per pixel part index, UV, shade and
    depth. Pixel (0, 0) is at (x0, y0) supersampled pixels from the ground point."""
    projected = []
    for i, part, points, triangles, uv, normals in parts:
        x, y, z, n = rotate(binding, direction, points, normals)
        sx, sy, d = screen(x, y, z)
        projected.append((sx * supersample, sy * supersample, d, z, n))
    xs = np.concatenate([p[0] for p in projected]); ys = np.concatenate([p[1] for p in projected])
    x0, y0 = int(np.floor(xs.min())) - 1, int(np.floor(ys.min())) - 1
    w, h = int(np.ceil(xs.max())) - x0 + 2, int(np.ceil(ys.max())) - y0 + 2
    depth = np.full((h, w), -np.inf); which = np.full((h, w), -1)
    tu = np.zeros((h, w)); tv = np.zeros((h, w)); shade = np.zeros((h, w)); normal_map = np.zeros((h, w, 3))
    for (i, part, points, triangles, uv, normals), (px, py, pd, pz, pn) in zip(parts, projected):
        for a, b, t in triangles:
            if pz[a] < 0 and pz[b] < 0 and pz[t] < 0:
                continue
            ax, ay, bx, by, cx, cy = px[a] - x0, py[a] - y0, px[b] - x0, py[b] - y0, px[t] - x0, py[t] - y0
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-9:
                continue
            left, right = int(max(0, np.floor(min(ax, bx, cx)))), int(min(w - 1, np.ceil(max(ax, bx, cx))))
            top, bottom = int(max(0, np.floor(min(ay, by, cy)))), int(min(h - 1, np.ceil(max(ay, by, cy))))
            gx, gy = np.meshgrid(np.arange(left, right + 1) + .5, np.arange(top, bottom + 1) + .5)
            u = ((bx - gx) * (cy - gy) - (by - gy) * (cx - gx)) / area
            v = ((cx - gx) * (ay - gy) - (cy - gy) * (ax - gx)) / area
            r = 1 - u - v
            inside = (u >= 0) & (v >= 0) & (r >= 0) & (u * pz[a] + v * pz[b] + r * pz[t] >= 0)
            d = u * pd[a] + v * pd[b] + r * pd[t]
            region = depth[top:bottom + 1, left:right + 1]
            win = inside & (d > region)
            if not win.any():
                continue
            region[win] = d[win]
            which[top:bottom + 1, left:right + 1][win] = i
            tu[top:bottom + 1, left:right + 1][win] = (u * uv[a, 0] + v * uv[b, 0] + r * uv[t, 0])[win]
            tv[top:bottom + 1, left:right + 1][win] = (u * uv[a, 1] + v * uv[b, 1] + r * uv[t, 1])[win]
            normal = (u[..., None] * pn[a] + v[..., None] * pn[b] + r[..., None] * pn[t])
            normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
            normal_map[top:bottom + 1, left:right + 1][win] = normal[win]
            shade[top:bottom + 1, left:right + 1][win] = (.38 + .62 * np.clip((normal * LIGHT).sum(-1), 0, 1))[win]
    return {"which": which, "u": tu, "v": tv, "shade": shade, "normal": normal_map, "depth": depth,
            "x0": x0, "y0": y0, "supersample": supersample}


def box(mask: np.ndarray):
    rows, cols = np.nonzero(mask)
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


# ---- sprite fit ------------------------------------------------------------------

def bilinear(grid: dict, x, y):
    """Control-grid cell corners and weights of model points (clamped)."""
    gx = np.clip((np.asarray(x, float) - grid["gx0"]) / grid["g"], 0, grid["nx"] - 1.001)
    gy = np.clip((np.asarray(y, float) - grid["gy0"]) / grid["g"], 0, grid["ny"] - 1.001)
    i, j = gy.astype(int), gx.astype(int)
    fy, fx = gy - i, gx - j
    corners = np.stack([i * grid["nx"] + j, i * grid["nx"] + j + 1, (i + 1) * grid["nx"] + j, (i + 1) * grid["nx"] + j + 1], -1)
    weights = np.stack([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], -1)
    return corners, weights


def to_sprite(fit: dict, x, y):
    """Pixels from the ground point -> sprite row/column (float): the global
    fit plus the local warps (coarse, then fine)."""
    r, c = fit["cy"] + y * fit["sy"] + fit["dy"], fit["cx"] + x * fit["sx"] + fit["dx"]
    for grid in fit.get("warps", ()):
        corners, weights = bilinear(grid, x, y)
        r = r + (grid["warp"][corners, 0] * weights).sum(-1)
        c = c + (grid["warp"][corners, 1] * weights).sum(-1)
    return r, c


def edges(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Softened gradient magnitude, unit mean over `valid`."""
    e = np.zeros(image.shape[:2])
    for axis in (0, 1):
        step = np.abs(np.diff(image, axis=axis))
        step = step.sum(-1) if step.ndim == 3 else step
        pad = [(0, 0), (0, 0)]; pad[axis] = (0, 1)
        e += np.pad(step, pad)
    e = coverage.soften(coverage.soften(e))
    return e / max(float(e[valid].mean()), 1e-9)


def refine(fit: dict, view: dict, sprite: np.ndarray, luma: np.ndarray,
           levels=((24, 28, (8, 4, 2)), (10, 8, (2, 1))), edge_weight: float = .5, stiffness: float = .2) -> dict:
    """Local warps on control grids so the sprite's parts land on the
    model's (a turret on the turret, a coat on the coat, wingtips on longer
    wings): silhouettes agree and the sprite's shading edges follow the
    model's geometry edges. Greedy coordinate search with a smoothness cost,
    per level (grid spacing, reach in pixels, steps): coarse, then fine."""
    for spacing, reach, steps in levels:
        fit = warp_level(fit, view, sprite, luma, spacing, reach, steps, edge_weight, stiffness)
    return fit


def warp_level(fit: dict, view: dict, sprite: np.ndarray, luma: np.ndarray, spacing: int, reach: int, steps,
               edge_weight: float, stiffness: float) -> dict:
    ss = view["supersample"]
    visible = view["which"] >= 0
    near = visible.copy()
    for _ in range(6):
        near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
    depth = np.where(visible, view["depth"], 0.0)
    span = max(float(np.ptp(depth[visible])), 1e-9) if visible.any() else 1.0
    geometry = np.concatenate([view["normal"], (depth / span * 4)[..., None], visible[..., None] * 2.0], -1)
    model_edge = edges(geometry, near)
    body = sprite < 224
    sprite_edge = edges(np.where(body, luma, 0.0), body)
    rows, cols = np.nonzero(near)
    x = (cols + .5 + view["x0"]) / ss; y = (rows + .5 + view["y0"]) / ss
    inside = visible[rows, cols]; e_m = model_edge[rows, cols]
    grid = {"g": spacing, "gx0": float(x.min()), "gy0": float(y.min())}
    grid["nx"] = int(math.ceil((x.max() - x.min()) / spacing)) + 2
    grid["ny"] = int(math.ceil((y.max() - y.min()) / spacing)) + 2
    warp = np.zeros((grid["nx"] * grid["ny"], 2))
    base_r, base_c = to_sprite(fit, x, y)
    corners, weights = bilinear(grid, x, y)
    support = [[] for _ in range(len(warp))]
    for k in range(4):
        for p, (corner, w) in enumerate(zip(corners[:, k], weights[:, k])):
            if w > 0:
                support[corner].append((p, w))
    support = [(np.array([p for p, _ in s], int), np.array([w for _, w in s])) for s in support]
    neighbours = [[] for _ in range(len(warp))]
    for i in range(grid["ny"]):
        for j in range(grid["nx"]):
            for di, dj in ((0, 1), (1, 0)):
                if i + di < grid["ny"] and j + dj < grid["nx"]:
                    a, b = i * grid["nx"] + j, (i + di) * grid["nx"] + j + dj
                    neighbours[a].append(b); neighbours[b].append(a)

    def score(idx, r, c):
        r, c = r.astype(int), c.astype(int)
        ok = (r >= 0) & (r < sprite.shape[0]) & (c >= 0) & (c < sprite.shape[1])
        on = np.zeros(len(idx), bool); on[ok] = body[r[ok], c[ok]]
        e = np.zeros(len(idx)); e[ok] = sprite_edge[r[ok], c[ok]]
        return float((on == inside[idx]).sum() + edge_weight * (e_m[idx] * e).sum())

    def offsets(idx):
        cw = weights[idx][..., None] * warp[corners[idx]]
        return cw[..., 0].sum(-1), cw[..., 1].sum(-1)

    for step in steps:
        for _ in range(2):
            for k, (idx, w) in enumerate(support):
                if not len(idx):
                    continue
                dr, dc = offsets(idx)
                current = score(idx, base_r[idx] + dr, base_c[idx] + dc)
                bend = sum(((warp[k] - warp[n]) ** 2).sum() for n in neighbours[k])
                best, choice = 0.0, None
                for sr, sc in ((step, 0), (-step, 0), (0, step), (0, -step), (step, step), (-step, -step), (step, -step), (-step, step)):
                    moved = warp[k] + (sr, sc)
                    if np.abs(moved).max() > reach:
                        continue
                    gain = score(idx, base_r[idx] + dr + w * sr, base_c[idx] + dc + w * sc) - current
                    gain -= stiffness * (sum(((moved - warp[n]) ** 2).sum() for n in neighbours[k]) - bend)
                    if gain > best:
                        best, choice = gain, moved
                if choice is not None:
                    warp[k] = choice
    grid["warp"] = warp
    return dict(fit, warps=list(fit.get("warps", ())) + [grid])


def align(view: dict, sprite: np.ndarray) -> dict:
    """Fit the sprite to the model's silhouette: the frame centre is the
    ground point; per-axis scale and shift maximise the overlap (IoU)."""
    ss = view["supersample"]
    visible = view["which"] >= 0
    body = sprite < 224
    rows, cols = np.nonzero(visible)
    x = (cols + .5 + view["x0"]) / ss
    y = (rows + .5 + view["y0"]) / ss
    model_area = len(rows) / ss ** 2
    base = math.sqrt(body.sum() / max(model_area, 1e-9))
    fit = {"cy": sprite.shape[0] / 2, "cx": sprite.shape[1] / 2}

    def iou(sx, sy, dx, dy):
        r = (fit["cy"] + y * sy + dy).astype(int); c = (fit["cx"] + x * sx + dx).astype(int)
        ok = (r >= 0) & (r < sprite.shape[0]) & (c >= 0) & (c < sprite.shape[1])
        hit = np.zeros(len(r), bool); hit[ok] = body[r[ok], c[ok]]
        both = hit.sum() / ss ** 2 * sx * sy
        return float(both / max(model_area * sx * sy + body.sum() - both, 1e-9))

    best = max((iou(base, base, dx, dy), base, base, dx, dy) for dy in range(-12, 13, 2) for dx in range(-12, 13, 2))
    _, _, _, dx0, dy0 = best
    for fx in (.86, .93, 1, 1.07, 1.14):
        for fy in (.86, .93, 1, 1.07, 1.14):
            for dy in range(dy0 - 4, dy0 + 5, 2):
                for dx in range(dx0 - 4, dx0 + 5, 2):
                    best = max(best, (iou(base * fx, base * fy, dx, dy), base * fx, base * fy, dx, dy))
    fit.update(iou=best[0], sx=best[1], sy=best[2], dx=best[3], dy=best[4])
    return fit


# ---- texels as surface points ------------------------------------------------------

def components(tri: np.ndarray, count: int) -> np.ndarray:
    """Triangle -> connected component over shared vertex ids (0..n-1)."""
    label = np.arange(count)
    while True:
        lowest = label[tri].min(1)
        new = label.copy()
        np.minimum.at(new, tri.ravel(), np.repeat(lowest, 3))
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    return np.unique(label[tri[:, 0]], return_inverse=True)[1].ravel()


def texel_samples(parts: list, texture: str, size, extent: float = 1.0) -> dict:
    """Every texel of `texture` covered by a triangle, as a posed surface
    point: texel coordinates, position, normal, UV island (triangles
    connected through shared vertices) and piece (connected through shared
    positions: a turret, a hull, a sleeve). Mirrored UVs give a texel
    several points."""
    th, tw = size
    out = {"ty": [], "tx": [], "pos": [], "nrm": [], "island": [], "piece": []}
    next_island = next_piece = 0
    for i, part, points, triangles, uv, normals in parts:
        if part["texture"] != texture or not len(triangles):
            continue
        tri = np.asarray(triangles)
        islands = components(tri, len(points)) + next_island
        next_island = islands.max() + 1
        _, weld = np.unique(np.round(points / (extent * 2e-3)).astype(np.int64), axis=0, return_inverse=True)
        weld = weld.ravel()
        pieces = components(weld[tri], weld.max() + 1) + next_piece
        next_piece = pieces.max() + 1
        us, vs = uv[:, 0] * tw, uv[:, 1] * th
        for (a, b, t), island, piece in zip(tri, islands, pieces):
            ax, ay, bx, by, cx, cy = us[a], vs[a], us[b], vs[b], us[t], vs[t]
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-12:
                continue
            left, right = int(np.floor(min(ax, bx, cx))), int(np.ceil(max(ax, bx, cx)))
            top, bottom = int(np.floor(min(ay, by, cy))), int(np.ceil(max(ay, by, cy)))
            gx, gy = np.meshgrid(np.arange(left, right + 1) + .5, np.arange(top, bottom + 1) + .5)
            u = ((bx - gx) * (cy - gy) - (by - gy) * (cx - gx)) / area
            v = ((cx - gx) * (ay - gy) - (cy - gy) * (ax - gx)) / area
            r = 1 - u - v
            inside = (u >= -.02) & (v >= -.02) & (r >= -.02)
            if not inside.any():   # sub-texel triangle: its centroid
                gx, gy = np.array([[(ax + bx + cx) / 3]]), np.array([[(ay + by + cy) / 3]])
                u = v = r = np.full((1, 1), 1 / 3); inside = np.ones((1, 1), bool)
            u, v, r = u[inside], v[inside], r[inside]
            out["tx"].append(np.mod(gx[inside].astype(int), tw)); out["ty"].append(np.mod(gy[inside].astype(int), th))
            out["pos"].append(u[:, None] * points[a] + v[:, None] * points[b] + r[:, None] * points[t])
            n = u[:, None] * normals[a] + v[:, None] * normals[b] + r[:, None] * normals[t]
            out["nrm"].append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9))
            out["island"].append(np.full(len(u), island)); out["piece"].append(np.full(len(u), piece))
    if not out["tx"]:
        return {k: np.zeros((0, 3) if k in ("pos", "nrm") else 0, float if k in ("pos", "nrm") else int) for k in out}
    return {k: np.concatenate(v) for k, v in out.items()}


def sample_views(samples: dict, binding: dict, direction: int, view: dict, fit: dict, sprite: np.ndarray):
    """Civ colour the sprite shows where each texel point projects, and the
    weight of that look: zero when hidden or off the sprite, else the
    facing toward the camera, sharpened so the most frontal view decides."""
    x, y, z, n = rotate(binding, direction, samples["pos"], samples["nrm"])
    sx, sy, d = screen(x, y, z)
    ss = view["supersample"]
    col = np.floor(sx * ss - view["x0"]).astype(int); row = np.floor(sy * ss - view["y0"]).astype(int)
    h, w = view["depth"].shape
    ok = (row >= 0) & (row < h) & (col >= 0) & (col < w) & (z >= 0)
    front = np.full(len(d), -np.inf); front[ok] = view["depth"][row[ok], col[ok]]
    span = np.ptp(view["depth"][np.isfinite(view["depth"])]) if np.isfinite(view["depth"]).any() else 1.0
    ok &= d >= front - .015 * span
    sr, sc = to_sprite(fit, sx, sy)
    sr, sc = sr.astype(int), sc.astype(int)
    ok &= (sr >= 0) & (sr < sprite.shape[0]) & (sc >= 0) & (sc < sprite.shape[1])
    index = np.full(len(d), 255); index[ok] = sprite[sr[ok], sc[ok]]
    ok &= index < 224
    facing = np.clip((n * TOWARD_CAMERA).sum(1), 0, 1)
    weight = np.where(ok, facing ** FACING_POWER, 0.0)
    return np.where(ok, CIVNESS[np.minimum(index, 255)], 0.0), weight


def segments(rgb: np.ndarray, islands: np.ndarray, k: int = CLUSTERS) -> np.ndarray:
    """Texel -> segment: its UV island split by material colour (k-means on
    the softened texture's chromaticity, lightness weighted down so lit and
    shadowed shades of one paint stay one material)."""
    blurred = rgb
    for _ in range(3):
        blurred = np.stack([coverage.soften(blurred[..., c]) for c in range(3)], -1)
    total = np.maximum(blurred.sum(-1, keepdims=True), 1e-3)
    luma = (blurred * [.2126, .7152, .0722]).sum(-1, keepdims=True)
    feature = np.concatenate([blurred[..., :2] / total, .35 * luma], -1)
    covered = islands >= 0
    pick = feature[covered]
    if not len(pick):
        return np.full(islands.shape, -1)
    pick = pick[np.linspace(0, len(pick) - 1, min(len(pick), 6000)).astype(int)]
    order = np.argsort(pick[:, 2])
    centres = pick[order[np.linspace(0, len(pick) - 1, k).astype(int)]].copy()
    for _ in range(15):
        assign = ((pick[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (assign == j).any():
                centres[j] = pick[assign == j].mean(0)
    # One paint is one material: centres closer than MATERIAL merge.
    label = np.arange(k)
    for i in range(k):
        for j in range(i):
            if np.sqrt(((centres[i] - centres[j]) ** 2).sum()) < MATERIAL:
                label[label == label[i]] = label[j]
    cluster = label[((feature[..., None, :] - centres) ** 2).sum(-1).argmin(-1)]
    return np.where(covered, islands * k + cluster, -1)


def surface_graph(pos, normal, material, cell: float) -> dict:
    """The model's surface as a graph of small patches: samples grouped into
    3D cells of `cell` per material and facing (the dominant normal axis and
    sign). Patches in touching cells link when they share a material and do
    not face opposite ways, so evidence and growth follow the surface around
    a turret or along a sleeve, but never cross into another material or
    through a thin sheet to its other side."""
    ijk = np.floor((pos - pos.min(0)) / cell).astype(np.int64) + 1
    axis = np.abs(normal).argmax(1)
    facing = axis * 2 + (normal[np.arange(len(normal)), axis] > 0)
    span = int(ijk.max()) + 2

    def key(m, f, i, j, k):
        return (((m * 6 + f) * span + i) * span + j) * span + k

    nodes, node = np.unique(key(material.astype(np.int64), facing, ijk[:, 0], ijk[:, 1], ijk[:, 2]), return_inverse=True)
    k = nodes % span; j = nodes // span % span; i = nodes // span ** 2 % span
    f = nodes // span ** 3 % 6; m = nodes // span ** 3 // 6
    a, b = [], []
    for other in range(6):
        compatible = (other // 2 != f // 2) | (other == f)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    want = key(m, np.full_like(f, other), i + di, j + dj, k + dk)
                    at = np.minimum(np.searchsorted(nodes, want), len(nodes) - 1)
                    hit = compatible & (nodes[at] == want) & (at != np.arange(len(nodes)))
                    a.append(np.nonzero(hit)[0]); b.append(at[hit])
    return {"node": node, "count": len(nodes), "a": np.concatenate(a), "b": np.concatenate(b)}


def mirror(pos, value, weight, material, cell: float):
    """Units are left/right symmetric: each sample's evidence is pooled with
    its mirror image across the model's centre plane (the horizontal axis
    that maps the surface onto itself best), so a mark one view missed on
    one side comes from the other. Unmatched samples keep their own."""
    best = (0.0, None)
    for axis in (0, 1):
        centre = (pos[:, axis].min() + pos[:, axis].max()) / 2
        flipped = pos.copy(); flipped[:, axis] = 2 * centre - pos[:, axis]
        lo = np.minimum(pos.min(0), flipped.min(0))
        span = int(np.ceil((np.maximum(pos.max(0), flipped.max(0)) - lo).max() / cell)) + 3

        def key(q):
            ijk = np.floor((q - lo) / cell).astype(np.int64) + 1
            return ((material.astype(np.int64) * span + ijk[:, 0]) * span + ijk[:, 1]) * span + ijk[:, 2]

        nodes, node = np.unique(key(pos), return_inverse=True)
        want = key(flipped)
        at = np.minimum(np.searchsorted(nodes, want), len(nodes) - 1)
        found = nodes[at] == want
        if found.mean() > best[0]:
            best = (found.mean(), (node.ravel(), at, found, len(nodes)))
    share, match = best
    if match is None or share < .5:
        return value, weight
    node, at, found, count = match
    total = np.bincount(node, value * weight, count); mass = np.bincount(node, weight, count)
    pooled_total = total[node] + np.where(found, total[at], 0.0)
    pooled_mass = mass[node] + np.where(found, mass[at], 0.0)
    return np.where(pooled_mass > 1e-12, pooled_total / np.maximum(pooled_mass, 1e-12), value), pooled_mass


def graph_smooth(graph: dict, value, weight, passes: int = 1):
    """Weighted neighbour averaging over the surface graph; unseen samples
    take their neighbours' value."""
    total = np.bincount(graph["node"], value * weight, graph["count"])
    mass = np.bincount(graph["node"], weight, graph["count"])
    for _ in range(passes):
        total = total + np.bincount(graph["a"], total[graph["b"]], graph["count"])
        mass = mass + np.bincount(graph["a"], mass[graph["b"]], graph["count"])
    return np.where(mass > 1e-12, total / np.maximum(mass, 1e-12), np.nan)[graph["node"]]


def band_pool(pos, value, weight, piece, normal, bins: int = 14, report=None) -> np.ndarray:
    """Civ III's civ colour follows simple structure: a ring around a
    turret, a tunic from shoulder to waist, wingtip bands, a stripe across a
    hull. Per mesh piece, evidence is replaced (in part) by a 1D profile:
    an upright piece whose evidence goes most of the way around it becomes
    a ring (profile by height); otherwise bands run across its longest axis,
    as far as that profile explains the evidence."""
    out = value.copy()
    up = np.array([0.0, 0.0, 1.0])
    for k in np.unique(piece):
        sel = np.nonzero(piece == k)[0]
        w = weight[sel]
        if len(sel) < 64 or w.sum() < 1e-6:
            continue
        p = pos[sel] - (pos[sel] * w[:, None]).sum(0) / w.sum()
        v = value[sel]
        mean = (v * w).sum() / w.sum()
        spread = ((v - mean) ** 2 * w).sum()
        if spread < 1e-9:
            continue

        def profile(axis, only=None):
            c = (p * axis).sum(1)
            lo, hi = c.min(), c.max()
            b = np.minimum(((c - lo) / max(hi - lo, 1e-9) * bins).astype(int), bins - 1)
            m = w if only is None else w * only
            mass = np.bincount(b, m, bins)
            prof = np.where(mass > 1e-9, np.bincount(b, v * m, bins) / np.maximum(mass, 1e-12), mean)
            centres = lo + (np.arange(bins) + .5) * (hi - lo) / bins
            return float((mass * (prof - mean) ** 2).sum() / spread), np.interp(c, centres, prof)

        # Ring: upright, and the side walls carry evidence most of the way round.
        horizontal = np.ptp(p[:, :2], 0).max()
        side = np.abs(normal[sel, 2]) < .6
        ring = False
        if np.ptp(p[:, 2]) >= .3 * horizontal and side.sum() > 32:
            angle = np.arctan2(p[side, 1], p[side, 0])
            sector = ((angle + np.pi) / (2 * np.pi) * 8).astype(int) % 8
            level = np.bincount(sector, v[side] * w[side], 8) / np.maximum(np.bincount(sector, w[side], 8), 1e-12)
            ring = level.max() >= .1 and (level >= .4 * level.max()).mean() >= .6
        if ring:
            # Walls only: a roof seen from every view would take the band's
            # place (Civ III bands are on walls); roofs keep their evidence.
            out[sel[side]] = (1 - BAND) * v[side] + BAND * profile(up, side)[1][side]
            if report is not None:
                report.append({"piece": int(k), "samples": len(sel), "mean": round(float(mean), 3), "mode": "ring"})
            continue
        else:
            # Bands run across the piece's longest axis: a wing's span, a
            # hull's length, a figure's height.
            _, _, axes = np.linalg.svd(p[np.linspace(0, len(p) - 1, min(len(p), 4000)).astype(int)], full_matrices=False)
            share, pooled = profile(axes[0])
        trust = BAND * min(1.0, share / .2)
        out[sel] = (1 - trust) * v + trust * pooled
        if report is not None:
            report.append({"piece": int(k), "samples": len(sel), "mean": round(float(mean), 3), "mode": "band",
                           "explained": round(share, 3), "trust": round(trust, 3)})
    return out


def graph_grow(graph: dict, value, high: float, low: float) -> np.ndarray:
    """Hysteresis: patches at `high` seed; connected patches at `low` join,
    so a ring or a garment completes where one view saw it weakly."""
    eligible = value >= low
    chosen = value >= high
    while True:
        reach = np.bincount(graph["a"], chosen[graph["b"]].astype(float), graph["count"]) > 0
        grown = chosen | (eligible & reach)
        if (grown == chosen).all():
            return chosen
        chosen = grown


def soft(value, tau):
    t = np.clip((np.nan_to_num(value, nan=0.0) - tau + EDGE) / (2 * EDGE), 0, 1)
    return t * t * (3 - 2 * t)


# ---- transfer ------------------------------------------------------------------------

def texel_mean(per_sample, samples: dict, size) -> np.ndarray:
    """Per-sample values -> texel map (mean; NaN where uncovered)."""
    total, count = np.zeros(size), np.zeros(size)
    np.add.at(total, (samples["ty"], samples["tx"]), per_sample); np.add.at(count, (samples["ty"], samples["tx"]), 1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def visible_texels(result: dict, view: dict):
    """Per part: its visible pixels, texture and texel coordinates."""
    for i, part, *_ in result["parts"]:
        mine = view["which"] == i
        if not mine.any():
            continue
        t = part["texture"]; th, tw = result["size"][t]
        tx = np.floor(np.mod(view["u"][mine], 1) * tw).astype(int) % tw
        ty = np.floor(np.mod(view["v"][mine], 1) * th).astype(int) % th
        yield mine, part, t, ty, tx


def visible_share(result: dict, mask_of) -> float:
    shares = []
    for view in result["views"].values():
        visible = view["which"] >= 0
        weight = np.zeros(visible.shape)
        for mine, part, t, ty, tx in visible_texels(result, view):
            weight[mine] = mask_of(t)[ty, tx]
        if visible.any():
            shares.append(float(weight[visible].mean()))
    return float(np.median(shares)) if shares else 0.0


def authored(key: str, parts: list, size: dict) -> dict:
    """Texel -> the source asset's own owner mask (0..1) per texture, from the
    pack the readability heuristics started from: where the model's author
    put team colour, i.e. paintable cloth and panels. Same UV layout."""
    prior = {t: np.full(shape, .5) for t, shape in size.items()}
    bindings = json.loads((SOURCE_PACK / "bindings.json").read_text()) if (SOURCE_PACK / "bindings.json").is_file() else {}
    source = next((v for v in bindings.values() if isinstance(v, dict) and v.get("key0") == key), None)
    if source is None or source["idle"]["part_count"] != len(parts):
        return prior
    seen = set()
    for i, part, *_ in parts:
        original = source["idle"][f"part{i}"]
        t = part["texture"]
        alpha = coverage.dds_alpha(SOURCE_PACK / original["texture"])
        if alpha.shape != size[t]:
            continue
        weight = coverage.owner_weight(original, alpha)
        prior[t] = weight if t not in seen else np.maximum(prior[t], weight)
        seen.add(t)
    return prior


def transfer(key: str) -> dict:
    bindings = json.loads((PACK / "bindings.json").read_text())
    binding = next(v for v in bindings.values() if isinstance(v, dict) and v.get("key0") == key)
    parts = list(coverage.idle_parts(PACK, binding))
    sprites, palette = sprite_views(key, binding)
    luma = (palette[:, :3].astype(float) * [.2126, .7152, .0722]).sum(1) / 255
    textures = sorted({part["texture"] for _, part, *_ in parts})
    rgb = {t: coverage.dds_rgb(PACK / t) for t in textures}
    size = {t: rgb[t].shape[:2] for t in textures}
    extent = max(float(np.ptp(np.concatenate([points for _, _, points, *_ in parts]), 0).max()), 1e-9)
    samples = {t: texel_samples(parts, t, size[t], extent) for t in textures}
    num = {t: np.zeros(len(samples[t]["tx"])) for t in textures}
    den = {t: np.zeros(len(samples[t]["tx"])) for t in textures}
    views, fits, civ_share = {}, {}, []
    for direction, sprite in sprites.items():
        view = raster(parts, binding, direction)
        coarse = raster(parts, binding, direction, 1)
        fit = refine(align(coarse, sprite), coarse, sprite, luma[sprite])
        views[direction], fits[direction] = view, fit
        body = sprite < 224
        civ_share.append(float(CIVNESS[sprite[body]].mean()))
        for t in textures:
            if len(samples[t]["tx"]):
                value, weight = sample_views(samples[t], binding, direction, view, fit, sprite)
                num[t] += value * weight; den[t] += weight
    prior = authored(key, parts, size)
    fields, islands, graphs, per_sample, stages = {}, {}, {}, {}, {}
    for t in textures:
        th, tw = size[t]; s = samples[t]
        island_map = np.full((th, tw), -1)
        island_map[s["ty"], s["tx"]] = s["island"]
        islands[t] = island_map
        if not len(s["tx"]):
            fields[t] = np.full((th, tw), np.nan)
            continue
        seg = segments(rgb[t], island_map)[s["ty"], s["tx"]]
        graphs[t] = graph = surface_graph(s["pos"], s["nrm"], np.maximum(seg, 0) % CLUSTERS, SMOOTH * extent)
        # Evidence smoothed over the surface within each material, so whole
        # areas decide (a tunic, a ring around a turret), never single texels;
        # part of it is the texel's segment (UV island split by colour), so a
        # garment piece switches whole; authored paintable surfaces win ties.
        value = np.where(den[t] > 1e-12, num[t] / np.maximum(den[t], 1e-12), 0.0)
        value, seen_weight = mirror(s["pos"], value, den[t], np.maximum(seg, 0) % CLUSTERS, SMOOTH * extent)
        surface = np.nan_to_num(graph_smooth(graph, value, seen_weight))
        report = []
        smooth = band_pool(s["pos"], surface, den[t], s["piece"], s["nrm"], report=report)
        flat = np.maximum(seg, 0)
        mean = np.bincount(flat, smooth * den[t], flat.max() + 1) / np.maximum(np.bincount(flat, den[t], flat.max() + 1), 1e-12)
        seen = np.bincount(flat, den[t], flat.max() + 1) > 1e-9
        field = np.where(seen[flat], (1 - SEGMENT) * smooth + SEGMENT * mean[flat], smooth)
        per_sample[t] = field + PRIOR * (2 * prior[t][s["ty"], s["tx"]] - 1)
        fields[t] = texel_mean(per_sample[t], s, size[t])
        stages[t] = {"pieces": report, "value": value, "weight": den[t], "surface": surface, "banded": smooth, "field": per_sample[t], "segment": seg}
    result = {"key": key, "binding": binding, "parts": parts, "sprites": sprites, "views": views, "fits": fits,
              "rgb": rgb, "size": size, "fields": fields, "samples": samples, "stages": stages,
              "civ3_share": float(np.median(civ_share)) if civ_share else 0.0,
              "iou": float(np.median([f["iou"] for f in fits.values()])) if fits else 0.0}

    def masks_at(high):
        low = high * GROW
        out = {}
        for t in textures:
            if t not in graphs:
                out[t] = np.zeros(size[t])
                continue
            graph, f = graphs[t], per_sample[t]
            patch = np.bincount(graph["node"], f, graph["count"]) / np.maximum(np.bincount(graph["node"], minlength=graph["count"]), 1)
            chosen = graph_grow(graph, patch, high, low)[graph["node"]]
            stages[t]["mask"] = np.where(chosen, soft(f, low), 0.0)
            out[t] = np.nan_to_num(texel_mean(stages[t]["mask"], samples[t], size[t]))
        return out

    # Threshold so the model shows Civ III's share of civ colour. Growth makes
    # the share jump where a connected area joins, so scan, then refine
    # around the closest; ties prefer less colour.
    def miss(tau):
        candidate = masks_at(tau)
        return abs(visible_share(result, lambda t: candidate[t]) - result["civ3_share"])
    taus = list(np.linspace(.12, .9, 27))
    tau = min(taus, key=lambda x: (miss(x), -x))
    for x in np.linspace(tau - .03, tau + .03, 7):
        if miss(x) < miss(tau):
            tau = x
    result["tau"] = float(tau)
    result["masks"] = {t: np.where(islands[t] >= 0, coverage.soften(m), 0.0) for t, m in masks_at(tau).items()}
    result["new_share"] = visible_share(result, lambda t: result["masks"][t])
    result["current_share"] = float(coverage.measure(PACK, binding)["coverage"])
    return result


# ---- gallery ---------------------------------------------------------------------------

def owner_colour(civ_palette: np.ndarray) -> np.ndarray:
    return civ_palette[4:8].astype(float).mean(0) / 255  # the civ ramp's saturated middle


def flattened(rgb: np.ndarray, block: int = 16) -> np.ndarray:
    """Albedo with its luminance pulled toward the local mean (FLATTEN):
    folds and grime stay faintly, the owner colour reads as one colour."""
    h, w = rgb.shape[:2]
    block = max(1, min(block, h, w))   # tiny textures (missiles) flatten per texel block
    luma = (rgb * [.2126, .7152, .0722]).sum(-1)
    hb, wb = h // block * block, w // block * block
    coarse = luma[:hb, :wb].reshape(hb // block, block, wb // block, block).mean((1, 3))
    local = np.repeat(np.repeat(coarse, block, 0), block, 1)
    local = np.pad(local, ((0, h - hb), (0, w - wb)), mode="edge")
    for _ in range(3):
        local = coverage.soften(local)
    target = (1 - FLATTEN) * luma + FLATTEN * local
    return np.clip(rgb * (target / np.maximum(luma, 1e-3))[..., None], 0, 1)


def shade_model(result: dict, direction: int, mask_of, owner: np.ndarray, debug: bool = False) -> np.ndarray:
    view = result["views"][direction]
    out = np.zeros(view["which"].shape + (3,)); alpha = view["which"] >= 0
    for mine, part, t, ty, tx in visible_texels(result, view):
        weight = np.asarray(mask_of(part, t, ty, tx), float)
        if debug:
            seen = np.isfinite(weight)
            w = np.nan_to_num(weight)
            colour = np.where(seen[:, None], np.stack([w, 1 - w, np.zeros_like(w)], 1), .45)
            out[mine] = colour * view["shade"][mine][:, None]
            continue
        tint = np.array([part.get("tint_r", 1), part.get("tint_g", 1), part.get("tint_b", 1)])
        albedo = result["rgb"][t][ty, tx] * tint
        if "flat" not in result:
            result["flat"] = {}
        if t not in result["flat"]:
            result["flat"][t] = flattened(result["rgb"][t])
        # The unit shader's owner ramp, over the flattened albedo the baked
        # texture would carry under the mask (one even colour, not mottled).
        luma = (result["flat"][t][ty, tx] * tint * [.2126, .7152, .0722]).sum(1)
        own = float((owner * np.array([.2126, .7152, .0722])).sum())
        ramp = np.clip(owner[None] * np.minimum(1, luma[:, None] / max(own, .03)), 0, 1)
        out[mine] = (albedo * (1 - weight[:, None]) + ramp * weight[:, None]) * view["shade"][mine][:, None]
    return np.concatenate([out, alpha[..., None]], -1)


def place(canvas: np.ndarray, image: np.ndarray, x: int, y: int):
    h, w = image.shape[:2]
    h, w = min(h, canvas.shape[0] - y), min(w, canvas.shape[1] - x)
    if h <= 0 or w <= 0:
        return
    a = image[:h, :w, 3:4]
    canvas[y:y + h, x:x + w] = canvas[y:y + h, x:x + w] * (1 - a) + image[:h, :w, :3] * a


def resize(image: np.ndarray, factor: float, size: int) -> np.ndarray:
    """Crop to the alpha box, scale by `factor` (box filter when shrinking,
    nearest when growing) and centre-crop to a `size` cell."""
    rows, cols = np.nonzero(image[..., 3] > 0)
    if not len(rows):
        return image[:1, :1]
    image = image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    step = max(1, int(1 / factor))
    if step > 1:
        h, w = image.shape[0] // step * step, image.shape[1] // step * step
        image = image[:h, :w].reshape(h // step, step, w // step, step, 4).mean((1, 3))
        factor *= step
    h, w = max(1, round(image.shape[0] * factor)), max(1, round(image.shape[1] * factor))
    image = image[(np.arange(h) / factor).astype(int).clip(0, image.shape[0] - 1)][:, (np.arange(w) / factor).astype(int).clip(0, image.shape[1] - 1)]
    top, left = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return image[top:top + size, left:left + size]


def gallery(results: list, civ_palette: np.ndarray, path: Path, panels=("civ3", "current", "new", "field"),
            shown=SHOWN, cell: int = 220, rows: bool = True):
    """Per unit and direction: Civ III sprite | current 3D owner colour |
    transferred | texel civ field (green none, red all; grey unseen)."""
    owner = owner_colour(civ_palette)
    canvas = np.full((len(results) * cell, cell * len(panels) * len(shown), 3), .23)
    alphas = {}
    roots = game_roots(CIV3_ROOT)
    for row, result in enumerate(results):
        y = row * cell
        palette = civ3_flc.read(civ3_art([result["binding"][f"key{i}"] for i in range(result["binding"]["key_count"])])).palette

        def current(part, t, ty, tx):
            if t not in alphas:
                alphas[t] = coverage.dds_alpha(PACK / t)
            return coverage.owner_weight(part, alphas[t][ty, tx])

        boxes = [box(result["sprites"][d] < 224) for d in shown if d in result["sprites"]]
        factor = min(3.0, cell * .9 / max(max(r1 - r0, c1 - c0) for r0, r1, c0, c1 in boxes))
        sources = {"current": (current, False), "new": (lambda part, t, ty, tx: result["masks"][t][ty, tx], False),
                   "field": (lambda part, t, ty, tx: result["fields"][t][ty, tx], True)}
        for slot, direction in enumerate(shown):
            for k, panel in enumerate(panels):
                x = (slot * len(panels) + k) * cell
                if panel == "civ3":
                    sprite = result["sprites"].get(direction)
                    if sprite is None:
                        continue
                    image = civ3_flc.rgba(sprite, palette, civ_palette).astype(float) / 255
                    image[..., 3] = (sprite < 224).astype(float)
                    image = resize(image, factor, cell)
                else:
                    mask_of, debug = sources[panel]
                    image = shade_model(result, direction, mask_of, owner, debug)
                    fit = result["fits"][direction]
                    image = resize(image, factor * math.sqrt(fit["sx"] * fit["sy"]) / SUPERSAMPLE, cell)
                place(canvas, image, x + (cell - image.shape[1]) // 2, y + (cell - image.shape[0]) // 2)
    image = (np.clip(canvas, 0, 1) * 255 + .5).astype(np.uint8)
    write_png(path, image)
    half = cell * len(panels) * len(shown) // 2
    for row, result in enumerate(results if rows else ()):   # per unit, two directions a line, for close review
        band = image[row * cell:(row + 1) * cell]
        write_png(path.with_name(f"{path.stem}-{result['key']}.png"), np.concatenate([band[:, :half], band[:, half:]], 0))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", default=",".join(DEFAULT_UNITS))
    parser.add_argument("--civ", type=int, default=1, help="ntpNN civ palette for the gallery")
    parser.add_argument("--out", default="gallery.png")
    args = parser.parse_args(argv)
    global CIVNESS
    CIVNESS = civness()
    OUT.mkdir(parents=True, exist_ok=True)
    civ_palette = civ3_flc.read_pcx_palette(CIV3_ROOT / "Art/Units/Palettes" / f"ntp{args.civ:02d}.pcx")
    results, summary = [], {}
    for key in filter(None, args.units.split(",")):
        result = transfer(key)
        results.append(result)
        summary[key] = {k: round(result[k], 4) for k in ("civ3_share", "current_share", "new_share", "iou", "tau")}
        print(key, summary[key], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(gallery(results, civ_palette, OUT / args.out))
    print(gallery(results, civ_palette, OUT / "review.png", ("civ3", "current", "new"), (5, 1), 240, False))


if __name__ == "__main__":
    main()
