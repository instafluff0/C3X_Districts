"""Bake Civ III ground-state decals (pollution, craters, ruins) from GroundStatesNormalized.

Each look is a stack of Civ VI decal stamps composed once, in tile units, into
one BC3 texture cell: the runtime drapes a single flat decal per state and
lights it as ground. Relief from the source height maps is baked as a
direction-free occlusion (bowls darker, rims lighter), since a decal carries
no normal. Numpy only; the art direction lives in tile_object_render_strategy.json.
"""
from __future__ import annotations

import math
import random
import struct
from pathlib import Path

import numpy as np

from Renderer.tools.asset_compiler.build_resource_compositions import dds_header
from Renderer.tools.asset_compiler.unit_owner_coverage import dds_alpha, dds_rgb


def dds_red(path: Path) -> np.ndarray:
    """Top mip first channel (0..1) of a BC4/BC5 DXGI DDS (source height maps)."""
    data = path.read_bytes()
    height, width = struct.unpack_from("<II", data, 12)
    dxgi = struct.unpack_from("<I", data, 128)[0]
    size = {80: 8, 81: 8, 83: 16, 84: 16}.get(dxgi)
    if size is None:
        raise ValueError(f"unsupported height format {dxgi}: {path}")
    bw, bh = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    blocks = np.frombuffer(data, np.uint8, bw * bh * size, 148).reshape(bh, bw, size)[..., :8]
    r0, r1 = blocks[..., 0].astype(float), blocks[..., 1].astype(float)
    bits = np.zeros((bh, bw), np.uint64)
    for i in range(6):
        bits |= blocks[..., 2 + i].astype(np.uint64) << np.uint64(8 * i)
    index = np.stack([(bits >> np.uint64(3 * k)) & np.uint64(7) for k in range(16)], -1).astype(int)
    table = np.zeros((bh, bw, 8))
    table[..., 0], table[..., 1] = r0, r1
    six = r0 > r1
    for k in range(2, 8):
        table[..., k] = np.where(six, ((8 - k) * r0 + (k - 1) * r1) / 7,
                                 ((6 - k) * r0 + (k - 1) * r1) / 5 if k < 6 else (0 if k == 6 else 255))
    red = np.take_along_axis(table, index, -1) / 255
    return red.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)[:height, :width]


def dds_rgba(path: Path) -> np.ndarray:
    return np.dstack([dds_rgb(path), dds_alpha(path)]).astype(np.float32)


def blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Three box passes (about Gaussian) along both axes, edges clamped."""
    out = image.astype(np.float64)
    for _ in range(3):
        for axis in (0, 1):
            padded = np.pad(out, [(radius + 1, radius) if a == axis else (0, 0) for a in range(out.ndim)], mode="edge")
            total = np.cumsum(padded, axis=axis)
            hi = np.take(total, np.arange(2 * radius + 1, total.shape[axis]), axis=axis)
            lo = np.take(total, np.arange(0, total.shape[axis] - 2 * radius - 1), axis=axis)
            out = (hi - lo) / (2 * radius + 1)
    return out


def occlusion(height: np.ndarray, strength: float, radius: int = 10) -> np.ndarray:
    """Direction-free relief: below the local mean darker, above it lighter."""
    return np.clip(1.0 + strength * (height - blur(height, radius)), 0.45, 1.35)


def sample(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = texture.shape[:2]
    x = np.clip(u * w - .5, 0, w - 1.001); y = np.clip(v * h - .5, 0, h - 1.001)
    x0 = x.astype(int); y0 = y.astype(int); fx = x - x0; fy = y - y0
    if texture.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    a, b = texture[y0, x0], texture[y0, x0 + 1]
    c, d = texture[y0 + 1, x0], texture[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def quad_map(vertices: list[dict]):
    """Affine tile-position -> uv map and footprint of a decoded decal quad."""
    positions = np.array([v["position"] for v in vertices], float)
    uvs = np.array([v["uv0"] for v in vertices], float)
    affine = np.linalg.lstsq(np.c_[positions, np.ones(len(positions))], uvs, rcond=None)[0]
    return affine, positions.min(0), positions.max(0)


def full_map(half_x: float, half_y: float):
    """A whole texture laid over [-half_x, half_x] x [-half_y, half_y]."""
    return (np.array([[1 / (2 * half_x), 0], [0, -1 / (2 * half_y)], [.5, .5]]),
            np.array([-half_x, -half_y]), np.array([half_x, half_y]))


class Cell:
    """A premultiplied RGBA canvas over a square of `span` tiles about the tile centre."""
    def __init__(self, span: float, pixels: int = 512):
        self.span, self.pixels = span, pixels
        grid = (np.arange(pixels) + .5) / pixels * span - span / 2
        self.x, self.y = np.meshgrid(grid, grid)
        self.colour = np.zeros((pixels, pixels, 3)); self.alpha = np.zeros((pixels, pixels))

    def stamp(self, texture, mapping, scale=1.0, rotation=0.0, offset=(0.0, 0.0), tint=(1, 1, 1),
              opacity=1.0, alpha_power=1.0, shade=None, darken=None, height=None, sun=None, relief=0.0):
        """Lay one decal over the cell. `darken` makes it a black veil of that strength.
        `height` with `sun` bakes the stamp's relief lit from that tile-space direction
        (the decal then must not turn at runtime)."""
        affine, low, high = mapping
        c, s = math.cos(-rotation), math.sin(-rotation)
        dx, dy = self.x - offset[0], self.y - offset[1]
        px, py = (dx * c - dy * s) / scale, (dx * s + dy * c) / scale
        inside = (px >= low[0]) & (px <= high[0]) & (py >= low[1]) & (py <= high[1])
        u = affine[0, 0] * px + affine[1, 0] * py + affine[2, 0]
        v = affine[0, 1] * px + affine[1, 1] * py + affine[2, 1]
        texel = sample(texture, u, v)
        a = np.clip(texel[..., 3], 0, 1) ** alpha_power * inside * opacity
        if darken is not None:
            rgb, a = np.zeros_like(texel[..., :3]), a * darken
        else:
            rgb = texel[..., :3] * np.array(tint)
            if shade is not None:
                rgb = rgb * sample(shade, u, v)[..., None]
            if height is not None and sun is not None:
                step = 0.004
                def h(x, y):
                    return sample(height, affine[0, 0] * x + affine[1, 0] * y + affine[2, 0],
                                  affine[0, 1] * x + affine[1, 1] * y + affine[2, 1])
                gx = (h(px + step, py) - h(px - step, py)) / (2 * step * scale)
                gy = (h(px, py + step) - h(px, py - step)) / (2 * step * scale)
                c2, s2 = math.cos(rotation), math.sin(rotation)
                wx, wy = gx * c2 - gy * s2, gx * s2 + gy * c2
                n = np.stack([-wx * relief, -wy * relief, np.ones_like(wx)], -1)
                n /= np.linalg.norm(n, axis=-1, keepdims=True)
                light = np.array(sun, float) / np.linalg.norm(sun)
                rgb = rgb * np.clip((n * light).sum(-1) / light[2], 0.35, 1.5)[..., None]
        self.colour = rgb * a[..., None] + self.colour * (1 - a[..., None])
        self.alpha = a + self.alpha * (1 - a)

    def rgba(self, feather: float = 0.0, round_edge=None) -> np.ndarray:
        """Straight RGBA; empty texels take the nearby colour so filtering never fringes.
        `round_edge` (inner, outer radius in tiles) fades a square stamp to a round one."""
        alpha = self.alpha.copy()
        if feather:
            edge = self.span / 2 - np.maximum(np.abs(self.x), np.abs(self.y))
            alpha *= np.clip(edge / feather, 0, 1)
        if round_edge:
            radius = np.sqrt(self.x ** 2 + self.y ** 2)
            t = np.clip((round_edge[1] - radius) / (round_edge[1] - round_edge[0]), 0, 1)
            alpha *= t * t * (3 - 2 * t)
        weight = blur(self.alpha, 12)
        fill = blur(self.colour, 12) / np.maximum(weight, 1e-6)[..., None]
        colour = np.where(self.alpha[..., None] > 1e-4, self.colour / np.maximum(self.alpha, 1e-4)[..., None], fill)
        return np.dstack([np.clip(colour, 0, 1), np.clip(alpha, 0, 1)])


def encode_bc3(rgba: np.ndarray) -> bytes:
    """A BC3_UNORM_SRGB DDS with a full mip chain (alpha-weighted colour downsampling)."""
    levels, image = [], rgba.astype(np.float64)
    while True:
        h, w = image.shape[:2]
        bh, bw = max(1, h // 4), max(1, w // 4)
        padded = np.zeros((bh * 4, bw * 4, 4)); padded[:h, :w] = image[:bh * 4, :bw * 4]
        blocks = padded.reshape(bh, 4, bw, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4) * 255
        alpha = np.round(blocks[..., 3])
        a0, a1 = alpha.max(1), alpha.min(1)
        palette = np.stack([a0, a1] + [((7 - k) * a0 + k * a1) / 7 for k in range(1, 7)], -1)
        aindex = np.abs(alpha[:, :, None] - palette[:, None, :]).argmin(-1)
        abits = np.zeros(len(alpha), np.uint64)
        for k in range(16):
            abits |= aindex[:, k].astype(np.uint64) << np.uint64(3 * k)
        rgb = blocks[..., :3]
        luma = rgb @ np.array([.3, .59, .11])
        hi = np.take_along_axis(rgb, luma.argmax(1)[:, None, None].repeat(3, -1), 1)[:, 0]
        lo = np.take_along_axis(rgb, luma.argmin(1)[:, None, None].repeat(3, -1), 1)[:, 0]
        def code(c):
            c = np.clip(np.round(c), 0, 255).astype(int)
            return (c[:, 0] >> 3) << 11 | (c[:, 1] >> 2) << 5 | (c[:, 2] >> 3)
        c0, c1 = code(hi), code(lo)
        def expand(v):
            return np.stack(((v >> 11) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31), -1).astype(float)
        e0, e1 = expand(c0), expand(c1)
        colours = np.stack([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3], 1)
        cindex = ((rgb[:, :, None, :] - colours[:, None, :, :]) ** 2).sum(-1).argmin(-1)
        cbits = np.zeros(len(rgb), np.uint64)
        for k in range(16):
            cbits |= cindex[:, k].astype(np.uint64) << np.uint64(2 * k)
        out = np.zeros((len(rgb), 16), np.uint8)
        out[:, 0], out[:, 1] = a0, a1
        for i in range(6):
            out[:, 2 + i] = (abits >> np.uint64(8 * i)) & np.uint64(255)
        out[:, 8], out[:, 9] = c0 & 255, c0 >> 8
        out[:, 10], out[:, 11] = c1 & 255, c1 >> 8
        for i in range(4):
            out[:, 12 + i] = (cbits >> np.uint64(8 * i)) & np.uint64(255)
        levels.append(out.tobytes())
        if h <= 4 or w <= 4:
            break
        a = image[..., 3:4]
        weighted = (image[..., :3] * a).reshape(h // 2, 2, w // 2, 2, 3).sum((1, 3))
        alpha_sum = a.reshape(h // 2, 2, w // 2, 2, 1).sum((1, 3))
        colour = image[..., :3].reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3))
        image = np.dstack([np.where(alpha_sum > 1e-6, weighted / np.maximum(alpha_sum, 1e-6), colour), alpha_sum / 4])
    h, w = rgba.shape[:2]
    return dds_header(w, h, len(levels), 78, len(levels[0])) + b"".join(levels)


def atlas(cells: list[np.ndarray]) -> np.ndarray:
    """Two-by-two atlas of equal square cells (missing cells stay empty)."""
    n = cells[0].shape[0]
    out = np.zeros((2 * n, 2 * n, 4))
    for i, cell in enumerate(cells[:4]):
        out[(i // 2) * n:(i // 2 + 1) * n, (i % 2) * n:(i % 2 + 1) * n] = cell
    return out


class Sources:
    def __init__(self, pack: Path, manifest: dict):
        self.pack, self.manifest, self.cache = pack, manifest, {}

    def texture(self, relative: str, kind: str = "rgba") -> np.ndarray:
        key = (relative, kind)
        if key not in self.cache:
            path = self.pack / relative
            self.cache[key] = dds_rgba(path) if kind == "rgba" else dds_red(path)
        return self.cache[key]

    def decal(self, name: str, index: int = 0):
        record = self.manifest["decals"][name]["decals"][index]
        colour = self.texture(record["channels"]["base_color"]["texture"])
        height = record["channels"].get("height")
        return colour, quad_map(record["vertices"]), (self.texture(height["texture"], "red") if height else None)

    def plain(self, name: str, kind: str = "rgba") -> np.ndarray:
        return self.texture(self.manifest["textures"][name]["texture"], kind)


def scorch_veil(sources: Sources) -> np.ndarray:
    """Civ VI's scorch mark is a white mask; keep its coverage as alpha."""
    scorch = sources.plain("scorch")
    return np.dstack([scorch[..., :3], scorch[..., :3].mean(-1) * scorch[..., 3]])


def pollution_cell(sources: Sources, look: dict, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    cell = Cell(look["span"])
    veil = scorch_veil(sources)
    cell.stamp(veil, full_map(.5, .5), scale=look["scorch_scale"], rotation=rng.uniform(0, 6.283),
               darken=look["scorch_strength"])
    light, light_map, light_height = sources.decal("ash_light")
    cell.stamp(light, light_map, scale=look["light_scale"], rotation=rng.uniform(0, 6.283),
               tint=look["light_tint"], shade=occlusion(light_height, look["relief"]) if light_height is not None else None)
    heavy, heavy_map, heavy_height = sources.decal("ash_heavy")
    cell.stamp(heavy, heavy_map, scale=look["heavy_scale"], rotation=rng.uniform(0, 6.283),
               tint=look["heavy_tint"], alpha_power=look["heavy_alpha_power"],
               shade=occlusion(heavy_height, look["relief"]) if heavy_height is not None else None)
    return cell.rgba(feather=look["feather"], round_edge=look.get("round_edge"))


def crater_cell(sources: Sources, look: dict, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    cell = Cell(look["span"])
    cell.stamp(scorch_veil(sources), full_map(.5, .5), scale=look["scorch_scale"], rotation=rng.uniform(0, 6.283),
               darken=look["scorch_strength"])
    spots = look["spots"][: look["spots_min"] + seed % (len(look["spots"]) - look["spots_min"] + 1)]
    for k, offset in enumerate(spots):
        colour, mapping, height = sources.decal(f"crater_{(k + seed) % 4 + 1}")
        scale = look["scale"] * rng.uniform(*look["scale_jitter"]); rotation = rng.uniform(0, 6.283)
        cell.stamp(colour, mapping, scale=scale, rotation=rotation, offset=offset, tint=look["tint"],
                   height=height, sun=look["sun"], relief=look["relief"])
        # A darker floor: the same stamp, shrunk, as a veil.
        cell.stamp(colour, mapping, scale=scale * look["floor_scale"], rotation=rotation, offset=offset,
                   alpha_power=1.5, darken=look["floor_strength"])
    return cell.rgba(feather=look["feather"])


def rubble_cell(sources: Sources, look: dict) -> np.ndarray:
    """The light debris decal and two dark rubble heaps, sunlit by their height map."""
    cell = Cell(look["span"])
    height = sources.plain("rubble_height", "red")
    half = (.5, .25)
    cell.stamp(sources.plain("rubble_light"), full_map(*half), scale=look["main_scale"], rotation=-math.pi / 4,
               tint=look["main_tint"], height=height, sun=look["sun"], relief=look["relief"])
    for offset, scale, rotation in look["patches"]:
        cell.stamp(sources.plain("rubble"), full_map(*half), scale=look["main_scale"] * scale, rotation=rotation,
                   offset=offset, tint=look["patch_tint"], height=height, sun=look["sun"], relief=look["relief"])
    return cell.rgba(feather=look["feather"])
