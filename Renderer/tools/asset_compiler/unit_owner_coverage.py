#!/usr/bin/env python3
"""Measure visible owner-colour coverage of packed units (offline pipeline).

Each binding's idle pose is rasterized with depth at 128-pixel tiles (the live
vertex shader projection), per pixel recording the nearest part and its
texture coordinate. The owner weight follows the unit material shader: owner
mask mode 1 uses smoothstep(.06,.94,1-alpha), mode 2 the whole part, times the
part's owner strength; material model 1 uses alpha when strength > .5. The
result is the mean weight over the unit's visible pixels plus each part's
visible share, median over the four diagonal facings units usually show on
the map. No unit or part names are used.

    python3 -m Renderer.tools.asset_compiler.unit_owner_coverage [--pack UnitAnimationFidelity]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
Z_PIXELS = 150 * 128 / 224


def dds_alpha(path: Path) -> np.ndarray:
    """Top mip alpha (0..1) of a BC1/BC3 DXGI DDS, as an HxW float array."""
    data = path.read_bytes()
    height, width = struct.unpack_from("<II", data, 12)
    if data[84:88] != b"DX10":
        raise ValueError(f"unsupported DDS header: {path}")
    dxgi = struct.unpack_from("<I", data, 128)[0]
    bw, bh = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    if dxgi in (77, 78):  # BC3: interpolated alpha block, then a BC1 colour block
        blocks = np.frombuffer(data, np.uint8, bw * bh * 16, 148).reshape(bh, bw, 16)
        a0, a1 = blocks[..., 0].astype(float), blocks[..., 1].astype(float)
        bits = np.zeros((bh, bw), np.uint64)
        for i in range(6):
            bits |= blocks[..., 2 + i].astype(np.uint64) << np.uint64(8 * i)
        index = np.stack([(bits >> np.uint64(3 * k)) & np.uint64(7) for k in range(16)], -1).astype(int)
        six = a0 > a1
        table = np.zeros((bh, bw, 8))
        table[..., 0], table[..., 1] = a0, a1
        for k in range(2, 8):
            table[..., k] = np.where(six, ((8 - k) * a0 + (k - 1) * a1) / 7,
                                     ((6 - k) * a0 + (k - 1) * a1) / 5 if k < 6 else (0 if k == 6 else 255))
        alpha = np.take_along_axis(table, index, -1) / 255
    elif dxgi in (71, 72):  # BC1: alpha 0 only for punch-through index 3
        blocks = np.frombuffer(data, np.uint8, bw * bh * 8, 148).reshape(bh, bw, 8)
        c0 = blocks[..., 0].astype(int) | blocks[..., 1].astype(int) << 8
        c1 = blocks[..., 2].astype(int) | blocks[..., 3].astype(int) << 8
        bits = np.zeros((bh, bw), np.uint64)
        for i in range(4):
            bits |= blocks[..., 4 + i].astype(np.uint64) << np.uint64(8 * i)
        index = np.stack([(bits >> np.uint64(2 * k)) & np.uint64(3) for k in range(16)], -1).astype(int)
        alpha = np.where((c0 <= c1)[..., None] & (index == 3), 0.0, 1.0)
    else:
        raise ValueError(f"unsupported DXGI format {dxgi}: {path}")
    alpha = alpha.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)
    return alpha[:height, :width]


def dds_rgb(path: Path) -> np.ndarray:
    """Top mip colour (0..1, sRGB-encoded values) of a BC1/BC3 DXGI DDS."""
    data = path.read_bytes()
    height, width = struct.unpack_from("<II", data, 12)
    dxgi = struct.unpack_from("<I", data, 128)[0]
    bw, bh = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    size, offset = (16, 8) if dxgi in (77, 78) else (8, 0)
    if dxgi not in (71, 72, 77, 78):
        raise ValueError(f"unsupported DXGI format {dxgi}: {path}")
    blocks = np.frombuffer(data, np.uint8, bw * bh * size, 148).reshape(bh, bw, size)[..., offset:offset + 8]
    c0 = blocks[..., 0].astype(int) | blocks[..., 1].astype(int) << 8
    c1 = blocks[..., 2].astype(int) | blocks[..., 3].astype(int) << 8
    def rgb(c):
        return np.stack(((c >> 11) * 255 // 31, ((c >> 5) & 63) * 255 // 63, (c & 31) * 255 // 31), -1).astype(float) / 255
    e0, e1 = rgb(c0), rgb(c1)
    four = (c0 > c1)[..., None] | (size == 16)
    table = np.stack((e0, e1, np.where(four, (2 * e0 + e1) / 3, (e0 + e1) / 2),
                      np.where(four, (e0 + 2 * e1) / 3, 0)), -2)
    bits = np.zeros((bh, bw), np.uint64)
    for i in range(4):
        bits |= blocks[..., 4 + i].astype(np.uint64) << np.uint64(8 * i)
    index = np.stack([(bits >> np.uint64(2 * k)) & np.uint64(3) for k in range(16)], -1).astype(int)
    colours = np.take_along_axis(table, index[..., None].repeat(3, -1), -2)
    colours = colours.reshape(bh, bw, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)
    return colours[:height, :width]


def idle_parts(pack: Path, binding: dict):
    from Renderer.native.environment_refresh.prepare_units import idle_geometry
    idle = binding["idle"]
    for i in range(idle["part_count"]):
        part = idle[f"part{i}"]
        blob = (pack / part["mesh"]).read_bytes()
        points, triangles = idle_geometry(blob)
        version, n = struct.unpack_from("<2I", blob, 8)
        stride = 88 if version == 2 else 64
        raw = np.frombuffer(blob, np.uint8, n * stride, 32).reshape(n, stride)
        uv = raw[:, 24:32].copy().view("<f4").reshape(n, 2)
        yield i, part, points, triangles, uv, skinned_normals(blob, raw, n, stride)


def skinned_normals(blob: bytes, raw: np.ndarray, n: int, stride: int) -> np.ndarray:
    """First-frame normals under the same skin palette (rotation part)."""
    normal = raw[:, 12:24].copy().view("<f4").reshape(n, 3)
    joints = raw[:, 32:48].copy().view("<u4").reshape(n, 4)
    weights = raw[:, 48:64].copy().view("<f4").reshape(n, 4)
    indices, bones = struct.unpack_from("<2I", blob, 16)
    palette = np.frombuffer(blob, "<f4", bones * 16, 32 + n * stride + indices * 4).reshape(bones, 4, 4)
    out = np.zeros((n, 3))
    for k in range(4):
        m = palette[joints[:, k]]
        out += weights[:, k, None] * (normal[:, 0, None] * m[:, 0, :3] + normal[:, 1, None] * m[:, 1, :3] +
                                      normal[:, 2, None] * m[:, 2, :3])
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def owner_weight(part: dict, alpha: np.ndarray) -> np.ndarray:
    strength = float(part.get("owner_strength", 0))
    if part.get("material_model", 0) >= .5:
        return alpha if strength > .5 else np.zeros_like(alpha)
    mode = part.get("owner_mask", 0)
    if mode == 1:
        t = np.clip((1 - alpha - .06) / .88, 0, 1)
        return t * t * (3 - 2 * t) * strength
    return np.full_like(alpha, strength if mode == 2 else 0.0)


def measure(pack: Path, binding: dict, directions=(1, 3, 5, 7), supersample=2, cache=None, collect=None) -> dict:
    cache = {} if cache is None else cache
    samples, aspects = [], []
    parts = list(idle_parts(pack, binding))
    scale, offset, yaw = binding["scale"], binding["offset_z"], binding.get("yaw_offset", 225.0)
    rows = []
    for direction in directions:
        angle = math.radians(yaw + direction * 45)
        c, s = math.cos(angle), math.sin(angle)
        projected = []
        for i, part, points, triangles, uv, normals in parts:
            x = (points[:, 0] * c - points[:, 1] * s) * scale
            y = (points[:, 0] * s + points[:, 1] * c) * scale
            z = (points[:, 2] + offset) * scale
            projected.append(((x - y) * 64 * supersample, ((x + y) * 32 - z * Z_PIXELS) * supersample,
                              (x + y) * 5 + z * .1, z))
        xs_all = np.concatenate([p[0] for p in projected]); ys_all = np.concatenate([p[1] for p in projected])
        x0, y0 = int(np.floor(xs_all.min())) - 1, int(np.floor(ys_all.min())) - 1
        w, h = int(np.ceil(xs_all.max())) - x0 + 2, int(np.ceil(ys_all.max())) - y0 + 2
        depth = np.full((h, w), -np.inf); owner = np.zeros((h, w)); which = np.full((h, w), -1)
        texel_x = np.zeros((h, w), int); texel_y = np.zeros((h, w), int); height = np.zeros((h, w))
        for (i, part, points, triangles, uv, normals), (px_, py_, pd, pz) in zip(parts, projected):
            key = part["texture"]
            if key not in cache:
                cache[key] = dds_alpha(pack / key)
            alpha_map = cache[key]; th, tw = alpha_map.shape
            for a, b, t in triangles:
                if pz[a] < 0 and pz[b] < 0 and pz[t] < 0:
                    continue
                ax, ay, bx, by, cx, cy = px_[a] - x0, py_[a] - y0, px_[b] - x0, py_[b] - y0, px_[t] - x0, py_[t] - y0
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
                tu = u * uv[a, 0] + v * uv[b, 0] + r * uv[t, 0]
                tv = u * uv[a, 1] + v * uv[b, 1] + r * uv[t, 1]
                tx = (np.floor(np.mod(tu, 1) * tw).astype(int)) % tw
                ty = (np.floor(np.mod(tv, 1) * th).astype(int)) % th
                weight = owner_weight(part, alpha_map[ty, tx])
                region[win] = d[win]
                owner[top:bottom + 1, left:right + 1][win] = weight[win]
                which[top:bottom + 1, left:right + 1][win] = i
                if collect is not None and i in collect:
                    texel_x[top:bottom + 1, left:right + 1][win] = tx[win]
                    texel_y[top:bottom + 1, left:right + 1][win] = ty[win]
                    height[top:bottom + 1, left:right + 1][win] = (u * pz[a] + v * pz[b] + r * pz[t])[win]
        visible = which >= 0
        if not visible.any():
            continue  # an edge-on view shows nothing to measure
        total = int(visible.sum())
        rows_v, cols_v = np.nonzero(visible)
        aspects.append((np.ptp(rows_v) + 1) / (np.ptp(cols_v) + 1))
        if collect is not None:
            chosen = np.isin(which, list(collect))
            samples.append((total, which[chosen], texel_x[chosen], texel_y[chosen], height[chosen]))
        rows.append((float(owner[visible].sum() / total),
                     [float((which == i).sum() / total) for i in range(len(parts))]))
    if not rows:
        return {"coverage": 0.0, "part_share": [0.0] * len(parts), "samples": samples, "aspect": 1.0}
    coverage = float(np.median([r[0] for r in rows]))
    shares = [float(np.median([r[1][i] for r in rows])) for i in range(len(parts))]
    return {"coverage": coverage, "part_share": shares, "samples": samples,
            "aspect": float(np.median(aspects))}


def paintable(rgb: np.ndarray) -> np.ndarray:
    """Neutral, light texels: the tintable paint/cloth of tint-authored art."""
    luminance = rgb[..., 0] * .2126 + rgb[..., 1] * .7152 + rgb[..., 2] * .0722
    saturation = (rgb.max(-1) - rgb.min(-1)) / np.maximum(rgb.max(-1), 1e-3)
    return (luminance >= .35) & (saturation <= .38)


def texel_attributes(pack: Path, binding: dict, indices, size, scale, offset, lateral=None) -> tuple:
    """Mean idle-pose height of every texel the chosen parts' UVs cover, and
    sign-free normal moments (xx, xy, yy, |z|): mirrored UVs often share
    texels between opposite sides, whose signed normals would cancel. With
    `lateral` (side direction, centre), a fifth moment holds the mean unsigned
    distance from that centreline."""
    th, tw = size
    total, count, moments = np.zeros((th, tw)), np.zeros((th, tw)), np.zeros((th, tw, 5))
    for i, part, points, triangles, uv, normals in idle_parts(pack, binding):
        if i not in indices:
            continue
        z = (points[:, 2] + offset) * scale
        across = np.zeros(len(points)) if lateral is None else \
            np.abs((points[:, 0] - lateral[1][0]) * lateral[0][0] + (points[:, 1] - lateral[1][1]) * lateral[0][1])
        us, vs = uv[:, 0] * tw, uv[:, 1] * th
        for a, b, t in triangles:
            ax, ay, bx, by, cx, cy = us[a], vs[a], us[b], vs[b], us[t], vs[t]
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-9:
                continue
            left, right = int(np.floor(min(ax, bx, cx))), int(np.ceil(max(ax, bx, cx)))
            top, bottom = int(np.floor(min(ay, by, cy))), int(np.ceil(max(ay, by, cy)))
            gx, gy = np.meshgrid(np.arange(left, right + 1) + .5, np.arange(top, bottom + 1) + .5)
            u = ((bx - gx) * (cy - gy) - (by - gy) * (cx - gx)) / area
            v = ((cx - gx) * (ay - gy) - (cy - gy) * (ax - gx)) / area
            r = 1 - u - v
            inside = (u >= -.02) & (v >= -.02) & (r >= -.02)
            if not inside.any():
                continue
            yy, xx = np.mod(gy[inside].astype(int), th), np.mod(gx[inside].astype(int), tw)
            np.add.at(total, (yy, xx), (u * z[a] + v * z[b] + r * z[t])[inside])
            np.add.at(count, (yy, xx), 1)
            face = normals[a] + normals[b] + normals[t]
            face = face / max(float(np.linalg.norm(face)), 1e-9)
            np.add.at(moments, (yy, xx), np.array([face[0] * face[0], face[0] * face[1], face[1] * face[1], abs(face[2]), 0.0]))
            np.add.at(moments[..., 4], (yy, xx), (u * across[a] + v * across[b] + r * across[t])[inside])
    covered = count > 0
    moments /= np.maximum(count, 1)[..., None]
    return np.where(covered, total / np.maximum(count, 1), np.nan), moments, covered


def hull_band(pack: Path, binding: dict, indices, scale: float, offset: float):
    """Side direction and hull height range of a body component.

    The hull is the lowest run of height slices whose vertices span at least
    80% of the body's length along its ground footprint's long axis: below a
    turret, deck house, conning tower, masts or sails. Sparse slices (small
    fittings) are ignored and one short slice inside the run is tolerated.
    A hull starts at the ground or water: a run first found above the lowest
    fifth of the body is rigging (lateen yards longer than a short hull), and
    the hull is the body below it."""
    points = np.concatenate([p for i, _, p, *_ in idle_parts(pack, binding) if i in indices])
    xy = points[:, :2] - points[:, :2].mean(0)
    values, vectors = np.linalg.eigh(np.cov(xy.T))
    major = vectors[:, 1]
    along = xy[:, 0] * major[0] + xy[:, 1] * major[1]
    z = (points[:, 2] + offset) * scale
    edges = np.linspace(max(0.0, float(z.min())), float(z.max()), 21)
    sparse = max(8, int(.004 * len(z)))
    state = []  # None for a sparse slice, else whether it spans the body length
    for k in range(20):
        inside = (z >= edges[k]) & (z <= edges[k + 1])
        # Spans of the bulk of each slice's vertices: a thin gun barrel,
        # bowsprit or rigging line does not make a slice full-length.
        bulk = lambda values: float(np.percentile(values, 95) - np.percentile(values, 5))
        state.append(None if inside.sum() < sparse else bulk(along[inside]) >= .8 * bulk(along))
    first = next((k for k, f in enumerate(state) if f), 0)
    last, short = first, 0
    for k in range(first + 1, 20):
        if state[k] is False:
            short += 1
            if short > 1:
                break
        elif state[k]:
            last, short = k, 0
    side = np.array([-major[1], major[0], 0.0])
    low, high = float(edges[first]), float(edges[last + 1])
    if first >= 4:
        low, high = float(edges[0]), low
    centre = points[:, :2].mean(0)
    hull = (z >= low) & (z <= high)
    across = np.abs((points[:, 0] - centre[0]) * side[0] + (points[:, 1] - centre[1]) * side[1])
    return side, low, high, centre, float(np.percentile(across[hull] if hull.any() else across, 95))


def bc3_with_alpha(path: Path, alpha: np.ndarray) -> bytes:
    """Copy a BC1/BC3 DDS as BC3 with a new alpha channel on every mip."""
    data = path.read_bytes()
    height, width = struct.unpack_from("<II", data, 12)
    mips = max(1, struct.unpack_from("<I", data, 28)[0])
    dxgi = struct.unpack_from("<I", data, 128)[0]
    bc1 = dxgi in (71, 72)
    header = bytearray(data[:148])
    struct.pack_into("<I", header, 128, {71: 77, 72: 78}.get(dxgi, dxgi))
    struct.pack_into("<I", header, 20, max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * 16)
    out, offset, level_alpha = [bytes(header)], 148, alpha.astype(float)
    for level in range(mips):
        w, h = max(1, width >> level), max(1, height >> level)
        bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
        size = 8 if bc1 else 16
        source = np.frombuffer(data, np.uint8, bw * bh * size, offset).reshape(bh * bw, size).copy()
        offset += bw * bh * size
        colour = source[:, -8:].copy()
        if bc1:  # BC3 colour blocks always use four-colour interpolation
            c0 = colour[:, 0].astype(int) | colour[:, 1].astype(int) << 8
            c1 = colour[:, 2].astype(int) | colour[:, 3].astype(int) << 8
            three = c0 < c1
            if three.any():
                bits = colour[three, 4:8].copy().view("<u4").ravel()
                idx = np.stack([(bits >> (2 * k)) & 3 for k in range(16)], -1)
                mapped = np.choose(idx, [1, 0, 2, 1])  # swap endpoints; black -> darker end
                packed = np.zeros(len(bits), np.uint32)
                for k in range(16):
                    packed |= mapped[:, k].astype(np.uint32) << np.uint32(2 * k)
                swapped = colour[three].copy()
                swapped[:, 0:2], swapped[:, 2:4] = colour[three, 2:4], colour[three, 0:2]
                swapped[:, 4:8] = packed.view(np.uint8).reshape(-1, 4)
                colour[three] = swapped
        padded = np.ones((bh * 4, bw * 4))
        padded[:min(h, level_alpha.shape[0]), :min(w, level_alpha.shape[1])] = level_alpha[:h, :w]
        values = np.round(np.clip(padded, 0, 1) * 255).reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3).reshape(-1, 16)
        a0, a1 = values.max(1), values.min(1)
        palette = np.stack([a0, a1] + [((7 - k) * a0 + k * a1) / 7 for k in range(1, 7)], -1)
        index = np.abs(values[:, :, None] - palette[:, None, :]).argmin(-1)
        index[a0 == a1] = 0
        bits = np.zeros(len(values), np.uint64)
        for k in range(16):
            bits |= index[:, k].astype(np.uint64) << np.uint64(3 * k)
        block = np.zeros((len(values), 16), np.uint8)
        block[:, 0], block[:, 1] = a0, a1
        for i in range(6):
            block[:, 2 + i] = (bits >> np.uint64(8 * i)) & np.uint64(255)
        block[:, 8:] = colour
        out.append(block.tobytes())
        if level_alpha.shape[0] > 1 or level_alpha.shape[1] > 1:
            ph, pw = (level_alpha.shape[0] + 1) // 2 * 2, (level_alpha.shape[1] + 1) // 2 * 2
            grown = np.pad(level_alpha, ((0, ph - level_alpha.shape[0]), (0, pw - level_alpha.shape[1])), mode="edge")
            level_alpha = grown.reshape(ph // 2, 2, pw // 2, 2).mean((1, 3))
    return b"".join(out)


def soften(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="edge")
    return sum(padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]] for dy in range(3) for dx in range(3)) / 9


def paint_mask(pack: Path, binding: dict, indices, texture: str, target: float, band: float = .03,
               keep_alpha: bool = False, stripe=None, top_only_outboard=None):
    """Owner mask over a component's texture.

    Default: paintable texels from the top of the component down until the
    unit's visible owner coverage reaches `target`. With `stripe` (side
    direction and hull height range of a body), a horizontal stripe on
    side-facing, not-dark texels instead, along the upper hull, grown towards
    `target` within the hull range."""
    measured = measure(pack, binding, collect=set(indices))
    rgb, alpha = dds_rgb(pack / texture), dds_alpha(pack / texture)
    sides = stripe if stripe is not None else top_only_outboard
    texel_z, texel_n, covered = texel_attributes(pack, binding, set(indices), rgb.shape[:2],
                                                 binding["scale"], binding["offset_z"],
                                                 None if sides is None else (sides[0], sides[3]))
    zs = texel_z[covered]
    low, span = (float(zs.min()), max(1e-6, float(np.ptp(zs)))) if covered.any() else (0.0, 1.0)
    if stripe is None:
        eligible = paintable(rgb)
        if top_only_outboard is not None:
            # A body without a hull side: mark its top, but not a centreline gun.
            eligible &= texel_n[..., 4] >= .6 * top_only_outboard[4]
    else:
        side, hull_low, hull_high, centre, half_width = stripe
        luminance = rgb[..., 0] * .2126 + rgb[..., 1] * .7152 + rgb[..., 2] * .0722
        # Hull sides only: a gun barrel, antenna or centreline fitting at hull
        # height is not on the side of the hull.
        outboard = texel_n[..., 4] >= .6 * half_width
        facing = np.sqrt(np.maximum(side[0] ** 2 * texel_n[..., 0] + 2 * side[0] * side[1] * texel_n[..., 1] +
                                    side[1] ** 2 * texel_n[..., 2], 0)) >= .45
        eligible = (luminance >= .10) & facing & (texel_n[..., 3] <= .7) & outboard
    pixels = sum(total for total, *_ in measured["samples"])
    ok = np.concatenate([eligible[ty, tx] for _, _, tx, ty, _ in measured["samples"]])
    heights = np.concatenate([z for *_, z in measured["samples"]])
    wanted = int(target * pixels)
    z = np.nan_to_num(texel_z, nan=-np.inf)
    if stripe is None:
        candidates = np.sort(heights[ok])[::-1]
        threshold = -np.inf if len(candidates) <= wanted or wanted <= 0 else candidates[wanted]
        region = np.clip((z - threshold) / (band * span) + .5, 0, 1)
    else:
        hull = max(1e-6, hull_high - hull_low)
        centre = hull_low + .65 * hull
        distance = np.sort(np.abs(heights[ok] - centre))
        half = distance[min(wanted, len(distance) - 1)] if wanted > 0 and len(distance) else 0.0
        half = float(np.clip(half, .12 * hull, .3 * hull))
        region = np.clip((half - np.abs(z - centre)) / (band * span) + .5, 0, 1)
    mask = soften(np.where(covered & eligible, region, 0.0))
    # Only an existing owner mask (mode 1) keeps its alpha; otherwise alpha
    # carried no owner meaning and the new mask replaces it.
    return (np.minimum(alpha, 1 - mask) if keep_alpha else 1 - mask), mask


def measured_samples(pack: Path, binding: dict, indices):
    return measure(pack, binding, collect=set(indices))["samples"]


def choose_garment(idle: dict, assets: list, part_share: list, garment_share: float):
    """The largest visible owner-marked component if it covers at least
    `garment_share` of the unit, else the largest visible component."""
    share, marked = {}, set()
    for i in range(idle["part_count"]):
        share[assets[i]] = share.get(assets[i], 0) + part_share[i]
        if idle[f"part{i}"].get("owner_mask", 0):
            marked.add(assets[i])
    eligible = [a for a in marked if share[a] >= garment_share]
    return max(eligible or share, key=lambda a: share[a]), share, bool(eligible)


def paint_garments(bindings: dict, pack: Path, components: dict, minimum: float, garment_share: float,
                   target: float, strength: float, stripe_target: float = .14) -> dict:
    """Author owner masks for low-coverage units (see paint_mask).

    The garment component comes from choose_garment. Each of its textures
    gains an owner mask written to a new BC3 texture in `pack`; its parts in
    every action use that texture with owner mask mode 1 at `strength`.
    """
    report, cache = {}, {}
    for key, binding in bindings.items():
        if not isinstance(binding, dict) or "idle" not in binding or key not in components:
            continue
        if binding.get("owner_marks"):
            continue  # hulls and aircraft carry their own marks
        measured = measure(pack, binding, cache=cache)
        if measured["coverage"] >= minimum:
            continue
        idle = binding["idle"]
        assets = components[key]["idle"]
        chosen, share, eligible = choose_garment(idle, assets, measured["part_share"], garment_share)
        indices = [i for i in range(idle["part_count"]) if assets[i] == chosen
                   and idle[f"part{i}"].get("material_model", 0) < .5 and not idle[f"part{i}"].get("cutout", 0)]
        if not indices:
            continue
        painted = {}
        # Figures stand taller than wide on screen; vehicles, siege and hulls do not.
        long_body = measured["aspect"] < 1.15
        stripe = hull_band(pack, binding, set(indices), binding["scale"], binding["offset_z"]) if long_body else None
        groups = {}
        for i in indices:
            groups.setdefault(idle[f"part{i}"]["texture"], []).append(i)
        def paint_all(band, goal):
            return {texture: paint_mask(pack, binding, group, texture, goal, stripe=band,
                                        keep_alpha=any(idle[f"part{i}"].get("owner_mask", 0) == 1 for i in group))
                    for texture, group in groups.items()}
        results = paint_all(stripe, stripe_target if long_body else target)
        if stripe is not None:
            # Judge the whole component: a barrel with its own texture rightly
            # gets no stripe. With no usable hull side at all (an open gun
            # carriage, say) mark the top instead.
            samples = measured_samples(pack, binding, indices)
            pixels = sum(total for total, *_ in samples)
            marked = sum(float(results[idle[f"part{int(i)}"]["texture"]][1][y, x])
                         for _, which, tx, ty, _ in samples for i, x, y in zip(which, tx, ty))
            if marked / max(1, pixels) < .04:
                results = {texture: paint_mask(pack, binding, group, texture, stripe_target, top_only_outboard=stripe,
                                               keep_alpha=any(idle[f"part{i}"].get("owner_mask", 0) == 1 for i in group))
                           for texture, group in groups.items()}
        for texture, (alpha, mask) in results.items():
            if not (mask > .01).any():
                continue  # nothing to mark on this texture; keep its part as it was
            blob = bc3_with_alpha(pack / texture, alpha)
            name = "textures/" + hashlib.sha256(blob).hexdigest() + ".dds"
            if not (pack / name).exists():
                (pack / name).write_bytes(blob)
            painted[texture] = name
        for action, ids in components[key].items():
            data = binding.get(action)
            for i, asset in enumerate(ids):
                part = data.get(f"part{i}") if isinstance(data, dict) else None
                if asset == chosen and part is not None and part.get("texture") in painted:
                    part["texture"] = painted[part["texture"]]
                    part["owner_mask"] = 1
                    part["owner_strength"] = max(float(part.get("owner_strength", 0)), strength)
        binding["owner_garment"] = {"component": chosen, "coverage_before": round(measured["coverage"], 4),
                                    "share": round(share[chosen], 4), "marked": eligible, "style": "side stripe" if long_body else "upper garment",
                                    "coverage_after": round(measure(pack, binding)["coverage"], 4)}
        report[binding["key0"]] = binding["owner_garment"]
    return report


def texel_values(pack: Path, binding: dict, indices, size, values) -> tuple:
    """Per-texel mean of per-vertex values(part_index, points) -> (n, k) over
    the UV coverage of the chosen parts, with the covered texels."""
    th, tw = size
    total, count = None, np.zeros((th, tw))
    for i, part, points, triangles, uv, normals in idle_parts(pack, binding):
        if i not in indices:
            continue
        vertex = values(i, points)
        if total is None:
            total = np.zeros((th, tw, vertex.shape[1]))
        us, vs = uv[:, 0] * tw, uv[:, 1] * th
        for a, b, t in triangles:
            ax, ay, bx, by, cx, cy = us[a], vs[a], us[b], vs[b], us[t], vs[t]
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-9:
                continue
            left, right = int(np.floor(min(ax, bx, cx))), int(np.ceil(max(ax, bx, cx)))
            top, bottom = int(np.floor(min(ay, by, cy))), int(np.ceil(max(ay, by, cy)))
            gx, gy = np.meshgrid(np.arange(left, right + 1) + .5, np.arange(top, bottom + 1) + .5)
            u = ((bx - gx) * (cy - gy) - (by - gy) * (cx - gx)) / area
            v = ((cx - gx) * (ay - gy) - (cy - gy) * (ax - gx)) / area
            r = 1 - u - v
            inside = (u >= -.02) & (v >= -.02) & (r >= -.02)
            if not inside.any():
                continue
            yy, xx = np.mod(gy[inside].astype(int), th), np.mod(gx[inside].astype(int), tw)
            np.add.at(total, (yy, xx), u[inside][:, None] * vertex[a] + v[inside][:, None] * vertex[b] +
                      r[inside][:, None] * vertex[t])
            np.add.at(count, (yy, xx), 1)
    covered = count > 0
    if total is None:
        return np.full((th, tw, 1), np.nan), covered
    return np.where(covered[..., None], total / np.maximum(count, 1)[..., None], np.nan), covered


def idle_motion(pack: Path, binding: dict, indices) -> dict:
    """Vertices that move relative to the body during the idle loop (rotors,
    propellers, flags), per part: off the whole unit's best rigid motion
    between frames by more than 2% of its size. A banking or bobbing body
    moves rigidly and is not marked."""
    frames = {}
    for i in indices:
        blob = (pack / binding["idle"][f"part{i}"]["mesh"]).read_bytes()
        version, n, ni, nb, nf = struct.unpack_from("<5I", blob, 8)
        stride = 88 if version == 2 else 64
        raw = np.frombuffer(blob, np.uint8, n * stride, 32).reshape(n, stride)
        position = raw[:, :12].copy().view("<f4").reshape(n, 3)
        joints = raw[:, 32:48].copy().view("<u4").reshape(n, 4)
        weights = raw[:, 48:64].copy().view("<f4").reshape(n, 4)
        poses = []
        for f in sorted({0, nf // 4, nf // 2, (3 * nf) // 4}):
            palette = np.frombuffer(blob, "<f4", nb * 16, 32 + n * stride + ni * 4 + f * nb * 64).reshape(nb, 4, 4)
            points = np.zeros((n, 3))
            for k in range(4):
                m = palette[joints[:, k]]
                points += weights[:, k, None] * (position[:, 0, None] * m[:, 0, :3] + position[:, 1, None] * m[:, 1, :3] +
                                                 position[:, 2, None] * m[:, 2, :3] + m[:, 3, :3])
            poses.append(points)
        frames[i] = poses
    if not frames:
        return {}
    count = min(len(p) for p in frames.values())
    whole = [np.concatenate([frames[i][f] for i in frames]) for f in range(count)]
    limit = .02 * float(np.ptp(whole[0], axis=0).max())
    residual = np.zeros(len(whole[0]))
    for pose in whole[1:]:
        # Fit the body's rigid motion, refitting to the vertices that follow
        # it so a large spinning rotor does not drag the fit.
        inliers = np.ones(len(pose), bool)
        for _ in range(4):
            ca, cb = whole[0][inliers].mean(0), pose[inliers].mean(0)
            a, b = whole[0][inliers] - ca, pose[inliers] - cb
            h = np.array([[float((a[:, r] * b[:, c]).sum()) for c in range(3)] for r in range(3)])
            u, _, vt = np.linalg.svd(h)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            rotation = vt.T @ np.diag([1, 1, d]) @ u.T
            fitted = sum((whole[0][:, k, None] - ca[k]) * rotation[:, k] for k in range(3)) + cb
            error = np.linalg.norm(fitted - pose, axis=1)
            # Start from the better-fitting half, then keep the body's vertices.
            follow = error <= (max(limit, float(np.median(error))) if inliers.all() else limit)
            if follow.sum() < 3 or np.array_equal(follow, inliers):
                break
            inliers = follow
        residual = np.maximum(residual, error)
    result, start = {}, 0
    for i in frames:
        n = len(frames[i][0])
        result[i] = residual[start:start + n] > limit
        start += n
    return result


def symmetry_axis(xy: np.ndarray) -> np.ndarray:
    """Ground direction of the vertical mirror plane through the footprint's
    centre that best matches it: an aircraft's fuselage axis."""
    xy = xy - xy.mean(0)
    reach = max(float(np.abs(xy).max()), 1e-9)
    best, axis = -1.0, np.array([1.0, 0.0])
    for degrees in range(0, 180, 2):
        a = math.radians(degrees)
        d = np.array([math.cos(a), math.sin(a)])
        along = xy[:, 0] * d[0] + xy[:, 1] * d[1]
        across = xy[:, 1] * d[0] - xy[:, 0] * d[1]
        grid = np.zeros((48, 48), bool)
        grid[np.clip(((along / reach + 1) * 24).astype(int), 0, 47), np.clip(((across / reach + 1) * 24).astype(int), 0, 47)] = True
        score = (grid & grid[:, ::-1]).sum() / max(1, (grid | grid[:, ::-1]).sum())
        if score > best:
            best, axis = score, d
    return axis


def mark_masks(pack: Path, binding: dict, indices, style: str, targets: dict, band: float = .03) -> dict:
    """Owner masks per texture for a hull or an aircraft.

    `hull`: a side stripe along the hull (paint_mask stripe; oars and deck
    overhangs make the hull's outboard reach meaningless, so any side-facing
    hull texel qualifies) plus accents on the highest texels, within the top
    `accent_floor` of the height: mast tops, pennants, a conning tower's top,
    not the sails below them.
    `extremities`: wing tips (outermost static texels across the mirror axis)
    and tail tips (outermost or highest texels of the rear fifth, the end with
    the taller fin). Idle motion (rotors, propellers) is never marked. Each
    region is grown from its extreme towards its share of the visible unit."""
    idle = binding["idle"]
    groups = {}
    for i in indices:
        groups.setdefault(idle[f"part{i}"]["texture"], []).append(i)
    scale, offset = binding["scale"], binding["offset_z"]
    parts = {i: points for i, _, points, *_ in idle_parts(pack, binding) if i in indices}
    masks = {texture: np.zeros(dds_alpha(pack / texture).shape) for texture in groups}
    if style == "hull":
        stripe = hull_band(pack, binding, set(indices), scale, offset)[:4] + (0.0,)
        for texture, group in groups.items():
            masks[texture] = paint_mask(pack, binding, group, texture, targets["stripe"], stripe=stripe)[1]
        z = np.concatenate([(p[:, 2] + offset) * scale for p in parts.values()])
        top, low = float(z.max()), max(0.0, float(z.min()))
        floor = 1 - targets.get("accent_floor", .1)
        def height(i, p):
            value = (((p[:, 2:3] + offset) * scale) - low) / max(top - low, 1e-9)
            return np.where(value >= floor, value, -1.0)
        scores = [(height, targets["accents"])]
    else:
        moving = idle_motion(pack, binding, indices)
        static = np.concatenate([p[~moving[i]] for i, p in parts.items()])
        centre = static[:, :2].mean(0)
        axis = symmetry_axis(static[:, :2])
        def frame(p):
            xy = p[:, :2] - centre
            return xy[:, 0] * axis[0] + xy[:, 1] * axis[1], np.abs(xy[:, 1] * axis[0] - xy[:, 0] * axis[1])
        along, across = frame(static)
        span = max(float(np.percentile(across, 98)), 1e-9)
        first, last = np.percentile(along, 1), np.percentile(along, 99)
        length = max(float(last - first), 1e-9)
        ends = [along <= first + .2 * length, along >= last - .2 * length]
        rear = int(static[ends[1], 2].max(initial=-np.inf) > static[ends[0], 2].max(initial=-np.inf))
        tail = ends[rear]
        tail_span = max(float(np.percentile(across[tail], 98)), 1e-9) if tail.any() else span
        tail_low, tail_top = (float(static[tail, 2].min()), float(static[tail, 2].max())) if tail.any() else (0.0, 1.0)
        def tips(i, p):
            return np.where(moving[i], -1.0, frame(p)[1] / span)[:, None]
        def tail_tips(i, p):
            a, c = frame(p)
            rearward = (a - first) / length if rear == 0 else (last - a) / length
            value = np.maximum(c / tail_span, (p[:, 2] - tail_low) / max(tail_top - tail_low, 1e-9))
            return np.where(moving[i] | (rearward > .2), -1.0, value)[:, None]
        scores = [(tips, targets["tips"]), (tail_tips, targets["tail"])]
    samples = measured_samples(pack, binding, indices)
    pixels = sum(total for total, *_ in samples)
    for score, share in scores:
        maps = {texture: texel_values(pack, binding, set(group), masks[texture].shape, score)[0][..., 0]
                for texture, group in groups.items()}
        seen = np.array([maps[idle[f"part{int(w)}"]["texture"]][y, x] for _, which, tx, ty, _ in samples
                         for w, x, y in zip(which, tx, ty)], float)
        ranked = np.sort(seen[np.isfinite(seen) & (seen >= 0)])[::-1]
        wanted = int(share * pixels)
        if not len(ranked) or wanted <= 0:
            continue
        threshold = ranked[min(wanted, len(ranked) - 1)]
        for texture in groups:
            value = np.nan_to_num(maps[texture], nan=-np.inf)
            masks[texture] = np.maximum(masks[texture], soften(np.clip((value - threshold) / band + .5, 0, 1)))
    return masks


def paint_marks(bindings: dict, pack: Path, components: dict, styles: dict, targets: dict, strength: float) -> dict:
    """Replace the owner masks of hulls and aircraft (styles: native key ->
    `hull` or `extremities`) with mark_masks over their largest component,
    written like paint_garments; their broad authored masks would otherwise
    tint the whole vessel."""
    report = {}
    for key, binding in bindings.items():
        if not isinstance(binding, dict) or "idle" not in binding or key not in components:
            continue
        style = styles.get(binding["key0"])
        if style is None:
            continue
        idle = binding["idle"]
        before = measure(pack, binding)
        assets = components[key]["idle"]
        share = {}
        for i in range(idle["part_count"]):
            share[assets[i]] = share.get(assets[i], 0) + before["part_share"][i]
        chosen = max(share, key=lambda a: share[a])
        indices = [i for i in range(idle["part_count"]) if assets[i] == chosen
                   and idle[f"part{i}"].get("material_model", 0) < .5 and not idle[f"part{i}"].get("cutout", 0)]
        if not indices:
            continue
        painted = {}
        for texture, mask in mark_masks(pack, binding, indices, style, targets).items():
            if not (mask > .01).any():
                continue  # a flat single-texel material (a procedural missile) cannot carry marks
            blob = bc3_with_alpha(pack / texture, 1 - mask)
            name = "textures/" + hashlib.sha256(blob).hexdigest() + ".dds"
            if not (pack / name).exists():
                (pack / name).write_bytes(blob)
            painted[texture] = name
        if not painted:
            continue
        for action, ids in components[key].items():
            data = binding.get(action)
            for i, asset in enumerate(ids):
                part = data.get(f"part{i}") if isinstance(data, dict) else None
                if part is None or part.get("material_model", 0) >= .5:
                    continue
                if asset == chosen and part.get("texture") in painted:
                    part["texture"] = painted[part["texture"]]
                    part["owner_mask"] = 1
                    part["owner_strength"] = strength
                elif part.get("owner_mask", 0):
                    part["owner_mask"] = 0  # other components keep no broad tint
        binding["owner_marks"] = {"component": chosen, "style": style, "coverage_before": round(before["coverage"], 4),
                                  "coverage_after": round(measure(pack, binding)["coverage"], 4)}
        report[binding["key0"]] = binding["owner_marks"]
    return report


def components_by_binding(manifest: dict, bindings: dict) -> dict:
    """Component ids per binding and action, from the pack manifest."""
    result = {}
    for unit in manifest["units"].values():
        for key, binding in bindings.items():
            if isinstance(binding, dict) and "key_count" in binding and \
                    set(unit["civ3_ids"]) == {binding["key" + str(i)] for i in range(binding["key_count"])}:
                result[key] = {action: [part.get("asset") for part in data["parts"]]
                               for action, data in unit["actions"].items()}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", default="UnitAnimationFidelity")
    parser.add_argument("--keys", default="", help="comma-separated native keys to measure (default all)")
    args = parser.parse_args(argv)
    pack = ROOT / "Renderer/packs" / args.pack
    bindings = json.loads((pack / "bindings.json").read_text())
    wanted = set(filter(None, args.keys.split(",")))
    cache = {}
    for key, binding in sorted(bindings.items()):
        if not isinstance(binding, dict) or "idle" not in binding:
            continue
        if wanted and binding["key0"] not in wanted:
            continue
        result = measure(pack, binding, cache=cache)
        shares = " ".join(f"{s:.2f}" for s in result["part_share"])
        print(f"{binding['key0'][5:]:22} coverage {100 * result['coverage']:5.1f}%  parts {shares}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
