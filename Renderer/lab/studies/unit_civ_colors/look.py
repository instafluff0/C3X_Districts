#!/usr/bin/env python3
"""Lab: unit lighting looks against Civ III's sprites (Mac preview).

Civ III units read at a glance. Measured on the DEFAULT sprites (owner
palette ntp01, display values): body luma about 90 of 255 with a wide spread
(standard deviation about 45, 5th-95th percentile about 35-180), saturation
0.35-0.5, no dark outline, and a key light from the upper right.

This previews the live unit material in NumPy -- the unit BeautyFrame
(key light, sky, view), base x tint, owner modulation and ramp, sky-weighted
ambient with AO, key diffuse, GGX specular from gloss, rim, the look
gain/saturation and highlight roll-off, then the production filmic display
(no self-shadow or normal maps) -- at the live projection, 128-pixel tiles, for named looks, side by side with the Civ III sprite on grass. It
is an approximation for comparing looks; an accepted look is ported to
prepare_unit_shader.py and verified in game.

    python3 Renderer/lab/studies/unit_civ_colors/look.py [--units ...] [--looks now,dusk,...]

Writes Renderer/lab/out/unit-civ-colors/look.png and look.json (tone
statistics per unit and look, next to Civ III's).
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transfer as T  # noqa: E402
from image_io import write_png  # noqa: E402
from Renderer.tools.asset_compiler import civ3_flc  # noqa: E402
from Renderer.tools.asset_compiler.civ3_unit_sprites import default_flc, game_roots, pedia_names  # noqa: E402

LUMA = np.array([.2126, .7152, .0722])
GRASS_DISPLAY = np.array([112, 136, 52], float) / 255   # unit_readability/sheet.py ground
FILMIC = .5
Z_SHADE = 150.0 / (112 * .82)   # direct_units.h: shading-frame normal z divisor
# The preview has no self-shadow or normal maps and reads brighter than the
# D3D unit path: this scale matches the mean body luma of the live render
# lab/out/unit-readability/tone-new (noon, SW; Infantry, Rifleman, Tank,
# Bomber, Battleship) -- see calibrate().
CALIBRATION = .45

# Looks: environment hour, and changes to the live material response.
#   gain, saturation   pack look_gain / look_saturation (production .5 / .25)
#   ambient, sun       scale of the environment's ambient and sun
#   key                None: the shared key light; else a unit key light fixed
#                      to the camera (screen right, screen up, toward viewer)
#   flatten            albedo detail pulled toward its local mean (0..1)
#   specular, knee     specular scale (x .52) and highlight roll-off start (1.6)
#   ao_key             share of the AO texture that also darkens the key light
#   sharpen            unsharp amount on the unit layer at map scale
#   neutral            warm key / cool sky moved toward grey (0..1)
#   steady             floor on the hour's light, as a share of noon (0..1)
#   outline            darkening of a 1-pixel ring just outside the silhouette
#   masks              "current": the pack's owner masks; default the transferred ones
#   flat_owner         owner ramp over flattened albedo (default; production: False)
PROPOSED = {"sharpen": .6, "specular": 2.2, "knee": 2.4, "ao_key": .5, "saturation": 0.0,
            "sun": 1.15, "ambient": .85, "neutral": .6, "steady": .75}
LOOKS = {
    "now": {"hour": 12},
    "proposed": {"hour": 12, **PROPOSED},
    "proposed outline": {"hour": 12, **PROPOSED, "outline": .35},
    "now dusk": {"hour": 18},
    "proposed dusk": {"hour": 18, **PROPOSED},
    "now night": {"hour": 0},
    "proposed night": {"hour": 0, **PROPOSED},
    "crisp": {"hour": 12, "sharpen": .6},
    "neutral": {"hour": 12, "neutral": .6},
}


def environment(hour: float) -> dict:
    """environment_runtime.cpp evaluate_environment (season 0)."""
    def smooth(a, b, x):
        t = min(1, max(0, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    hour %= 24
    daylight = smooth(5, 7, hour) * (1 - smooth(17, 19, hour))
    elevation = max(0.0, math.sin((hour - 5) * math.pi / 14))
    warm = min(1, max(0, 1 - abs(elevation - .18) / .32)) * smooth(.02, .16, elevation)
    dusk, dawn = warm * smooth(12, 18, hour), warm * (1 - smooth(12, 18, hour))
    phase = hour * math.pi / 12
    cast_x, cast_y = math.cos(phase), -math.sin(phase)
    wx, wy = cast_x + 2 * cast_y, cast_x - 2 * cast_y
    h = math.hypot(wx, wy)
    sun = np.array([-wx / h, -wy / h, max(.08, elevation)])
    return {"sun_direction": sun / np.linalg.norm(sun),
            "sun_color": np.array([1, .96 - .30 * dusk - .16 * dawn, .88 - .46 * dusk - .28 * dawn]),
            "sun_intensity": min(1, max(0, elevation * (.45 + .35 * daylight) + .20 * daylight)),
            "ambient": np.array([.18 + .55 * daylight + .20 * dusk + .04 * dawn,
                                 .23 + .55 * daylight + .07 * dusk + .02 * dawn,
                                 .42 + .40 * daylight - .12 * dusk + .03 * dawn]),
            "moon_color": np.array([.42, .56, 1.0]),
            "moon_intensity": (1 - daylight) * (.18 + .16 * (1 - daylight)),
            "exposure": .79 + .21 * daylight}


def ground(hour: float) -> np.ndarray:
    """Grass lit by the hour, relative to noon: a flat ground's ambient plus
    sun and moon at their elevation (a proxy for the terrain shader)."""
    def light(e):
        sun = e["sun_color"] * e["sun_intensity"] * max(e["sun_direction"][2], 0)
        return (e["ambient"] + sun + e["moon_color"] * e["moon_intensity"] * .8) * e["exposure"]
    return light(environment(hour)) / light(environment(12))


def frame(hour: float, neutral: float = 0.0, steady: float = 0.0) -> dict:
    """The unit BeautyFrame (sandbox/direct_units.h, scene_lighting.h
    key_light): the key follows the sun/moon azimuth at the shared shadow
    slope; warm key 2.05 x (1, 4.5/6.2, 3.5/6.2) and cool sky .62 x
    (.34, .45, .60), both scaled by the hour against noon; fixed view."""
    env, noon = environment(hour), environment(12)
    x, y = env["sun_direction"][:2]
    h = math.hypot(x, y)
    x, y = (x / h, y / h) if h > 1e-6 else (-1.0, 0.0)
    n = math.sqrt(1 + 1.35 ** 2)
    chromatic = np.array([1, 4.5 / 6.2, 3.5 / 6.2])
    sky = np.array([.34, .45, .60])
    # neutral: the warm key and cool sky move toward grey of equal luma, so
    # paint keeps its own hue (grey hulls, tan armour, the civ colour).
    chromatic = chromatic + neutral * ((chromatic * LUMA).sum() - chromatic)
    sky = sky + .6 * neutral * ((sky * LUMA).sum() - sky)
    # steady: units keep this share of noon light whatever the hour (the
    # hour still tints them); Civ III units never darken.
    total = env["sun_intensity"] + env["moon_intensity"]
    tint = (env["sun_color"] * env["sun_intensity"] + env["moon_color"] * env["moon_intensity"]) / max(total, 1e-6)
    key = max(total / (noon["sun_intensity"] + noon["moon_intensity"]), steady)
    fill = env["ambient"] / noon["ambient"]
    fill = fill + steady * np.maximum(0, 1 - fill)
    return {"light": np.array([x / n, y / n, 1.35 / n]),
            "sun": chromatic * tint / noon["sun_color"] * 2.05 * key,
            "ambient": sky * fill * .62,
            "view": np.array([.490290, -.735435, .469979]),
            "exposure": env["exposure"] + steady * (1 - env["exposure"])}


def linear(srgb):
    return np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)


def display(rgb):
    """native/scene_display.h scene_display_srgb (unit_readability/sheet.py)."""
    x = np.maximum(rgb, 0)
    neutral = x / (1 + x.max(axis=-1, keepdims=True))
    f = np.minimum(x * .65, 64)
    f = np.clip((f * (2.51 * f + .03)) / (f * (2.43 * f + .59) + .14), 0, 1)
    v = neutral + (f - neutral) * FILMIC
    return np.where(v <= .0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0), 1 / 2.4) - .055)


def inverse_display(target, iterations=80):
    x = np.full_like(target, .2)
    for _ in range(iterations):
        x = x * np.clip(target / np.maximum(display(x), 1e-5), .5, 2)
    return x


def bc4(path: Path) -> np.ndarray:
    """Top mip of a BC4_UNORM DXGI DDS (0..1)."""
    data = path.read_bytes()
    height, width = struct.unpack_from("<II", data, 12)
    bw, bh = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    blocks = np.frombuffer(data, np.uint8, bw * bh * 8, 148).reshape(bh, bw, 8)
    e0, e1 = blocks[..., 0].astype(float), blocks[..., 1].astype(float)
    six = (e0 > e1)[..., None]
    k = np.arange(8)
    table = np.where(six, np.where(k == 0, e0[..., None], np.where(k == 1, e1[..., None],
                     ((8 - k) * e0[..., None] + (k - 1) * e1[..., None]) / 7)),
                     np.where(k == 0, e0[..., None], np.where(k == 1, e1[..., None], np.where(k < 6,
                     ((6 - k) * e0[..., None] + (k - 1) * e1[..., None]) / 5, np.where(k == 6, 0, 255)))))
    bits = np.zeros((bh, bw), np.uint64)
    for i in range(6):
        bits |= blocks[..., 2 + i].astype(np.uint64) << np.uint64(8 * i)
    index = np.stack([(bits >> np.uint64(3 * j)) & np.uint64(7) for j in range(16)], -1).astype(int)
    values = np.take_along_axis(table, index, -1).reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)
    return values[:height, :width] / 255


def ggx(n, light, view, roughness, f0):
    halfway = light + view
    halfway = halfway / np.linalg.norm(halfway)
    ndl = np.clip((n * light).sum(-1), 0, 1); ndv = np.clip((n * view).sum(-1), 0, 1)
    ndh = np.clip((n * halfway).sum(-1), 0, 1); vdh = float(np.clip((view * halfway).sum(), 0, 1))
    alpha = np.maximum(.055, roughness ** 2); a2 = alpha ** 2
    d = a2 / np.maximum(math.pi * (ndh ** 2 * (a2 - 1) + 1) ** 2, 1e-4)
    k = (roughness + 1) ** 2 * .125
    g = ndv / np.maximum(ndv * (1 - k) + k, 1e-4) * ndl / np.maximum(ndl * (1 - k) + k, 1e-4)
    fresnel = f0 + (1 - f0) * (1 - vdh) ** 5
    return d * g * fresnel / np.maximum(4 * ndv * ndl, 1e-4) * ndl


def key_direction(screen_right: float, screen_up: float, toward_viewer: float) -> np.ndarray:
    """A camera-fixed light in the shading frame, from screen axes."""
    right = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)                  # tile space
    toward = np.array([1.0, 1.0, 64 / T.coverage.Z_PIXELS]); toward /= np.linalg.norm(toward)
    up = np.cross(right, toward); up /= np.linalg.norm(up)
    d = screen_right * right + screen_up * up + toward_viewer * toward
    d = np.array([d[0], -d[1], d[2] * Z_SHADE])                         # tile -> shading frame (direction)
    return d / np.linalg.norm(d)


class Materials:
    def __init__(self):
        self.cache = {}

    def alpha(self, name: str):
        key = (name, "alpha")
        if key not in self.cache:
            self.cache[key] = T.coverage.dds_alpha(T.PACK / name)
        return self.cache[key]

    def get(self, name: str, kind: str):
        key = (name, kind)
        if key not in self.cache:
            path = T.PACK / name
            self.cache[key] = bc4(path) if kind == "ao" else T.coverage.dds_rgb(path)
        return self.cache[key]


def shade(result: dict, direction: int, look: dict, owner_linear: np.ndarray, materials: Materials) -> np.ndarray:
    """Linear radiance (premultiplied RGBA) at the transfer view's supersampling."""
    env = frame(look.get("hour", 12), look.get("neutral", 0), look.get("steady", 0))
    view = result["views"][direction]
    out = np.zeros(view["which"].shape + (4,))
    light = env["light"] if look.get("key") is None else key_direction(*look["key"])
    sun = env["sun"] * look.get("sun", 1)
    ambient = env["ambient"] * look.get("ambient", 1)
    eye = env["view"]
    gain, saturation = look.get("gain", .5), look.get("saturation", .25)
    for mine, part, t, ty, tx in T.visible_texels(result, view):
        n = view["normal"][mine]
        n = np.stack([n[:, 0], -n[:, 1], n[:, 2] / Z_SHADE], 1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        tint = np.array([part.get("tint_r", 1), part.get("tint_g", 1), part.get("tint_b", 1)])
        base = result["rgb"][t]
        if look.get("flatten"):
            base = (1 - look["flatten"]) * base + look["flatten"] * T.flattened(base)
        b = linear(base[ty, tx])
        albedo = b * tint
        if look.get("masks") == "current":   # the pack's owner mask, as the shader reads it
            mask = T.coverage.owner_weight(part, materials.alpha(t)[ty, tx])
        else:                                 # the transferred mask at garment strength
            mask = result["masks"][t][ty, tx] * .9
        albedo = albedo * (1 - mask[:, None]) + albedo * (.45 + owner_linear * 1.1) * mask[:, None]
        flat = b * tint
        if look.get("flat_owner", True):
            if t not in result.setdefault("flat", {}):
                result["flat"][t] = T.flattened(result["rgb"][t])
            flat = linear(result["flat"][t][ty, tx]) * tint
        ramp = np.clip(owner_linear[None] * np.minimum(1, (flat * LUMA).sum(1, keepdims=True) / max((owner_linear * LUMA).sum(), .03)), 0, 1)
        owned = np.clip(mask, 0, 1)
        albedo = albedo * (1 - owned[:, None]) + ramp * owned[:, None]
        ao = 1.0
        if "ao_texture" in part:
            ao = .48 + .52 * materials.get(part["ao_texture"], "ao")[ty % materials.get(part["ao_texture"], "ao").shape[0],
                                                                      tx % materials.get(part["ao_texture"], "ao").shape[1]]
        gloss = .08
        if "gloss_texture" in part:
            g = materials.get(part["gloss_texture"], "rgb")
            gloss = linear(g[ty % g.shape[0], tx % g.shape[1], 0])
        sky = np.clip(n[:, 2] * .5 + .5, 0, 1)
        diffuse = np.clip((n * light).sum(1), 0, 1)
        crease = 1 - look.get("ao_key", 0) * (1 - np.asarray(ao))
        lit = ambient[None] * (.48 + .52 * sky)[:, None] * np.asarray(ao)[..., None] + \
            sun[None] * ((.035 + .965 * diffuse) * crease)[:, None]
        radiance = albedo * lit
        roughness = .91 + (.31 - .91) * np.clip(gloss, 0, 1)
        radiance += sun[None] * .52 * look.get("specular", 1) * ggx(n, light, eye, roughness, .045)[:, None]
        rim = (1 - np.clip((n * eye).sum(1), 0, 1)) ** 3
        radiance += ambient[None] * rim[:, None] * .08
        l = (radiance * LUMA).sum(1, keepdims=True)
        radiance = np.maximum(0, l + (radiance - l) * (1 + saturation * (1 - owned[:, None]))) * (1 + gain)
        peak = radiance.max(1, keepdims=True)
        knee = look.get("knee", 1.6)
        rolled = np.where(peak > knee, (knee + (peak - knee) / (1 + (peak - knee) / .8)) / np.maximum(peak, 1e-6), 1)
        out[mine, :3] = radiance * rolled * env["exposure"] * CALIBRATION * look.get("exposure", 1)
        out[mine, 3] = 1
    return out


def downsample(image: np.ndarray, step: int) -> np.ndarray:
    h, w = image.shape[0] // step * step, image.shape[1] // step * step
    return image[:h, :w].reshape(h // step, step, w // step, step, image.shape[2]).mean((1, 3))


def tone(rgb_display: np.ndarray, body: np.ndarray, lawn: np.ndarray = GRASS_DISPLAY) -> dict:
    """Civ III-comparable statistics over body pixels (0..255 display), and
    figure-ground separation: how far the silhouette's rim pixels are from
    the grass in luma and in colour."""
    rgb = rgb_display[body] * 255
    luma = (rgb * LUMA).sum(1)
    sat = (rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1)
    inner = body.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        inner &= np.roll(body, (dy, dx), (0, 1))
    rim = rgb_display[body & ~inner] * 255
    ground = lawn * 255
    return {"luma": round(float(luma.mean()), 1), "std": round(float(luma.std()), 1),
            "p5": round(float(np.percentile(luma, 5)), 1), "p95": round(float(np.percentile(luma, 95)), 1),
            "sat": round(float(sat.mean()), 3),
            "rim_luma": round(float(np.abs((rim * LUMA).sum(1) - (ground * LUMA).sum()).mean()), 1),
            "rim_colour": round(float(np.linalg.norm(rim - ground, axis=1).mean()), 1),
            "body_colour": round(float(np.linalg.norm(rgb - ground, axis=1).mean()), 1)}


def render(result: dict, direction: int, look: dict, owner: np.ndarray, materials: Materials, grass: np.ndarray):
    """One unit in one look at map scale over the hour's grass: display RGB,
    unit-layer alpha (after sharpen/outline), the ground colour and tone."""
    radiance = downsample(shade(result, direction, look, owner, materials), T.SUPERSAMPLE)
    alpha = radiance[..., 3:4]
    lawn = grass * ground(look.get("hour", 12))
    shown = display(radiance[..., :3] + lawn * (1 - alpha))
    cell = display(lawn[None, :])[0]
    if look.get("sharpen"):
        # Unsharp mask on the unit layer only (ground untouched).
        blur = np.stack([T.coverage.soften(shown[..., c]) for c in range(3)], -1)
        shown = np.where(alpha > .02, np.clip(shown + look["sharpen"] * (shown - blur), 0, 1), shown)
    if look.get("outline"):
        # Faint dark line just outside the silhouette (unit layer alpha).
        solid = alpha[..., 0] > .5
        grown = solid.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            grown |= np.roll(solid, (dy, dx), (0, 1))
        ring = grown & ~solid
        shown[ring] *= 1 - look["outline"]
        alpha = np.where(ring[..., None], np.maximum(alpha, look["outline"]), alpha)
    return shown, alpha, cell, tone(shown, alpha[..., 0] > .5, cell)


def civ3_sprite(sprite: np.ndarray, palette, civ_palette):
    """Civ III frame as display RGB and body alpha (shadow and smoke left out)."""
    body = sprite < 224
    rgba = civ3_flc.rgba(sprite, palette, civ_palette).astype(float) / 255
    return rgba[..., :3], body[..., None].astype(float), tone(rgba[..., :3], body)


def place(canvas, shown, alpha, cy, cx, cell_w, cell_h, ground_rgb=None):
    """Crop to the alpha box and centre in a cell (cell filled with the ground)."""
    if ground_rgb is not None:
        canvas[cy - cell_h // 2:cy - cell_h // 2 + cell_h, cx - cell_w // 2:cx - cell_w // 2 + cell_w] = ground_rgb
    rows, cols = np.nonzero(alpha[..., 0] > .02)
    if not len(rows):
        return
    shown = shown[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    alpha = alpha[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = min(shown.shape[0], cell_h), min(shown.shape[1], cell_w)
    top, left = (shown.shape[0] - h) // 2, (shown.shape[1] - w) // 2
    shown, alpha = shown[top:top + h, left:left + w], alpha[top:top + h, left:left + w]
    y0, x0 = cy - h // 2, cx - w // 2
    canvas[y0:y0 + h, x0:x0 + w] = canvas[y0:y0 + h, x0:x0 + w] * (1 - alpha) + shown * alpha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", default=",".join(T.DEFAULT_UNITS))
    parser.add_argument("--looks", default=",".join(LOOKS))
    parser.add_argument("--direction", type=int, default=5)
    parser.add_argument("--civ", type=int, default=1)
    parser.add_argument("--zoom", type=int, default=2)
    parser.add_argument("--out", default="look")
    args = parser.parse_args(argv)
    T.CIVNESS = T.civness()
    looks = [name for name in args.looks.split(",") if name]
    civ_palette = civ3_flc.read_pcx_palette(T.CIV3_ROOT / "Art/Units/Palettes" / f"ntp{args.civ:02d}.pcx")
    owner = linear(T.owner_colour(civ_palette))
    grass = inverse_display(GRASS_DISPLAY[None, :].copy())[0]
    materials, roots = Materials(), game_roots(T.CIV3_ROOT)
    cell_w, cell_h = 150, 130
    canvas = np.zeros((len(args.units.split(",")) * cell_h, (1 + len(looks)) * cell_w, 3))
    canvas[:] = GRASS_DISPLAY
    stats = {}
    for row, key in enumerate(filter(None, args.units.split(","))):
        result = T.transfer(key)
        palette = civ3_flc.read(T.civ3_art([result["binding"][f"key{i}"] for i in range(result["binding"]["key_count"])])).palette
        cy = row * cell_h + cell_h // 2
        shown, alpha, stats[key] = civ3_sprite(result["sprites"][args.direction], palette, civ_palette)
        stats[key] = {"civ3": stats[key]}
        place(canvas, shown, alpha, cy, cell_w // 2, cell_w, cell_h)
        for k, name in enumerate(looks, 1):
            shown, alpha, cell, stats[key][name] = render(result, args.direction, LOOKS[name], owner, materials, grass)
            place(canvas, shown, alpha, cy, k * cell_w + cell_w // 2, cell_w, cell_h, cell)
        print(key, stats[key], flush=True)
    image = np.repeat(np.repeat((np.clip(canvas, 0, 1) * 255 + .5).astype(np.uint8), args.zoom, 0), args.zoom, 1)
    header = np.concatenate([label_strip(name, cell_w * args.zoom) for name in ["civ3"] + looks], 1)
    T.OUT.mkdir(parents=True, exist_ok=True)
    write_png(T.OUT / f"{args.out}.png", np.concatenate([header, image], 0))
    (T.OUT / f"{args.out}.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(T.OUT / f"{args.out}.png")


def label_strip(text: str, width: int) -> np.ndarray:
    sys.path.insert(0, str(HERE.parent / "unit_readability"))
    import sheet
    return sheet.label_strip(text, width)


if __name__ == "__main__":
    main()
