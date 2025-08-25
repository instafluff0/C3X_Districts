#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Civ 3 day-night PCX transformer (pure Python, Pillow) — parameterized
- Reads PCX files from Terrain/1200 (base/noon).
- Writes transformed copies to sibling folders: 100, 200, ..., 2400 (skips 1200).
- Keeps ignored colors #ff00ff and #00ff00 EXACTLY unchanged (same pixels & RGB).
- Preserves image size/format.
- Look is controlled by CLI params (see --help).

Examples:
  python civ3_daynight_palette_cli.py --terrain "/path/Terrain" --noon 1200
  python civ3_daynight_palette_cli.py --terrain "/path/Terrain" --noon 1200 \
    --only-hour 2400 --only-file "terrain1.pcx" \
    --darken-max 0.30 --value-floor 0.04 --desat-max 0.20 --hue-shift 0.45 --hue-target-deg 220
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image

MAGENTA = (255, 0, 255)   # ff00ff
GREEN   = (0, 255, 0)     # 00ff00


# ---------- Look controls ----------

@dataclass
class LookParams:
    darken_max: float = 0.25
    value_floor: float = 0.05
    desat_max: float = 0.15
    hue_shift: float = 0.35
    hue_target_deg: float = 220.0
    hue_shift_warm: float = 0.0
    warm_min_deg: float = 10.0
    warm_max_deg: float = 70.0
    hue_steer_mode: str = "shortest"
    sat_gate: float = 0.25
    val_gate: float = 0.35
    # NEW (blue-sector)
    blue_min_deg: float = 185.0     # start of “blue range”
    blue_max_deg: float = 255.0     # end of “blue range”
    blue_sat_boost: float = 0.0     # extra saturation added in blue sector at midnight (0..1); 0 keeps old behavior
    blue_val_lift: float = 0.0      # optional tiny brightness lift in blue sector at midnight (0..1)

def _hue_in_range(h01: float, start_deg: float, end_deg: float) -> bool:
    a = (start_deg % 360.0) / 360.0
    b = (end_deg   % 360.0) / 360.0
    return (a <= b and a <= h01 <= b) or (a > b and (h01 >= a or h01 <= b))\

def _smoothstep(x, e0, e1):
    if e1 <= e0:  # guard
        return 1.0 if x >= e1 else 0.0
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2*t)

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def hour_factor(hour_1_24: int) -> float:
    """0 at 12 (no change), 1 at 24/0 (midnight). Smooth cosine easing."""
    d = abs(hour_1_24 - 12)
    if d > 12:
        d = 24 - d
    return 0.5 * (1.0 - math.cos(math.pi * (d / 12.0)))

def _to_byte(x: float) -> int:
    return int(min(255, max(0, round(x))))

def transform_rgb_triplet(r: int, g: int, b: int, f: float, p: LookParams):
    import colorsys
    dm = max(0.0, min(1.0, p.darken_max))
    vf = max(0.0, min(1.0, p.value_floor))
    ds = max(0.0, min(1.0, p.desat_max))
    hs = max(0.0, min(1.0, p.hue_shift))
    hs_warm = max(0.0, min(1.0, p.hue_shift_warm))
    ht = (p.hue_target_deg % 360.0) / 360.0

    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)

    # darker but playable
    v = v * (1 - dm*f) + vf*f
    # modest desaturation into night
    s = s * (1 - ds*f)

    # Gating: only hue-shift noticeably if the color is reasonably saturated & bright
    sat_w = _smoothstep(s, p.sat_gate, min(1.0, p.sat_gate + 0.25))
    val_w = _smoothstep(v, p.val_gate, min(1.0, p.val_gate + 0.25))
    gate = sat_w * val_w

    # Warm-hue sector detection (brown/orange)
    warm_min = (p.warm_min_deg % 360.0) / 360.0
    warm_max = (p.warm_max_deg % 360.0) / 360.0
    is_warm = warm_min <= warm_max and (warm_min <= h <= warm_max)

    # Effective hue-shift strength
    hs_eff = (hs_warm if is_warm else hs) * f * gate

    # Steering: default shortest; optionally force CCW/CW
    delta = ht - h
    if p.hue_steer_mode == "shortest":
        if delta > 0.5: delta -= 1.0
        elif delta < -0.5: delta += 1.0
    elif p.hue_steer_mode == "ccw":     # increase hue only (avoids crossing 0° from warm hues)
        if delta < 0.0: delta += 1.0
    elif p.hue_steer_mode == "cw":      # decrease hue only
        if delta > 0.0: delta -= 1.0

    # Extra guard: if it's warm and our delta would head through magenta, go CCW instead
    if is_warm and delta < 0.0:
        delta += 1.0

    h = (h + hs_eff * delta) % 1.0

    # NEW: Blue-sector saturation (and optional value) boost
    if p.blue_sat_boost > 0.0 or p.blue_val_lift > 0.0:
        if _hue_in_range(h, p.blue_min_deg, p.blue_max_deg):
            # push saturation toward 1.0 by up to blue_sat_boost at midnight
            s = s + (1.0 - s) * (max(0.0, min(1.0, p.blue_sat_boost)) * f)
            # tiny optional brightness lift, if you set it (>0) on the CLI
            v = v + (max(0.0, min(1.0, p.blue_val_lift)) * f)

    # clamp and convert back
    s = max(0.0, min(1.0, s))
    v = max(0.0, min(1.0, v))
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)

    return int(round(rr*255.0)), int(round(gg*255.0)), int(round(bb*255.0))



# ---------- Palette helpers (Pillow) ----------

def get_palette(image: Image.Image) -> List[int]:
    """Return palette list [r0,g0,b0, r1,g1,b1, ...], padded to 256*3."""
    pal = image.getpalette() or []
    if len(pal) < 256*3:
        pal = pal + [0] * (256*3 - len(pal))
    return pal[:256*3]

def put_palette(image: Image.Image, palette: List[int]) -> None:
    if len(palette) < 256*3:
        palette = palette + [0] * (256*3 - len(palette))
    image.putpalette(palette[:256*3])

def find_color_index(palette: List[int], rgb: Tuple[int, int, int]) -> int:
    r, g, b = rgb
    for i in range(256):
        if palette[3*i] == r and palette[3*i+1] == g and palette[3*i+2] == b:
            return i
    return -1

def ensure_palette_has_colors(image: Image.Image, colors: List[Tuple[int, int, int]]) -> List[int]:
    """
    Ensure palette contains given colors. If missing and palette is full, replace least-used slots.
    Returns updated palette.
    """
    palette = get_palette(image)
    to_add = [c for c in colors if find_color_index(palette, c) == -1]
    if not to_add:
        return palette

    hist = image.histogram()
    if image.mode != 'P' or len(hist) != 256:
        candidates = list(range(255, -1, -1))
    else:
        candidates = sorted(range(256), key=lambda i: hist[i])

    protected = set()
    for rgb in colors + [MAGENTA, GREEN]:
        idx = find_color_index(palette, rgb)
        if idx != -1:
            protected.add(idx)

    replace_slots = []
    for i in candidates:
        if i in protected:
            continue
        replace_slots.append(i)
        if len(replace_slots) >= len(to_add):
            break

    if len(replace_slots) < len(to_add):
        tail = [i for i in range(255, -1, -1) if i not in protected][:len(to_add)-len(replace_slots)]
        replace_slots.extend(tail)

    for slot, rgb in zip(replace_slots, to_add):
        palette[3*slot:3*slot+3] = list(rgb)

    return palette


# ---------- Image processing ----------

def process_indexed_pcx(in_path: str, out_path: str, hour_1_24: int, params: LookParams) -> None:
    print(f"Processing {in_path} for hour {hour_1_24}")
    f = hour_factor(hour_1_24)
    im = Image.open(in_path)
    if im.mode != 'P':
        im = im.convert('RGBA')
        im = im.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)

    palette = get_palette(im)
    new_palette = palette.copy()

    for i in range(256):
        r, g, b = palette[3*i], palette[3*i+1], palette[3*i+2]
        if (r, g, b) == MAGENTA or (r, g, b) == GREEN:
            continue
        nr, ng, nb = transform_rgb_triplet(r, g, b, f, params)
        new_palette[3*i:3*i+3] = [nr, ng, nb]

    put_palette(im, new_palette)

    # Guarantee exact magic colors, then enforce exact indices for magic pixels
    pal_after = get_palette(im)
    idx_mag = find_color_index(pal_after, MAGENTA)
    idx_grn = find_color_index(pal_after, GREEN)

    src_rgb = Image.open(in_path).convert('RGB')
    w, h = src_rgb.size
    src_pixels = src_rgb.load()

    if idx_mag == -1 or idx_grn == -1:
        pal_after = ensure_palette_has_colors(im, [MAGENTA, GREEN])
        put_palette(im, pal_after)
        idx_mag = find_color_index(pal_after, MAGENTA)
        idx_grn = find_color_index(pal_after, GREEN)

    dst_px = im.load()  # palette indices
    for y in range(h):
        for x in range(w):
            r, g, b = src_pixels[x, y]
            if (r, g, b) == MAGENTA:
                dst_px[x, y] = idx_mag
            elif (r, g, b) == GREEN:
                dst_px[x, y] = idx_grn

    im.save(out_path, format='PCX')


def process_folder(terrain_dir: str, noon_subfolder: str,
                   params: LookParams, only_hour: int = None, only_file: str = None) -> None:
    base_dir = os.path.join(terrain_dir, noon_subfolder)
    if not os.path.isdir(base_dir):
        raise SystemExit(f"No such folder: {base_dir}")

    pcx_names = [f for f in os.listdir(base_dir) if f.lower().endswith('.pcx')]
    if not pcx_names:
        raise SystemExit(f"No PCX files found in {base_dir}")

    hours = [h for h in range(100, 2500, 100) if h != 1200]
    if only_hour is not None:
        if only_hour == 1200:
            raise SystemExit("1200 is the base (left untouched). Choose another hour.")
        if only_hour % 100 != 0 or only_hour < 100 or only_hour > 2400:
            raise SystemExit("--only-hour must be one of 100, 200, ..., 2400.")
        hours = [only_hour]

    if only_file is not None:
        pcx_names = [only_file] if only_file in pcx_names else []

    if not pcx_names:
        raise SystemExit("The requested --only-file was not found in the noon folder." if only_file else
                         "No PCX files to process.")

    for hhh in hours:
        hour_1_24 = hhh // 100
        out_dir = os.path.join(terrain_dir, f"{hhh}")
        os.makedirs(out_dir, exist_ok=True)

        for name in pcx_names:
            in_path = os.path.join(base_dir, name)
            out_path = os.path.join(out_dir, name)
            process_indexed_pcx(in_path, out_path, hour_1_24, params)


def main():
    ap = argparse.ArgumentParser(description="Civ 3 day-night PCX transformer (pure Python, Pillow)")
    ap.add_argument("--terrain", required=True, help="Path to Terrain folder (parent of noon subfolder)")
    ap.add_argument("--noon", default="1200", help="Name of noon subfolder (default: 1200)")
    ap.add_argument("--only-hour", type=int, help="Process a single hour folder (e.g., 2400)")
    ap.add_argument("--only-file", help="Process a single PCX filename from the noon folder")

    # Look params (all optional)
    ap.add_argument("--darken-max", type=float, default=0.25,
                    help="Max dimming of V at midnight (0..1). Default: 0.25")
    ap.add_argument("--value-floor", type=float, default=0.05,
                    help="Brightness floor added at midnight (0..1). Default: 0.05")
    ap.add_argument("--desat-max", type=float, default=0.15,
                    help="Max desaturation at midnight (0..1). Default: 0.15")
    ap.add_argument("--hue-shift", type=float, default=0.35,
                    help="Fraction of the way toward hue-target at midnight (0..1). Default: 0.35")
    ap.add_argument("--hue-target-deg", type=float, default=220.0,
                    help="Target hue in degrees (0..360). Default: 220 (cool blue)")
    ap.add_argument("--hue-shift-warm", type=float, default=0.0,
                help="Hue-shift strength for warm hues at midnight (0..1). Default: 0.0")
    ap.add_argument("--warm-min-deg", type=float, default=10.0,
                    help="Warm sector start in degrees. Default: 10")
    ap.add_argument("--warm-max-deg", type=float, default=70.0,
                    help="Warm sector end in degrees. Default: 70")
    ap.add_argument("--hue-steer-mode", choices=["shortest","ccw","cw"], default="shortest",
                    help="Direction for hue steering. Default: shortest")
    ap.add_argument("--sat-gate", type=float, default=0.25,
                    help="Minimum saturation for full hue shift. Default: 0.25")
    ap.add_argument("--val-gate", type=float, default=0.35,
                    help="Minimum value (brightness) for full hue shift. Default: 0.35")
    ap.add_argument("--blue-min-deg", type=float, default=185.0, help="Blue sector start in degrees. Default: 185")
    ap.add_argument("--blue-max-deg", type=float, default=255.0, help="Blue sector end in degrees. Default: 255")
    ap.add_argument("--blue-sat-boost", type=float, default=0.0, help="Extra saturation for blue sector at midnight (0..1). Default: 0 (off)")
    ap.add_argument("--blue-val-lift", type=float, default=0.0, help="Optional value (brightness) lift for blue sector at midnight (0..1). Default: 0 (off)")



    args = ap.parse_args()

    params = LookParams(
        darken_max=args.darken_max,
        value_floor=args.value_floor,
        desat_max=args.desat_max,
        hue_shift=args.hue_shift,
        hue_target_deg=args.hue_target_deg,
    )

    process_folder(args.terrain, args.noon, params, args.only_hour, args.only_file)


if __name__ == "__main__":
    main()
