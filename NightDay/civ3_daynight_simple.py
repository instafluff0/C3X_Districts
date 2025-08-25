#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Civ 3 day-night PCX transformer (SIMPLE)
- Reads PCX from Terrain/1200 and writes 100..2400 siblings.
- Preserves #ff00ff and #00ff00 exactly.
- Three controls only:
    --night  (0..1)  midnight darkness amount
    --blue   (0..1)  blue emphasis strength (teal->blue + blue sat boost)
    --sunset (0..1)  evening warmth around ~19:00 (7pm)

Defaults tuned for a cool dusk with readable terrain.
"""

import argparse, math, os
from typing import List, Tuple
from PIL import Image

MAGENTA = (255, 0, 255)   # ff00ff
GREEN   = (0, 255, 0)     # 00ff00

# --------- helpers ---------

def hour_factor(hour_1_24: int) -> float:
    """0 at 12 (no change), 1 at 24/0 (midnight). Smooth cosine easing."""
    d = abs(hour_1_24 - 12)
    if d > 12: d = 24 - d
    return 0.5 * (1.0 - math.cos(math.pi * (d / 12.0)))

def evening_weight(hour_1_24: int, center: float = 19.0, width: float = 3.0) -> float:
    """Cosine window centered at `center`, 1 at center, 0 at center±width."""
    d = abs(hour_1_24 - center)
    d = min(d, 24.0 - d)
    if d >= width or width <= 0: return 0.0
    t = d / width
    return 0.5 * (1.0 + math.cos(math.pi * t))

def get_palette(image: Image.Image) -> List[int]:
    pal = image.getpalette() or []
    if len(pal) < 256*3: pal += [0] * (256*3 - len(pal))
    return pal[:256*3]

def put_palette(image: Image.Image, palette: List[int]) -> None:
    if len(palette) < 256*3: palette += [0] * (256*3 - len(palette))
    image.putpalette(palette[:256*3])

def find_color_index(palette: List[int], rgb: Tuple[int,int,int]) -> int:
    r,g,b = rgb
    for i in range(256):
        if palette[3*i]==r and palette[3*i+1]==g and palette[3*i+2]==b:
            return i
    return -1

def ensure_palette_has_colors(image: Image.Image, colors: List[Tuple[int,int,int]]) -> List[int]:
    palette = get_palette(image)
    to_add = [c for c in colors if find_color_index(palette, c) == -1]
    if not to_add: return palette

    hist = image.histogram() if image.mode == 'P' else None
    if hist is None or len(hist) != 256:
        candidates = list(range(255, -1, -1))
    else:
        candidates = sorted(range(256), key=lambda i: hist[i])

    protected = set()
    for rgb in colors + [MAGENTA, GREEN]:
        idx = find_color_index(palette, rgb)
        if idx != -1: protected.add(idx)

    replace_slots = []
    for i in candidates:
        if i in protected: continue
        replace_slots.append(i)
        if len(replace_slots) >= len(to_add): break
    if len(replace_slots) < len(to_add):
        tail = [i for i in range(255, -1, -1) if i not in protected][:len(to_add)-len(replace_slots)]
        replace_slots.extend(tail)

    for slot, rgb in zip(replace_slots, to_add):
        palette[3*slot:3*slot+3] = list(rgb)
    return palette

def _hue_in_range(h01: float, start_deg: float, end_deg: float) -> bool:
    a = (start_deg % 360.0) / 360.0
    b = (end_deg   % 360.0) / 360.0
    return (a <= b and a <= h01 <= b) or (a > b and (h01 >= a or h01 <= b))

# --------- color grading (simple) ---------

def transform_rgb_triplet(r: int, g: int, b: int, hour_1_24: int, night: float, blue: float, sunset: float) -> Tuple[int,int,int]:
    import colorsys

    # clamp controls
    night  = max(0.0, min(1.0, night))
    blue   = max(0.0, min(1.0, blue))
    sunset = max(0.0, min(1.0, sunset))

    # global curves
    f  = hour_factor(hour_1_24)                  # 0 at noon, 1 at midnight
    ew = evening_weight(hour_1_24, 19.0, 3.0)    # 0..1 around ~7pm

    h0, s0, v0 = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    h,  s,  v  = h0, s0, v0

    # 1) DARKNESS: deeper than before + gentle gamma for midtone roll-off
    max_darken = 0.78          # was 0.60 → allows darker nights at --night 1.0
    floor_at_midnight = 0.02   # was 0.03 → slightly lower floor
    v = v * (1 - (max_darken*night)*f) + (floor_at_midnight*night)*f
    # Night gamma (darkens midtones without crushing highlights)
    v = pow(max(0.0, min(1.0, v)), 1.0 + 0.25*night*f)

    # 2) Mild global cool-down for non-warm hues (sky/water), gated by s & v
    warm_min, warm_max = 10/360.0, 75/360.0       # protect browns/oranges
    is_warm = warm_min <= h <= warm_max
    if not is_warm and s > 0.20 and v > 0.20:
        target_cool = 220/360.0                   # dusk blue
        delta = target_cool - h
        if delta > 0.5: delta -= 1.0
        elif delta < -0.5: delta += 1.0
        h = (h + (0.40*blue)*f * delta) % 1.0     # slightly softened from 0.45

    # 3) WATER CONTINUUM (teal ↔ ocean) with a gentle feathered blend
    if _hue_in_range(h0, 140, 215) and s0 > 0.12:
        # t = 0 at 140° (teal) … 1 at 215° (ocean), smoothed
        h0deg = h0 * 360.0
        t_lin = max(0.0, min(1.0, (h0deg - 140.0) / (215.0 - 140.0)))
        t = t_lin * t_lin * (3 - 2*t_lin)  # smoothstep

        # Interpolate targets & strengths so there’s no “edge” between bands
        target = ((232.0/360.0) * (1 - t)) + ((238.0/360.0) * t)         # 232° → 238°
        pull   = ((0.55)          * (1 - t)) + ((0.95)          * t)      # 0.55 → 0.95
        satb   = ((0.35)          * (1 - t)) + ((0.60)          * t)      # 0.35 → 0.60

        # Apply hue shift toward blended target
        d = target - h
        if d > 0.5: d -= 1.0
        elif d < -0.5: d += 1.0
        h = (h + (pull * blue * f) * d) % 1.0

        # Apply saturation boost (also blended)
        s = s + (1.0 - s) * (satb * blue * f)


    # 4) GREEN LAND → cooler/blue: original hue in 95–140°
    if _hue_in_range(h0, 95, 140) and s0 > 0.18:
        green_target = 232/360.0
        d3 = green_target - h
        if d3 > 0.5: d3 -= 1.0
        elif d3 < -0.5: d3 += 1.0
        h = (h + (0.95*blue)*(f**1.20) * d3) % 1.0   # CHANGED: stronger + midnight-biased
        s = s + (1.0 - s) * (0.40*blue)*(f**1.10)    # CHANGED: slightly higher sat, midnight-biased


    # 5) Blue-sector chroma boost (after shifts)
    if _hue_in_range(h, 190, 260):
        s = s + (1.0 - s) * (0.35*blue)*f         # was 0.30 → slightly up

    # 6) Sunset warmth (evenings only), skip warm hues to avoid reddening mountains
    if ew > 0 and not is_warm:
        target_warm = 15/360.0
        d4 = target_warm - h
        if d4 > 0.5: d4 -= 1.0
        elif d4 < -0.5: d4 += 1.0
        h = (h + (0.20*sunset) * ew * d4) % 1.0
        s = s + (1.0 - s) * (0.05*sunset) * ew

    # clamp & back to RGB
    s = max(0.0, min(1.0, s))
    v = max(0.0, min(1.0, v))
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
    return int(round(rr*255.0)), int(round(gg*255.0)), int(round(bb*255.0))

# --------- processing ---------

def process_indexed_pcx(in_path: str, out_path: str, hour_1_24: int, night: float, blue: float, sunset: float) -> None:
    print(f"Processing {in_path} for hour {hour_1_24}")
    im = Image.open(in_path)
    # quantize to palette if needed
    if im.mode != 'P':
        try:
            dnone = Image.Dither.NONE
        except Exception:
            dnone = getattr(Image, "NONE", 0)
        im = im.convert('RGBA').quantize(colors=256, method=Image.MEDIANCUT, dither=dnone)

    palette = get_palette(im)
    new_palette = palette.copy()

    # remap palette (skip magenta/green)
    for i in range(256):
        r,g,b = palette[3*i], palette[3*i+1], palette[3*i+2]
        if (r,g,b) == MAGENTA or (r,g,b) == GREEN:
            continue
        nr,ng,nb = transform_rgb_triplet(r,g,b, hour_1_24, night, blue, sunset)
        new_palette[3*i:3*i+3] = [nr,ng,nb]

    put_palette(im, new_palette)

    # ensure magic colors exist & pixels point to them exactly
    pal_after = get_palette(im)
    idx_mag = find_color_index(pal_after, MAGENTA)
    idx_grn = find_color_index(pal_after, GREEN)

    src_rgb = Image.open(in_path).convert('RGB')
    w,h = src_rgb.size
    src_px = src_rgb.load()

    if idx_mag == -1 or idx_grn == -1:
        pal_after = ensure_palette_has_colors(im, [MAGENTA, GREEN])
        put_palette(im, pal_after)
        idx_mag = find_color_index(pal_after, MAGENTA)
        idx_grn = find_color_index(pal_after, GREEN)

    dst_px = im.load()
    for y in range(h):
        for x in range(w):
            r,g,b = src_px[x,y]
            if (r,g,b) == MAGENTA:
                dst_px[x,y] = idx_mag
            elif (r,g,b) == GREEN:
                dst_px[x,y] = idx_grn

    im.save(out_path, format='PCX')

def process_folder(terrain_dir: str, noon_subfolder: str, night: float, blue: float, sunset: float,
                   only_hour: int = None, only_file: str = None) -> None:
    base_dir = os.path.join(terrain_dir, noon_subfolder)
    if not os.path.isdir(base_dir): raise SystemExit(f"No such folder: {base_dir}")

    pcx_names = [f for f in os.listdir(base_dir) if f.lower().endswith('.pcx')]
    if not pcx_names: raise SystemExit(f"No PCX files found in {base_dir}")

    hours = [h for h in range(100, 2500, 100) if h != 1200]
    if only_hour is not None:
        if only_hour == 1200: raise SystemExit("1200 is the base (left untouched). Choose another hour.")
        if only_hour % 100 != 0 or only_hour < 100 or only_hour > 2400:
            raise SystemExit("--only-hour must be one of 100, 200, ..., 2400.")
        hours = [only_hour]

    if only_file is not None:
        pcx_names = [only_file] if only_file in pcx_names else []
        if not pcx_names: raise SystemExit("The requested --only-file was not found in the noon folder.")

    for hhh in hours:
        hour_1_24 = hhh // 100
        out_dir = os.path.join(terrain_dir, f"{hhh}")
        os.makedirs(out_dir, exist_ok=True)
        for name in pcx_names:
            process_indexed_pcx(
                os.path.join(base_dir, name),
                os.path.join(out_dir, name),
                hour_1_24, night, blue, sunset
            )

def main():
    ap = argparse.ArgumentParser(description="Civ 3 day-night (SIMPLE)")
    ap.add_argument("--terrain", required=True, help="Path to Terrain folder (parent of noon subfolder)")
    ap.add_argument("--noon", default="1200", help="Name of noon subfolder (default: 1200)")
    ap.add_argument("--only-hour", type=int, help="Process a single hour (e.g., 2400)")
    ap.add_argument("--only-file", help="Process a single PCX filename from the noon folder")
    ap.add_argument("--night",  type=float, default=0.60, help="Midnight darkness strength (0..1). Default: 0.60")
    ap.add_argument("--blue",   type=float, default=0.85, help="Blue emphasis strength (0..1). Default: 0.85")
    ap.add_argument("--sunset", type=float, default=0.12, help="Evening warmth at ~19:00 (0..1). Default: 0.12")
    args = ap.parse_args()
    process_folder(args.terrain, args.noon, args.night, args.blue, args.sunset, args.only_hour, args.only_file)

if __name__ == "__main__":
    main()
