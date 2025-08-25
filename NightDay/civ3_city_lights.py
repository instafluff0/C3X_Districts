#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Civ 3 city lights glow compositor for *_lights.pcx

- Detects a placeholder color marking windows/torches and replaces it with a
  warm yellow-orange light + soft halo whose intensity varies by time of day:
    * Visible from 19:00 to 06:00 (inclusive), peak at 24:00 (midnight).
    * Outside that window, fills the placeholder with shadowy brown/gray
      sampled from surroundings (or fallback color).

- Operates only on files whose name contains "_lights" (case-insensitive).
- Preserves #ff00ff and #00ff00 exactly in the final PCX.
- Works alongside your day-night color grading:
    * If an hour folder already has the file (e.g., after grading), this script
      composites onto that version using the mask from the noon file.
    * Otherwise, it uses the noon file as the backdrop.

CLI knobs:
  --light-key R,G,B         Placeholder color to detect (default 255,0,0)
  --core-radius, --halo-radius
  --core-gain, --halo-gain  Strength of inner and outer glow (0..1)
  --core-color, --glow-color
  --shadow auto|#rrggbb     Auto = sample ring around windows; else fixed color
"""

import argparse, os, math, re
from typing import List, Tuple, Optional
from PIL import Image, ImageChops, ImageFilter

MAGENTA = (255, 0, 255)   # ff00ff
GREEN   = (0, 255, 0)     # 00ff00

# ---------- small utils ----------

def parse_rgb(s: str) -> Tuple[int,int,int]:
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
        if len(s) != 6: raise ValueError("Hex color must be #rrggbb")
        return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
    if "," in s:
        parts = [int(p) for p in s.split(",")]
        if len(parts)!=3: raise ValueError("RGB must be R,G,B")
        return tuple(max(0,min(255,v)) for v in parts)  # type: ignore
    raise ValueError("Color must be '#rrggbb' or 'R,G,B'")

def hour_weight_lights(hour_1_24: int, start: float=19.0, end: float=6.0, peak: float=24.0) -> float:
    """
    Cosine window centered at midnight (24/0), visible from ~19:00 to ~06:00.
    Returns 0..1. Slightly inclusive at the edges.
    """
    # cyclic distance to peak (midnight)
    h = float(hour_1_24) % 24.0
    d = min((h - peak) % 24.0, (peak - h) % 24.0)  # 0..12
    # half-window length (choose ~6.05 to give a faint 06:00 glow)
    half = 6.05
    if d > half: return 0.0
    t = d / half  # 0..1
    return 0.5 * (1.0 + math.cos(math.pi * t))

def get_palette(image: Image.Image) -> List[int]:
    pal = image.getpalette() or []
    if len(pal) < 256*3: pal += [0]*(256*3 - len(pal))
    return pal[:256*3]

def put_palette(image: Image.Image, palette: List[int]) -> None:
    if len(palette) < 256*3: palette += [0]*(256*3 - len(palette))
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

def color_equal(img_rgb: Image.Image, color: Tuple[int,int,int]) -> Image.Image:
    """Return an 'L' mask: 255 where pixel == color else 0, no tolerance."""
    w,h = img_rgb.size
    solid = Image.new('RGB', (w,h), color)
    diff = ImageChops.difference(img_rgb, solid)     # 0 where equal
    # Sum channels and threshold: 0 => equal, else nonzero
    diffL = diff.convert('L')
    # map 0 -> 255, others -> 0
    mask = diffL.point(lambda v: 255 if v == 0 else 0, mode='L')
    return mask

def scale_L(imgL: Image.Image, factor: float) -> Image.Image:
    factor = max(0.0, min(10.0, factor))
    return imgL.point(lambda v: int(round(max(0, min(255, v*factor)))))

def subtract_L(a: Image.Image, b: Image.Image) -> Image.Image:
    # a - b, clipped to [0,255]
    return ImageChops.subtract(a, b, scale=1.0, offset=0)

def any_true(maskL: Image.Image) -> bool:
    # fastest simple check
    bbox = maskL.getbbox()
    return bbox is not None

def average_color_in_ring(base_rgb: Image.Image, ring_mask: Image.Image) -> Optional[Tuple[int,int,int]]:
    # Compute average color over pixels where ring_mask==255
    base_data = base_rgb.getdata()
    ring_data = ring_mask.getdata()
    total = [0,0,0]
    count = 0
    for (r,g,b), m in zip(base_data, ring_data):
        if m == 255:
            total[0]+=r; total[1]+=g; total[2]+=b
            count += 1
    if count == 0:
        return None
    return (total[0]//count, total[1]//count, total[2]//count)

# ---------- glow compositor per image ----------

def process_lights_for_hour(
    src_noon_path: str,
    base_hour_path: Optional[str],
    out_path: str,
    hour_1_24: int,
    light_key: Tuple[int,int,int],
    core_radius: float,
    halo_radius: float,
    core_gain: float,
    halo_gain: float,
    core_color: Tuple[int,int,int],
    glow_color: Tuple[int,int,int],
    shadow_mode: str,
    shadow_color_const: Tuple[int,int,int]
) -> None:

    # Load source (noon) with mask and the hour backdrop (if any)
    im_noon = Image.open(src_noon_path).convert('RGBA')
    src_rgb = im_noon.convert('RGB')  # for magic color mapping

    if base_hour_path and os.path.exists(base_hour_path):
        base = Image.open(base_hour_path).convert('RGBA')
    else:
        base = im_noon.copy()

    w, h = base.size

    # Build binary mask from placeholder color
    mask_bin = color_equal(src_rgb, light_key)  # L 0/255
    if not any_true(mask_bin):
        # Nothing to do; just save base out as PCX while preserving magic colors
        save_with_palette_and_magic(base, out_path, src_rgb)
        return

    # Build blurred masks for glow
    blur_core = mask_bin.filter(ImageFilter.GaussianBlur(radius=core_radius))
    # Dilate to create a ring for sampling & to separate halo from core
    dilated = mask_bin.filter(ImageFilter.GaussianBlur(radius=1.0)).point(lambda v: 255 if v>0 else 0, mode='L')
    ring_mask = subtract_L(dilated, mask_bin)
    blur_halo_full = mask_bin.filter(ImageFilter.GaussianBlur(radius=halo_radius))
    halo_only = subtract_L(blur_halo_full, blur_core)  # avoids double in core

    # Time-of-day weight
    wtime = hour_weight_lights(hour_1_24)  # 0..1

    # Compose:
    comp = base.copy()

    if wtime > 0.0:
        # 1) Fill window cores toward a warm color (strong alpha inside shapes)
        core_alpha = scale_L(mask_bin, wtime * max(0.0, min(1.0, 0.95)))  # fixed fill strength
        core_solid = Image.new('RGBA', (w,h), (*core_color, 255))
        comp = Image.composite(core_solid, comp, core_alpha)  # lerp toward core color

        # 2) Add soft halo using blurred mask(s)
        # Scale core and halo alphas by gains and time
        core_glow_alpha = scale_L(blur_core, wtime * max(0.0, min(1.0, core_gain)))
        halo_glow_alpha = scale_L(halo_only, wtime * max(0.0, min(1.0, halo_gain)))
        # Combine, saturating
        glow_alpha = ImageChops.add(core_glow_alpha, halo_glow_alpha, scale=1.0, offset=0)

        glow_solid = Image.new('RGBA', (w,h), (*glow_color, 255))
        comp = Image.composite(glow_solid, comp, glow_alpha)

    else:
        # Lights "off": fill placeholder with shadow
        base_rgb = comp.convert('RGB')
        if shadow_mode == "auto":
            sampled = average_color_in_ring(base_rgb, ring_mask)
            if sampled is None:
                sampled = shadow_color_const
            # slightly darken sampled for shadow feel
            sr, sg, sb = sampled
            sampled = (int(sr*0.85), int(sg*0.83), int(sb*0.80))
            fill_color = sampled
        else:
            fill_color = shadow_color_const

        shadow_solid = Image.new('RGBA', (w,h), (*fill_color, 255))
        comp = Image.composite(shadow_solid, comp, mask_bin)

    # Preserve MAGENTA and GREEN exactly
    save_with_palette_and_magic(comp, out_path, src_rgb)

def save_with_palette_and_magic(comp_rgba: Image.Image, out_path: str, src_rgb_for_magic: Image.Image) -> None:
    # Quantize to P with 256 colors
    try:
        dnone = Image.Dither.NONE
    except Exception:
        dnone = getattr(Image, "NONE", 0)
    imP = comp_rgba.quantize(colors=256, method=Image.MEDIANCUT, dither=dnone)
    palette = get_palette(imP)

    # Ensure magic colors exist
    pal_after = ensure_palette_has_colors(imP, [MAGENTA, GREEN])
    put_palette(imP, pal_after)
    idx_mag = find_color_index(pal_after, MAGENTA)
    idx_grn = find_color_index(pal_after, GREEN)

    # Force pixels that were magic in the original to remain magic
    w,h = imP.size
    dst_px = imP.load()
    src_px = src_rgb_for_magic.load()
    for y in range(h):
        for x in range(w):
            r,g,b = src_px[x,y]
            if (r,g,b) == MAGENTA:
                dst_px[x,y] = idx_mag
            elif (r,g,b) == GREEN:
                dst_px[x,y] = idx_grn

    imP.save(out_path, format='PCX')

# ---------- folder driver ----------

def process_folder(terrain_dir: str, noon_subfolder: str,
                   light_key: Tuple[int,int,int],
                   core_radius: float, halo_radius: float,
                   core_gain: float, halo_gain: float,
                   core_color: Tuple[int,int,int],
                   glow_color: Tuple[int,int,int],
                   shadow_mode: str, shadow_color: Tuple[int,int,int],
                   only_hour: int = None, only_file: str = None) -> None:

    base_dir = os.path.join(terrain_dir, noon_subfolder)
    if not os.path.isdir(base_dir): raise SystemExit(f"No such folder: {base_dir}")

    def is_lights(name: str) -> bool:
        return name.lower().endswith('.pcx') and ('_lights' in name.lower())

    names = [f for f in os.listdir(base_dir) if is_lights(f)]
    if not names:
        raise SystemExit(f"No *_lights.pcx files found in {base_dir}")

    hours = [h for h in range(100, 2500, 100) if h != 1200]
    if only_hour is not None:
        if only_hour == 1200: raise SystemExit("1200 is the base (left untouched). Choose another hour.")
        if only_hour % 100 != 0 or only_hour < 100 or only_hour > 2400:
            raise SystemExit("--only-hour must be one of 100, 200, ..., 2400.")
        hours = [only_hour]

    if only_file is not None:
        names = [only_file] if only_file in names else []
        if not names: raise SystemExit("The requested --only-file was not found in the noon folder (or does not match *_lights.pcx).")

    for hhh in hours:
        hour_1_24 = hhh // 100
        out_dir = os.path.join(terrain_dir, f"{hhh}")
        os.makedirs(out_dir, exist_ok=True)
        for name in names:
            src_noon_path = os.path.join(base_dir, name)
            base_hour_path = os.path.join(out_dir, name)  # if exists, composite on it
            out_path = os.path.join(out_dir, name)

            process_lights_for_hour(
                src_noon_path,
                base_hour_path if os.path.exists(base_hour_path) else None,
                out_path,
                hour_1_24,
                light_key,
                core_radius, halo_radius,
                core_gain, halo_gain,
                core_color, glow_color,
                shadow_mode, shadow_color
            )

def main():
    ap = argparse.ArgumentParser(description="Civ 3 city lights glow compositor for *_lights.pcx")
    ap.add_argument("--terrain", required=True, help="Path to Terrain folder (parent of noon subfolder)")
    ap.add_argument("--noon", default="1200", help="Name of noon subfolder (default: 1200)")
    ap.add_argument("--only-hour", type=int, help="Process a single hour (e.g., 2400)")
    ap.add_argument("--only-file", help="Process a single *_lights.pcx filename from the noon folder")

    ap.add_argument("--light-key", type=str, default="255,0,0", help="Placeholder RGB (R,G,B or #rrggbb). Default: 255,0,0")
    ap.add_argument("--core-radius", type=float, default=1.6, help="Gaussian blur radius (px) for inner glow. Default: 1.6")
    ap.add_argument("--halo-radius", type=float, default=6.0, help="Gaussian blur radius (px) for soft halo. Default: 6.0")
    ap.add_argument("--core-gain", type=float, default=0.95, help="Inner glow strength (0..1). Default: 0.95")
    ap.add_argument("--halo-gain", type=float, default=0.55, help="Halo strength (0..1). Default: 0.55")
    ap.add_argument("--core-color", type=str, default="#ffd27a", help="Core window color (lit). Default: #ffd27a")
    ap.add_argument("--glow-color", type=str, default="#ffae4a", help="Glow tint color. Default: #ffae4a")
    ap.add_argument("--shadow", type=str, default="auto", help="Shadow fill when lights off: 'auto' to sample surrounding ring, or a color (#rrggbb). Default: auto")
    ap.add_argument("--shadow-fallback", type=str, default="#5a4e44", help="Fallback shadow color if sampling fails. Default: #5a4e44")

    args = ap.parse_args()

    # parse colors
    light_key = parse_rgb(args.light_key)
    core_color = parse_rgb(args.core_color)
    glow_color = parse_rgb(args.glow_color)

    shadow_mode = "auto"
    shadow_const = parse_rgb(args.shadow_fallback)
    if args.shadow.lower() != "auto":
        shadow_mode = "const"
        shadow_const = parse_rgb(args.shadow)

    process_folder(
        terrain_dir=args.terrain, noon_subfolder=args.noon,
        light_key=light_key,
        core_radius=args.core_radius, halo_radius=args.halo_radius,
        core_gain=args.core_gain, halo_gain=args.halo_gain,
        core_color=core_color, glow_color=glow_color,
        shadow_mode=shadow_mode, shadow_color=shadow_const,
        only_hour=args.only_hour, only_file=args.only_file
    )

if __name__ == "__main__":
    main()
