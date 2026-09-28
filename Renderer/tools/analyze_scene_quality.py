"""Mac analysis of explicit standalone HDR captures; never an in-game render path."""
from pathlib import Path
import argparse
import json
import gzip
import struct

import numpy as np
from PIL import Image, ImageDraw


def read_linear(path):
    opener = path.open
    if not path.is_file():
        path = path.with_suffix(path.suffix + '.gz')
        opener = lambda mode: gzip.open(path, mode)
    with opener('rb') as f:
        magic, count = struct.unpack('<II', f.read(8))
        if magic not in (0x31485243, 0x32485243) or count != 3:
            raise ValueError('Invalid HDR capture')
        exposure, gain = struct.unpack('<ff', f.read(8))
        filmic, _ = struct.unpack('<ff', f.read(8)) if magic == 0x32485243 else (0, 0)
        surfaces = []
        for _ in range(count):
            w, h = struct.unpack('<II', f.read(8))
            if not 1 <= w <= 8192 or not 1 <= h <= 8192:
                raise ValueError('Invalid capture extent')
            raw = f.read(w * h * 8)
            surfaces.append(np.frombuffer(raw, '<f2').reshape(h, w, 4).astype(np.float32))
        if f.read(1):
            raise ValueError('Unexpected trailing data')
    if not all(np.isfinite(a).all() for a in surfaces):
        raise ValueError('Non-finite scene radiance')
    return surfaces, exposure, gain, filmic


def bilinear(image, x, y):
    h, w = image.shape[:2]
    x = np.clip(x, 0, w - 1); y = np.clip(y, 0, h - 1)
    ix = x.astype(int); iy = y.astype(int)
    fx = (x - ix)[None, :, None]; fy = (y - iy)[:, None, None]
    nx = np.minimum(ix + 1, w - 1); ny = np.minimum(iy + 1, h - 1)
    return ((image[iy[:, None], ix] * (1 - fx) + image[iy[:, None], nx] * fx) * (1 - fy)
            + (image[ny[:, None], ix] * (1 - fx) + image[ny[:, None], nx] * fx) * fy)


def radiance(surfaces, exposure, gain):
    base, moving, bloom = surfaces
    c = moving + base * (1 - moving[:, :, 3:4])
    h, w = c.shape[:2]
    c = c[4:-4, 4:-4]
    glow = bilinear(bloom, (np.arange(w - 8) + 5) * bloom.shape[1] / w - .5,
                    (np.arange(h - 8) + 5) * bloom.shape[0] / h - .5)
    return np.maximum(0, (c[:, :, :3] / np.maximum(c[:, :, 3:4], 1e-6)
                         + glow[:, :, :3] * gain) * exposure)


def srgb(rgb):
    return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0)**(1 / 2.4) - .055)


def display(rgb, curve='baseline', exposure=1, amount=.5):
    x = rgb * exposure
    if curve == 'baseline':
        x = x / (1 + np.max(x, axis=2, keepdims=True))
    elif curve == 'filmic':
        # Narkowicz's published ACES-like fit, not a full ACES transform:
        # https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
        x = np.clip((x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14), 0, 1)
    elif curve == 'mixed':
        neutral = x / (1 + np.max(x, axis=2, keepdims=True))
        f = np.minimum(x * .65, 64)
        f = np.clip((f * (2.51 * f + .03)) / (f * (2.43 * f + .59) + .14), 0, 1)
        x = neutral + (f - neutral) * amount
    else:
        raise ValueError(curve)
    return np.clip(srgb(x), 0, 1)


# Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

def cas(rgb, amount=.35):
    # Numeric mirror of native/scene_detail_filter.h; that file retains AMD's
    # complete MIT notice. All input in this diagnostic is opaque scene color.
    p = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode='edge')
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    lo = np.minimum.reduce([d, rgb, f, b, h]); hi = np.maximum.reduce([d, rgb, f, b, h])
    lo += np.minimum.reduce([lo, a, c, g, i]); hi += np.maximum.reduce([hi, a, c, g, i])
    weight = -np.sqrt(np.clip(np.minimum(lo, 2 - hi) / np.maximum(hi, 1e-6), 0, 1)) / 8
    sharpened = np.clip(((b + d + f + h) * weight + rgb) / (1 + 4 * weight), 0, 1)
    return rgb + (sharpened - rgb) * amount


def save(path, rgb):
    Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', type=Path)
    args = p.parse_args()
    surfaces, exposure, gain, filmic = read_linear(args.case / 'scene.crh')
    hdr = radiance(surfaces, exposure, gain)
    control = display(hdr, 'mixed', amount=filmic)
    gpu = np.asarray(Image.open(args.case / 'frame.png').convert('RGB'), dtype=np.float32) / 255
    error = np.abs(np.rint(control * 255) - np.rint(gpu * 255))
    results = {'capture_exposure': exposure, 'bloom_gain': gain, 'filmic_amount': filmic,
               'radiance_percentiles': np.percentile(hdr, [0, 5, 50, 95, 99, 100]).tolist(),
               'baseline_mean_error_255': float(error.mean()), 'baseline_max_error_255': float(error.max())}
    if error.max() > 2:
        raise RuntimeError('CPU study does not match GPU display: ' + str(results))
    variants = [('baseline-cas', cas(display(hdr))),
                ('filmic-065', cas(display(hdr, 'filmic', .65))),
                ('filmic-080', cas(display(hdr, 'filmic', .8))),
                ('mixed-050', cas(display(hdr, 'mixed', amount=.5)))]
    sheet = Image.new('RGB', (960, 700), '#181818')
    draw = ImageDraw.Draw(sheet)
    for n, (name, rgb) in enumerate(variants):
        save(args.case / (name + '.png'), rgb)
        # Native-pixel scene crop, not a resized or sharpened-up enlargement.
        crop = Image.open(args.case / (name + '.png')).crop((400, 180, 880, 510))
        x, y = (n % 2) * 480, (n // 2) * 350
        sheet.paste(crop, (x, y + 20)); draw.text((x + 8, y + 3), name, fill='white')
    sheet.save(args.case / 'display-comparison.png')
    (args.case / 'analysis.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
