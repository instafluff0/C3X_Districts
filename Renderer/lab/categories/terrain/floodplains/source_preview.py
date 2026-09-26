#!/usr/bin/env python3
"""Lab-only study of normalized, source-shaped Civ VI floodplain decals.

The source triangles, UVs, color, height and gloss come from the local pack.
The study's scatter and projection onto saved Lab captures are inferred; this
script does not modify the production renderer, packs or fixed references.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import DdsBc3Texture, write_png
from Renderer.tools.asset_compiler.generic_decal_compiler import compile_decal_pack, default_assets_root


CATEGORY = Path(__file__).resolve().parent
OUT = ROOT / "Renderer/lab/out/floodplains"
PACK = OUT / "source-pack"
REFERENCES = ROOT / "Renderer/lab/references/floodplains/approved"


def read_bmp(path: Path) -> Canvas:
    data = path.read_bytes()
    offset = struct.unpack_from("<I", data, 10)[0]
    width, signed_height = struct.unpack_from("<ii", data, 18)
    bits = struct.unpack_from("<H", data, 28)[0]
    if data[:2] != b"BM" or width <= 0 or not signed_height or bits not in (24, 32):
        raise ValueError("Expected an uncompressed 24/32-bit BMP")
    if struct.unpack_from("<I", data, 30)[0]:
        raise ValueError("Compressed BMP is unsupported")
    height = abs(signed_height)
    pixel_size = bits // 8
    stride = (width * pixel_size + 3) & ~3
    if offset + stride * height > len(data):
        raise ValueError("Truncated BMP")
    canvas = Canvas(width, height)
    for y in range(height):
        source_y = y if signed_height < 0 else height - y - 1
        row = data[offset + source_y * stride:][:width * pixel_size]
        canvas.pixels[y * width:(y + 1) * width] = [
            (row[x + 2], row[x + 1], row[x])
            for x in range(0, len(row), pixel_size)
        ]
    return canvas


def bc4_pixel(block: bytes, x: int, y: int) -> float:
    first, second = block[:2]
    if first > second:
        palette = [first, second] + [round(((8 - i) * first + (i - 1) * second) / 7)
                                     for i in range(2, 8)]
    else:
        palette = [first, second] + [round(((6 - i) * first + (i - 1) * second) / 5)
                                     for i in range(2, 6)] + [0, 255]
    selection = int.from_bytes(block[2:8], "little") >> (3 * (4 * y + x)) & 7
    return palette[selection] / 255


class Material:
    def __init__(self, pack: Path, decal: dict):
        channels = decal["channels"]
        self.color = DdsBc3Texture.from_file(pack / channels["base_color"]["texture"])
        self.height = (pack / channels["height"]["texture"]).read_bytes()
        self.gloss = (pack / channels["specular"]["texture"]).read_bytes()
        self.width = self.color.width
        self.color_pixels = [
            self.color.sample_rgba((x + .5) / self.width, (y + .5) / self.width)
            for y in range(self.width) for x in range(self.width)
        ]
        self.pixel_cache = {}
        assert self.width == self.color.height
        assert struct.unpack_from("<I", self.height, 128)[0] == 83
        assert struct.unpack_from("<I", self.gloss, 128)[0] == 80

    def sample(self, u: float, v: float):
        sx = min(self.width - 1, max(0, u * self.width - .5))
        sy = min(self.width - 1, max(0, v * self.width - .5))
        x0, y0 = math.floor(sx), math.floor(sy)
        x1, y1 = min(self.width - 1, x0 + 1), min(self.width - 1, y0 + 1)
        tx, ty = sx - x0, sy - y0
        samples = (
            (self.color_pixels[y0 * self.width + x0], (1 - tx) * (1 - ty)),
            (self.color_pixels[y0 * self.width + x1], tx * (1 - ty)),
            (self.color_pixels[y1 * self.width + x0], (1 - tx) * ty),
            (self.color_pixels[y1 * self.width + x1], tx * ty),
        )
        alpha = sum(color[3] * weight for color, weight in samples)
        if alpha < .01:
            return (0, 0, 0), 0
        color = tuple(sum(sample[channel] * sample[3] * weight
                          for sample, weight in samples) / alpha
                      for channel in range(3))
        x = min(self.width - 1, max(0, round(sx)))
        y = min(self.width - 1, max(0, round(sy)))
        key = (x, y)
        if key not in self.pixel_cache:
            block_index = ((y // 4) * (self.width // 4) + x // 4)
            h = self.height[148 + block_index * 16:][:16]
            g = self.gloss[148 + block_index * 8:][:8]
            nx = bc4_pixel(h[:8], x % 4, y % 4) * 2 - 1
            ny = bc4_pixel(h[8:], x % 4, y % 4) * 2 - 1
            gloss = bc4_pixel(g, x % 4, y % 4)
            # The same packed RG normal interpretation used by the terrain
            # decal shader; source material channels stay in their own UVs.
            shade = max(.80, min(1.12, 1 + .11 * (nx * -.55 + ny * -.35)))
            specular = .025 * gloss
            self.pixel_cache[key] = (shade, specular)
        shade, specular = self.pixel_cache[key]
        return (tuple(min(255, round(c * shade + 255 * specular))
                      for c in color), round(alpha))


def ground_pixel(rgb: tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return r - b > 43 and g - b > 29 and r > 120 and g > 105


def raster_decal(canvas: Canvas, decal: dict, material: Material, center: tuple[float, float],
                 angle: float, scale: float, ground_mask: list[bool], opacity_scale: float):
    vertices = decal["mesh"]["vertices"]
    indices = decal["mesh"]["indices"]
    co, si = math.cos(angle), math.sin(angle)
    projected = []
    for vertex in vertices:
        xx, yy = vertex["position"]
        xx, yy = scale * (co * xx - si * yy), scale * (si * xx + co * yy)
        projected.append((center[0] + (xx - yy) * 28,
                          center[1] + (xx + yy) * 13,
                          *vertex["uv0"]))
    for i in range(0, len(indices), 3):
        a, b, c = (projected[indices[i + k]] for k in range(3))
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(area) < .001:
            continue
        x0 = max(0, math.floor(min(a[0], b[0], c[0])))
        x1 = min(canvas.width - 1, math.ceil(max(a[0], b[0], c[0])))
        y0 = max(0, math.floor(min(a[1], b[1], c[1])))
        y1 = min(canvas.height - 1, math.ceil(max(a[1], b[1], c[1])))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                px, py = x + .5, y + .5
                w1 = ((px - a[0]) * (c[1] - a[1]) - (py - a[1]) * (c[0] - a[0])) / area
                w2 = ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) / area
                w0 = 1 - w1 - w2
                if min(w0, w1, w2) < 0:
                    continue
                index = y * canvas.width + x
                old = canvas.pixels[index]
                if not ground_mask[index]:
                    continue
                u = w0 * a[2] + w1 * b[2] + w2 * c[2]
                v = w0 * a[3] + w1 * b[3] + w2 * c[3]
                color, opacity = material.sample(u, v)
                alpha = opacity / 255 * opacity_scale
                if alpha > .005:
                    canvas.pixels[index] = tuple(round(old[channel] * (1 - alpha) +
                                                       color[channel] * alpha)
                                                 for channel in range(3))


def make_scene(base: Canvas, decals: list[dict], material: Material, spacing: int,
               strength: float, opacity_scale: float) -> Canvas:
    rng = random.Random(0xF100D)
    ground_mask = [
        ground_pixel(base.pixels[y * base.width + x]) and
        abs(y - (.52 * x + 45)) > 14
        for y in range(base.height) for x in range(base.width)
    ]
    for row in range(-1, base.height // spacing + 2):
        for col in range(-1, base.width // spacing + 2):
            if rng.random() > strength:
                continue
            variant = rng.randrange(4)
            cx = (col + .5 * (row % 2)) * spacing + rng.uniform(-.22, .22) * spacing
            cy = row * spacing + rng.uniform(-.22, .22) * spacing
            raster_decal(base, decals[variant], material, (cx, cy),
                         rng.random() * math.tau, 1.15 * (1 + rng.uniform(-.15, .15)),
                         ground_mask, opacity_scale)
    return base


def main():
    mapping = CATEGORY / "source_decals.json"
    compile_decal_pack(default_assets_root(), mapping, PACK, OUT / "source-build.json")
    manifest = json.loads((PACK / "manifest.json").read_text())
    groups = {}
    for family in ("plains", "grassland"):
        group = manifest["decal_groups"][f"terrain/floodplain/{family}_surface"]
        decals = [json.loads((PACK / manifest["assets"][key]["decal"]).read_text())
                  for key in group["variants"]]
        groups[family] = (decals, Material(PACK, decals[0]))
    target = OUT / "source-preview"
    target.mkdir(parents=True, exist_ok=True)
    outputs = []
    references = {}
    for family, (decals, material) in groups.items():
        closeup = Canvas(720, 520, (185, 169, 112))
        for index, decal in enumerate(decals):
            center = (180 + (index % 2) * 360, 130 + (index // 2) * 260)
            raster_decal(closeup, decal, material, center, 0, 2.3,
                         [True] * (closeup.width * closeup.height), 1)
        output = target / f"source-{family}-variants.png"
        write_png(closeup, output)
        outputs.append(str(output.relative_to(ROOT)))
    for case in ("detail", "gameplay"):
        reference = REFERENCES / case / f"{case}-h12-z128.bmp"
        references[case] = hashlib.sha256(reference.read_bytes()).hexdigest()
        control = target / f"{case}-control.png"
        write_png(read_bmp(reference), control)
        outputs.append(str(control.relative_to(ROOT)))
        for family, (decals, material) in groups.items():
            for label, spacing, strength, opacity in (("sparse", 35, .72, .42),
                                                      ("full", 26, .9, .48)):
                canvas = make_scene(read_bmp(reference), decals, material, spacing,
                                    strength, opacity)
                output = target / f"{case}-{family}-{label}.png"
                write_png(canvas, output)
                outputs.append(str(output.relative_to(ROOT)))
                if case == "detail" and label == "full":
                    before = read_bmp(reference)
                    difference = Canvas(canvas.width, canvas.height, (20, 20, 20))
                    difference.pixels = [
                        tuple(min(255, 4 * abs(after[channel] - prior[channel]))
                              for channel in range(3))
                        for prior, after in zip(before.pixels, canvas.pixels)
                    ]
                    diff_output = target / f"detail-{family}-difference.png"
                    write_png(difference, diff_output)
                    outputs.append(str(diff_output.relative_to(ROOT)))
    active = {ROOT / output for output in outputs}
    for stale in target.glob("*.png"):
        if stale not in active:
            stale.unlink()
    (target / "manifest.json").write_text(json.dumps({
        "method": "source-triangle/UV/material lab study over fixed synthetic D3D11 captures",
        "limitation": "inferred orthographic decal placement; not the production terrain shader",
        "mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(),
        "pack_sha256": hashlib.sha256((PACK / "manifest.json").read_bytes()).hexdigest(),
        "reference_sha256": references,
        "outputs": outputs,
    }, indent=2) + "\n")
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
