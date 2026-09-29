#!/usr/bin/env python3
"""Render diagnostic views of the four installed Civ VI volcano height fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.tools.asset_compiler import terrain_relief_builder as relief
from Renderer.lab.studies.volcanoes.inventory import DEFAULT_ASSETS


SOURCES = (
    ("Ordinary volcano", "Shared by 143 place names", "DLC/Expansion2",
     "ART_DEF_TERRAIN_ELEMENT_FEATURE_VOLCANO_01"),
    ("Vesuvius", "Natural wonder · reserved", "DLC/Expansion2",
     "ART_DEF_TERRAIN_ELEMENT_NWON_VESUVIUS"),
    ("Kilimanjaro", "Natural wonder · reserved", "Base",
     "ART_DEF_TERRAIN_ELEMENT_NWON_KILIMANJARO"),
    ("Eyjafjallajökull", "Natural wonder · reserved", "DLC/VikingsLandmarks",
     "ART_DEF_TERRAIN_ELEMENT_NWON_EYJAFJALLAJOKULL"),
)


def source_field(assets: Path, package: Path, key: str):
    import numpy as np
    _, elements, report = relief.inspect_terrain_element_package(package)
    entry = elements[key]
    channel = next(item for item in entry["channels"]["height"] if item["level"] == 0)
    with package.open("rb") as stream:
        stream.seek(report["big_data_offset"] + channel["relative_offset"])
        raw = stream.read(channel["bytes"])
    if len(raw) != channel["bytes"]:
        raise ValueError(f"Incomplete source height field: {key}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(channel["height"], channel["width"]), {
        "source_package": package.relative_to(assets).as_posix(),
        "source_entry": key,
        "source_resource": channel["name"],
        "height_sha256": hashlib.sha256(raw).hexdigest(),
        "dimensions": [channel["width"], channel["height"]],
        "height_scale": entry["parameters"]["height_scale"],
    }


def panel(height, title: str, subtitle: str):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    scale = 2
    width, tall = 570, 390
    canvas = Image.new("RGB", (width * scale, tall * scale), "#212831")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial Unicode.ttf", 20 * scale)
        small = ImageFont.truetype("Arial Unicode.ttf", 13 * scale)
    except OSError:
        font = ImageFont.load_default(size=20 * scale)
        small = ImageFont.load_default(size=13 * scale)
    draw.text((24 * scale, 17 * scale), title, fill="#f4f1e8", font=font)
    draw.text((24 * scale, 50 * scale), subtitle, fill="#bdc8ca", font=small)

    resolution = 96
    low = Image.fromarray(height, "L").resize((resolution, resolution), Image.Resampling.BILINEAR)
    h = np.asarray(low, dtype=np.float32) / 255.0
    h = np.maximum(0.0, (h - 0.035) / 0.965)
    # Oblique orthographic view of the source height channel only. Terrain
    # material, neighbor geometry, lava, smoke and Firaxis shaders are absent.
    gx, gy = np.gradient(h)
    cx, base, extent_x, extent_y, rise = 285 * scale, 252 * scale, 193 * scale, 85 * scale, 188 * scale

    def point(i: int, j: int):
        u = (i / (resolution - 1) - .5) * 2
        v = (j / (resolution - 1) - .5) * 2
        return (cx + (u - v) * extent_x,
                base + (u + v) * extent_y - float(h[j, i]) * rise)

    for diagonal in range(2 * resolution - 3):
        for j in range(max(0, diagonal - (resolution - 2)),
                       min(resolution - 2, diagonal) + 1):
            i = diagonal - j
            z = float(h[j, i])
            light = max(.28, min(1.0, .72 - float(gx[j, i]) * 2.0
                                  + float(gy[j, i]) * 2.4))
            rock = (92 + 65 * z, 96 + 44 * z, 82 + 49 * z)
            grass = (79, 105, 66)
            blend = min(1.0, z * 6.0)
            color = tuple(int(max(0, min(255, (grass[k] * (1-blend) +
                                               rock[k] * blend) * light))) for k in range(3))
            draw.polygon((point(i, j), point(i+1, j), point(i+1, j+1), point(i, j+1)),
                         fill=color)
    draw.text((24 * scale, 360 * scale), "Civ VI source height field · diagnostic shading",
              fill="#a7b5b9", font=small)
    return canvas.resize((width, tall), Image.Resampling.LANCZOS)


def main() -> None:
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "Renderer/lab/out/volcanoes/source-gallery")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gallery = Image.new("RGB", (1140, 780), "#212831")
    records = {}
    for index, (title, subtitle, origin, key) in enumerate(SOURCES):
        package = (args.assets / origin / "Platforms/Windows/BLPs/terrain/"
                   "TerrainElementSet_Base.blp")
        field, record = source_field(args.assets, package, key)
        records[key] = record
        gallery.paste(panel(field, title, subtitle), ((index % 2) * 570, (index // 2) * 390))
    image = args.out / "source-heightfields.png"
    gallery.save(image, optimize=True)
    (args.out / "source-heightfields.json").write_text(json.dumps(records, indent=2) + "\n")
    print(image)


if __name__ == "__main__":
    main()
