#!/usr/bin/env python3
"""Contact sheet of every imported palace, for choosing late-era capitals.

The industrial and modern capitals share one generic palace; this sheet shows
every normalized palace at one footprint so a replacement (or one per Civ III
culture) can be picked by eye. Mac-only preview; no runtime data changes.

    python3 Renderer/lab/studies/city_readability/palace_audition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.studies.cities.sheet import font, render_cell

PACK = Path("Renderer/packs/CityPalacesNormalized")
OUT = ROOT / "Renderer/lab/out/city-study/palace-audition.png"
CURRENT = "c9fb1862f0f18efe"  # the generic palace every late-era capital uses


def label(entry: dict) -> str:
    names = []
    for selector in entry["source_selectors"]:
        name = selector["culture"].replace("CIVILIZATION_", "").replace("_STK", "").title()
        if selector["era"] != "DEFAULT":
            name += " " + selector["era"].replace("ARTERA_", "").title()
        names.append(name)
    return " / ".join(dict.fromkeys(names))


def main():
    catalog = json.loads((ROOT / PACK / "palace_catalog.json").read_text())["palaces"]
    width, height, caption, columns = 300, 230, 26, 6
    rows = (len(catalog) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + caption)), (37, 34, 43))
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(catalog):
        asset = entry["asset_id"]
        body = component(asset, PACK)
        span = [body["hi"][j] - body["lo"][j] for j in range(3)]
        scale = min(.62 / max(span[:2]), .4 / span[2])
        instance = {"asset": asset, "pack": str(PACK), "scale": scale, "rotation": 0.0, "offset": [0.0, 0.0]}
        design = {"culture_name": "Source", "era_name": "Audition",
                  "tier_designs": [{"houses": [], "base_centerpiece": instance, "palace": instance}],
                  "capital_replaces_centerpiece": True}
        image, _ = render_cell(design, 0, False, True, (width, height), 384)
        x, y = (index % columns) * width, (index // columns) * (height + caption)
        sheet.paste(image, (x, y))
        tag = asset.split("/")[-1]
        current = " (current)" if tag == CURRENT else ""
        draw.text((x + 6, y + height + 4), f"{index:02} {label(entry)}{current}", font=font(14),
                  fill=(255, 210, 120) if current else (244, 235, 240))
        print(index, tag, label(entry), flush=True)
    # The review renderer keys on magenta; show a neutral olive ground instead.
    import numpy as np
    pixels = np.asarray(sheet).astype(np.float64)
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    key = np.clip((np.minimum(r, b) - g - 60) / 120, 0, 1)[..., None]
    ground = np.array([88, 92, 70], float)
    Image.fromarray((pixels * (1 - key) + ground * key).astype(np.uint8)).save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
