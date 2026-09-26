#!/usr/bin/env python3
"""Contact sheet of every usable component in a focused local city pool."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.studies.cities.sheet import font, render_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    selected = report["pools"][0]["selected"]
    width, height, caption, columns = 420, 310, 55, 5
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + caption)), (37, 34, 43))
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(selected):
        body = component(item["asset_id"], args.pack)
        span = [body["hi"][j] - body["lo"][j] for j in range(3)]
        scale = min(.62 / max(span[:2]), .35 / span[2])
        instance = {"asset": item["asset_id"], "pack": str(args.pack),
                    "scale": scale, "rotation": 0.0, "offset": [0.0, 0.0]}
        design = {"culture_name": "Source", "era_name": "Audition",
                  "tier_designs": [{"houses": [instance], "base_centerpiece": instance,
                                    "palace": instance}], "capital_replaces_centerpiece": True}
        image, _ = render_cell(design, 0, False, False, (width, height), 512)
        x = (index % columns) * width
        y = (index // columns) * (height + caption)
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + height + 5), f"{index:02}  {item['entry']}",
                  font=font(15), fill=(244, 235, 240))
        draw.text((x + 8, y + height + 29),
                  "source span " + " × ".join(f"{v:.3f}" for v in span),
                  font=font(13), fill=(190, 180, 195))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
