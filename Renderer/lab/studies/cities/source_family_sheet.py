#!/usr/bin/env python3
"""Show representative normalized source-city bodies before mapping cultures."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.studies.cities.sheet import (MAGENTA, font, prepared_asset,
                                                transformed)
from Renderer.preview.render_city_day_night_sheet import _draw_mesh
from Renderer.preview.render_iso import Canvas


def representatives(entries):
    blocks = [entry for entry in entries if "_Block_LG_SQ_" in entry["entry"]]
    squares = [entry for entry in entries if "_Block_SQ_" in entry["entry"]]
    single = [entry for entry in entries if "_Bld_" in entry["entry"]]
    chosen = blocks[:1] + squares[:1]
    if single:
        chosen += [single[0], single[len(single)//2], single[-1]]
    for entry in entries:
        if len(chosen) >= 5:
            break
        if entry not in chosen:
            chosen.append(entry)
    return chosen[:5]


def draw_body(asset_id, pack):
    width, height = 350, 230
    canvas = Canvas(width, height, MAGENTA)
    depth = [-math.inf] * (width * height)
    body = component(asset_id, pack)
    span = max(body["hi"][0]-body["lo"][0], body["hi"][1]-body["lo"][1])
    rise = body["hi"][2]-body["lo"][2]
    # Comparison cells zoom each body separately; these are source auditions,
    # not a claim that all bodies should have the same size on a map tile.
    scale = min(18.0, .85/max(span, .001), .45/max(rise, .001))
    instance = {"scale": scale, "rotation": 0.0, "offset": [0.0, 0.0],
                "vertical_metric": 1.0, "ground_z_offset": -body["lo"][2]}
    for mesh, base, emissive in prepared_asset(asset_id, pack.as_posix()):
        _draw_mesh(canvas, depth, transformed(mesh, instance), base, emissive,
                   (width//2, 155), 350, 0.0, False)
    image = Image.new("RGB", (width, height))
    image.putdata(canvas.pixels)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.source_report.read_text())
    pools = [pool for pool in report["pools"] if pool["selected"]]
    width, height = 350, 230
    sheet = Image.new("RGB", (width*len(pools), (height+48)*5+48), (33, 27, 42))
    draw = ImageDraw.Draw(sheet)
    for column, pool in enumerate(pools):
        title = pool["source_culture"].removeprefix("CIVILIZATION_")
        draw.text((column*width+8, 12), title, font=font(17), fill=(250, 232, 250))
        for row, entry in enumerate(representatives(pool["selected"])):
            y = 48+row*(height+48)
            image = draw_body(entry["asset_id"], args.pack)
            sheet.paste(image, (column*width, y))
            draw.text((column*width+8, y+height+5), entry["entry"].split("_", 3)[-1][:38],
                      font=font(14), fill=(237, 216, 237))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
