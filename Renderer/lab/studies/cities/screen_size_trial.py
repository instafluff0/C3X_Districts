#!/usr/bin/env python3
"""Audition larger intact AncientWood houses without changing canonical layouts."""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.build_layouts import ROOT, footprint, inside_wall
from Renderer.lab.studies.cities.test_biq_gallery import OUT, PACK, render
from Renderer.native.city_fidelity.prepare_pack import build_pack


TRIAL = ROOT / "Renderer/lab/out/cities/test-biq/screen-size-trial"
SOURCE = ROOT / "Renderer/lab/studies/cities/layouts.json"
FRAMES = ROOT / "Renderer/lab/out/cities/source-frames/all-city-source-frames.json"
POSITIONS = {
    0: [[-.25, .18], [0, .17], [.25, .18], [.30, -.08]],
    1: [[-.39, .27], [-.14, .27], [.11, .27], [.34, .27],
        [-.42, -.07], [.42, -.07]],
}


def bounds(instance: dict) -> list[float]:
    source = component(instance["asset"], ROOT / instance["pack"])
    return footprint({"low": source["lo"], "high": source["hi"]}, instance)


def make_layouts() -> dict:
    layouts = json.loads(SOURCE.read_text())
    design = next(item for item in layouts["designs"]
                  if (item["culture"], item["era"]) == (4, 0))
    for size, tier in enumerate(design["tier_designs"]):
        houses = tier["houses"]
        for slot in (0, 2):
            houses[slot]["scale"] = round(houses[slot]["scale"] * 1.5, 8)
        if size in POSITIONS:
            if len(houses) != len(POSITIONS[size]):
                raise ValueError(f"unexpected house count at size {size}")
            for house, position in zip(houses, POSITIONS[size]):
                house["offset"] = position
        boxes = [bounds(house) for house in houses]
        cores = [bounds(tier["base_centerpiece"]), bounds(tier["palace"])]
        if any(not inside_wall(box, size) for box in boxes):
            raise ValueError(f"house escapes size {size} wall")
        if any(overlaps(box, other) for index, box in enumerate(boxes)
               for other in boxes[:index] + cores):
            raise ValueError(f"house overlaps at size {size}")
    for slot in range(4):
        scales = [tier["houses"][slot]["scale"] for tier in design["tier_designs"]]
        if len(set(scales)) != 1:
            raise ValueError(f"house {slot} changes size across population tiers")
    return layouts


def replace_pack(data: bytes) -> None:
    temporary = PACK.with_name(PACK.name + ".screen-size-trial")
    temporary.write_bytes(data)
    os.replace(temporary, PACK)


def main() -> None:
    TRIAL.mkdir(parents=True, exist_ok=True)
    layout_path = TRIAL / "layouts.json"
    layout_path.write_text(json.dumps(make_layouts(), indent=2) + "\n")
    build_pack(TRIAL / "pack", layout_path, ("asian", "ancient"), FRAMES)
    original = PACK.read_bytes()
    candidate = (TRIAL / "pack/city.bin").read_bytes()
    cases = (("town", 0, (20, 64), False),
             ("town-walled", 0, (20, 64), True),
             ("city-walled", 1, (20, 64), True),
             ("metropolis-walled", 2, (20, 64), True),
             ("hill-town-walled", 0, (15, 63), True))
    records = [render(4, 0, 0, (20, 64), "size-trial-baseline-town-walled",
                      capital=True, walls=True)]
    try:
        replace_pack(candidate)
        for name, size, site, walls in cases:
            records.append(render(4, 0, size, site, "size-trial-" + name,
                                  capital=True, walls=walls))
    finally:
        replace_pack(original)
    (OUT / "screen-size-trial.json").write_text(json.dumps(records, indent=2) + "\n")
    comparison = Image.new("RGB", (1080, 270), (20, 16, 24))
    draw = ImageDraw.Draw(comparison)
    for index, label in enumerate(("baseline-town-walled", "town-walled",
                                   "hill-town-walled")):
        with Image.open(OUT / ("size-trial-" + label + ".png")) as rendered:
            crop = rendered.convert("RGB").crop((480, 305, 760, 485))
        x = index * 360
        comparison.paste(crop, (x + 40, 45))
        draw.text((x + 40, 16), label, fill="white")
    comparison.save(OUT / "screen-size-trial-comparison.png")
    cards = "".join(
        f'<article><h2>{case["name"]}</h2><img src="{case["image"]}"></article>'
        for case in records)
    (OUT / "screen-size-trial.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>City building size trial</title>'
        '<style>body{font:16px system-ui;background:#211b29;color:white;margin:20px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}'
        'article{background:#372b3e;padding:12px}img{width:100%;height:auto}'
        'h2{margin:0 0 8px}</style><h1>Unaccepted screen-size trial</h1><main>' +
        cards + '</main>')
    print("Wrote", OUT / "screen-size-trial.html")


if __name__ == "__main__":
    main()
