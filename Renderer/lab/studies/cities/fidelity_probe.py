#!/usr/bin/env python3
"""Trace city image quality from one source body to a complete hill scene."""

from __future__ import annotations

import hashlib
import json
import os

from PIL import Image, ImageDraw

from Renderer.lab.studies.cities.build_layouts import ROOT
from Renderer.lab.studies.cities.isolate_pack import isolate
from Renderer.lab.studies.cities.test_biq_gallery import OUT, PACK, render


SOURCE = ROOT / "Renderer/lab/out/cities/test-biq/all-designs-pack/city.bin"


def replace_pack(data: bytes) -> None:
    temporary = PACK.with_name(PACK.name + ".fidelity-probe")
    temporary.write_bytes(data)
    os.replace(temporary, PACK)


def main() -> None:
    original = SOURCE.read_bytes()
    house = isolate(original, 4, 0, 0, False)
    palace = isolate(original, 4, 0, 0, True)
    medium_house = isolate(original, 4, 0, 0, False, 1.5)
    large_house = isolate(original, 4, 0, 0, False, 2.0)
    cases = [
        ("one-house", house, (20, 64), False, False),
        ("one-house-1.5x", medium_house, (20, 64), False, False),
        ("one-house-2x", large_house, (20, 64), False, False),
        ("palace-only", palace, (20, 64), True, False),
        ("whole-town", original, (20, 64), True, False),
        ("whole-town-walled", original, (20, 64), True, True),
        ("hill-town-walled", original, (15, 63), True, True),
    ]
    records = []
    try:
        for label, pack, site, capital, walls in cases:
            replace_pack(pack)
            result = render(4, 0, 0, site, "fidelity-" + label,
                            capital=capital, walls=walls)
            result["stage"] = label
            result["isolated_pack_sha256"] = hashlib.sha256(pack).hexdigest()
            records.append(result)
    finally:
        replace_pack(original)
    manifest = {"schema": "c3x.lab.city_fidelity_probe.v1",
                "source_pack_sha256": hashlib.sha256(original).hexdigest(),
                "cases": records}
    (OUT / "fidelity-probe.json").write_text(json.dumps(manifest, indent=2) + "\n")
    comparison = Image.new("RGB", (1080, 540), (20, 16, 24))
    labels = ("one-house", "one-house-1.5x", "one-house-2x",
              "palace-only", "whole-town", "whole-town-walled")
    draw = ImageDraw.Draw(comparison)
    for index, label in enumerate(labels):
        with Image.open(OUT / ("fidelity-" + label + ".png")) as rendered:
            crop = rendered.convert("RGB").crop((480, 305, 760, 485))
        x, y = (index % 3) * 360, (index // 3) * 270
        comparison.paste(crop, (x + 40, y + 45))
        draw.text((x + 40, y + 16), label, fill="white")
    comparison.save(OUT / "fidelity-stages-crops.png")
    cards = "".join(
        f'<article><h2>{case["stage"]}</h2><img src="{case["image"]}">'
        f'<p>test.biq site {case["site"][0]},{case["site"][1]}</p></article>'
        for case in records)
    (OUT / "fidelity-probe.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>City fidelity stages</title>'
        '<style>body{font:16px system-ui;background:#211b29;color:white;margin:20px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}'
        'article{background:#372b3e;padding:12px}img{width:100%;height:auto}'
        'h2{margin:0 0 8px}</style><h1>City fidelity stages</h1><main>' + cards + '</main>')
    print("Wrote", OUT / "fidelity-probe.html")


if __name__ == "__main__":
    main()
