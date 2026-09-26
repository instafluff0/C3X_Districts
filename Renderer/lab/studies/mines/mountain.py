#!/usr/bin/env python3
"""Test water-aware, camera-facing mountain-base sites for the 3x central mine."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mines import study, terrain


LABEL = "mountain-safe"


def build() -> Path:
    output = study.OUT / "mountain-safe.dll"
    if output.is_file():
        return output
    path = study.PRIVATE / "Renderer/native/object_preparation.h"
    source = path.read_text()
    source = study.replace_once(source,
        "{.72f,.6f},{.6f,.72f}};",
        "{.72f,.6f},{.6f,.72f},{.92f,.92f},{.92f,.78f},"
        "{.78f,.92f},{.86f,.86f},{.88f,.7f},{.7f,.88f}};")
    source = study.replace_once(source,
        "        if(tile.real_terrain_type==6){mine_u=.92f;mine_v=.92f;}\n", "")
    source = study.replace_once(source,
        "            float score=(high-low)+std::max(0.f,.03f-shore)*120.f+",
        "            if(tile.real_terrain_type==6 && (site[0]>.8f||site[1]>.8f) &&"
        " (shore<.03f||river_distance<6.f))continue;\n"
        "            float score=(high-low)+std::max(0.f,.03f-shore)*120.f+")
    path.write_text(source)
    shutil.copy2(study.build(), output)
    return output


def sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    names = ("mountain-ridge", "mountain-wooded")
    canvas = Image.new("RGB", (1208, 1350), "#20272b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    draw.text((12, 10), "test.biq | 3x mountain mine: current base vs water-aware front base",
              fill="white", font=font)
    for row, name in enumerate(names):
        for col, label in enumerate(("central-building", LABEL)):
            source = Image.open(study.OUT / label / name / "preview.png").convert("RGB")
            crop = source.crop((650, 300, 950, 620)).resize((600, 640), Image.Resampling.BICUBIC)
            x, y = col * 604, row * 648 + 40
            canvas.paste(crop, (x, y + 32))
            draw.text((x + 10, y + 4), f"{name} | {'current' if col == 0 else 'front'}",
                      fill="white", font=font)
    output = study.OUT / "test-biq-mine-mountain-front.png"
    canvas.save(output)
    return output


def main() -> None:
    dll = build()
    pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    bare = study.OUT / "bare-pack.bin"
    shutil.copy2(study.OUT / "central-pack.bin", pack)
    study.SITES = {name: terrain.TERRAIN_SITES[name]
                   for name in ("mountain-ridge", "mountain-wooded")}
    try:
        for name in study.SITES:
            for attempt in range(3):
                try:
                    print(study.render(LABEL, name, study.OUT / (name + ".csv"), dll), flush=True)
                    break
                except RuntimeError:
                    if attempt == 2:
                        raise
                    time.sleep(3)
    finally:
        shutil.copy2(bare, pack)
    print(sheet(), flush=True)


if __name__ == "__main__":
    main()
