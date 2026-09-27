#!/usr/bin/env python3
"""Close hill comparisons: all source mine parts versus the central building."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mines import central, study, terrain_refresh
from Renderer.lab.studies.mines.build_pack import build as build_full


CASES = (("full-1p5-lift20", "All buildings, 1.5x", 1.5, True),
         ("main-1p5-lift20", "Main only, 1.5x", 1.5, False),
         ("full-1p8-lift20", "All buildings, 1.8x", 1.8, True),
         ("main-1p8-lift20", "Main only, 1.8x", 1.8, False))
HILLS = ("hill-inland", "hill-coast")
HILL_LIFT = 20.0


def render() -> None:
    dll = terrain_refresh.setup_hill_lift()
    scenes = study.scenes()
    pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    original = study.digest(ROOT / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin")
    for label, _title, scale, full in CASES:
        stats = (build_full(pack.parent, include_ground=False, scale=scale,
                            conform_parts=True, first_two_only=True)
                 if full else central.build(pack.parent, scale))
        shutil.copy2(pack, study.OUT / f"{label}-pack.bin")
        (study.OUT / f"{label}-pack.json").write_text(json.dumps(stats, indent=2) + "\n")
        try:
            for name in HILLS:
                receipt = study.OUT / label / name / "capture.json"
                if receipt.is_file():
                    prior = json.loads(receipt.read_text())
                    if (prior["scene_sha256"] == study.digest(scenes[name]) and
                            prior["dll_sha256"] == study.digest(dll) and
                            prior["pack_sha256"] == study.digest(pack) and
                            prior["tile_width"] == 192 and
                            prior.get("hill_lift") == HILL_LIFT):
                        continue
                for attempt in range(3):
                    try:
                        print(label, name,
                              study.render(label, name, scenes[name], dll, 192,
                                           HILL_LIFT), flush=True)
                        break
                    except RuntimeError:
                        if attempt == 2:
                            raise
                        time.sleep(3)
        finally:
            shutil.copy2(ROOT / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin", pack)
        if study.digest(pack) != original:
            raise RuntimeError("Detached mine bundle did not restore")
    sheet()


def sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (4*520, 2*570+48), "#20272b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=21)
    draw.text((16, 10), "Current test.biq hills | smaller full Civ VI mines vs main-only | 192px tiles",
              fill="white", font=font)
    for row, name in enumerate(HILLS):
        for col, (label, title, _scale, _full) in enumerate(CASES):
            source = Image.open(study.OUT / label / name / "preview.png").convert("RGB")
            crop = source.crop((650, 300, 950, 620)).resize((500, 530), Image.Resampling.BICUBIC)
            x,y=col*520+10,row*570+48
            canvas.paste(crop,(x,y+27))
            draw.text((x+8,y),f"{name} | {title}",fill="white",font=font)
    target = study.OUT / "test-biq-hill-small-full-vs-main-raised.png"
    canvas.save(target)
    print(target, flush=True)
    return target


if __name__ == "__main__":
    render()
