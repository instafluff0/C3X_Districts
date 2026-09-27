#!/usr/bin/env python3
"""Test whether lifting the rigid mine body while draping its feet clears hills."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mines import central, study, terrain_refresh


LIFTS = (0, 10, 20)
HILLS = ("hill-inland", "hill-coast")


def main() -> None:
    dll = terrain_refresh.setup_hill_lift()
    scenes = study.scenes()
    pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    central.build(pack.parent, 3.0)
    try:
        for lift in LIFTS:
            for name in HILLS:
                for attempt in range(3):
                    try:
                        print(lift, name, study.render(f"lift-{lift}", name,
                                                       scenes[name], dll, 192, lift), flush=True)
                        break
                    except RuntimeError:
                        if attempt == 2:
                            raise
                        time.sleep(3)
    finally:
        shutil.copy2(ROOT / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin", pack)
    sheet()


def sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (3*520, 2*570+48), "#20272b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    draw.text((12, 10), "Current test.biq hills | 3x central mine | ground contact lift",
              fill="white", font=font)
    for row, name in enumerate(HILLS):
        for col, lift in enumerate(LIFTS):
            source = Image.open(study.OUT / f"lift-{lift}" / name / "preview.png").convert("RGB")
            crop = source.crop((650, 300, 950, 620)).resize((500, 530), Image.Resampling.BICUBIC)
            x,y=col*520+10,row*570+48
            canvas.paste(crop,(x,y+27))
            draw.text((x+8,y),f"{name} | +{lift} terrain units",fill="white",font=font)
    target = study.OUT / "test-biq-hill-lift-probe.png"
    canvas.save(target)
    print(target, flush=True)
    return target


if __name__ == "__main__":
    main()
