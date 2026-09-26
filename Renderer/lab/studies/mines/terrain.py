#!/usr/bin/env python3
"""Hill and mountain witnesses for the isolated mine Lab candidate."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mines import study
from Renderer.lab.studies.mines.build_pack import build as build_mine_pack


TERRAIN_SITES = {
    "hill-inland": (21, 85, 3, 12),
    "hill-coast": (73, 47, 0, 12),
    "mountain-ridge": (63, 57, 2, 12),
    "mountain-wooded": (85, 25, 2, 12),
}
ERA_SITES = {
    "coast-ancient": "Ancient | preindustrial",
    "hill-central": "Medieval | preindustrial",
    "mountain-wooded": "Industrial | industrial",
    "hill-inland": "Modern | industrial",
}


def render_missing(label: str, name: str, scene: Path, dll: Path) -> None:
    if (study.OUT / label / name / "preview.png").is_file():
        return
    for attempt in range(3):
        try:
            print(study.render(label, name, scene, dll), flush=True)
            return
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(3)


def terrain_sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    names = tuple(TERRAIN_SITES)
    width, height = 1208, 2668
    canvas = Image.new("RGB", (width, height), "#222831")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    draw.text((18, 12), "test.biq hills and mountains | centered 1.8x vs grounded parts 2.3x",
              fill="white", font=font)
    for row, name in enumerate(names):
        for col, label in enumerate(("terrain-control", "part-ground")):
            frame = Image.open(study.OUT / label / name / "preview.png").convert("RGB")
            crop = frame.crop((650, 300, 950, 620)).resize((600, 640), Image.Resampling.BICUBIC)
            x, y = col * 604, row * 648 + 40
            canvas.paste(crop, (x, y + 32))
            caption = "centered" if col == 0 else (
                "on hill" if name.startswith("hill") else "at mountain base")
            draw.text((x + 12, y + 4), f"{name} | {caption}", fill="white", font=font)
    target = study.OUT / "test-biq-hill-mountain-mine-study.png"
    canvas.save(target)
    return target


def era_sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1208, 1284), "#222831")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=21)
    draw.text((16, 12), "test.biq | four Civ III eras, two authored mine families | 2.3x",
              fill="white", font=font)
    for index, (name, caption) in enumerate(ERA_SITES.items()):
        frame = Image.open(study.OUT / "part-ground" / name / "preview.png").convert("RGB")
        crop = frame.crop((650, 300, 950, 600)).resize((600, 600), Image.Resampling.BICUBIC)
        x, y = (index % 2) * 604, (index // 2) * 620 + 40
        canvas.paste(crop, (x, y + 24))
        draw.text((x + 12, y), f"{caption} | {name}", fill="white", font=font)
    target = study.OUT / "test-biq-mine-era-study.png"
    canvas.save(target)
    return target


def main() -> None:
    if not (study.OUT / "candidate.dll").is_file() or not (study.OUT / "bare-pack.bin").is_file():
        raise RuntimeError("Run Renderer/lab/studies/mines/study.py first")
    pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    bare = study.OUT / "bare-pack.bin"
    if study.digest(pack) != study.digest(bare):
        shutil.copy2(bare, pack)
    final = study.OUT / "part-ground.dll"
    if not final.is_file():
        if "float relief_front=" not in (study.PRIVATE / "Renderer/native/object_compiler.h").read_text():
            study.patch_relief_site()
        if "float river_distance=input.river_ready?" not in (
                study.PRIVATE / "Renderer/native/object_preparation.h").read_text():
            study.patch_rigid_site()
        if "bool hill_mine=" not in (
                study.PRIVATE / "Renderer/native/rigid_object_instance.h").read_text():
            study.patch_hill_ground()
        if "float part_u=.5f+placement.scale" not in (
                study.PRIVATE / "Renderer/native/object_compiler.h").read_text():
            study.patch_component_contacts()
        time.sleep(2)
        shutil.copy2(study.build(), final)
    study.SITES.update(TERRAIN_SITES)
    study.SITES["hill-central"] = (19, 59, 1, 12)
    scenes = study.scenes()
    expected = {(x, y): 5 if name.startswith("hill") else 6
                for name, (x, y, _era, _hour) in TERRAIN_SITES.items()}
    expected[(19, 59)] = 5
    with (study.OUT / "test-biq-original.csv").open(newline="") as stream:
        actual = {(int(row[0]), int(row[1])): int(row[3]) for row in csv.reader(stream)
                  if len(row) > 3 and row[0].isdigit()}
    if any(actual.get(site) != terrain for site, terrain in expected.items()):
        raise RuntimeError("The test.biq hill or mountain witness changed")
    for name in TERRAIN_SITES:
        render_missing("terrain-control", name, scenes[name], study.OUT / "candidate.dll")
    grounded = study.OUT / "part-pack.bin"
    if not grounded.is_file():
        stats = build_mine_pack(pack.parent, include_ground=False, scale=2.3,
                                conform_parts=True)
        shutil.copy2(pack, grounded)
        (study.OUT / "part-pack.json").write_text(json.dumps(stats, indent=2) + "\n")
    else:
        shutil.copy2(grounded, pack)
    try:
        for name in (*TERRAIN_SITES, *ERA_SITES, "forest-river-medieval", "forest-river-night",
                     "plain-modern"):
            render_missing("part-ground", name, scenes[name], final)
    finally:
        shutil.copy2(bare, pack)
    print(terrain_sheet(), flush=True)
    print(era_sheet(), flush=True)


if __name__ == "__main__":
    main()
