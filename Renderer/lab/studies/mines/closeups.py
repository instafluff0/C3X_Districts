#!/usr/bin/env python3
"""Render every authored mine variant on unobstructed grassland in each Civ III era."""

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


# Existing test.biq tile coordinates whose stable seeds select variants 0, 1, 2.
SITES = ((50, 48), (50, 50), (52, 50))
ERAS = ("Ancient", "Medieval", "Industrial", "Modern")
LABEL = "grassland-closeups"
TILE_WIDTH = 256


def variant(x: int, y: int) -> int:
    value = 2166136261
    value = ((value ^ x) * 16777619) & 0xffffffff
    value = ((value ^ y) * 16777619) & 0xffffffff
    return value % 3


def prepare() -> dict[str, Path]:
    raw = study.OUT / "test-biq-original.csv"
    if not raw.is_file():
        raise RuntimeError("Run Renderer/lab/studies/mines/study.py first")
    with raw.open(newline="") as stream:
        header = next(stream).strip()
        rows = list(csv.reader(stream))
    if [variant(*site) for site in SITES] != [0, 1, 2]:
        raise RuntimeError("Selected source tile seeds no longer cover all mine variants")
    result = {}
    for era in range(4):
        for index, (x, y) in enumerate(SITES):
            name = f"era-{era}-variant-{index}"
            data = []
            for source in rows:
                row = source.copy()
                row[2:7] = ["2", "2", "0", "0", "0"]
                if (int(row[0]), int(row[1])) == (x, y):
                    row[5] = str(0xe0000000 | era)
                data.append(",".join(row))
            scene = study.OUT / (name + "-grassland.csv")
            scene.write_text(header + "\n" + "\n".join(data) + "\n")
            study.SITES[name] = (x, y, era, 12)
            result[name] = scene
    return result


def render_all(scenes: dict[str, Path], single: bool) -> None:
    pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    bare = study.OUT / "bare-pack.bin"
    candidate = study.OUT / "part-pack.bin"
    dll = study.OUT / "part-ground.dll"
    if not all(item.is_file() for item in (bare, candidate, dll)):
        raise RuntimeError("Run Renderer/lab/studies/mines/terrain.py first")
    shutil.copy2(candidate, pack)
    try:
        for name, scene in list(scenes.items())[:1 if single else None]:
            output = study.OUT / LABEL / name
            capture = output / "capture.json"
            if capture.is_file() and (output / "preview.png").is_file():
                previous = json.loads(capture.read_text())
                if (previous["scene_sha256"] == study.digest(scene) and
                        previous["dll_sha256"] == study.digest(dll) and
                        previous["pack_sha256"] == study.digest(pack) and
                        previous["tile_width"] == TILE_WIDTH):
                    continue
            for attempt in range(3):
                try:
                    print(study.render(LABEL, name, scene, dll, TILE_WIDTH), flush=True)
                    break
                except RuntimeError:
                    if attempt == 2:
                        raise
                    time.sleep(3)
    finally:
        shutil.copy2(bare, pack)


def sheets() -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont

    output = study.OUT / LABEL
    output.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=32)
    family = ("preindustrial", "preindustrial", "industrial", "industrial")
    results = []
    for era, label in enumerate(ERAS):
        sheet = Image.new("RGB", (1200, 2830), "#20272b")
        draw = ImageDraw.Draw(sheet)
        draw.text((18, 10), f"{label} | {family[era]} | complete mine art on grassland",
                  fill="white", font=font)
        for index in range(3):
            name = f"era-{era}-variant-{index}"
            frame = Image.open(output / name / "preview.png").convert("RGB")
            crop = frame.crop((665, 325, 935, 535)).resize((1160, 900), Image.Resampling.LANCZOS)
            single = output / f"{label.lower()}-variant-{index + 1}-large.png"
            crop.save(single)
            results.append(single)
            y = 55 + index * 925
            sheet.paste(crop, (20, y))
            draw.text((34, y + 12), f"Variant {index + 1}", fill="white", font=font,
                      stroke_width=3, stroke_fill="#20272b")
        path = output / f"{label.lower()}-grassland-closeups.png"
        sheet.save(path)
        results.append(path)
    overview = Image.new("RGB", (1200, 4 * 2830), "#20272b")
    for era, label in enumerate(ERAS):
        with Image.open(output / f"{label.lower()}-grassland-closeups.png") as sheet:
            overview.paste(sheet, (0, era * 2830))
    path = output / "all-four-eras-grassland.png"
    overview.save(path)
    results.append(path)
    return results


def main() -> None:
    scenes = prepare()
    single = "--single" in sys.argv[1:]
    render_all(scenes, single)
    if not single:
        for path in sheets():
            print(path, flush=True)


if __name__ == "__main__":
    main()
