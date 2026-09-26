#!/usr/bin/env python3
"""Lift the centered hill building above its highest contact, then fit its base."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mines import study, terrain


LABEL = "hill-support"


def build() -> Path:
    output = study.OUT / "hill-support.dll"
    if output.is_file():
        return output
    path = study.PRIVATE / "Renderer/native/object_compiler.h"
    source = path.read_text()
    source = study.replace_once(source,
        "    float center_x = left + half_w + (local_u - local_v) * half_w;",
        "    bool hill_mine=tile.real_terrain_type==5 &&\n"
        "        (tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE) &&\n"
        "        asset.id.rfind(\"mine_\",0)==0;\n"
        "    if(hill_mine){\n"
        "        float ca=std::cos(rotation),sa=std::sin(rotation);\n"
        "        for(auto const& foot:asset.vertices){\n"
        "            if(foot.position[2]>.008f)continue;\n"
        "            float x=(foot.position[0]*ca-foot.position[1]*sa)*scale;\n"
        "            float y=(foot.position[0]*sa+foot.position[1]*ca)*scale;\n"
        "            ground_sample[0]=std::max(ground_sample[0],\n"
        "                natural_height_at(tile_world_u+local_u+x,\n"
        "                    tile_world_v+1.f-local_v-y)-2.5f+.01f);\n"
        "        }\n"
        "    }\n"
        "    float center_x = left + half_w + (local_u - local_v) * half_w;")
    source = study.replace_once(source,
        "    bool hill_mine=tile.real_terrain_type==5 &&\n"
        "        (tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE) &&\n"
        "        asset.id.rfind(\"mine_\",0)==0;\n"
        "    std::vector<Vertex> transformed(asset.vertices.size());",
        "    std::vector<Vertex> transformed(asset.vertices.size());")
    path.write_text(source)
    shutil.copy2(study.build(), output)
    return output


def sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1208, 1370), "#20272b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    draw.text((12, 10), "test.biq | 3x hill mine: centered base fit vs lifted supported base",
              fill="white", font=font)
    for row, name in enumerate(("hill-inland", "hill-coast")):
        for col, label in enumerate(("hill-adapt", LABEL)):
            source = Image.open(study.OUT / label / name / "preview.png").convert("RGB")
            crop = source.crop((650, 300, 950, 620)).resize((600, 640), Image.Resampling.BICUBIC)
            x, y = col * 604, row * 648 + 40
            canvas.paste(crop, (x, y + 32))
            draw.text((x + 10, y + 4), f"{name} | {'base fit' if col == 0 else 'supported'}",
                      fill="white", font=font)
    output = study.OUT / "test-biq-mine-hill-support.png"
    canvas.save(output)
    return output


def final_sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1208, 2668), "#20272b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    draw.text((12, 10), "test.biq | complete 2.3x mine vs 3x central-only terrain fit",
              fill="white", font=font)
    for row, name in enumerate(terrain.TERRAIN_SITES):
        for col, label in enumerate(("part-ground", LABEL)):
            source = Image.open(study.OUT / label / name / "preview.png").convert("RGB")
            crop = source.crop((650, 300, 950, 620)).resize((600, 640), Image.Resampling.BICUBIC)
            x, y = col * 604, row * 648 + 40
            canvas.paste(crop, (x, y + 32))
            draw.text((x + 10, y + 4), f"{name} | {'full' if col == 0 else 'central'}",
                      fill="white", font=font)
    output = study.OUT / "test-biq-mine-central-final.png"
    canvas.save(output)
    return output


def main() -> None:
    dll = build()
    pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    bare = study.OUT / "bare-pack.bin"
    shutil.copy2(study.OUT / "central-pack.bin", pack)
    study.SITES = dict(terrain.TERRAIN_SITES)
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
    print(final_sheet(), flush=True)


if __name__ == "__main__":
    main()
