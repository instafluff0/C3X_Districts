#!/usr/bin/env python3
"""Center the 3x mine on hills and fit its lowest vertices to the hill surface."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mines import study, terrain


LABEL = "hill-adapt"


def build() -> Path:
    output = study.OUT / "hill-adapt.dll"
    if output.is_file():
        return output
    path = study.PRIVATE / "Renderer/native/object_preparation.h"
    source = study.replace_once(path.read_text(),
        "        for(auto& instance:plan.instances)if(instance.family==mine_family){",
        "        if(tile.real_terrain_type==5){mine_u=.5f;mine_v=.5f;}\n"
        "        for(auto& instance:plan.instances)if(instance.family==mine_family){")
    source = study.replace_once(source,
        "            if(shared_rigid_mesh(asset)){",
        "            if(shared_rigid_mesh(asset) && !(instance.family==mine_family &&"
        " input.projection.tile.real_terrain_type==5)){")
    path.write_text(source)
    path = study.PRIVATE / "Renderer/native/object_compiler.h"
    source = study.replace_once(path.read_text(),
        "    std::vector<Vertex> transformed(asset.vertices.size());",
        "    bool hill_mine=tile.real_terrain_type==5 &&\n"
        "        (tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE) &&\n"
        "        asset.id.rfind(\"mine_\",0)==0;\n"
        "    std::vector<Vertex> transformed(asset.vertices.size());")
    source = study.replace_once(source,
        "            : ground_sample;\n        if (ground_decal) ground_shore[vertex_index] = vertex_ground[2];",
        "            : ground_sample;\n"
        "        if(hill_mine){\n"
        "            float surface=natural_height_at(tile_world_u+local_u+local_x,\n"
        "                tile_world_v+1.f-local_v-local_y)-2.5f;\n"
        "            float contact=std::clamp((.045f-source.position[2])/.045f,0.f,1.f);\n"
        "            vertex_ground[0]+=(surface-ground_sample[0])*contact;\n"
        "        }\n"
        "        if (ground_decal) ground_shore[vertex_index] = vertex_ground[2];")
    path.write_text(source)
    shutil.copy2(study.build(), output)
    return output


def sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    names = ("hill-inland", "hill-coast", "mountain-ridge", "mountain-wooded")
    canvas = Image.new("RGB", (1208, 2668), "#20272b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    draw.text((12, 10), "test.biq | 3x central building: current vs hill-adaptive + mountain-safe",
              fill="white", font=font)
    for row, name in enumerate(names):
        for col, label in enumerate(("central-building", LABEL)):
            source = Image.open(study.OUT / label / name / "preview.png").convert("RGB")
            crop = source.crop((650, 300, 950, 620)).resize((600, 640), Image.Resampling.BICUBIC)
            x, y = col * 604, row * 648 + 40
            canvas.paste(crop, (x, y + 32))
            draw.text((x + 10, y + 4), f"{name} | {'current' if col == 0 else 'adapted'}",
                      fill="white", font=font)
    output = study.OUT / "test-biq-mine-hill-adapt.png"
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


if __name__ == "__main__":
    main()
