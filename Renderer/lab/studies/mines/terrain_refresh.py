#!/usr/bin/env python3
"""Compare central mine sizes against the current, flatter test.biq relief."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mines import central, study, terrain


OUT = study.OUT / "current-terrain"
SCALES = (2.0, 2.5, 3.0, 3.5)


def setup() -> Path:
    study.OUT = OUT
    study.PRIVATE = OUT / "root"
    study.SITES = dict(terrain.TERRAIN_SITES)
    OUT.mkdir(parents=True, exist_ok=True)
    study.copy_sources()
    source_hashes = json.loads((OUT / "inputs.json").read_text())["inputs_sha256"]
    changed = [name for name, expected in source_hashes.items()
               if study.digest(ROOT / name) != expected]
    if changed and not (OUT / "terrain-refresh.dll").is_file():
        raise RuntimeError("Current renderer inputs changed before this isolated build: " +
                           ", ".join(changed))
    border_headers = study.PRIVATE / "Renderer/lab/studies/borders"
    border_headers.mkdir(parents=True, exist_ok=True)
    for name in ("depth_export.h", "mesh_export.h"):
        shutil.copy2(ROOT / "Renderer/lab/studies/borders" / name, border_headers / name)
    candidate = OUT / "terrain-refresh.dll"
    if candidate.is_file():
        return candidate
    preparation = study.PRIVATE / "Renderer/native/object_preparation.h"
    compiler = study.PRIVATE / "Renderer/native/object_compiler.h"
    if ("constexpr float mine_sites[][2]={{.72f,.72f}" in preparation.read_text() and
            "bool hill_mine=tile.real_terrain_type==5" in compiler.read_text()):
        client = study.PRIVATE / "Renderer/sandbox/reference_x64.cpp"
        if "MINE_LAB_FRAME fallback=" in client.read_text():
            study.build()
        else:
            study.prepare_one_shot_driver()
        shutil.copy2(study.PRIVATE / "Renderer/sandbox/out/C3XReference_x64.dll", candidate)
        return candidate
    source = preparation.read_text()
    source = study.replace_once(source,
        "    result->instances=unsigned(plan.instances.size());result->routes=unsigned(plan.routes.size());",
        "    if(tile.real_terrain_type==6 &&\n"
        "            (tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE)){\n"
        "        float world_u=float(tile.tile_x+tile.tile_y)*.5f;\n"
        "        float world_v=float(tile.tile_x-tile.tile_y)*.5f;\n"
        "        constexpr float mine_sites[][2]={{.72f,.72f},{.78f,.78f},{.86f,.86f},"
        "{.86f,.72f},{.72f,.86f},{.92f,.78f},{.78f,.92f},{.92f,.92f}};\n"
        "        constexpr float feet[][2]={{0,0},{-.11f,-.11f},{.11f,-.11f},"
        "{-.11f,.11f},{.11f,.11f}};\n"
        "        float best=1e30f,mine_u=.5f,mine_v=.5f;\n"
        "        for(auto const& site:mine_sites){\n"
        "            float low=1e30f,high=-1e30f,shore=1e30f,river_clearance=1e30f;\n"
        "            for(auto const& foot:feet){\n"
        "                float u=world_u+site[0]+foot[0];\n"
        "                float v=world_v+1.f-site[1]-foot[1];\n"
        "                float h=relief(u,v)[0];\n"
        "                low=std::min(low,h);high=std::max(high,h);\n"
        "                shore=std::min(shore,float(queries.shore(u,v).distance));\n"
        "                if(input.river_ready)river_clearance=std::min(river_clearance,\n"
        "                    float(scratch.rivers.river_sample({u,v}).distance));\n"
        "            }\n"
        "            if(shore<.025f||river_clearance<6.f)continue;\n"
        "            float front=std::max(0.f,site[0]+site[1]-1.f);\n"
        "            float score=(high-low)*.55f+high*.01f-front*70.f;\n"
        "            if(score<best){best=score;mine_u=site[0];mine_v=site[1];}\n"
        "        }\n"
        "        for(auto& instance:plan.instances)if(instance.family==mine_family){\n"
        "            instance.u+=mine_u-.5f;instance.v+=mine_v-.5f;\n"
        "        }\n"
        "    }\n"
        "    result->instances=unsigned(plan.instances.size());result->routes=unsigned(plan.routes.size());")
    source = study.replace_once(source,
        "            if(shared_rigid_mesh(asset)){",
        "            if(shared_rigid_mesh(asset) && !(instance.family==mine_family &&\n"
        "                    input.projection.tile.real_terrain_type==5)){")
    preparation.write_text(source)
    source = compiler.read_text()
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
        "                    tile_world_v+1.f-local_v-y)-2.5f+.02f);\n"
        "        }\n"
        "    }\n"
        "    float center_x = left + half_w + (local_u - local_v) * half_w;")
    source = study.replace_once(source,
        "        if (farm_decal) {\n            farm_shore[vertex_index] = vertex_ground[2];",
        "        if(hill_mine){\n"
        "            float surface=natural_height_at(tile_world_u+local_u+local_x,\n"
        "                tile_world_v+1.f-local_v-local_y)-2.5f;\n"
        "            float contact=std::clamp((.045f-source.position[2])/.045f,0.f,1.f);\n"
        "            vertex_ground[0]+=(surface-ground_sample[0])*contact;\n"
        "        }\n"
        "        if (farm_decal) {\n            farm_shore[vertex_index] = vertex_ground[2];")
    compiler.write_text(source)
    study.prepare_one_shot_driver()
    shutil.copy2(study.PRIVATE / "Renderer/sandbox/out/C3XReference_x64.dll", candidate)
    return candidate


def render_all() -> None:
    dll = setup()
    scenes = study.scenes()
    pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    base = study.digest(pack)
    for scale in SCALES:
        label = f"scale-{scale:.1f}".replace(".", "p")
        stats = central.build(pack.parent, scale)
        shutil.copy2(pack, OUT / f"{label}-pack.bin")
        (OUT / f"{label}-pack.json").write_text(json.dumps(stats, indent=2) + "\n")
        try:
            for name, scene in scenes.items():
                receipt = OUT / label / name / "capture.json"
                if receipt.is_file():
                    prior = json.loads(receipt.read_text())
                    if (prior["scene_sha256"] == study.digest(scene) and
                            prior["dll_sha256"] == study.digest(dll) and
                            prior["pack_sha256"] == study.digest(pack)):
                        continue
                for attempt in range(3):
                    try:
                        print(scale, name, study.render(label, name, scene, dll), flush=True)
                        break
                    except RuntimeError:
                        if attempt==2:
                            raise
                        time.sleep(3)
        finally:
            shutil.copy2(ROOT / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin", pack)
        if study.digest(pack) != base:
            raise RuntimeError("Detached private pack did not restore")
    sheet()


def setup_full_parts() -> Path:
    setup()
    candidate = OUT / "terrain-refresh-parts.dll"
    if candidate.is_file():
        return candidate
    compiler = study.PRIVATE / "Renderer/native/object_compiler.h"
    source = compiler.read_text()
    if "float part_u=.5f+placement.scale" not in source:
        source = study.replace_once(source,
            "                append_feature_instance(mine_bundle, placement,\n"
            "                    0.5f, 0.5f, rotation, placement.scale, 21.0f,",
            "                float part_u=.5f+placement.scale*(placement.width*std::cos(rotation)-\n"
            "                    placement.low_end_reduction*std::sin(rotation));\n"
            "                float part_v=.5f+placement.scale*(placement.width*std::sin(rotation)+\n"
            "                    placement.low_end_reduction*std::cos(rotation));\n"
            "                append_feature_instance(mine_bundle, placement,\n"
            "                    part_u, part_v, rotation, placement.scale, 21.0f,")
        compiler.write_text(source)
    for attempt in range(3):
        try:
            study.build()
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(3)
    shutil.copy2(study.PRIVATE / "Renderer/sandbox/out/C3XReference_x64.dll", candidate)
    return candidate


def setup_hill_lift() -> Path:
    setup_full_parts()
    candidate = OUT / "hill-lift.dll"
    if candidate.is_file():
        return candidate
    compiler = study.PRIVATE / "Renderer/native/object_compiler.h"
    source = compiler.read_text()
    if 'std::getenv("C3X_MINE_HILL_LIFT")' in source:
        source = study.replace_once(source,
            "        if(auto const* lift=std::getenv(\"C3X_MINE_HILL_LIFT\"))\n"
            "            ground_sample[0]+=float(std::atof(lift));",
            "        char lift_text[24]={};\n"
            "        if(GetEnvironmentVariableA(\"C3X_MINE_HILL_LIFT\",lift_text,sizeof(lift_text)))\n"
            "            ground_sample[0]+=float(std::atof(lift_text));")
        compiler.write_text(source)
    elif 'GetEnvironmentVariableA("C3X_MINE_HILL_LIFT"' not in source:
        source = study.replace_once(source,
            "                    tile_world_v+1.f-local_v-y)-2.5f+.02f);\n"
            "        }\n"
            "    }\n"
            "    float center_x = left + half_w + (local_u - local_v) * half_w;",
            "                    tile_world_v+1.f-local_v-y)-2.5f+.02f);\n"
            "        }\n"
            "        char lift_text[24]={};\n"
            "        if(GetEnvironmentVariableA(\"C3X_MINE_HILL_LIFT\",lift_text,sizeof(lift_text)))\n"
            "            ground_sample[0]+=float(std::atof(lift_text));\n"
            "    }\n"
            "    float center_x = left + half_w + (local_u - local_v) * half_w;")
        compiler.write_text(source)
    for attempt in range(3):
        try:
            study.build()
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(3)
    shutil.copy2(study.PRIVATE / "Renderer/sandbox/out/C3XReference_x64.dll", candidate)
    return candidate


def sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    names = tuple(study.SITES)
    canvas = Image.new("RGB", (4*462, len(names)*510+40), "#20272b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    draw.text((12, 8), "Current test.biq terrain | central mine art | hill contour fit", "white", font=font)
    for row, name in enumerate(names):
        for col, scale in enumerate(SCALES):
            label = f"scale-{scale:.1f}".replace(".", "p")
            source = Image.open(OUT / label / name / "preview.png").convert("RGB")
            crop = source.crop((650, 300, 950, 620)).resize((450, 480), Image.Resampling.BICUBIC)
            x,y=col*462+6,row*510+40
            canvas.paste(crop,(x,y+25))
            draw.text((x+8,y),f"{name} | {scale:.1f}x", "white", font=font)
    target = OUT / "test-biq-mine-size-comparison.png"
    canvas.save(target)
    print(target, flush=True)
    return target


if __name__ == "__main__":
    if sys.argv[1:] == ["--mountain-probe"]:
        dll = setup()
        scenes = study.scenes()
        pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
        central.build(pack.parent, 3.0)
        try:
            for name in ("mountain-ridge", "mountain-wooded"):
                print(study.render("mountain-front-probe", name, scenes[name], dll), flush=True)
        finally:
            shutil.copy2(ROOT / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin", pack)
    else:
        render_all()
