#!/usr/bin/env python3
"""Matched mine controls on unchanged test.biq terrain, with Lab-only mine markers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab import platform
from Renderer.lab.studies.mines.build_pack import build as build_mine_pack

OUT = ROOT / "Renderer/lab/out/mines"
PRIVATE = OUT / "root"
BIQ = ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"
SITES = {
    "coast-ancient": (79, 13, 0, 12),
    "forest-river-medieval": (59, 79, 1, 12),
    "mountain-wooded": (85, 25, 2, 12),
    "plain-modern": (77, 25, 3, 12),
    "forest-river-night": (59, 79, 1, 21),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Mine study patch target changed: {old[:75]}")
    return source.replace(old, new)


def windows(path: Path) -> str:
    return "..\\..\\" + path.relative_to(ROOT).as_posix().replace("/", "\\")


def copy_sources() -> None:
    if PRIVATE.exists():
        if not (OUT / "inputs.json").is_file():
            raise RuntimeError("Partial isolated mine study root; inspect it before reusing")
        return
    (PRIVATE / "Renderer").mkdir(parents=True)
    shutil.copytree(ROOT / "Renderer/native", PRIVATE / "Renderer/native",
                    ignore=shutil.ignore_patterns("build", "*.obj", "*.ilk", "*.pdb"))
    shutil.copytree(ROOT / "Renderer/sandbox", PRIVATE / "Renderer/sandbox",
                    ignore=shutil.ignore_patterns("out"))
    shutil.copytree(ROOT / "Renderer/lab/shared", PRIVATE / "Renderer/lab/shared")
    for name in ("default.custom_rendering.txt", "custom.custom_rendering.txt"):
        shutil.copy2(ROOT / "Renderer" / name, PRIVATE / "Renderer" / name)
    # Linked source art is read-only. Detach the one generated bundle before rebuilding it.
    shutil.copytree(ROOT / "Renderer/packs", PRIVATE / "Renderer/packs",
                    copy_function=os.link)
    pack_dir = PRIVATE / "Renderer/packs/ImprovementsNormalized"
    pack = pack_dir / "mine_runtime.bin"
    pack.unlink()
    shutil.copy2(ROOT / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin", pack)
    client = PRIVATE / "Renderer/sandbox/reference_x64.cpp"
    client.write_text(replace_once(client.read_text(),
        "        if (source.real == 7) tile.feature_flags = C3X_RENDERER_FEATURE_FOREST;",
        "        if ((source.overlays & 0xfffffffcu) == 0xe0000000u) {\n"
        "            tile.improvement_flags = C3X_RENDERER_IMPROVEMENT_MINE;\n"
        "            tile.route_style = int(source.overlays & 3u);\n"
        "            tile.terrain_overlays = 0;\n"
        "        }\n"
        "        if (source.real == 7) tile.feature_flags = C3X_RENDERER_FEATURE_FOREST;"))
    paths = ("Renderer/native/object_compiler.h", "Renderer/native/c3x_renderer.cpp",
             "Renderer/native/source_fidelity/geometry.h", "Renderer/sandbox/resident_scene.cpp",
             "Renderer/sandbox/reference_x64.cpp", "Renderer/tools/asset_compiler/build_mine_runtime.py")
    (OUT / "inputs.json").write_text(json.dumps({
        "test_biq_sha256": digest(BIQ),
        "inputs_sha256": {name: digest(ROOT / name) for name in paths},
        "marker": "0xe0000000 | Civ III era, Lab CSV only",
    }, indent=2) + "\n")


def build() -> Path:
    result = platform.native_command_result(
        "Renderer/lab/out/mines/root/Renderer/sandbox",
        "call build_reference_x64.bat", timeout_seconds=600)
    dll = PRIVATE / "Renderer/sandbox/out/C3XReference_x64.dll"
    if result["status"] != "pass" or not dll.is_file():
        raise RuntimeError("Isolated mine renderer build failed:\n" + result["output_tail"])
    return dll


def prepare_one_shot_driver() -> None:
    marker = OUT / "one-shot-driver.txt"
    if marker.is_file():
        return
    client = PRIVATE / "Renderer/sandbox/reference_x64.cpp"
    client.write_text(replace_once(client.read_text(),
        "#endif\n    if(bootstrap){",
        "#endif\n    char mine_lab_one_shot[8]={};\n"
        "    if(GetEnvironmentVariableA(\"C3X_MINE_LAB_ONESHOT\",mine_lab_one_shot,"
        "sizeof(mine_lab_one_shot))){\n"
        "        std::printf(\"MINE_LAB_FRAME fallback=%u result=%d\\n\","
        "output.fallback_tile_count,ok?0:1);\n"
        "        return ok?0:1;\n"
        "    }\n"
        "    if(bootstrap){"))
    build()
    marker.write_text("Reference executable exits after the first off-screen frame.\n")


def patch_candidate() -> None:
    header = PRIVATE / "Renderer/native/object_compiler.h"
    source = header.read_text()
    source = source.replace("farm_decal", "ground_decal").replace("farm_shore", "ground_shore")
    source = replace_once(source,
        "    std::vector<Vertex> transformed(asset.vertices.size());",
        "    ground_decal = ground_decal || asset.id.find(\":ground:\") != std::string::npos;\n"
        "    std::vector<Vertex> transformed(asset.vertices.size());")
    source = replace_once(source,
        "    for(auto const& instance:plan.instances){\n"
        "        FeaturePlacement placement{};placement.asset_index=instance.asset;\n"
        "        append_instance(input,assets[instance.family],placement,instance.u,instance.v,instance.rotation,",
        "    float mine_u=.5f,mine_v=.5f,best=1e30f;\n"
        "    if(input.tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE){\n"
        "        float world_u=float(input.tile.tile_x+input.tile.tile_y)*.5f;\n"
        "        float world_v=float(input.tile.tile_x-input.tile.tile_y)*.5f;\n"
        "        constexpr float mine_sites[][2]={{.5f,.5f},{.36f,.36f},{.64f,.36f},"
        "{.36f,.64f},{.64f,.64f},{.5f,.32f},{.5f,.68f},{.32f,.5f},{.68f,.5f}};\n"
        "        for(auto const& site:mine_sites){\n"
        "            float low=1e30f,high=-1e30f,shore=1e30f;\n"
        "            constexpr float corners[][2]={{0,0},{-.23f,-.23f},{.23f,-.23f},"
        "{-.23f,.23f},{.23f,.23f}};\n"
        "            for(auto const& corner:corners){\n"
        "                auto sample=relief(world_u+site[0]+corner[0],"
        "world_v+1.f-site[1]-corner[1]);\n"
        "                low=std::min(low,sample[0]);high=std::max(high,sample[0]);"
        "shore=std::min(shore,sample[2]);\n"
        "            }\n"
        "            float score=(high-low)+std::max(0.f,.03f-shore)*120.f+"
        "high*.012f+std::abs(site[0]-.5f)+std::abs(site[1]-.5f);\n"
        "            if(score<best){best=score;mine_u=site[0];mine_v=site[1];}\n"
        "        }\n"
        "    }\n"
        "    for(auto const& instance:plan.instances){\n"
        "        FeaturePlacement placement{};placement.asset_index=instance.asset;\n"
        "        float u=instance.family==mine_family?mine_u:instance.u;\n"
        "        float v=instance.family==mine_family?mine_v:instance.v;\n"
        "        append_instance(input,assets[instance.family],placement,u,v,instance.rotation,")
    header.write_text(source)
    native = PRIVATE / "Renderer/native/c3x_renderer.cpp"
    native.write_text(replace_once(native.read_text(),
        "float farm_shore=(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)",
        "float farm_shore=(tile.improvement_flags&(C3X_RENDERER_IMPROVEMENT_IRRIGATION|C3X_RENDERER_IMPROVEMENT_MINE))"))
    geometry = PRIVATE / "Renderer/native/source_fidelity/geometry.h"
    geometry.write_text(replace_once(geometry.read_text(),
        "auto const&city=it->occurrence;if(city.city_id<0)continue;",
        "auto const&city=it->occurrence;\n"
        "            if(city.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE)\n"
        "                buildings.push_back({float(c)+.18f,float(r)+.18f,"
        "float(c)+.82f,float(r)+.82f});\n"
        "            if(city.city_id<0)continue;"))
    pack = PRIVATE / "Renderer/packs/ImprovementsNormalized"
    stats = build_mine_pack(pack)
    (OUT / "candidate-pack.json").write_text(json.dumps(stats, indent=2) + "\n")


def patch_relief_site() -> None:
    """Lab placement trial: favor visible near slopes on hill/mountain tiles."""
    header = PRIVATE / "Renderer/native/object_compiler.h"
    source = header.read_text()
    source = replace_once(source,
        "{.5f,.68f},{.32f,.5f},{.68f,.5f}};",
        "{.5f,.68f},{.32f,.5f},{.68f,.5f},"
        "{.72f,.72f},{.78f,.78f},{.72f,.60f},{.60f,.72f}};")
    source = replace_once(source,
        "float score=(high-low)+std::max(0.f,.03f-shore)*120.f+high*.012f+"
        "std::abs(site[0]-.5f)+std::abs(site[1]-.5f);",
        "float relief_front=(input.tile.real_terrain_type==5||"
        "input.tile.real_terrain_type==6)?\n"
        "                std::max(0.f,site[0]+site[1]-1.f)*52.f:0.f;\n"
        "            float score=(high-low)+std::max(0.f,.03f-shore)*120.f+"
        "high*.012f+std::abs(site[0]-.5f)+std::abs(site[1]-.5f)-relief_front;")
    header.write_text(source)


def patch_rigid_site() -> None:
    """Apply mine siting before shared rigid records bypass mesh compilation."""
    path = PRIVATE / "Renderer/native/object_preparation.h"
    source = path.read_text()
    source = replace_once(source,
        "input.farm_ready && (input.projection.tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)",
        "(input.farm_ready||input.mine_ready) && (input.projection.tile.improvement_flags&"
        "(C3X_RENDERER_IMPROVEMENT_IRRIGATION|C3X_RENDERER_IMPROVEMENT_MINE))")
    source = replace_once(source,
        "    result->instances=unsigned(plan.instances.size());result->routes=unsigned(plan.routes.size());",
        "    if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE){\n"
        "        float world_u=float(tile.tile_x+tile.tile_y)*.5f;\n"
        "        float world_v=float(tile.tile_x-tile.tile_y)*.5f;\n"
        "        constexpr float mine_sites[][2]={{.5f,.5f},{.36f,.36f},{.64f,.36f},"
        "{.36f,.64f},{.64f,.64f},{.42f,.42f},{.58f,.42f},"
        "{.42f,.58f},{.58f,.58f},{.72f,.72f},{.78f,.78f},"
        "{.72f,.6f},{.6f,.72f}};\n"
        "        float mine_u=.5f,mine_v=.5f,best=1e30f;\n"
        "        for(auto const& site:mine_sites){\n"
        "            float low=1e30f,high=-1e30f,shore=1e30f;\n"
        "            constexpr float corners[][2]={{0,0},{-.22f,-.22f},"
        "{.22f,-.22f},{-.22f,.22f},{.22f,.22f}};\n"
        "            for(auto const& corner:corners){\n"
        "                auto sample=relief(world_u+site[0]+corner[0],"
        "world_v+1.f-site[1]-corner[1]);\n"
        "                low=std::min(low,sample[0]);high=std::max(high,sample[0]);"
        "shore=std::min(shore,sample[2]);\n"
        "            }\n"
        "            float front=(tile.real_terrain_type==5||tile.real_terrain_type==6)?"
        "std::max(0.f,site[0]+site[1]-1.f)*52.f:0.f;\n"
        "            float river_distance=input.river_ready?"
        "float(scratch.rivers.river_sample({world_u+site[0],"
        "world_v+1.f-site[1]}).distance):1000.f;\n"
        "            float score=(high-low)+std::max(0.f,.03f-shore)*120.f+"
        "std::max(0.f,20.f-river_distance)*3.f+high*.012f+"
        "std::abs(site[0]-.5f)+std::abs(site[1]-.5f)-front;\n"
        "            if(tile.real_terrain_type==5){\n"
        "                float crest=relief(world_u+site[0],"
        "world_v+1.f-site[1])[0];\n"
        "                score=(high-low)*.2f-crest*.75f+"
        "(std::abs(site[0]-.5f)+std::abs(site[1]-.5f))*38.f+"
        "std::max(0.f,.03f-shore)*120.f+"
        "std::max(0.f,20.f-river_distance)*3.f;\n"
        "            }\n"
        "            if(score<best){best=score;mine_u=site[0];mine_v=site[1];}\n"
        "        }\n"
        "        for(auto& instance:plan.instances)if(instance.family==mine_family){"
        "instance.u=mine_u;instance.v=mine_v;}\n"
        "    }\n"
        "    result->instances=unsigned(plan.instances.size());result->routes=unsigned(plan.routes.size());")
    path.write_text(source)


def patch_hill_ground() -> None:
    """Place hill mine bodies on the visible natural height used by hill meshes."""
    path = PRIVATE / "Renderer/native/rigid_object_instance.h"
    source = path.read_text()
    source = replace_once(source,
        "    if(source.family==site_family)ground=height(u+source.u,v+1-source.v)-2.5f;",
        "    bool hill_mine=source.family==mine_family && input.tile.real_terrain_type==5;\n"
        "    if(source.family==site_family||hill_mine)"
        "ground=height(u+source.u,v+1-source.v)-2.5f;")
    source = replace_once(source,
        "source.material,source.owner,false,source.family==site_family,relief,height,vertices,shadows,&indices);",
        "source.material,source.owner,false,source.family==site_family||hill_mine,"
        "relief,height,vertices,shadows,&indices);")
    path.write_text(source)


def patch_component_contacts() -> None:
    """Keep authored part offsets while sampling terrain under each rigid part."""
    path = PRIVATE / "Renderer/native/object_compiler.h"
    source = path.read_text()
    source = replace_once(source,
        "                append_feature_instance(mine_bundle, placement,\n"
        "                    0.5f, 0.5f, rotation, placement.scale, 21.0f,",
        "                float part_u=.5f+placement.scale*(placement.width*std::cos(rotation)-"
        "placement.low_end_reduction*std::sin(rotation));\n"
        "                float part_v=.5f+placement.scale*(placement.width*std::sin(rotation)+"
        "placement.low_end_reduction*std::cos(rotation));\n"
        "                append_feature_instance(mine_bundle, placement,\n"
        "                    part_u, part_v, rotation, placement.scale, 21.0f,")
    source = replace_once(source,
        "float u=instance.family==mine_family?mine_u:instance.u;\n"
        "        float v=instance.family==mine_family?mine_v:instance.v;",
        "float u=instance.family==mine_family?mine_u+instance.u-.5f:instance.u;\n"
        "        float v=instance.family==mine_family?mine_v+instance.v-.5f:instance.v;")
    source = replace_once(source,
        "instance.shadow,instance.family==site_family,\n"
        "            relief,height,output.layers",
        "instance.shadow,instance.family==site_family||"
        "(instance.family==mine_family&&input.tile.real_terrain_type==5),\n"
        "            relief,height,output.layers")
    path.write_text(source)
    path = PRIVATE / "Renderer/native/object_preparation.h"
    source = replace_once(path.read_text(),
        "instance.u=mine_u;instance.v=mine_v;",
        "instance.u+=mine_u-.5f;instance.v+=mine_v-.5f;")
    path.write_text(source)


def scenes() -> dict[str, Path]:
    raw = OUT / "test-biq-original.csv"
    subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"),
                    str(BIQ), str(raw)], cwd=ROOT, check=True)
    with raw.open(newline="") as stream:
        header = next(stream).strip()
        rows = list(csv.reader(stream))
    lookup = {(int(row[0]), int(row[1])): row for row in rows}
    result = {}
    for name, (x, y, era, _hour) in SITES.items():
        if (x, y) not in lookup or int(lookup[(x, y)][2]) >= 11:
            raise ValueError(f"Mine example has no land tile in test.biq: {name}")
        output = OUT / (name + ".csv")
        data = [row.copy() for row in rows]
        for row in data:
            if int(row[0]) == x and int(row[1]) == y:
                row[5] = str(0xe0000000 | era)
        output.write_text(header + "\n" + "\n".join(",".join(row) for row in data) + "\n")
        result[name] = output
    return result


def render(label: str, name: str, scene: Path, dll: Path, tile_width: int = 128) -> Path:
    x, y, _era, hour = SITES[name]
    output = OUT / label / name
    output.mkdir(parents=True, exist_ok=True)
    capture = output / "frame.bmp"
    invocation = output / "run.bat"
    exe = PRIVATE / "Renderer/sandbox/out/reference_x64.exe"
    invocation.write_text(
        "@echo off\nsetlocal\n"
        "set \"C3X_RENDERER_VISUAL_PROFILE=\"\n"
        "set \"C3X_RENDERER_TRACE=0\"\n"
        "set \"C3X_RENDERER_SHARED_SCENE_SURFACE=1\"\n"
        "set \"C3X_RENDERER_WATER_MOTION=1\"\n"
        "set \"C3X_RENDERER_WAVES=1\"\n"
        "set \"C3X_SANDBOX_SHADOW_PATCHES=1\"\n"
        "set \"C3X_SANDBOX_WHOLE_WORLD=1\"\n"
        "set \"C3X_MINE_LAB_ONESHOT=1\"\n"
        "set \"C3X_RENDERER_PREVIEW_UNITS=1\"\n"
        "set \"C3X_SANDBOX_UNITS=1\"\n"
        f"set \"C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS={windows(PRIVATE / 'Renderer/custom.custom_rendering.txt')}\"\n"
        f'"{windows(exe)}" "{windows(dll)}" "{windows(PRIVATE)}" '
        f'"{windows(PRIVATE / "Renderer/default.custom_rendering.txt")}" '
        f'"{windows(scene)}" "{windows(capture)}" '
        f"1600 900 {x} {y} {tile_width} {hour}\nexit /b %errorlevel%\n")
    status = platform.native_command_result("Renderer/native",
        f'call "{windows(invocation)}"', timeout_seconds=600)
    if status["status"] != "pass" or "fallback=0" not in status["output_tail"] or not capture.is_file():
        raise RuntimeError(f"Mine {label}/{name} render failed:\n" + status["output_tail"])
    from PIL import Image
    image = output / "preview.png"
    Image.open(capture).convert("RGB").save(image)
    (output / "capture.json").write_text(json.dumps({
        "test_biq_sha256": digest(BIQ), "scene_sha256": digest(scene),
        "dll_sha256": digest(dll), "camera": [x, y], "hour": hour,
        "pack_sha256": digest(PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"),
        "tile_width": tile_width, "fallback": 0,
    }, indent=2) + "\n")
    return image


def sheet(label: str) -> Path:
    from PIL import Image, ImageDraw
    names = list(SITES)
    canvas = Image.new("RGB", (1300, len(names) * 440 + 42), "#222831")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 13), "test.biq terrain | current mine (left) | mine without brown ground decal (right)", fill="white")
    for row, name in enumerate(names):
        for column, version in enumerate(("control", label)):
            frame = Image.open(OUT / version / name / "preview.png").convert("RGB")
            crop = frame.crop((480, 240, 1120, 650))
            canvas.paste(crop, (column * 650, row * 440 + 42))
        draw.text((12, row * 440 + 47), name, fill="white")
    target = OUT / f"test-biq-mine-{label}-comparison.png"
    canvas.save(target)
    return target


def close_sheet() -> Path:
    from PIL import Image, ImageDraw
    names = list(SITES)
    versions = ("candidate", "bare-match", "bare")
    labels = ("Brown ground, 1.34x", "No ground, 1.34x", "No ground, 1.8x")
    canvas = Image.new("RGB", (1452, len(names) * 460 + 38), "#222831")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "Matched decal control and a separate uniform-scale trial | test.biq", fill="white")
    for row, name in enumerate(names):
        for column, version in enumerate(versions):
            frame = Image.open(OUT / version / name / "preview.png").convert("RGB")
            crop = frame.crop((680, 340, 920, 550)).resize((480, 420), Image.Resampling.BICUBIC)
            x, y = column * 484, row * 460 + 38
            canvas.paste(crop, (x, y + 32))
            draw.text((x + 8, y + 8), f"{name} | {labels[column]}", fill="white")
    target = OUT / "test-biq-mine-detail-study.png"
    canvas.save(target)
    return target


def main() -> None:
    copy_sources()
    prepare_one_shot_driver()
    fixtures = scenes()
    pack_dir = PRIVATE / "Renderer/packs/ImprovementsNormalized"
    pack = pack_dir / "mine_runtime.bin"
    original_pack = ROOT / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    if digest(pack) != digest(original_pack):
        shutil.copy2(original_pack, pack)
    baseline = OUT / "control.dll"
    if not baseline.is_file():
        shutil.copy2(build(), baseline)
    for name, scene in fixtures.items():
        target = OUT / "control" / name / "preview.png"
        if not target.is_file():
            print(render("control", name, scene, baseline), flush=True)
    candidate = OUT / "candidate.dll"
    if not candidate.is_file():
        if not (OUT / "candidate-pack.json").is_file():
            patch_candidate()
        # Allow the shared-folder metadata for the patched native source to settle.
        time.sleep(2)
        shutil.copy2(build(), candidate)
    ground_pack = OUT / "ground-decal-pack.bin"
    if not ground_pack.is_file():
        shutil.copy2(pack, ground_pack)
    if digest(pack) != digest(ground_pack):
        shutil.copy2(ground_pack, pack)
    for name, scene in fixtures.items():
        target = OUT / "candidate" / name / "preview.png"
        if not target.is_file():
            print(render("candidate", name, scene, candidate), flush=True)
    bare_match_pack = OUT / "bare-match-pack.bin"
    if not bare_match_pack.is_file():
        stats = build_mine_pack(pack_dir, include_ground=False, scale=1.34)
        shutil.copy2(pack, bare_match_pack)
        (OUT / "bare-match-pack.json").write_text(json.dumps(stats, indent=2) + "\n")
    elif digest(pack) != digest(bare_match_pack):
        shutil.copy2(bare_match_pack, pack)
    for name, scene in fixtures.items():
        target = OUT / "bare-match" / name / "preview.png"
        if not target.is_file():
            print(render("bare-match", name, scene, candidate), flush=True)
    bare_pack = OUT / "bare-pack.bin"
    if not bare_pack.is_file():
        stats = build_mine_pack(pack_dir, include_ground=False, scale=1.8)
        shutil.copy2(pack, bare_pack)
        (OUT / "bare-pack.json").write_text(json.dumps(stats, indent=2) + "\n")
    elif digest(pack) != digest(bare_pack):
        shutil.copy2(bare_pack, pack)
    for name, scene in fixtures.items():
        target = OUT / "bare" / name / "preview.png"
        if not target.is_file():
            print(render("bare", name, scene, candidate), flush=True)
    print(sheet("bare"), flush=True)
    print(close_sheet(), flush=True)


if __name__ == "__main__":
    main()
