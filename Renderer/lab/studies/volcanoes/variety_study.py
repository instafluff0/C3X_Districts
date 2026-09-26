#!/usr/bin/env python3
"""Render source-derived volcano orientations at the accepted shape and scale.

The private build owns every edited source. Local source art is linked read-only;
the repository candidate, staging area, sandbox and fixed references are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.lab import platform, preparation

OUT = ROOT / "Renderer/lab/out/volcanoes/variety-study"
PRIVATE = OUT / "root"
CORE = Path("Renderer/native/render_core")
MATERIAL = Path("Renderer/lab/shared/shaders/relief/volcano_material.hlsl")
MESH = Path("Renderer/native/source_fidelity/terrain_mesh_body.h")
POSITIONS = [(x, y) for y in (26, 30, 34, 38) for x in (26, 30, 34, 38)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one source expression, found {count}: {old[:80]}")
    return source.replace(old, new)


def prepare_private() -> None:
    if PRIVATE.exists():
        if not (OUT / "inputs.txt").is_file():
            raise RuntimeError("Partial isolated study exists; inspect it before rerunning")
        return
    (PRIVATE / "Renderer").mkdir(parents=True)
    shutil.copytree(ROOT / "Renderer/native", PRIVATE / "Renderer/native",
                    ignore=shutil.ignore_patterns("build", "*.obj", "*.ilk", "*.pdb"))
    shutil.copytree(ROOT / "Renderer/lab/shared", PRIVATE / "Renderer/lab/shared")
    for name in ("default.custom_rendering.txt", "custom.custom_rendering.txt"):
        shutil.copy2(ROOT / "Renderer" / name, PRIVATE / "Renderer" / name)
    # Hard links preserve the ignored local art without making a second 6 GB copy.
    # No study step writes to these paths.
    shutil.copytree(ROOT / "Renderer/packs", PRIVATE / "Renderer/packs",
                    copy_function=os.link)

    terrain = PRIVATE / CORE / "terrain_query.h"
    source = terrain.read_text()
    marker = "inline bool water(Tile t) { return t.base>=11 && t.base<=13; }"
    source = replace_once(source, marker, """// Lab-only orientations of the single authored volcano field. The height
// and material use this same transform, so the crater and ridges stay aligned.
inline unsigned volcano_orientation(int raw_x,int raw_y) {
    return hash(std::uint32_t(raw_x)*73856093u ^ std::uint32_t(raw_y)*19349663u)&7u;
}
inline std::array<float,2> volcano_source_offset(float x,float y,unsigned orientation) {
    if(orientation&4u)x=-x;
    switch(orientation&3u) {
        case 1:return {-y,x};
        case 2:return {-x,-y};
        case 3:return {y,-x};
        default:return {x,y};
    }
}
""" + marker)
    terrain.write_text(source)

    relief = PRIVATE / CORE / "relief_query.h"
    source = relief.read_text()
    source = replace_once(source,
                          "        float u=.5f+(x-.5f)*footprint,v=.5f+(y-.5f)*footprint;",
                          """        if(tile.real==10) {
            auto offset=volcano_source_offset(x-.5f,y-.5f,
                volcano_orientation(coordinate[0],coordinate[1]));
            x=.5f+offset[0];y=.5f+offset[1];
        }
        float u=.5f+(x-.5f)*footprint,v=.5f+(y-.5f)*footprint;""")
    source = replace_once(source,
                          "            result.owner={.5f+ox*float(volcano_footprint),.5f-oy*float(volcano_footprint),",
                          """            auto raw_owner=raw(owner_c,owner_r);
            auto oriented=volcano_source_offset(ox,-oy,
                volcano_orientation(raw_owner[0],raw_owner[1]));
            result.owner={.5f+oriented[0]*float(volcano_footprint),
                          .5f+oriented[1]*float(volcano_footprint),""")
    relief.write_text(source)

    mesh = PRIVATE / MESH
    source = mesh.read_text()
    source = replace_once(source, "    std::vector<std::array<float,2>> volcano_centers;",
                          "    std::vector<std::array<float,3>> volcano_centers;")
    source = replace_once(source,
                          "            volcano_centers.push_back({float(nc+dc)+.5f,float(nr+dr)+.5f});",
                          """            volcano_centers.push_back({float(nc+dc)+.5f,float(nr+dr)+.5f,
                float(c3x_renderer::render_core::volcano_orientation(
                    nc+dc+nr+dr,nc+dc-nr-dr))});""")
    source = replace_once(source,
                          "                    v.relief_owner_u=dx;v.relief_owner_v=dy;",
                          """                    auto offset=c3x_renderer::render_core::volcano_source_offset(
                        dx,-dy,unsigned(center[2]));
                    v.relief_owner_u=offset[0];v.relief_owner_v=-offset[1];""")
    mesh.write_text(source)

    material = PRIVATE / MATERIAL
    source = material.read_text()
    source = replace_once(source, "Texture2D VolcanoLavaColor : register(t71);\n", "")
    start = source.index("    // Measured local art registration;")
    end = source.index("\n}", start)
    source = source[:start] + "    return albedo;" + source[end:]
    material.write_text(source)
    preparation.generate(PRIVATE)
    (OUT / "inputs.txt").write_text(
        "Lab-only source-derived volcano: accepted height and width; "
        "eight deterministic orientations; no static lava or smoke.\n"
        f"source_relief_sha256={digest(ROOT / CORE / 'relief_query.h')}\n"
        f"source_material_sha256={digest(ROOT / MATERIAL)}\n"
        f"private_relief_sha256={digest(relief)}\n"
        f"private_material_sha256={digest(material)}\n")


def build_private() -> Path:
    dll = PRIVATE / "Renderer/native/build/candidate/C3XRenderer.dll"
    receipt = OUT / "private-build.txt"
    if dll.is_file() and receipt.is_file() and f"dll_sha256={digest(dll)}" in receipt.read_text():
        return dll
    result = platform.native_command_result(
        "Renderer/lab/out/volcanoes/variety-study/root/Renderer/native",
        "call BUILD.bat candidate-compile", timeout_seconds=900)
    if result["status"] != "pass" or not dll.is_file():
        raise RuntimeError("Isolated native volcano build failed: " + result["output_tail"])
    receipt.write_text(f"dll_sha256={digest(dll)}\n")
    return dll


def variety_scene(_category: str, _case: str, destination: Path, *, world_size: int = 32) -> None:
    size = 64
    selected = set(POSITIONS)
    rows = [f"{x},{y},2,{10 if (x,y) in selected else 2},0,0,0"
            for y in range(size) for x in range(y % 2, size, 2)]
    destination.write_text(f"C3X_BIQ_TERRAIN_V3,{size},{size},{len(rows)}\n" +
                           "\n".join(rows) + "\n")


def render(label: str, dll: Path, *, variety: bool = False, closeup: bool = False) -> Path:
    renderer.ensure_preview_tool()
    target = OUT / label / ("variety" if variety else "closeup" if closeup else "gameplay")
    target.mkdir(parents=True, exist_ok=True)
    native_run = platform.run_native_fixture
    original_scene = renderer.scene

    def isolated(directory: Path, command: str, run_id: str):
        batch = directory / "render.bat"
        body = batch.read_text()
        if label == "proposal":
            needle = 'C3XRenderer.dll" ..\\.. '
            replacement = 'C3XRenderer.dll" ..\\lab\\out\\volcanoes\\variety-study\\root '
            body = replace_once(body, needle, replacement)
        if variety:
            body = replace_once(body, "640 480 16 16 128 12", "1280 800 32 32 128 12")
        batch.write_text(body)
        return native_run(directory, command, run_id)

    try:
        platform.run_native_fixture = isolated
        if variety:
            renderer.scene = variety_scene
        record = renderer.native_render("volcanoes", "detail" if variety or closeup else "gameplay",
                                        12, 224 if closeup else 128, target, candidate=dll,
                                        preview=renderer.LAB / ".cache/native_preview.exe")
    finally:
        platform.run_native_fixture = native_run
        renderer.scene = original_scene
    from PIL import Image
    image = target / "preview.png"
    Image.open(ROOT / record["image"]).convert("RGB").save(image)
    (target / "capture.txt").write_text(
        f"dll_sha256={digest(dll)}\nimage_sha256={digest(image)}\n"
        f"scene_sha256={digest(target / 'scene.csv')}\n"
        "renderer=production D3D11 via native Lab preview\n"
        f"fallback=0\nlabel={label}\n")
    return image


def sheets() -> None:
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=13)
    for label in ("current", "proposal"):
        frame = Image.open(OUT / label / "variety/preview.png").convert("RGB")
        atlas = Image.new("RGB", (4 * 240, 4 * 175), "#20252d")
        draw = ImageDraw.Draw(atlas)
        for index, (x, y) in enumerate(POSITIONS):
            col, row = index % 4, index // 4
            cx = 64 * (x - 32) + 640
            cy = 32 * (y - 32) + 368
            atlas.paste(frame.crop((cx-120,cy-72,cx+120,cy+73)),
                        (col*240,row*175+30))
            draw.text((col*240+9,row*175+8), f"tile {x},{y}", fill="white", font=small)
        atlas.save(OUT / label / "grassland-variety.png")

    before = Image.open(OUT / "current/grassland-variety.png").convert("RGB")
    after = Image.open(OUT / "proposal/grassland-variety.png").convert("RGB")
    sheet = Image.new("RGB", (before.width*2,before.height+42), "#20252d")
    sheet.paste(before,(0,42));sheet.paste(after,(before.width,42))
    draw = ImageDraw.Draw(sheet)
    draw.text((14,10),"Existing volcano art | staged renderer",fill="white",font=font)
    draw.text((before.width+14,10),"Lab: current size, eight orientations, no lava",fill="white",font=font)
    sheet.save(OUT / "grassland-comparison.png")

    before = Image.open(OUT / "current/gameplay/preview.png").convert("RGB")
    after = Image.open(OUT / "proposal/gameplay/preview.png").convert("RGB")
    sheet = Image.new("RGB", (before.width*2,before.height+42), "#20252d")
    sheet.paste(before,(0,42));sheet.paste(after,(before.width,42))
    draw = ImageDraw.Draw(sheet)
    draw.text((14,10),"Staged renderer | mountain context",fill="white",font=font)
    draw.text((before.width+14,10),"Lab proposal | same scene",fill="white",font=font)
    sheet.save(OUT / "gameplay-comparison.png")

    closeups = [OUT / label / "closeup/preview.png" for label in ("current", "proposal")]
    if all(path.is_file() for path in closeups):
        before, after = (Image.open(path).convert("RGB") for path in closeups)
        sheet = Image.new("RGB", (before.width*2,before.height+42), "#20252d")
        sheet.paste(before,(0,42));sheet.paste(after,(before.width,42))
        draw = ImageDraw.Draw(sheet)
        draw.text((14,10),"Staged renderer | 224 px tile",fill="white",font=font)
        draw.text((before.width+14,10),"Lab proposal | 224 px tile",fill="white",font=font)
        sheet.save(OUT / "closeup-comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets-only", action="store_true")
    parser.add_argument("--closeups-only", action="store_true")
    args = parser.parse_args()
    if args.sheets_only:
        sheets();return
    prepare_private()
    proposal = build_private()
    current = OUT / "current/C3XRenderer.dll"
    if not current.is_file():
        staged = ROOT / "Renderer/bin/C3XRenderer.dll"
        if staged.is_file():
            current.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, current)
    if not current.is_file():
        raise RuntimeError("Staged baseline DLL missing")
    for label, dll in (("current", current), ("proposal", proposal)):
        if not args.closeups_only:
            render(label, dll, variety=True)
            render(label, dll)
        render(label, dll, closeup=True)
    sheets()
    print(OUT / "grassland-comparison.png")
    print(OUT / "gameplay-comparison.png")


if __name__ == "__main__":
    main()
