#!/usr/bin/env python3
"""Render explicit ordinary-volcano slots with bare, forest and jungle feet.

All source edits and source-art hard links live in an isolated Lab output root.
The snow treatment and natural-wonder art are deliberately outside this study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.lab import platform
from Renderer.lab.studies.volcanoes import foundation_study as earlier

OUT = ROOT / "Renderer/lab/out/volcanoes/slot-study"
PRIVATE = OUT / "root"
SEED = earlier.PRIVATE
TERRAIN = Path("Renderer/native/render_core/terrain_query.h")
RELIEF = Path("Renderer/native/render_core/relief_query.h")
NATIVE = Path("Renderer/native/c3x_renderer.cpp")
PREVIEW_SOURCE = Path("Renderer/native/biq_preview.cpp")
POSITIONS = earlier.POSITIONS
FAMILIES = (
    "Current crater cone", "Smooth cone study", "Broad crater cone",
    "Offset steep cone", "Broken ridge complex", "Breached rim study",
    "Paired shoulders", "Eroded cone",
)
MARKERS = {"bare": 0, "forest": 0xF0000001, "jungle": 0xF0000002}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one frozen source expression: {old[:80]}")
    return source.replace(old, new)


def prepare() -> None:
    if not (SEED / RELIEF).is_file():
        raise RuntimeError("The earlier foundation Lab source is required")
    if not PRIVATE.exists():
        (PRIVATE / "Renderer").mkdir(parents=True)
        shutil.copytree(SEED / "Renderer/native", PRIVATE / "Renderer/native",
                        ignore=shutil.ignore_patterns("build", "*.obj", "*.ilk", "*.pdb"))
        shutil.copytree(SEED / "Renderer/lab/shared", PRIVATE / "Renderer/lab/shared")
        for name in ("default.custom_rendering.txt", "custom.custom_rendering.txt"):
            shutil.copy2(SEED / "Renderer" / name, PRIVATE / "Renderer" / name)
        # These packs are existing ignored local art. Hard links add no art copy.
        shutil.copytree(SEED / "Renderer/packs", PRIVATE / "Renderer/packs",
                        copy_function=os.link)
    elif not (OUT / "inputs.txt").is_file():
        raise RuntimeError("Partial Lab root exists; inspect it before reusing")
    lab = PRIVATE / "Renderer/lab"
    lab.mkdir(parents=True, exist_ok=True)
    for name in ("native_preview.cpp", "build_native_preview.bat", "volcano_witness.h"):
        shutil.copy2(SEED / "Renderer/lab" / name, lab / name)

    source = (SEED / TERRAIN).read_text()
    start = source.index("inline unsigned volcano_family(int raw_x,int raw_y) {")
    end = source.index("inline std::array<float,2> volcano_source_offset", start)
    source = source[:start] + """// These raw coordinates are the 4x4 Lab grid, in Civ III PCX row order.
// Other tiles get a stable slot without depending on the visible viewport.
inline unsigned volcano_slot(int raw_x,int raw_y) {
    if(raw_x>=26 && raw_x<=38 && raw_y>=26 && raw_y<=38 &&
       (raw_x-26)%4==0 && (raw_y-26)%4==0)
        return unsigned((raw_y-26)/4*4+(raw_x-26)/4);
    return hash(std::uint32_t(raw_x)*73856093u ^
                std::uint32_t(raw_y)*19349663u)&15u;
}
inline unsigned volcano_family(int raw_x,int raw_y) {
    return volcano_slot(raw_x,raw_y)>>1;
}
inline unsigned volcano_orientation(int raw_x,int raw_y) {
    unsigned slot=volcano_slot(raw_x,raw_y);
    return (slot==5u || slot==15u)?7u:(slot&1u)?5u:0u;
}
""" + source[end:]
    (PRIVATE / TERRAIN).write_text(source)

    source = (SEED / RELIEF).read_text()
    start = source.index("            // Keep one exact current cone. Other families reuse")
    end = source.index("        return result;", start)
    source = source[:start] + """            unsigned slot=volcano_slot(coordinate[0],coordinate[1]);
            unsigned family=slot>>1;
            if(family!=0) {
                float dx=x-.5f,dy=y-.5f;
                float radius=std::sqrt(dx*dx+dy*dy);
                float crater_clear=smooth01((radius-.10f)/.18f);
                float outer=1-smooth01((radius-.50f)/.24f);
                float gate=crater_clear*outer;
                auto foundation=[&](unsigned source_variant,float shift_x,
                                    float shift_y,float span,float exponent) {
                    float fu=.5f+(dx+shift_x)*span;
                    float fv=.5f+(dy+shift_y)*span;
                    float h=source(16,source_variant,0,fu,fv);
                    float b=source(16,source_variant,1,fu,fv);
                    return std::array<float,2>{std::pow(std::max(0.f,h),exponent),b};
                };
                unsigned field=(slot*3u)%5u;
                if(family==1) {
                    // Smoother, nearly symmetric flank, with an open crater.
                    float cone=(1-smooth01((radius-.08f)/.49f))*
                        smooth01((radius-.045f)/.11f);
                    result.height=std::max(result.height*.28f,cone*.82f);
                    result.blend=std::max(result.blend,cone*.83f);
                } else if(family==2) {
                    auto broad=foundation(field,.04f,-.03f,.36f,.72f);
                    result.height=std::max(result.height*.82f,broad[0]*.78f*gate);
                    result.blend=std::max(result.blend,broad[1]*.86f*gate);
                } else if(family==3) {
                    auto steep=foundation(field,.15f,.03f,.52f,1.12f);
                    float lean=std::clamp(1.f+.70f*dx-.25f*dy,.55f,1.30f);
                    result.height=std::max(result.height*lean*.88f,
                                           steep[0]*.92f*gate);
                    result.blend=std::max(result.blend,steep[1]*.82f*gate);
                } else if(family==4) {
                    auto west=foundation(field,-.14f,.07f,.43f,.84f);
                    auto east=foundation((field+2)%5,.13f,-.09f,.45f,.88f);
                    result.height=std::max({result.height*.72f,
                        west[0]*.90f*gate,east[0]*.82f*gate});
                    result.blend=std::max({result.blend,
                        west[1]*.75f*gate,east[1]*.60f*gate});
                } else if(family==5) {
                    float breach=smooth01((dx+dy+.04f)/.20f);
                    float upper=1-smooth01((radius-.13f)/.34f);
                    auto flank=foundation(field,-.12f,.11f,.46f,.87f);
                    result.height=std::max(result.height*(1-.62f*breach*upper),
                                           flank[0]*.60f*gate);
                    result.blend=std::max(result.blend,flank[1]*.65f*gate);
                } else if(family==6) {
                    auto left=foundation(field,-.18f,.06f,.45f,.85f);
                    auto right=foundation((field+3)%5,.17f,-.10f,.47f,.88f);
                    result.height=std::max({result.height*.77f,
                        left[0]*.82f*gate,right[0]*.79f*gate});
                    result.blend=std::max({result.blend,left[1]*.79f*gate,
                        right[1]*.76f*gate});
                } else {
                    auto eroded=foundation(field,.10f,-.04f,.38f,.73f);
                    float broad=(1-smooth01((radius-.11f)/.55f))*.64f*gate;
                    result.height=std::max({result.height*.82f,
                        eroded[0]*.68f*gate,broad});
                    result.blend=std::max(result.blend,eroded[1]*.72f*gate);
                }
            }
        }
""" + source[end:]
    (PRIVATE / RELIEF).write_text(source)

    source = (SEED / PREVIEW_SOURCE).read_text()
    source = replace_once(source,
        "        if (source.real == 10) {tile.feature_flags = C3X_RENDERER_FEATURE_VOLCANO;tile.has_effect=active?1:0;}",
        "        if (source.real == 10) {\n"
        "            tile.feature_flags = C3X_RENDERER_FEATURE_VOLCANO;\n"
        "            if(source.overlays==0xF0000001u) {tile.feature_flags|=C3X_RENDERER_FEATURE_FOREST;tile.terrain_overlays=0;}\n"
        "            if(source.overlays==0xF0000002u) {tile.feature_flags|=C3X_RENDERER_FEATURE_JUNGLE;tile.terrain_overlays=0;}\n"
        "            tile.has_effect=active?1:0;\n"
        "        }")
    (PRIVATE / PREVIEW_SOURCE).write_text(source)

    source = (SEED / NATIVE).read_text()
    source = replace_once(source,
        "            bool draw_feature = feature_assets_ready &&\n"
        "                (tile.real_terrain_type == 7 || tile.real_terrain_type == 8);",
        "            bool draw_feature = feature_assets_ready &&\n"
        "                ((tile.feature_flags & (C3X_RENDERER_FEATURE_FOREST | C3X_RENDERER_FEATURE_JUNGLE)) != 0);")
    start = source.index("            if (feature_assets_ready &&\n"
                         "                (tile.real_terrain_type == 7 || tile.real_terrain_type == 8) &&")
    end = source.index("            if (river_rock_group != nullptr", start)
    section = source[start:end]
    section = section.replace("tile.real_terrain_type == 7 || tile.real_terrain_type == 8",
                              "canopy_kind == 7 || canopy_kind == 8")
    section = section.replace("tile.real_terrain_type", "canopy_kind")
    section = replace_once(section,
        "!(fidelity_profile && canopy_kind == 7)",
        "!(fidelity_profile && canopy_kind == 7 && tile.real_terrain_type != 10)")
    section = "            int canopy_kind = (tile.feature_flags & C3X_RENDERER_FEATURE_FOREST) ? 7 :\n" \
              "                (tile.feature_flags & C3X_RENDERER_FEATURE_JUNGLE) ? 8 : -1;\n" + section
    section = replace_once(section,
        "                        unsigned row = instance / grid_side;",
        "                        unsigned row = instance / grid_side;\n"
        "                        // Light foot ring; the crater and rock flanks stay exposed.\n"
        "                        if(tile.real_terrain_type==10 &&\n"
        "                           ((row>0 && row+1<grid_side && column>0 && column+1<grid_side) ||\n"
        "                            (row+column<grid_side-1) ||\n"
        "                            (instance&1u))) continue;")
    section = replace_once(section,
        "float scene_feature_scale = canopy_kind == 7 ? 0.42f : 0.40f;",
        "float scene_feature_scale = tile.real_terrain_type==10 ?\n"
        "                            (canopy_kind==7 ? .27f : .25f) :\n"
        "                            (canopy_kind==7 ? .42f : .40f);")
    source = source[:start] + section + source[end:]
    (PRIVATE / NATIVE).write_text(source)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs.txt").write_text(
        "Lab-only explicit slots; source art hard-linked, no snow or natural wonders.\n"+
        "".join(f"{path.name}_sha256={digest(PRIVATE/path)}\n"
                for path in (TERRAIN, RELIEF, NATIVE, PREVIEW_SOURCE)))
    (OUT / "slots.json").write_text(json.dumps([
        {"slot": index, "tile": list(position), "family": FAMILIES[index//2],
         "orientation": 7 if index in (5,15) else 5 if index%2 else 0}
        for index,position in enumerate(POSITIONS)], indent=2)+"\n")


def build() -> tuple[Path,Path]:
    dll = PRIVATE / "Renderer/native/build/candidate/C3XRenderer.dll"
    exe = PRIVATE / "Renderer/lab/.cache/native_preview.exe"
    source_hash = hashlib.sha256("".join(digest(PRIVATE/path)
        for path in (TERRAIN, RELIEF, NATIVE, PREVIEW_SOURCE)).encode()).hexdigest()
    receipt=OUT / "build.txt"
    if not (dll.is_file() and exe.is_file() and receipt.is_file() and
            f"source_sha256={source_hash}" in receipt.read_text() and
            f"dll_sha256={digest(dll)}" in receipt.read_text() and
            f"exe_sha256={digest(exe)}" in receipt.read_text()):
        result=platform.native_command_result(
            "Renderer/lab/out/volcanoes/slot-study/root/Renderer/native",
            "call BUILD.bat candidate-compile", timeout_seconds=900)
        if result["status"]!="pass" or not dll.is_file():
            raise RuntimeError("Lab candidate build failed: "+result["output_tail"])
        result=platform.native_command_result(
            "Renderer/lab/out/volcanoes/slot-study/root/Renderer/lab",
            "call build_native_preview.bat", timeout_seconds=900)
        if result["status"]!="pass" or not exe.is_file():
            raise RuntimeError("Lab preview build failed: "+result["output_tail"])
        receipt.write_text(f"source_sha256={source_hash}\n"
                           f"dll_sha256={digest(dll)}\nexe_sha256={digest(exe)}\n")
    return dll,exe


def scene(kind: str, destination: Path, *, context: bool=False) -> None:
    marker=MARKERS[kind]
    selected={(30,30)} if context else set(POSITIONS)
    neighbors={(29,29),(29,31),(31,29),(31,31)} if context else set()
    neighbor_type=7 if kind=="forest" else 8
    rows=[f"{x},{y},2,{10 if (x,y) in selected else neighbor_type if (x,y) in neighbors else 2},0,"
          f"{marker if (x,y) in selected else 0},0"
          for y in range(64) for x in range(y%2,64,2)]
    destination.write_text(f"C3X_BIQ_TERRAIN_V3,64,64,{len(rows)}\n"+
                           "\n".join(rows)+"\n")


def render(kind: str, dll: Path, exe: Path, *, context: bool=False,
           target_name: str | None=None) -> Path:
    from PIL import Image
    target=OUT/(target_name or (kind+"-context" if context else kind))
    target.mkdir(parents=True,exist_ok=True)
    native_run=platform.run_native_fixture
    original_scene=renderer.scene

    def isolated(directory: Path, command: str, run_id: str):
        batch=directory/"render.bat"
        body=batch.read_text()
        body=replace_once(body,'C3XRenderer.dll" ..\\.. ',
                          'C3XRenderer.dll" ..\\lab\\out\\volcanoes\\slot-study\\root ')
        if not context:
            body=replace_once(body,"640 480 32 32 128 12","1280 800 32 32 128 12")
        batch.write_text(body)
        return native_run(directory,command,run_id)

    try:
        platform.run_native_fixture=isolated
        renderer.scene=lambda _category,_case,destination,**_:scene(kind,destination,context=context)
        record=renderer.native_render("volcanoes","detail",12,224 if context else 128,target,
            center=(30,30) if context else (32,32),candidate=dll,diagnostics=True,preview=exe)
    finally:
        platform.run_native_fixture=native_run
        renderer.scene=original_scene
    bitmap=ROOT/record["image"]
    image=target/"preview.png"
    Image.open(bitmap).convert("RGB").save(image)
    bitmap.unlink()
    (target/"capture.txt").write_text(
        f"dll_sha256={digest(dll)}\nscene_sha256={digest(target/'scene.csv')}\n"
        f"image_sha256={digest(image)}\nrenderer=native D3D11 Lab preview\n"
        "fallback=0\n")
    return image


def sheets() -> None:
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.load_default(size=13)
    for kind in MARKERS:
        frame=Image.open(OUT/kind/"preview.png").convert("RGB")
        sheet=Image.new("RGB",(960,720),"#20252d")
        draw=ImageDraw.Draw(sheet)
        for index,(x,y) in enumerate(POSITIONS):
            col,row=index%4,index//4
            cx=64*(x-32)+640
            cy=32*(y-32)+368
            sheet.paste(frame.crop((cx-120,cy-72,cx+120,cy+73)),
                        (col*240,row*180+32))
            draw.text((col*240+7,row*180+7),
                f"{index:02d} {FAMILIES[index//2]}",fill="white",font=font)
        sheet.save(OUT/f"{kind}-slots.png")
    comparison=Image.new("RGB",(2880,765),"#20252d")
    labels=ImageDraw.Draw(comparison)
    for index,kind in enumerate(MARKERS):
        comparison.paste(Image.open(OUT/f"{kind}-slots.png").convert("RGB"),
                         (index*960,45))
        labels.text((index*960+10,12),kind.title()+" · slots 00–15",
                    fill="white",font=ImageFont.load_default(size=19))
    comparison.save(OUT/"all-treatments.png")
    if all((OUT/(kind+"-context")/"preview.png").is_file()
           for kind in ("forest","jungle")):
        context_sheet=Image.new("RGB",(1280,520),"#20252d")
        labels=ImageDraw.Draw(context_sheet)
        for index,kind in enumerate(("forest","jungle")):
            context_sheet.paste(Image.open(OUT/(kind+"-context")/"preview.png").convert("RGB"),
                                (index*640,40))
            labels.text((index*640+12,12),
                        f"Slot 05 · {kind} neighbors and light foot ring",
                        fill="white",font=ImageFont.load_default(size=17))
        context_sheet.save(OUT/"neighbor-contexts.png")


def orientation_probe() -> None:
    """Compare all source-field orientations for the two B silhouettes in one build."""
    from PIL import Image, ImageDraw, ImageFont
    prepare()
    terrain_path=PRIVATE/TERRAIN
    relief_path=PRIVATE/RELIEF
    terrain=terrain_path.read_text()
    original_relief=relief_path.read_text()
    try:
        terrain_path.write_text(replace_once(terrain,
            "return (slot==5u || slot==15u)?7u:(slot&1u)?5u:0u;",
            "return volcano_slot(raw_x,raw_y)&7u;"))
        relief=replace_once(original_relief,"unsigned family=slot>>1;",
                            "unsigned family=slot<8u?2u:7u;")
        relief_path.write_text(replace_once(relief,
            "unsigned field=(slot*3u)%5u;","unsigned field=0u;"))
        dll,exe=build()
        frame=Image.open(render("bare",dll,exe,
                                target_name="orientation-probe-capture")).convert("RGB")
        sheet=Image.new("RGB",(960,720),"#20252d")
        draw=ImageDraw.Draw(sheet)
        font=ImageFont.load_default(size=13)
        for index,(x,y) in enumerate(POSITIONS):
            col,row=index%4,index//4
            cx=64*(x-32)+640
            cy=32*(y-32)+368
            sheet.paste(frame.crop((cx-120,cy-72,cx+120,cy+73)),
                        (col*240,row*180+32))
            draw.text((col*240+7,row*180+7),
                      f"{'Broad B' if index<8 else 'Eroded B'} · orientation {index%8}",
                      fill="white",font=font)
        sheet.save(OUT/"orientation-probe.png")
        print(OUT/"orientation-probe.png")
    finally:
        terrain_path.write_text(terrain)
        relief_path.write_text(original_relief)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets-only",action="store_true")
    parser.add_argument("--orientation-probe",action="store_true")
    args=parser.parse_args()
    assert len(POSITIONS)==16 and len(set(POSITIONS))==16
    if args.orientation_probe:
        orientation_probe();return
    if args.sheets_only:
        sheets();return
    prepare()
    dll,exe=build()
    for kind in MARKERS:
        render(kind,dll,exe)
    for kind in ("forest","jungle"):
        render(kind,dll,exe,context=True)
    sheets()
    print(OUT/"all-treatments.png")


if __name__=="__main__":
    main()
