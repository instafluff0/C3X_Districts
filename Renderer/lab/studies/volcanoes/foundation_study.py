#!/usr/bin/env python3
"""Compare ordinary volcanoes with authored mountain-field foundations in Lab.

This is C3X art exploration, not a reconstruction of Civ VI's composition rule.
The private native build starts from the prior lava-free orientation study and
hard-links its read-only packs. No production source or staged binary is changed.
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
from Renderer.lab import platform
from Renderer.lab.studies.volcanoes import variety_study as base

OUT = ROOT / "Renderer/lab/out/volcanoes/foundation-study"
PRIVATE = OUT / "root"
SEED = ROOT / "Renderer/lab/out/volcanoes/variety-study/root"
TERRAIN = Path("Renderer/native/render_core/terrain_query.h")
RELIEF = Path("Renderer/native/render_core/relief_query.h")
ADAPTER = Path("Renderer/lab/shared/natural/relief.h")
RENDERER = Path("Renderer/native/c3x_renderer.cpp")
POSITIONS = base.POSITIONS
NAMES = (
    "Current cone, no lava",
    "Broad caldera study",
    "Steep asymmetric study",
    "Ridge complex study",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one frozen-source expression: {old[:75]}")
    return source.replace(old, new)


def prepare_private() -> None:
    if not (SEED / RELIEF).is_file() or not (SEED / ADAPTER).is_file():
        raise RuntimeError("Run variety_study.py first to retain its frozen source")
    if not PRIVATE.exists():
        (PRIVATE / "Renderer").mkdir(parents=True)
        shutil.copytree(SEED / "Renderer/native", PRIVATE / "Renderer/native",
                        ignore=shutil.ignore_patterns("build", "*.obj", "*.ilk", "*.pdb"))
        shutil.copytree(SEED / "Renderer/lab/shared", PRIVATE / "Renderer/lab/shared")
        for name in ("default.custom_rendering.txt", "custom.custom_rendering.txt"):
            shutil.copy2(SEED / "Renderer" / name, PRIVATE / "Renderer" / name)
        # Existing normalized art stays linked read-only; do not copy gigabytes.
        shutil.copytree(SEED / "Renderer/packs", PRIVATE / "Renderer/packs",
                        copy_function=os.link)
    elif not (OUT / "inputs.txt").is_file():
        raise RuntimeError("Partial private build exists; inspect before replacing it")

    lab = PRIVATE / "Renderer/lab"
    lab.mkdir(parents=True, exist_ok=True)
    for name in ("native_preview.cpp", "build_native_preview.bat", "volcano_witness.h"):
        shutil.copy2(ROOT / "Renderer/lab" / name, lab / name)

    source = (SEED / TERRAIN).read_text()
    source = replace_once(source,
        "inline unsigned volcano_orientation(int raw_x,int raw_y) {\n"
        "    return hash(std::uint32_t(raw_x)*73856093u ^ std::uint32_t(raw_y)*19349663u)&7u;\n"
        "}",
        """inline unsigned volcano_family(int raw_x,int raw_y) {
    auto seed=hash(std::uint32_t(raw_x)*73856093u ^
                   std::uint32_t(raw_y)*19349663u);
    return (seed>>8)&3u;
}
inline unsigned volcano_orientation(int raw_x,int raw_y) {
    if(volcano_family(raw_x,raw_y)==0)return 0;
    return hash(std::uint32_t(raw_x)*73856093u ^
                std::uint32_t(raw_y)*19349663u)&7u;
}""")
    (PRIVATE / TERRAIN).write_text(source)

    source = (SEED / ADAPTER).read_text()
    source = replace_once(source,
        "                if(fidelity_profile && (kind==5 || kind==6))return 0.f; // replaced exact natural providers",
        "                if(fidelity_profile && (kind==5 || kind==6))return 0.f; // replaced exact natural providers\n"
        "                if(kind==16)kind=6; // Lab-only mountain foundation of an ordinary volcano")
    (PRIVATE / ADAPTER).write_text(source)

    source = (SEED / RELIEF).read_text()
    source = replace_once(source,
        "        result.blend=source(tile.real,variant,1,u,v)*edge;\n        return result;",
        """        result.blend=source(tile.real,variant,1,u,v)*edge;
        if(tile.real==10) {
            // Keep one exact current cone. Other families reuse the five
            // authored mountain fields beneath the same volcano crater.
            unsigned family=volcano_family(coordinate[0],coordinate[1]);
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
                unsigned field=(seed>>4)%5;
                if(family==1) {
                    auto broad=foundation(field,.04f,-.03f,.36f,.72f);
                    result.height=std::max(result.height*.60f,broad[0]*.68f*gate);
                    result.blend=std::max(result.blend,broad[1]*.86f*gate);
                } else if(family==2) {
                    auto steep=foundation(field,.15f,.03f,.52f,1.12f);
                    result.height=std::max(result.height,steep[0]*.88f*gate);
                    result.blend=std::max(result.blend,steep[1]*.82f*gate);
                } else {
                    auto west=foundation(field,-.14f,.07f,.43f,.84f);
                    auto east=foundation((field+2)%5,.13f,-.09f,.45f,.88f);
                    result.height=std::max({result.height*.91f,
                        west[0]*.76f*gate,east[0]*.62f*gate});
                    result.blend=std::max({result.blend,
                        west[1]*.75f*gate,east[1]*.60f*gate});
                }
            }
        }
        return result;""")
    (PRIVATE / RELIEF).write_text(source)
    # The frozen renderer expects city and farm packs in every preview, even
    # though this isolated grassland fixture uses neither. Those unrelated
    # source packs may change while this study is retained. Keep their load
    # attempt, but gate readiness only on the assets this fixture can draw.
    source = (SEED / RENDERER).read_text()
    source = replace_once(source,
        "!route_assets_ready || !resource_assets_ready || !city_assets_ready ||\n"
        "                !mine_assets_ready || !farm_assets_ready)",
        "!route_assets_ready || !resource_assets_ready)")
    (PRIVATE / RENDERER).write_text(source)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs.txt").write_text(
        "Lab-only source-derived mountain foundations; no lava or smoke.\n"
        f"seed_relief_sha256={digest(SEED / RELIEF)}\n"
        f"seed_adapter_sha256={digest(SEED / ADAPTER)}\n"
        f"private_terrain_sha256={digest(PRIVATE / TERRAIN)}\n"
        f"private_relief_sha256={digest(PRIVATE / RELIEF)}\n"
        f"private_adapter_sha256={digest(PRIVATE / ADAPTER)}\n"
        f"private_renderer_sha256={digest(PRIVATE / RENDERER)}\n")


def build_private() -> Path:
    dll = PRIVATE / "Renderer/native/build/candidate/C3XRenderer.dll"
    source_hash = hashlib.sha256("".join(digest(PRIVATE / path)
                                    for path in (TERRAIN, RELIEF, ADAPTER, RENDERER)).encode()).hexdigest()
    receipt = OUT / "private-build.txt"
    if dll.is_file() and receipt.is_file():
        record = receipt.read_text()
        if f"source_sha256={source_hash}" in record and f"dll_sha256={digest(dll)}" in record:
            return dll
    result = platform.native_command_result(
        "Renderer/lab/out/volcanoes/foundation-study/root/Renderer/native",
        "call BUILD.bat candidate-compile", timeout_seconds=900)
    if result["status"] != "pass" or not dll.is_file():
        raise RuntimeError("Private native build failed: " + result["output_tail"])
    receipt.write_text(f"source_sha256={source_hash}\ndll_sha256={digest(dll)}\n")
    return dll


def build_private_preview() -> Path:
    preview = PRIVATE / "Renderer/lab/.cache/native_preview.exe"
    receipt = OUT / "private-preview.txt"
    source_hash = hashlib.sha256((digest(PRIVATE / "Renderer/native/biq_preview.cpp") +
                                  digest(PRIVATE / "Renderer/lab/native_preview.cpp")).encode()).hexdigest()
    if preview.is_file() and receipt.is_file():
        record = receipt.read_text()
        if f"source_sha256={source_hash}" in record and f"exe_sha256={digest(preview)}" in record:
            return preview
    result = platform.native_command_result(
        "Renderer/lab/out/volcanoes/foundation-study/root/Renderer/lab",
        "call build_native_preview.bat", timeout_seconds=900)
    if result["status"] != "pass" or not preview.is_file():
        raise RuntimeError("Private Lab preview build failed: " + result["output_tail"])
    receipt.write_text(f"source_sha256={source_hash}\nexe_sha256={digest(preview)}\n")
    return preview


def render(name: str, dll: Path, *, center: tuple[int,int]=(32,32), atlas: bool=False) -> Path:
    preview = build_private_preview()
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    native_run = platform.run_native_fixture
    original_scene = renderer.scene

    def isolated(directory: Path, command: str, run_id: str):
        batch = directory / "render.bat"
        body = batch.read_text()
        body = replace_once(body, 'C3XRenderer.dll" ..\\.. ',
                            'C3XRenderer.dll" ..\\lab\\out\\volcanoes\\foundation-study\\root ')
        if atlas:
            body = replace_once(body, "640 480 32 32 128 12", "1280 800 32 32 128 12")
        batch.write_text(body)
        return native_run(directory, command, run_id)

    def single_scene(_category: str, _case: str, destination: Path, *, world_size: int = 32):
        size = 64
        rows = [f"{x},{y},2,{10 if (x,y)==center else 2},0,0,0"
                for y in range(size) for x in range(y % 2, size, 2)]
        destination.write_text(f"C3X_BIQ_TERRAIN_V3,{size},{size},{len(rows)}\n" +
                               "\n".join(rows) + "\n")

    try:
        platform.run_native_fixture = isolated
        renderer.scene = base.variety_scene if atlas else single_scene
        record = renderer.native_render("volcanoes", "detail", 12,
                                        128 if atlas else 224, target,
                                        center=center, candidate=dll, diagnostics=True,
                                        preview=preview)
    finally:
        platform.run_native_fixture = native_run
        renderer.scene = original_scene
    from PIL import Image
    output = target / "preview.png"
    bitmap = ROOT / record["image"]
    Image.open(bitmap).convert("RGB").save(output)
    bitmap.unlink()  # The PNG and capture receipt retain the visual evidence.
    (target / "capture.txt").write_text(
        f"dll_sha256={digest(dll)}\nscene_sha256={digest(target / 'scene.csv')}\n"
        f"image_sha256={digest(output)}\nrenderer=native D3D11 Lab preview\n"
        f"center={center[0]},{center[1]}\nfallback=0\n")
    return output


def sheets() -> None:
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    frame = Image.open(OUT / "atlas/preview.png").convert("RGB")
    atlas = Image.new("RGB", (960, 720), "#20252d")
    draw = ImageDraw.Draw(atlas)
    for index,(x,y) in enumerate(POSITIONS):
        family = family_for(x,y)
        col,row=index%4,index//4
        cx=64*(x-32)+640
        cy=32*(y-32)+368
        atlas.paste(frame.crop((cx-120,cy-72,cx+120,cy+73)),
                    (col*240,row*180+32))
        draw.text((col*240+7,row*180+6),f"{NAMES[family]} | {x},{y}",
                  fill="white",font=small)
    atlas.save(OUT / "grassland-16.png")

    detail_paths=[OUT / f"detail-{family}/preview.png" for family in range(4)]
    if all(path.is_file() for path in detail_paths):
        close = Image.new("RGB", (960, 590), "#20252d")
        labels = ImageDraw.Draw(close)
        for family,path in enumerate(detail_paths):
            col,row=family%2,family//2
            close.paste(Image.open(path).convert("RGB").crop((80,40,560,300)),
                        (col*480,row*295+35))
            labels.text((col*480+12,row*295+11),NAMES[family],fill="white",font=font)
        close.save(OUT / "four-forms-closeup.png")

    previous=ROOT / "Renderer/lab/out/volcanoes/variety-study/proposal/grassland-variety.png"
    if previous.is_file():
        old=Image.open(previous).convert("RGB")
        comparison=Image.new("RGB",(1920,762),"#20252d")
        comparison.paste(old,(0,42));comparison.paste(atlas,(960,42))
        labels=ImageDraw.Draw(comparison)
        labels.text((12,12),"Orientation-only Lab study",fill="white",font=font)
        labels.text((972,12),"Mountain-foundation Lab study",fill="white",font=font)
        comparison.save(OUT / "comparison.png")


def family_for(x: int,y: int) -> int:
    seed=((x+y)*73856093 ^ (x-y)*19349663)&0xffffffff
    seed ^= seed>>16;seed=seed*0x7feb352d&0xffffffff
    seed ^= seed>>15;seed=seed*0x846ca68b&0xffffffff
    seed ^= seed>>16
    return seed>>8&3


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets-only",action="store_true")
    parser.add_argument("--atlas-only",action="store_true")
    args=parser.parse_args()
    assert {family_for(x,y) for x,y in POSITIONS}==set(range(4))
    if args.sheets_only:
        sheets();return
    prepare_private()
    dll=build_private()
    render("atlas",dll,atlas=True)
    if not args.atlas_only:
        for family in range(4):
            center={0:(30,26),1:(26,30),2:(30,30),3:(38,26)}[family]
            assert family_for(*center)==family
            render(f"detail-{family}",dll,center=center)
    sheets()
    print(OUT / "grassland-16.png")
    if not args.atlas_only: print(OUT / "four-forms-closeup.png")


if __name__=="__main__":
    main()
