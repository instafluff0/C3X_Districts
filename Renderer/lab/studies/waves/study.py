"""Reproducible wave-source audit and isolated production-D3D surf experiment.

Run with a Python containing numpy/Pillow. No production shader/pack is edited.
Snapshot assets are independent copies; generated licensed files remain ignored.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import uuid

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from Renderer import renderer
from Renderer.lab.platform import run_native_fixture
from Renderer.tools.asset_compiler import wave_blp_extractor as waves
from Renderer.tools.asset_compiler.c3x_asset_compiler import make_dds_dx10_header, extract_civbig_to_dds

OUT = ROOT / "Renderer/lab/out/waves"
SNAPSHOT = OUT / "snapshot"
STUDY = Path(__file__).parent


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def source_images():
    report = waves.extract(output=OUT / "source")
    images = {}
    for role, data in report["textures"].items():
        dds = (OUT / "source" / data["texture"]).read_bytes()
        images[role] = Image.frombytes("RGBA", (data["width"], data["height"]),
                                      dds[148:148+data["width"]*data["height"]*4])
        images[role].save(OUT / "source" / (role + ".png"))
        panel = Image.new("RGB", (1024, 292), "#17222c")
        draw = ImageDraw.Draw(panel)
        for channel in range(4):
            tile = images[role].getchannel(channel)
            tile.thumbnail((256, 256))
            panel.paste(tile, (channel*256, 30))
            draw.text((channel*256+8, 8), role + " / " + "RGBA"[channel], fill="white")
        panel.save(OUT / (role + "-channels.png"))
    delays = np.frombuffer((OUT / "source/crest-delays.f32").read_bytes(), "<f4").reshape(16,512)
    atlas = np.asarray(images["crest"])
    sheet = Image.new("RGB", (1024, 616), "#142935")
    draw = ImageDraw.Draw(sheet)
    for page in range(16):
        x,y=page%8*128,page//8*512
        tile=atlas[y:y+512,x:x+128]
        # Rotate only for display: source pages are 128 across by 512 along.
        mask=Image.fromarray(tile[:,:,0]).rotate(90,expand=True)
        mask.thumbnail((240,70))
        px,py=(page%4)*256+8,(page//4)*148+28
        color=Image.new("RGB",mask.size,"#ecf4f7")
        sheet.paste(color,(px,py),mask)
        draw.text((px,py-20),f"Crest {page:02} / {(delays[page]<1e30).sum()} active samples",fill="white")
    sheet.save(OUT / "all-crests.png")
    stats = {}
    for name,im in images.items():
        stats[name] = {"RGBA"[i]:dict(min=int(a.min()),max=int(a.max()),std=float(a.std()))
            for i,a in enumerate(np.asarray(im).transpose(2,0,1))}
    (OUT / "channel-statistics.json").write_text(json.dumps(stats,indent=2)+"\n")
    names=("FX_Wave_Crash_Foam", "FireFX_Rock_Wake_Foam", "FX_LakeFoam", "FX_LakeFoam_a02",
           "FX_Crater_Waves", "FX_Crater_Waves2", "FX_Crater_Waves_a01", "FX_Crater_Waves_a02",
           "FX_SeaFoam1_a", "FX_SeaFoam2_a", "FX_WaterFoam06", "FXt_Wave_Flood")
    related=OUT/"source/related"
    related.mkdir(exist_ok=True)
    sheet=Image.new("RGB",(1024,len(names)*120),"#17222c")
    draw=ImageDraw.Draw(sheet)
    records=[]
    for row,name in enumerate(names):
        source=sorted(waves.DEFAULT_ASSETS.rglob("TEXTURE_"+name))[0]
        target=related/(name+".dds")
        info=extract_civbig_to_dds(source,target)
        info={k:v for k,v in info.items() if k not in ("source","output")}
        records.append(dict(name=name,source=source.relative_to(waves.DEFAULT_ASSETS).as_posix(),metadata=info))
        data=bytearray(target.read_bytes())
        fmt=struct.unpack_from("<I",data,128)[0]
        # Pillow lacks the sRGB aliases for BC1/2/3. Decode identical block
        # bytes through the linear alias for channel inspection only.
        struct.pack_into("<I",data,128,{72:71,75:74,78:77}.get(fmt,fmt))
        im=Image.open(io.BytesIO(data)).convert("RGBA")
        im.save(related/(name+".png"))
        draw.text((8,row*120+4),name,fill="white")
        for channel in range(4):
            tile=im.getchannel(channel);tile.thumbnail((232,90))
            sheet.paste(tile,(channel*256,row*120+26))
            draw.text((channel*256+235,row*120+26),"RGBA"[channel],fill="white")
    sheet.save(OUT/"related-art.png")
    (related/"inventory.json").write_text(json.dumps(records,indent=2)+"\n")
    return images, delays


def rg16(image):
    h,w,_=image.shape
    return make_dds_dx10_header(dict(width=w,height=h,mip_count=1,dxgi_format=35)) + np.round(np.clip(image,0,1)*65535).astype("<u2").tobytes()


def prepare():
    images,delays=source_images()
    # Copy only runtime shader inputs and packs. Keep the snapshot immutable
    # except for the two study texture channels and the study hydrology file.
    for source in (ROOT/"Renderer/native").rglob("*"):
        if source.is_file() and source.suffix in (".hlsl", ".h", ".cso"):
            target=SNAPSHOT/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    names=set()
    for config in ("default.custom_rendering.txt", "custom.custom_rendering.txt"):
        path=ROOT/"Renderer"/config
        if path.exists():
            text=path.read_text()
            names.update(re.findall(r"path = mod:Renderer\\packs\\([^\s]+)",text))
            write(SNAPSHOT/"Renderer"/config,text.encode())
    names.update(("NaturalFidelityRuntime","CityCompositionRuntime","TerrainProfileR1"))
    names.update(name.decode() for name in re.findall(rb"Renderer/packs/([A-Za-z0-9_]+)/",
                 (ROOT/"Renderer/packs/CityCompositionRuntime/city.bin").read_bytes()))
    for name in sorted(names):
        source=ROOT/"Renderer/packs"/name
        target=SNAPSHOT/"Renderer/packs"/name
        if not target.exists():
            shutil.copytree(source,target)
    # Two otherwise unused LEAN variance slots in the natural-water branch
    # carry source-derived scalar masks solely in this isolated evaluation.
    atlas=np.asarray(images["crest"])[:,:,0]/255.
    delay_map=np.ones_like(atlas)
    for page in range(16):
        x,y=page%8*128,page//8*512
        delay_map[y:y+512,x:x+128]=np.minimum(delays[page],1)[:,None]
    crest=rg16(np.stack((atlas,delay_map),axis=-1))
    auxiliary=np.asarray(images["auxiliary"])[:,:,:3].mean(axis=2)/255.
    aux=rg16(np.stack((auxiliary,np.ones_like(auxiliary)),axis=-1))
    for name in ("TerrainNormalized","Civ5EnvironmentSkin"):
        target=SNAPSHOT/"Renderer/packs"/name/"textures/water/surface"
        if target.exists():
            write(target/"large_lean1.dds",crest)
            write(target/"small_lean1.dds",aux)
    candidate=ROOT/"Renderer/native/build/candidate/C3XRenderer.dll"
    write(OUT/"C3XRenderer.dll",candidate.read_bytes())
    write(OUT/"native_preview.exe",(ROOT/"Renderer/lab/.cache/native_preview.exe").read_bytes())
    # Freeze the generated current shader as the control. Changes are injected
    # into its natural-water branch, before native overlays would be composed.
    write(OUT/"control.hlsl",(ROOT/"Renderer/native/city_fidelity/hydrology.hlsl").read_bytes())
    (OUT/"snapshot.json").write_text(json.dumps(dict(candidate_sha256=waves.digest(candidate.read_bytes()),
        preview_sha256=waves.digest((OUT/"native_preview.exe").read_bytes()),
        control_shader_sha256=waves.digest((OUT/"control.hlsl").read_bytes()),
        source_shader="Renderer/native/city_fidelity/hydrology.hlsl",packs=sorted(names),
        status="isolated experiment; not staged; source channel packing is study-only"),indent=2)+"\n")


def render(case="lowland", time=0, enabled=True, hour=12, zoom=128, width=960,height=640):
    if not 0 <= time <= 120:
        raise ValueError("Study time must be within its 120-second texture-carried clock")
    tag=f"{case}-{'waves' if enabled else 'control'}-t{time:g}-h{hour}-z{zoom}"
    output=OUT/"native"/tag
    output.mkdir(parents=True,exist_ok=True)
    renderer.scene("shorelines",case,output/"scene.csv")
    shader=(OUT/"control.hlsl").read_text()
    if enabled:
        # A study-only unused G texel carries absolute phase. This avoids a new
        # shader compilation for every sample without modifying the game DLL.
        # It is not a proposed production time transport or animation scheduler.
        for name in ("TerrainNormalized", "Civ5EnvironmentSkin"):
            path=SNAPSHOT/"Renderer/packs"/name/"textures/water/surface/small_lean1.dds"
            if path.exists():
                data=bytearray(path.read_bytes())
                struct.pack_into("<H",data,150,round(time/120*65535))
                write(path,data)
        fragment="#define STUDY_TIME (water_small_lean1_texture.Load(int3(0,0,0)).g*120.0)\n"+(STUDY/"surf.hlsl").read_text()+"\n"
        marker="float4 q3_natural_water(PixelInput input) {"
        if shader.count(marker)!=1:
            raise ValueError("Current natural-water shader insertion point changed")
        shader=shader.replace(marker,fragment+marker)
        marker="return float4(premult/max(alpha,.0001),alpha);"
        if shader.count(marker)!=1:
            raise ValueError("Current natural-water output changed")
        shader=shader.replace(marker,"return study_surf(input,float4(premult/max(alpha,.0001),alpha));")
    write(SNAPSHOT/"Renderer/native/city_fidelity/hydrology.hlsl",shader.encode())
    def win(path):
        return "..\\..\\"+path.relative_to(ROOT).as_posix().replace("/","\\")
    run_id=uuid.uuid4().hex
    image=output/"frame.bmp"
    env={"C3X_RENDERER_VISUAL_PROFILE":"", "C3X_RENDERER_PREVIEW_OBJECTS":"",
         "C3X_RENDERER_PREVIEW_REPLAY":"", "C3X_RENDERER_PREVIEW_EDITS":"",
         "C3X_RENDERER_PREVIEW_ANIMATION":"", "C3X_RENDERER_PREVIEW_UNITS":"",
         "C3X_RENDERER_TRACE":"2", "C3X_RENDERER_TRACE_FILE":win(output/"trace.log"),
         "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS":win(SNAPSHOT/"Renderer/custom.custom_rendering.txt"),
         "C3X_LAB_PID_FILE":win(output/"process.txt"), "C3X_LAB_RUN_ID":run_id}
    command="\n".join(f'set "{k}={v}"' for k,v in env.items())
    command+=f'\n"{win(OUT/"native_preview.exe")}" "{win(OUT/"C3XRenderer.dll")}" "{win(SNAPSHOT)}" "{win(SNAPSHOT/"Renderer/default.custom_rendering.txt")}" "{win(output/"scene.csv")}" "{win(image)}" {width} {height} 13 17 {zoom} {hour}'
    batch=output/"render.bat"
    batch.write_text("@echo off\nsetlocal\n"+command+f' > "{win(output/"native.log")}" 2>&1\nset "C3X_LAB_EXIT=%errorlevel%"\n> "{win(output/"completion.txt")}" echo {run_id} %C3X_LAB_EXIT%\nexit /b %C3X_LAB_EXIT%\n')
    result=run_native_fixture(output,f'call "{win(batch)}"',run_id)
    if result["status"]!="pass" or "0 fallback" not in result["output_tail"]:
        raise ValueError(str(result))
    Image.open(image).save(output/"frame.png")
    return image


def sequence(case="gameplay"):
    frames=[]
    receipts=[]
    for time in range(0,30,2):
        print(f"Native wave motion sample {time}/28 seconds",flush=True)
        path=render(case,time)
        frame=Image.open(path).convert("RGB")
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,960,25),fill="#17222c")
        draw.text((10,7),f"ISOLATED NATIVE WAVE STUDY | t={time:02}s | 2-second samples, 4x playback",fill="white")
        frames.append(frame)
        receipts.append(dict(time=time,pixels_sha256=waves.digest(Image.open(path).tobytes())))
    frames[0].save(OUT/(case+"-motion.gif"),save_all=True,append_images=frames[1:],
                   duration=500,loop=0,optimize=False)
    (OUT/(case+"-motion.json")).write_text(json.dumps(receipts,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("source","prepare","render","sequence"))
    parser.add_argument("--case",default="lowland",choices=("lowland","gameplay","detail"))
    parser.add_argument("--time",type=float,default=0)
    parser.add_argument("--off",action="store_true")
    parser.add_argument("--hour",type=int,default=12)
    parser.add_argument("--zoom",type=int,default=128)
    args=parser.parse_args()
    if args.action=="source":source_images()
    elif args.action=="prepare":prepare()
    elif args.action=="sequence":sequence(args.case)
    else:print(render(args.case,args.time,not args.off,args.hour,args.zoom).relative_to(ROOT))
