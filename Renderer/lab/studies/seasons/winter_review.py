#!/usr/bin/env python3
"""Bounded Winter Lab before/after review and five-biome color measurements."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/"Renderer/lab/out/seasons/programmatic/winter-focus"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wonderland",action="store_true",help="Compare the preceding frost pass with the current layered winter")
    args=parser.parse_args()
    try:
        from PIL import Image,ImageDraw,ImageFont
        import numpy as np
    except ImportError:
        python=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
        os.execv(str(python),[str(python),str(Path(__file__)),*sys.argv[1:]])
    previous="exposure-decals.png" if args.wonderland else "before.png"
    current="wonderland.png" if args.wonderland else "exposure-decals.png"
    suffix="wonderland-" if args.wonderland else ""
    receipt_label="wonderland" if args.wonderland else "exposure-decals"
    before=Image.open(OUT/"test-biq"/previous).convert("RGB")
    after=Image.open(OUT/"test-biq"/current).convert("RGB")
    font=ImageFont.load_default(size=22)
    sheet=Image.new("RGB",(1600,492),(19,29,38));draw=ImageDraw.Draw(sheet)
    for i,(image,label) in enumerate(((before,"Before: preceding winter prototype"),(after,"After: layered snow on the existing landscape"))):
        draw.text((i*800+12,10),label,font=font,fill="white")
        sheet.paste(image.resize((800,450),Image.Resampling.LANCZOS),(i*800,42))
    sheet.save(OUT/(suffix+"before-after.png"),optimize=True)
    regions=(("Terrain grain",(420,500,680,720)),("Tree frost",(800,315,1010,500)),("Rock relief",(545,180,740,420)))
    details=Image.new("RGB",(1440,596),(19,29,38));draw=ImageDraw.Draw(details)
    for i,(label,box) in enumerate(regions):
        for row,(image,caption) in enumerate(((before,"Before"),(after,"After"))):
            draw.text((i*480+10,row*298+7),caption+" | "+label,font=font,fill="white")
            crop=image.crop(box);crop.thumbnail((460,254),Image.Resampling.LANCZOS)
            # Enlarge only the comparison; source/target scene PNGs are untouched.
            ratio=min(460/crop.width,254/crop.height)
            crop=crop.resize((round(crop.width*ratio),round(crop.height*ratio)),Image.Resampling.LANCZOS)
            details.paste(crop,(i*480+(480-crop.width)//2,row*298+38))
    details.save(OUT/(suffix+"details.png"),optimize=True)
    biomes=("grassland","plains","desert","floodplains","tundra")
    image=Image.open(OUT/"biomes"/current).convert("RGB");pixels=np.array(image,dtype=float)/255
    colors=[];boxes=[]
    for i in range(5):
        center=int(image.width*(i+.5)/5);box=(center-64,360,center+64,460);boxes.append(list(box))
        colors.append(np.median(pixels[box[1]:box[3],box[0]:box[2]].reshape(-1,3),axis=0))
    colors=np.array(colors);linear=np.where(colors<=.04045,colors/12.92,((colors+.055)/1.055)**2.4)
    xyz=linear@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
    xyz/=np.array([.95047,1,1.08883]);f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29)
    lab=np.stack([116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])],axis=1)
    distances={biomes[i]+" / "+biomes[j]:round(float(np.linalg.norm(lab[i]-lab[j])),2) for i in range(5) for j in range(i+1,5)}
    report={"schema":"c3x.lab.winter_visual_diagnostics.v1","pack":"Civ5EnvironmentSkin",
        "profile":receipt_label,
        "median_srgb":dict(zip(biomes,[[round(float(c)*255,1) for c in row] for row in colors])),
        "crop_boxes":dict(zip(biomes,boxes)),"cie76_distances":distances,
        "interpretation":"Measured noon flat-strip color separation, not a subjective readability or production-parity pass.",
        "checks":{}}
    for case in ("test-biq","biomes"):
        receipt=json.loads((OUT/f"{case}-{receipt_label}-receipt.json").read_text());row=receipt["packs"][0]
        report["checks"][case]={"summer_pixel_exact":row["preceding_summer_pixel_exact"],"gpu_checks":row["gpu_checks"],
            "image_sha256":hashlib.sha256((ROOT/row["images"]["winter"]["path"]).read_bytes()).hexdigest()}
    (OUT/(suffix+"diagnostics.json")).write_text(json.dumps(report,indent=2)+"\n")
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Winter: Civ V skin, existing meshes</title><style>
body{margin:24px;background:#131d26;color:#e5edf5;font:16px system-ui;line-height:1.6}a{color:#96d6ff}
img{max-width:100%;height:auto}button{padding:9px;margin:5px;background:#294253;color:white;border:1px solid #658da4;border-radius:6px}
.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:#203240;padding:12px}.note{color:#b4c7d6}#current{width:1600px}
@media(max-width:800px){.views{grid-template-columns:1fr}}</style>
<h1>Winter on the Civ V skin</h1><p>Existing terrain and tree meshes, placements, opacity and relief remain intact.
Snow uses retained texture contrast, additive relief, source snow decals and small Blender-baked exposure masks.</p>
<p id="layerNote">Current layered winter: localized drifts, height-based snow-decal relief, source substrate contrast,
stronger canopy shelter, softer plane-correct shadows and source water slope textures. This is a shader render.</p>
<p class="note">Mac Metal material laboratory. Diagnostic water and shadow composition; no production-parity or visual-acceptance claim.
The source installation was unavailable. Both cases preserve the previous Summer pixels.</p>
<div><button data-scene="test-biq">test.biq scene</button><button data-scene="biomes">Five biomes</button></div>
<p id="biomeNote" hidden>Left to right: grassland, plains, desert, floodplains, tundra.</p>
<div class="views"><div class="panel">Before<a id="beforeLink" href="test-biq/before.png"><img id="before" src="test-biq/before.png"></a></div>
<div class="panel">After<a id="afterLink" href="test-biq/exposure-decals.png"><img id="after" src="test-biq/exposure-decals.png"></a></div></div>
<h2>Texture, frost and rock detail</h2><a href="details.png"><img src="details.png"></a>
<details><summary>Selected icy-blue art direction</summary><p class="note">Image concept, not a shader render. The existing Civ V forest silhouettes are preserved in this study.</p>
<img src="../../scene-concepts/winter-icy-blue.png"></details>
<p><a href="test-biq-exposure-decals-receipt.json">Scene receipt</a> · <a href="biomes-exposure-decals-receipt.json">Biome receipt</a> ·
<a href="diagnostics.json">Measurements</a></p>
<script>document.querySelectorAll('button[data-scene]').forEach(b=>b.addEventListener('click',()=>{const scene=b.dataset.scene;
document.querySelector('#before').src=document.querySelector('#beforeLink').href=scene+'/before.png';
document.querySelector('#after').src=document.querySelector('#afterLink').href=scene+'/exposure-decals.png';
document.querySelector('#biomeNote').hidden=scene!=='biomes';}));</script></html>'''
    html=html.replace("test-biq/before.png","test-biq/"+previous).replace("test-biq/exposure-decals.png","test-biq/"+current)
    html=html.replace("scene+'/before.png'","scene+'/"+previous+"'").replace("scene+'/exposure-decals.png'","scene+'/"+current+"'")
    html=html.replace('href="details.png"','href="'+suffix+'details.png"').replace('src="details.png"','src="'+suffix+'details.png"')
    html=html.replace("-exposure-decals-receipt.json","-"+receipt_label+"-receipt.json").replace('href="diagnostics.json"','href="'+suffix+'diagnostics.json"')
    if not args.wonderland:html=html.replace('id="layerNote"','id="layerNote" hidden')
    (OUT/"index.html").write_text(html)
    print("PASS winter review and diagnostics; minimum measured color distance="+str(min(distances.values())))


if __name__=="__main__":
    main()
