#!/usr/bin/env python3
"""Small autumn comparison gallery; measure rendered biome color and source grain."""
import json
import argparse
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/"Renderer/lab/out/seasons/programmatic"
OUT=BASE/"fall-focus"
PACK="Civ5EnvironmentSkin"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beauty",action="store_true",help="Compare the revised target-led candidate, with its corrected preview baseline")
    parser.add_argument("--target",action="store_true",help="Review crown irradiance/value reskin and source turf patterns; original proportions")
    args=parser.parse_args()
    if args.target:args.beauty=True
    suffix="target-" if args.target else "beauty-" if args.beauty else ""
    current="target.png" if args.target else "beauty.png" if args.beauty else "refined.png"
    preceding="beauty.png" if args.target else "refined.png"
    try:
        from PIL import Image,ImageDraw,ImageFont,ImageFilter
        import numpy as np
    except ImportError:
        python=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
        os.execv(str(python),[str(python),str(Path(__file__)),*sys.argv[1:]])
    before=Image.open(OUT/"test-biq"/preceding if args.beauty else BASE/"test-biq"/PACK/"fall.png").convert("RGB")
    after=Image.open(OUT/"test-biq"/current).convert("RGB")
    font=ImageFont.load_default(size=22)
    sheet=Image.new("RGB",(1600,492),(28,31,30));draw=ImageDraw.Draw(sheet)
    for i,(image,label) in enumerate(((before,"Before: preceding autumn"),(after,"After: golden crowns, quieter ground"))):
        draw.text((i*800+12,10),label,font=font,fill="white")
        sheet.paste(image.resize((800,450),Image.Resampling.LANCZOS),(i*800,42))
    if not args.beauty:sheet.save(OUT/"before-after.png",optimize=True)
    if args.beauty:
        target=Image.open(BASE.parent/"scene-concepts/fall.png").convert("RGB")
        comparison=Image.new("RGB",(1600,492),(28,31,30));draw=ImageDraw.Draw(comparison)
        for i,(image,label) in enumerate(((target,"Selected target"),(after,"Current shader candidate"))):
            draw.text((i*800+12,10),label,font=font,fill="white")
            comparison.paste(image.resize((800,450),Image.Resampling.LANCZOS),(i*800,42))
        comparison.save(OUT/(suffix+"target-comparison.png"),optimize=True)
    regions=(("Retained ground grain",(420,500,680,720)),("Golden foliage and litter",(775,320,1010,515)),("Rock relief and shade",(540,170,745,400)))
    details=Image.new("RGB",(1440,596),(28,31,30));draw=ImageDraw.Draw(details)
    for i,(label,box) in enumerate(regions):
        for row,(image,caption) in enumerate(((before,"Before"),(after,"After"))):
            draw.text((i*480+10,row*298+7),caption+" | "+label,font=font,fill="white")
            crop=image.crop(box);scale=min(460/crop.width,254/crop.height)
            crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
            details.paste(crop,(i*480+(480-crop.width)//2,row*298+38))
    details.save(OUT/(suffix+"details.png"),optimize=True)
    names=("grassland","plains","desert","floodplains","tundra")
    summer=Image.open(OUT/"biomes"/(suffix+"summer.png") if args.beauty else BASE/"biomes"/PACK/"summer.png").convert("L")
    report={"schema":"c3x.lab.autumn_visual_diagnostics.v1","pack":PACK,"profiles":{},"checks":{},
        "interpretation":"Noon flat-strip color and fine-pattern diagnostics; not visual acceptance, night readability or production parity."}
    for profile,path in (("before",OUT/"biomes"/preceding if args.beauty else BASE/"biomes"/PACK/"fall.png"),("refined",OUT/"biomes"/current)):
        image=Image.open(path).convert("RGB");pixels=np.array(image,dtype=float)/255
        colors=[];boxes={};correlations={}
        def grain(image,box):
            crop=image.convert("L").crop(box)
            return (np.array(crop,dtype=float)-np.array(crop.filter(ImageFilter.GaussianBlur(2)),dtype=float)).flatten()
        for i,name in enumerate(names):
            center=int(image.width*(i+.5)/5);box=(center-64,360,center+64,460);boxes[name]=list(box)
            colors.append(np.median(pixels[box[1]:box[3],box[0]:box[2]].reshape(-1,3),axis=0))
            correlations[name]=round(float(np.corrcoef(grain(summer,box),grain(image,box))[0,1]),4)
        colors=np.array(colors);linear=np.where(colors<=.04045,colors/12.92,((colors+.055)/1.055)**2.4)
        xyz=linear@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
        xyz/=np.array([.95047,1,1.08883]);f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29)
        lab=np.stack([116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])],axis=1)
        distances={names[i]+" / "+names[j]:round(float(np.linalg.norm(lab[i]-lab[j])),2) for i in range(5) for j in range(i+1,5)}
        report["profiles"][profile]={"crop_boxes":boxes,"median_srgb":dict(zip(names,colors.round(4).tolist())),
            "cie76_distances":distances,"fine_pattern_correlation_to_summer":correlations}
    for case in ("test-biq","biomes"):
        row=json.loads((OUT/(case+"-target-receipt.json" if args.target else case+"-beauty-receipt.json" if args.beauty else case+"-receipt.json")).read_text())["packs"][0]
        report["checks"][case]={"preceding_summer_pixel_exact":row.get("preceding_summer_pixel_exact"),
            "corrected_preview_baseline":args.beauty,
            "winter":row.get("preceding_winter_comparison"),"gpu_checks":row["gpu_checks"]}
    (OUT/(suffix+"diagnostics.json")).write_text(json.dumps(report,indent=2)+"\n")
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Autumn on the Civ V skin</title><style>
body{margin:24px;background:#1c1f1e;color:#f4efe0;font:16px system-ui;line-height:1.6}a{color:#efd38a}
img{max-width:100%;height:auto}select{padding:8px;margin:4px;background:#343b35;color:white;border:1px solid #948767;border-radius:6px}
.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:#2c322c;padding:12px}.note{color:#c9c4b4}
@media(max-width:800px){.views{grid-template-columns:1fr}}</style>
<h1>Golden autumn on the Civ V skin</h1><p>Stable gold, amber and occasional russet crowns above quieter dry-olive grass and paler honey-straw plains.
Existing tree geometry, bark, leaf detail, terrain texture and rock relief remain intact. Forest-floor decals carry a restrained litter tint.</p>
<p class="note">Actual Mac Metal shader render. Diagnostic water and shadows; no live-game or production-parity claim. All inputs are cached locally.</p>
<label>Scene <select id="scene"><option value="test-biq">test.biq</option><option value="biomes">Five biomes</option></select></label>
<label>Compare <select id="source"><option value="fall">Preceding autumn</option><option value="summer">Summer source</option></select></label>
<p id="bands" hidden>Left to right: grassland, plains, desert, floodplains, tundra. Forests above; relief below.</p>
<div class="views"><div class="panel"><strong id="caption">Before</strong><a id="beforeLink"><img id="before" alt="Comparison scene"></a></div>
<div class="panel"><strong>Refined autumn</strong><a id="afterLink"><img id="after" alt="Refined autumn shader scene"></a></div></div>
<h2>Texture and shading details</h2><a href="details.png"><img src="details.png" alt="Ground grain, foliage and rock detail comparison"></a>
<details><summary>Selected beauty target</summary><p class="note">Image concept; its altered tree shapes are not replacement assets.</p><img src="../../scene-concepts/fall.png" alt="Selected autumn art direction"></details>
<p class="note">Known source/harness slivers and shoreline geometry remain. The added autumn tint fades out at water boundaries.
Night calibration, production reflections and game integration remain in the playbook.</p>
<p><a href="test-biq-receipt.json">Scene receipt</a> · <a href="biomes-receipt.json">Biome receipt</a> · <a href="diagnostics.json">Measurements</a> ·
<a href="../../../../studies/seasons/FALL.md">Implementation notes</a></p>
<script>const $=id=>document.getElementById(id);function update(){const scene=$('scene').value,source=$('source').value;
$('before').src=$('beforeLink').href='../'+scene+'/Civ5EnvironmentSkin/'+source+'.png';
$('after').src=$('afterLink').href=scene+'/refined.png';$('caption').textContent=source==='summer'?'Summer source':'Preceding autumn';
$('bands').hidden=scene!=='biomes';}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));update();</script></html>'''
    if args.beauty:
        html=html.replace('<option value="fall">Preceding autumn</option>',
            '<option value="target">Selected target (full scene)</option><option value="fall">Preceding autumn</option>')
        html=html.replace("scene+'/refined.png'","scene+'/beauty.png'")
        html=html.replace("$('before').src=$('beforeLink').href='../'+scene+'/Civ5EnvironmentSkin/'+source+'.png';",
            "const compare=source==='target'?(scene==='test-biq'?'../../scene-concepts/fall.png':scene+'/refined.png'):"
            "source==='summer'?scene+'/beauty-summer.png':scene+'/refined.png';$('before').src=$('beforeLink').href=compare;")
        html=html.replace("$('caption').textContent=source==='summer'?'Summer source':'Preceding autumn';",
            "$('caption').textContent=source==='target'&&scene==='test-biq'?'Selected target':source==='summer'?'Corrected Lab Summer baseline':'Preceding autumn';")
        html=html.replace('href="details.png"','href="beauty-details.png"').replace('src="details.png"','src="beauty-details.png"')
        html=html.replace('href="test-biq-receipt.json"','href="test-biq-beauty-receipt.json"').replace('href="biomes-receipt.json"','href="biomes-beauty-receipt.json"').replace('href="diagnostics.json"','href="beauty-diagnostics.json"')
        html=html.replace('Known source/harness slivers and shoreline geometry remain. The added autumn tint fades out at water boundaries.',
            'The preview now masks coastal terrain spill, preserves underlay validity, uses biome textures on shore fringes, and adapts cached water facets with a localized reflection path. '
            'Its Summer baseline includes those harness corrections; it is distinct from the preceding Summer preview. Original tree silhouettes remain, and this candidate is not a match or accepted result.')
    if args.target:
        html=html.replace("scene+'/beauty.png'","scene+'/target.png'").replace("scene+'/beauty-summer.png'","scene+'/target-summer.png'")
        html=html.replace("scene+'/refined.png'","scene+'/beauty.png'")
        html=html.replace("beauty-details.png","target-details.png").replace("-beauty-receipt.json","-target-receipt.json").replace("beauty-diagnostics.json","target-diagnostics.json")
        html=html.replace("Refined autumn","Autumn material candidate")
        html=html.replace("Stable gold, amber and occasional russet crowns above quieter dry-olive grass and paler honey-straw plains.\nExisting tree geometry, bark, leaf detail, terrain texture and rock relief remain intact. Forest-floor decals carry a restrained litter tint.",
            "Broad crown irradiance and leaf-only value ramps preserve existing leaf detail. Bronze/olive turf carries source-derived golden dry patches; plains remains lighter straw. Tree proportions and placements are unchanged.")
        # Four reference-led material crops, aligned by stable map landmarks.
        target=target.resize(after.size,Image.Resampling.LANCZOS)
        crops=(("Crown volume and leaf detail",(155,75,360,260)),
               ("Bronze/olive turf",(350,470,575,635)),
               ("Rock facets and fissures",(865,170,1130,385)),
               ("Blue water and soft shore",(1280,120,1530,355)))
        board=Image.new("RGB",(1260,1064),(28,31,30));draw=ImageDraw.Draw(board)
        for row,(label,box) in enumerate(crops):
            for column,(im,caption) in enumerate(((target,"Selected target"),(before,"Rejected candidate"),(after,"Current material render"))):
                draw.text((column*420+8,row*266+5),caption,font=font,fill="white")
                crop=im.crop(box);crop.thumbnail((404,206),Image.Resampling.LANCZOS)
                board.paste(crop,(column*420+(420-crop.width)//2,row*266+36))
            draw.text((8,row*266+238),label,font=font,fill=(224,197,140))
        board.save(OUT/"target-reference-crops.png",optimize=True)
        html=html.replace('<h2>Texture and shading details</h2>',
            '<h2>Target, preceding candidate and current materials</h2><img src="target-reference-crops.png" alt="Four fixed reference-led material comparisons"><h2>Texture and shading details</h2>')
    (OUT/("target.html" if args.target else "beauty.html" if args.beauty else "index.html")).write_text(html)
    print(json.dumps(report["profiles"]["refined"],indent=2))


if __name__=="__main__":main()
