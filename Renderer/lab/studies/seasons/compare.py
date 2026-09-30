#!/usr/bin/env python3
"""Small read-only visual comparisons and measured five-biome diagnostics."""
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/"Renderer/lab/out/seasons/programmatic"


def main():
    try:
        from PIL import Image,ImageDraw,ImageFont,ImageFilter
        import numpy as np
    except ImportError:
        python=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
        os.execv(str(python),[str(python),str(Path(__file__)),*sys.argv[1:]])
    packs=("Civ5EnvironmentSkin","TerrainNormalized")
    seasons=("summer","fall","winter","spring")
    biomes=("grassland","plains","desert","floodplains","tundra")
    report={"schema":"c3x.lab.seasonal_visual_diagnostics.v1","packs":{},
        "interpretation":"Flat-strip median colors and texture correlation at noon. Not a subjective readability or production-parity pass."}
    for pack in packs:
        report["packs"][pack]={}
        original=Image.open(OUT/"biomes"/pack/"summer.png").convert("L")
        def detail(image):
            image=image.crop((96,310,288,490))
            return np.array(image,dtype=float)-np.array(image.filter(ImageFilter.GaussianBlur(2)),dtype=float)
        summer=detail(original).flatten()
        for season in seasons:
            im=Image.open(OUT/"biomes"/pack/(season+".png"));a=np.array(im,dtype=float)/255
            colors=[];crops={}
            for i,name in enumerate(biomes):
                center=int(im.width*(i+.5)/5)
                box=(center-64,360,center+64,460)
                colors.append(np.median(a[box[1]:box[3],box[0]:box[2]].reshape(-1,3),axis=0))
                crops[name]=list(box)
            c=np.array(colors);linear=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
            xyz=linear@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
            xyz/=np.array([.95047,1,1.08883]);f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29)
            lab=np.stack([116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])],axis=1)
            pairs={biomes[i]+" / "+biomes[j]:round(float(np.linalg.norm(lab[i]-lab[j])),2) for i in range(5) for j in range(i+1,5)}
            texture=detail(im.convert("L")).flatten()
            report["packs"][pack][season]={"crop_boxes":crops,"median_srgb":dict(zip(biomes,[[round(float(v)*255,1) for v in c] for c in colors])),
                "cie76_distances":pairs,"grassland_highpass_correlation_to_summer":round(float(np.corrcoef(summer,texture)[0,1]),4)}
    (OUT/"visual-diagnostics.json").write_text(json.dumps(report,indent=2)+"\n")
    font=ImageFont.load_default(size=22);small=ImageFont.load_default(size=17)
    def sheet(columns,rows,path,caption):
        w,h=800,450
        canvas=Image.new("RGB",(w*len(columns),54+(h+40)*len(rows)+32),(20,29,36));draw=ImageDraw.Draw(canvas)
        draw.text((16,14),caption,fill=(232,236,240),font=font)
        for ri,(season,label) in enumerate(rows):
            y=54+ri*(h+40)
            for ci,(column,title) in enumerate(columns):
                x=ci*w;draw.text((x+12,y+8),label+" | "+title,fill=(232,236,240),font=font)
                if column=="concept":
                    names={"fall":"fall","winter":"winter-icy-blue","spring":"spring-vibrant"}
                    image=Image.open(OUT.parent/"scene-concepts"/(names[season]+".png"))
                else:image=Image.open(OUT/"test-biq"/column/(season+".png"))
                canvas.paste(image.convert("RGB").resize((w,h),Image.Resampling.LANCZOS),(x,y+40))
        draw.text((12,canvas.height-26),"Mac Metal material experiment; diagnostic water/shadow composition. Concepts are art direction.",fill=(179,195,207),font=small)
        canvas.save(path,optimize=True)
    sheet([(p,p) for p in packs],[(s,s.title()) for s in seasons],OUT/"pack-comparison.png","Same world and source geometry; selected terrain materials remain authoritative")
    sheet([("concept","Selected concept"),(packs[0],"Executable shader / Civ V-style pack")],[(s,s.title()) for s in seasons[1:]],OUT/"concept-comparison.png","Selected beauty targets beside the actual programmatic Lab result")
    # A compact row is useful in the conversation; the gallery keeps full-size
    # images and the A/B tint comparison without copying any source assets.
    row=Image.new("RGB",(1600,334),(20,29,36));draw=ImageDraw.Draw(row)
    for i,s in enumerate(seasons[1:]):
        x=i*533;draw.text((x+12,8),s.title(),fill=(232,236,240),font=font)
        im=Image.open(OUT/"test-biq"/packs[0]/(s+".png"));row.paste(im.resize((533,300),Image.Resampling.LANCZOS),(x,34))
    row.save(OUT/"seasonal-scenes.png",optimize=True)
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Programmatic seasonal terrain Lab</title><style>
body{margin:28px;background:#121d26;color:#e5edf2;font:16px system-ui;line-height:1.5}h1{font-size:26px}label{margin-right:22px}select{padding:7px;border-radius:6px}a{color:#95d7ff}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}.views img{width:100%;display:block}.panel{background:#22313c;padding:10px}.wide{grid-column:1/-1}summary{cursor:pointer}.note{color:#b7cad7}.controls{margin:22px 0}#target{width:1000px;max-width:100%}#conceptPanel{margin-top:18px}pre{white-space:pre-wrap}@media(max-width:850px){.views{grid-template-columns:1fr}}</style>
<h1>Programmatic seasonal terrain Lab</h1><p>Existing summer textures, meshes, opacity and placements, with seasonal material policies applied before lighting.</p>
<p class="note">Mac Metal experiment. Water, shoreline composition and the shadow atlas are diagnostic; this is not the production render graph. Selected concepts remain beauty targets.</p>
<div class="controls"><label>Terrain pack <select id="pack"><option>Civ5EnvironmentSkin</option><option>TerrainNormalized</option></select></label><label>Season <select id="season"><option value="fall">Autumn</option><option value="winter">Winter</option><option value="spring">Spring</option></select></label><label>Scene <select id="scene"><option value="test-biq">test.biq, camera 20,75</option><option value="biomes">Five biome diagnostic strips</option></select></label><label>Autumn policy <select id="policy"><option value="fall">Tint existing source colors</option><option value="fall-hue-blend">Compare hue blend</option></select></label></div>
<div class="views"><div class="panel"><strong>Summer source</strong><a id="summerLink"><img id="summer" alt="Summer source scene"></a></div><div class="panel"><strong id="caption">Seasonal shader</strong><a id="resultLink"><img id="result" alt="Seasonal shader scene"></a></div></div>
<p id="bands" hidden>Strips left to right: grassland, plains, desert, floodplains, tundra. Forests above, relief below. Small dark shadow artifacts belong to this diagnostic shadow adapter.</p>
<details id="conceptPanel" open><summary>Selected beauty target</summary><img id="target" alt="Selected season concept"></details>
<details><summary>Evidence and implementation</summary><p><a href="../../../studies/seasons/PLAYBOOK.md">Real-game implementation playbook</a> · <a href="test-biq-receipt.json">Scene / source / GPU receipt</a> · <a href="biomes-receipt.json">Five-biome receipt</a> · <a href="visual-diagnostics.json">Color / grain diagnostics</a> · <a href="asset-census.json">Upstream asset census</a> · <a href="snow-decal-evidence.json">All 11 source snow-decal descriptors</a></p><p>Summer round trips and disabled seasonal policies are exact in the linear render target. Coverage remains unchanged; tests include map-wrapped noise and visible flowers, evergreen/wood protection, desert/stone exclusion and finite snow normals.</p></details>
<script>const $=x=>document.getElementById(x);function update(){const pack=$('pack').value,season=$('season').value,scene=$('scene').value;const mode=season==='fall'?$('policy').value:season;const source=scene+'/'+pack+'/summer.png',result=scene+'/'+pack+'/'+mode+'.png';$('summer').src=source;$('summerLink').href=source;$('result').src=result;$('resultLink').href=result;$('policy').disabled=season!=='fall';$('caption').textContent=season==='fall'?(mode==='fall'?'Autumn — multiplicative tint':'Autumn — hue blend'):season==='winter'?'Winter — source snow and retained relief':'Spring — gentle tint and blossom clusters';$('target').src='../scene-concepts/'+({fall:'fall',winter:'winter-icy-blue',spring:'spring-vibrant'}[season])+'.png';$('bands').hidden=scene!=='biomes';$('conceptPanel').hidden=scene==='biomes';}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));update();</script></html>'''
    (OUT/"index.html").write_text(html)
    print(json.dumps({p:{s:{"grass_plains_cie76":report["packs"][p][s]["cie76_distances"]["grassland / plains"],
        "minimum_pair_cie76":min(report["packs"][p][s]["cie76_distances"].values()),
        "texture_correlation":report["packs"][p][s]["grassland_highpass_correlation_to_summer"]} for s in seasons} for p in packs},indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
