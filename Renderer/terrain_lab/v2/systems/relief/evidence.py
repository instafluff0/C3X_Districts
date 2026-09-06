#!/usr/bin/env python3
"""Inspect retained Q4 outputs numerically and prepare labeled review sheets."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[5];BASE=ROOT/'Renderer/terrain_lab/v2/audits/relief';OUT=BASE/'out/Q1/Q4-relief'
def relative(p):return p.relative_to(ROOT).as_posix()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
for name in ['source-coast-09-check','real-primary-09-check','real-holdout-10-check','biomes-clearance-10-check']:
 d=OUT/name;report=json.loads((d/'report.json').read_text());sheet=Image.new('RGB',(1024,848),(25,25,25));draw=ImageDraw.Draw(sheet)
 for k,(h,z) in enumerate((h,z) for h in [12,18,0,6] for z in [1,2]):
  p=d/f'h{h:02}-z{z}-pan00.bmp';im=Image.open(p);a=np.asarray(im.convert('RGB')).astype(float)
  x,y=(k%2)*512,(k//2)*212
  # Matched actual-pixel contextual crop; no enlargement or resampling.
  box=(240,175,752,375) if z==1 else (120,87,376,187)
  crop=im.crop(box);crop.save(d/f'h{h:02}-z{z}-context.png');sheet.paste(crop,(x,y+12));draw.text((x+5,y),f'{name} h{h:02} zoom {z}',fill='white')
  origin=d/f'h{h:02}-z{z}-pan02.bmp';pan=d/f'h{h:02}-z{z}-pan01.bmp'
  repeat=(sha(p)==sha(origin));b=np.asarray(Image.open(pan).convert('RGB')).astype(float)
  # Offset is final-pixel [+16,-8]; compare the shared interior after undoing it.
  delta=np.abs(a[8:,:-16]-b[:-8,16:]); valid=np.max(a,axis=2)>15
  lum=a@np.array([.2126,.7152,.0722])/255
  results.append({'candidate':name,'hour':h,'zoom':z,'image':relative(p),'sha256':sha(p),'return_to_origin_identical':repeat,'integer_pan_mean_absolute_error':float(delta.mean()),'integer_pan_max_error':float(delta.max()),'foreground_mean_luminance':float(lum[valid].mean()) if valid.any() else 0})
 sheet.save(d/'context-inspection.png')
names=['source-rock','dunes','volcano','biomes','coast-r0','coast-r1','coast-r2','coast-r3','island','cove']
sheet=Image.new('RGB',(1024,1200),(25,25,25));draw=ImageDraw.Draw(sheet)
for k,name in enumerate(names):
 d=OUT/f'{name}-09-quick';p=d/'h12-z1-pan00.bmp';im=Image.open(p);im.save(p.with_suffix('.png'));im.thumbnail((512,230));x,y=k%2*512,k//2*240;sheet.paste(im,(x,y+10));draw.text((x+4,y),name+' — synthetic source-art diagnostic',fill='white')
sheet.save(BASE/'diagnostic-inspection.png')
(BASE/'METRICS.json').write_text(json.dumps({'schema':'c3x.q4.visual_metrics.v1','results':results,'visual_acceptance':False,'note':'Metrics describe deterministic/photometric behavior only; source fidelity and composed beauty remain separate gates.'},indent=2)+'\n')
print('wrote',len(results),'checkpoint metric records')
