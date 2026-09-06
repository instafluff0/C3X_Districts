#!/usr/bin/env python3
"""Numeric evidence and final-pixel inspection sheets; requires Pillow/numpy."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[5]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('report',type=Path);a=ap.parse_args()
    report=json.loads(a.report.read_text());out=a.report.parent
    baselines={};images={};metrics=[]
    for e in report['outputs']:
        p=ROOT/e['image'];im=Image.open(p).convert('RGB');im.save(p.with_suffix('.png'))
        x=np.asarray(im).astype(float);images[e['image']]=im
        key=(e['hour'],e['zoom'],tuple(e['offset']))
        if e['variant']=='baseline':baselines[key]=x
    for e in report['outputs']:
        x=np.asarray(images[e['image']]).astype(float)
        key=(e['hour'],e['zoom'],tuple(e['offset']));b=baselines[key]
        diff=np.abs(x-b);cost=json.loads((ROOT/e['cost']).read_text())
        row={**{k:e[k] for k in ('variant','hour','zoom','offset','repeat_identical')},
            'rgb_mae_bytes':float(diff.mean()),'p99_abs_bytes':float(np.percentile(diff,99)),
            'changed_pixels':int(np.any(diff>0,axis=2).sum()),
            'gradient_mean_bytes':float((np.abs(np.diff(x,axis=0)).mean()+np.abs(np.diff(x,axis=1)).mean())/2),
            'clipped_channel_fraction':float(np.logical_or(x==0,x==255).mean()),
            'gpu_ms':cost['gpu_ms_mean'],'sampled_gpu_high_water_bytes':cost['allocation_high_water_sampled_bytes']}
        metrics.append(row)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for zoom in sorted({e['zoom'] for e in report['outputs']}):
        rows=[e for e in report['outputs'] if e['zoom']==zoom and e['hour']==12 and e['offset']==[0,0]]
        # Final-pixel crops, not enlargement. Full .png files remain available.
        width,height=images[rows[0]['image']].size
        rect=(int(width*.26),int(height*.156),int(width*.73),int(height*.625))
        panels=[]
        for e in rows:
            crop=images[e['image']].crop(rect);panel=Image.new('RGB',(crop.width,crop.height+26),'#eeeeee')
            panel.paste(crop,(0,26));ImageDraw.Draw(panel).text((4,6),f"{e['variant']} / zoom {zoom}",fill='black');panels.append(panel)
        w=panels[0].width;h=panels[0].height;cols=min(3,len(panels))
        sheet=Image.new('RGB',(w*cols,h*((len(panels)+cols-1)//cols)),'#888888')
        for i,p in enumerate(panels):sheet.paste(p,((i%cols)*w,(i//cols)*h))
        sheet.save(out/f'final-pixel-crops-z{zoom}.png')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
