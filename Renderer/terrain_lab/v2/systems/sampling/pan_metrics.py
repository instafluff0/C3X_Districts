"""Aligned overlapping pan residuals and lossless animation witnesses (Pillow/numpy)."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[5]

def align(image,offset):
    h,w=image.shape[:2];dx,dy=offset
    # Interior common overlap; exact indexing for integers, bilinear otherwise.
    x=np.arange(8,w-int(np.ceil(abs(dx)))-8,dtype=float)+max(0,-dx)
    y=np.arange(8,h-int(np.ceil(abs(dy)))-8,dtype=float)+max(0,-dy)
    xx,yy=np.meshgrid(x,y);sx=xx+dx;sy=yy+dy
    ix=sx.astype(int);iy=sy.astype(int);fx=(sx-ix)[...,None];fy=(sy-iy)[...,None]
    aligned=(image[iy,ix]*(1-fx)*(1-fy)+image[iy,ix+1]*fx*(1-fy)+
             image[iy+1,ix]*(1-fx)*fy+image[iy+1,ix+1]*fx*fy)
    return aligned,xx.astype(int),yy.astype(int)

def main():
    p=Path(sys.argv[1]);r=json.loads(p.read_text());out=p.parent;metrics=[]
    for z in [1,2]:
        films={}
        for variant in sorted({e['variant'] for e in r['outputs']}):
            entries=[e for e in r['outputs'] if e['variant']==variant and e['zoom']==z]
            frames=[Image.open(ROOT/e['image']).convert('RGB') for e in entries]
            if not frames:continue
            base=np.asarray(frames[0],float);films[variant]=frames
            for e,im in zip(entries,frames):
                aligned,xx,yy=align(np.asarray(im,float),e['offset']);delta=np.abs(aligned-base[yy,xx])
                metrics.append({'zoom':z,'variant':variant,'offset':e['offset'],'mae_bytes':float(delta.mean()),
                    'p99_bytes':float(np.percentile(delta,99)),
                    'metric':'exact registered integer overlap' if all(float(x).is_integer() for x in e['offset']) else 'bilinear registered overlap; includes registration low-pass error'})
            frames[0].save(out/f'{variant}-z{z}.webp',save_all=True,append_images=frames[1:],duration=160,loop=0,lossless=True)
        keys=['baseline','finalist_control','finalist'];panels=[]
        for i in range(len(films['baseline'])):
            w,h=films['baseline'][i].size;canvas=Image.new('RGB',(w*3,h+24),'#eeeeee');d=ImageDraw.Draw(canvas)
            for j,key in enumerate(keys):canvas.paste(films[key][i],(j*w,24));d.text((j*w+4,6),key,fill='black')
            panels.append(canvas)
        panels[0].save(out/f'comparison-z{z}.webp',save_all=True,append_images=panels[1:],duration=160,loop=0,lossless=True)
        for i in [0,1,3,9]:panels[i].save(out/f'pan-sample-z{z}-{i}.png')
    (out/'aligned-pan-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Aligned pan metrics and lossless animations written')

if __name__=='__main__':main()
