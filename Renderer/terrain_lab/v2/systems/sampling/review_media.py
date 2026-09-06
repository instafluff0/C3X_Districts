#!/usr/bin/env python3
"""Lossless review media at final pixel size; never alters source render files."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[5]

def panel(rows,zoom):
    images=[Image.open(ROOT/r['image']).convert('RGB') for r in rows]
    w,h=images[0].size
    out=Image.new('RGB',(len(rows)*w,h+24),'#eeeeee');d=ImageDraw.Draw(out)
    for i,(row,im) in enumerate(zip(rows,images)):
        out.paste(im,(i*w,24));d.text((i*w+3,4),f"{row['variant']} / h{row['hour']} / z{zoom}",fill='black')
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['phases','animation']);ap.add_argument('reports',nargs='+',type=Path);ap.add_argument('--output',type=Path);a=ap.parse_args()
    reports=[json.loads(p.read_text()) for p in a.reports]
    out=a.output or a.reports[0].parent
    if not out.resolve().is_relative_to(ROOT/'Renderer/terrain_lab/v2/audits/sampling'):raise ValueError('output must be Q1 owned')
    out.mkdir(parents=True,exist_ok=True)
    for z in [1,2]:
        if a.mode=='phases':
            frames=[]
            for h in [12,18,0,6]:
                rows=[next(r for r in reports[0]['outputs'] if r['zoom']==z and r['hour']==h and r['variant']==v) for v in ['baseline','finalist']]
                frames.append(panel(rows,z))
            canvas=Image.new('RGB',(frames[0].width,frames[0].height*4),'#eeeeee')
            for i,frame in enumerate(frames):canvas.paste(frame,(0,i*frame.height))
            canvas.save(out/f'phase-comparison-z{z}.png')
        else:
            films={v:[] for v in ['baseline','finalist_control','finalist']};frames=[]
            for report in reports:
                rows=[next(r for r in report['outputs'] if r['zoom']==z and r['variant']==v) for v in films]
                frames.append(panel(rows,z))
                for row in rows:films[row['variant']].append(Image.open(ROOT/row['image']).convert('RGB'))
            films['comparison']=frames
            for v,images in films.items():
                images[0].save(out/f'{v}-z{z}.webp',save_all=True,append_images=images[1:]+[images[0]],duration=250,loop=0,lossless=True)
            for i,frame in enumerate(frames):frame.save(out/f'pose{i}-z{z}.png')
    print('Review media written:',out)

if __name__=='__main__':main()
