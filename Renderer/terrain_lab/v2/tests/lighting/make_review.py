#!/usr/bin/env python3
"""Optional Pillow review sheets; actual-size pixels, no renderer-input edits."""
from pathlib import Path
from PIL import Image,ImageDraw
V2=Path(__file__).resolve().parents[2];P=V2/'audits/lighting/out/Q1/Q6-lighting'
for system in ['q7_city','trees','rocks','units','improvements']:
 for suffix in ['','_contact_off','_shadows_off']:
  folder=P/(system+suffix+'-09');w,h=Image.open(folder/'h12-z1-pan00.bmp').size;sheet=Image.new('RGB',(w*2,(h+20)*4))
  for i,hour in enumerate([12,18,0,6]):
   for z in [1,2]:
    im=Image.open(folder/f'h{hour:02}-z{z}-pan00.bmp');x=i%2*w;y=(i//2*2+z-1)*(h+20);sheet.paste(im,(x,y+20));ImageDraw.Draw(sheet).text((x,y),f'{system}{suffix} h{hour} zoom{z}',fill='white')
  sheet.save(folder/'review.png')
 folder=P/(system+'-scroll-09');sheet=Image.new('RGB',(4*384,8*160))
 for row,(hour,z) in enumerate((hour,z) for hour in [12,18,0,6] for z in [1,2]):
  for pan in range(4):
   im=Image.open(folder/f'h{hour:02}-z{z}-pan{pan:02}.bmp');cx,cy=im.width//2,im.height//2
   crop=im.crop((max(0,cx-192),max(0,cy-68),min(im.width,cx+192),min(im.height,cy+68)));sheet.paste(crop,(pan*384,row*160+20));ImageDraw.Draw(sheet).text((pan*384,row*160),f'{system} h{hour} z{z} pan{pan}',fill='white')
 sheet.save(folder/'scroll-review.png')
sheet=Image.new('RGB',(1120,372))
for i,name in enumerate(['context-before','context-01']):
 im=Image.open(P/name/'h12-z1-pan00.bmp');sheet.paste(im.crop((104,40,664,392)),(560*i,20));ImageDraw.Draw(sheet).text((560*i,0),name+' | verified mixed | matched crop',fill='white')
sheet.save(P/'context-before-after.png')
