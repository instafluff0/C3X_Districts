"""Actual-size crops and occupancy metrics; use a Python with Pillow installed."""
import argparse,hashlib,json,math
from pathlib import Path
from PIL import Image,ImageDraw

def review(folder,layout):
    data=json.loads(layout.read_text());rows=[];metrics=[]
    for hour in [12,18,0,6]:
        images=[]
        for zoom in [1,2]:
            path=folder/f'h{hour:02}-z{zoom}-pan00.bmp'
            if not path.exists():continue
            im=Image.open(path).convert('RGB');bg=im.getpixel((0,0))
            mask=Image.new('L',im.size);mask.putdata([255 if max(abs(a-b) for a,b in zip(px,bg))>3 else 0 for px in im.getdata()])
            box=mask.getbbox();crop=im.crop(box) if box else im
            images.append((zoom,crop))
            repeat=path.with_name(path.name+'.repeat1.bmp')
            entry=dict(hour=hour,zoom=zoom,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),repeat_identical=repeat.exists() and path.read_bytes()==repeat.read_bytes(),occupied_pixels=sum(v>0 for v in mask.getdata()),bounds=box)
            if data.get('components'):
                objs=[]
                for r in data['components']:
                    b=r['bounds'];b=[math.floor(b[0]/zoom),math.floor(b[1]/zoom),math.ceil(b[2]/zoom),math.ceil(b[3]/zoom)]
                    c=mask.crop(b);objs.append(dict(asset=r['asset'],category=r.get('category','building'),bounds=b,width=b[2]-b[0],height=b[3]-b[1],visible_pixels_in_bounds=sum(v>0 for v in c.getdata()),measurement='bbox occupancy includes projected neighbors; isolated ID pass required for unique visibility'))
                entry['objects']=objs
            metrics.append(entry)
        if images:
            row=Image.new('RGB',(sum(im.width for _,im in images)+36,max(im.height for _,im in images)+28),(24,27,30));d=ImageDraw.Draw(row);x=8
            for zoom,im in images:
                d.text((x,5),f'{hour:02}:00 / zoom {zoom} / actual pixels',fill='white');row.paste(im,(x,24));x+=im.width+16
            rows.append(row)
    if rows:
        sheet=Image.new('RGB',(max(r.width for r in rows),sum(r.height for r in rows)+8*len(rows)),(24,27,30));y=0
        for row in rows:sheet.paste(row,(0,y));y+=row.height+8
        sheet.save(folder/'actual-size-review.png')
    (folder/'pixel-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(folder/'actual-size-review.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('layout',type=Path);a=p.parse_args();review(a.folder,a.layout)
