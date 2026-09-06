"""Read actual backend BMPs, save review PNGs and report pixel comparisons."""
from pathlib import Path
import hashlib,json,struct,sys
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'Renderer'))
from preview.render_iso import Canvas
from preview.render_textured_patch import write_png
OUT=ROOT/'Renderer/terrain_lab/v2/audits/networks/out/Q1/Q5-networks'

def read(path):
    b=path.read_bytes();w,h=struct.unpack_from('<ii',b,18);offset=struct.unpack_from('<I',b,10)[0]
    assert struct.unpack_from('<H',b,28)[0]==32
    rows=[]
    for y in range(abs(h)):
        src=y if h<0 else h-1-y
        row=b[offset+src*w*4:offset+(src+1)*w*4]
        rows.extend((row[i+2],row[i+1],row[i]) for i in range(0,len(row),4))
    c=Canvas(w,abs(h),(0,0,0));c.pixels=rows;return c

def sheets(name):
    p=OUT/name;report=json.loads((p/'report.json').read_text());frames=[]
    for o in report['outputs']:
        if o['offset']!=[0,0] or 'pan03' in o['image']:continue
        image=ROOT/o['image'];c=read(image);write_png(c,image.with_suffix('.png'));frames.append((o,c))
    # Native images remain separate and are inspected at actual display scale.
    reduced=[(o,c) for o,c in frames if o['zoom']==2]
    if reduced:
        w=max(c.width for _,c in reduced);h=max(c.height for _,c in reduced)
        sheet=Canvas(w*2,h*2,(12,12,12))
        for index,(o,c) in enumerate(reduced):
            x=index%2*w;y=index//2*h
            for r in range(c.height):sheet.pixels[(y+r)*sheet.width+x:(y+r)*sheet.width+x+c.width]=c.pixels[r*c.width:(r+1)*c.width]
        write_png(sheet,p/'phase-sheet-z2.png')
    return report

def compare(a,b):
    ca,cb=read(a),read(b);assert (ca.width,ca.height)==(cb.width,cb.height)
    d=[sum(abs(x-y) for x,y in zip(p,q))/3 for p,q in zip(ca.pixels,cb.pixels)]
    return dict(mean_absolute_channel_delta=sum(d)/len(d),changed_pixels=sum(x>0 for x in d),total_pixels=len(d),max_mean_channel_delta=max(d),byte_identical=a.read_bytes()==b.read_bytes())
if __name__=='__main__':
    for name in sys.argv[1:]:sheets(name)
