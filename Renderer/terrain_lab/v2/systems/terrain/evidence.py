"""Exact-size diagnostic contact sheets and metrics; never a visual pass oracle."""
import argparse
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
OUT=ROOT/'Renderer/terrain_lab/v2/audits/terrain/out/Q1/Q2-terrain'
AUDIT=ROOT/'Renderer/terrain_lab/v2/audits/terrain'
def read(path):
 b=path.read_bytes();w,h=struct.unpack_from('<ii',b,18);offset=struct.unpack_from('<I',b,10)[0]
 if b[:2]!=b'BM' or h>=0 or struct.unpack_from('<H',b,28)[0]!=32:raise ValueError('unexpected BMP')
 raw=b[offset:];return w,-h,[tuple(raw[i+c] for c in (2,1,0)) for i in range(0,len(raw),4)]
def png(path,w,h,pixels):
 def chunk(k,b):return struct.pack('>I',len(b))+k+b+struct.pack('>I',zlib.crc32(k+b)&0xffffffff)
 data=b''.join(b'\0'+bytes(c for pixel in pixels[y*w:(y+1)*w] for c in pixel) for y in range(h))
 path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(data))+chunk(b'IEND',b''))
def compare(a,b):
 wa,ha,aa=read(a);wb,hb,bb=read(b);assert(wa,ha)==(wb,hb)
 d=sorted(abs(x-y) for p,q in zip(aa,bb) for x,y in zip(p,q));return {'mean_absolute_rgb':sum(d)/len(d),'p99_rgb':d[int(.99*(len(d)-1))],'max_rgb':d[-1],'changed_channel_fraction':sum(x>0 for x in d)/len(d)}
def sheet(name,paths,columns):
 images=[read(p) for p in paths];cw=max(i[0] for i in images);ch=max(i[1] for i in images);w=columns*cw;h=math.ceil(len(images)/columns)*ch
 out=[(20,20,20)]*(w*h)
 for n,(iw,ih,pix) in enumerate(images):
  ox=(n%columns)*cw;oy=(n//columns)*ch
  for y in range(ih):out[(oy+y)*w+ox:(oy+y)*w+ox+iw]=pix[y*iw:(y+1)*iw]
 path=OUT/(name+'.png');png(path,w,h,out)
 return {'image':path.relative_to(ROOT).as_posix(),'layout':'row-major, native pixels; no scaling or sharpening','sources':[p.relative_to(ROOT).as_posix() for p in paths]}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--revision',default='r06');a=parser.parse_args();r=a.revision
 metrics={};sheets=[]
 def path(mode,h,z,pan=0):return OUT/f'context-{mode}-{r}'/f'h{h:02}-z{z}-pan{pan:02}.bmp'
 for z in [1,2]:
  sheets.append(sheet('context-before-after-z'+str(z),[path(m,h,z) for h in [12,0] for m in ['baseline','off','detail']],3))
  sheets.append(sheet('isolation-z'+str(z),[path(m,12,z) for m in ['albedo','height','normal','roughness','weights','detail']],3))
  metrics[f'detail_difference_z{z}']=compare(path('off',12,z),path('detail',12,z))
  metrics[f'baseline_difference_z{z}']=compare(path('baseline',12,z),path('detail',12,z))
  for hour in [12,18,0,6]:
   w,h,base=read(path('detail',hour,z));_,_,pan=read(path('detail',hour,z,1));error=[]
   for y in range(2,h-2):
    for x in range(2,w-3):error.extend(abs(base[y*w+x][c]-pan[y*w+x+1][c]) for c in range(3))
   metrics[f'scroll_h{hour}_z{z}']={'aligned_mean_absolute_rgb':sum(error)/len(error),'max_rgb':max(error),'return_to_origin_identical':path('detail',hour,z).read_bytes()==path('detail',hour,z,3).read_bytes()}
 for name in ['mixed','holdout','wrap','q2-dry','q2-cold','q2-wet','q2-dry-holdout','q2-cold-holdout','q2-wet-holdout']:
  folder=OUT/f'real-{name}-base-{r}'
  if folder.exists():sheets.append(sheet('real-'+name+'-base-only',[folder/f'h{h:02}-z{z}-pan00.bmp' for h in [12,18,0,6] for z in [1,2]],2))
 for hour in [12,18,0,6]:
  for zoom in [1,2]:
   for axis in ['column','row']:
    for reverse in [0,1]:
     paths=sorted(OUT.glob(f'pair_*_{axis}_r{reverse}_x98-{r}/h{hour:02}-z{zoom}-pan00.bmp'))
     if len(paths)==15:sheets.append(sheet(f'pairs-h{hour}-z{zoom}-{axis}-r{reverse}',paths,3))
 reports=list(OUT.glob('*-'+r+'/report.json'));metrics['rendered_images']=sum(len(json.loads(p.read_text())['outputs']) for p in reports)
 metrics['scope']='Base-only contextual proxies and controls; numerical metrics do not grant visual acceptance.'
 record={'schema':'c3x.q2.evidence.v1','revision':r,'metrics':metrics,'sheets':sheets}
 (AUDIT/'metrics_v1.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
