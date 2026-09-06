"""Lossless inspection PNGs and descriptive metrics for actual renderer BMPs."""
import argparse
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

def bmp(path):
 data=path.read_bytes();offset=struct.unpack_from('<I',data,10)[0];w,h=struct.unpack_from('<ii',data,18);bpp=struct.unpack_from('<H',data,28)[0]
 if bpp!=32: raise ValueError('expected renderer BGRA8')
 rows=[data[offset+i*w*4:offset+(i+1)*w*4] for i in range(abs(h))]
 if h>0: rows.reverse()
 pixels=bytes(v for row in rows for i in range(0,len(row),4) for v in (row[i+2],row[i+1],row[i],255))
 return w,abs(h),pixels

def png(path,w,h,pixels):
 def chunk(name,data):return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
 rows=b''.join(b'\0'+pixels[y*w*4:(y+1)*w*4] for y in range(h))
 path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))

def review(directory):
 report=json.loads((directory/'report.json').read_text());out=[]
 for entry in report['outputs']:
  path=Path(entry['image']);w,h,p=bmp(path);png(path.with_suffix('.png'),w,h,p)
  l=[(.2126*p[i]+.7152*p[i+1]+.0722*p[i+2])/255 for i in range(0,len(p),4)]
  covered=[v for v in l if v>.055];mean=sum(covered)/max(1,len(covered));var=sum((v-mean)**2 for v in covered)/max(1,len(covered))
  edge=sum(abs(l[y*w+x]-l[y*w+x-1]) for y in range(h) for x in range(1,w))/(h*(w-1))
  out.append({'image':str(path.with_suffix('.png')),'bmp_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pixels':[w,h],'mean_display_luminance':mean,'display_luminance_sd':math.sqrt(var),'mean_horizontal_luminance_difference':edge,'fraction_below_0055':1-len(covered)/len(l),'note':'Descriptive metrics only; no perceptual acceptance inferred.'})
 (directory/'inspection.json').write_text(json.dumps(out,indent=2)+'\n')
 # Each cell retains actual pixels; cells are indexed in inspection.json order.
 cellw=max(x['pixels'][0] for x in out);cellh=max(x['pixels'][1] for x in out);cols=2;rows=(len(out)+1)//2
 canvas=bytearray(bytes([20,20,20,255])*(cols*cellw*rows*cellh))
 for n,entry in enumerate(out):
  w,h,p=bmp(Path(report['outputs'][n]['image']));ox=(n%cols)*cellw;oy=(n//cols)*cellh
  for y in range(h):start=((oy+y)*cols*cellw+ox)*4;canvas[start:start+w*4]=p[y*w*4:(y+1)*w*4]
 png(directory/'contact.png',cellw*cols,cellh*rows,canvas)
 return out
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('directory',nargs='+',type=Path);args=parser.parse_args()
 for path in args.directory:print(json.dumps({'directory':str(path),'images':len(review(path))}))
