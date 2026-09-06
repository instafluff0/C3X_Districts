"""Generic normalized-texture + world-vertex diagnostic packet writer."""
import hashlib,struct
from pathlib import Path
from prepare_city import ROOT

def write_packet(target,vertices,pairs):
 if not 0<len(pairs)<=8:raise ValueError('diagnostic material slot limit')
 paths=list(dict.fromkeys(p for pair in pairs for p in pair if p));data=bytearray();hashes={}
 def u(*x):data.extend(struct.pack('<'+'I'*len(x),*x))
 def blob(b):u(len(b));data.extend(b)
 u(0x32514c43,2,768,512,1,len(paths))
 for path in paths:
  b=(ROOT/path).read_bytes();hashes[path]=hashlib.sha256(b).hexdigest();h,w=struct.unpack_from('<II',b,12);fmt,=struct.unpack_from('<I',b,128);nm,=struct.unpack_from('<I',b,28)
  if b[:4]!=b'DDS ' or b[84:88]!=b'DX10' or fmt not in [71,72,77,78]:raise ValueError('DDS contract')
  fmt=72 if fmt in [71,72] else 78;block=8 if fmt==72 else 16;u(w,h,fmt,max(1,nm));offset=148
  for _ in range(max(1,nm)):
   pitch=((w+3)//4)*block;length=pitch*((h+3)//4);u(pitch);blob(b[offset:offset+length]);offset+=length;w=max(1,w//2);h=max(1,h//2)
  if offset!=len(b):raise ValueError('DDS mip closure')
 u(2);blob(struct.pack('<'+'f'*len(vertices),*vertices));blob(bytes(160));u(1);u(0,1,len(vertices)//12,48,0,1,1);u(5)
 for components,offset in [(3,0),(2,12),(3,20),(3,32),(1,44)]:u(components,offset)
 for i in range(128):
  path=pairs[i][0] if i<len(pairs) else (pairs[i-8][1] if 8<=i<8+len(pairs) else '')
  u(paths.index(path)+1 if path else 0)
 target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 return hashes,hashlib.sha256(data).hexdigest()
