"""Diagnostic-only DDS rebinding; preserve every geometry/binding/shadow byte.

Replaces exact texture payload identities in a copied packet. The original
packet, cache and historical reports are never changed.
"""
import hashlib
from pathlib import Path
import struct
import os


def sha(b): return hashlib.sha256(b).hexdigest()


def dds(path):
    data=path.read_bytes()
    if data[:4]!=b'DDS ' or data[84:88]!=b'DX10': raise ValueError('DDS DX10 required')
    h,w=struct.unpack_from('<II',data,12);fmt=struct.unpack_from('<I',data,128)[0]
    n=struct.unpack_from('<I',data,28)[0]
    if fmt not in (10,11,35,77,78,80,83) or not 1<=n<=15: raise ValueError('unsupported replacement texture')
    block=8 if fmt==80 else 16;offset=148;mips=[]
    for level in range(n):
        pitch=max(1,w>>level)*(8 if fmt in (10,11) else 4) if fmt in (10,11,35) else ((max(1,w>>level)+3)//4)*block
        size=pitch*(max(1,h>>level) if fmt in (10,11,35) else ((max(1,h>>level)+3)//4))
        mips.append((pitch,data[offset:offset+size]));offset+=size
    if offset!=len(data): raise ValueError('replacement mip layout')
    return {'width':w,'height':h,'format':fmt,'mips':mips,'payload_sha256':sha(data[148:])}


def rebind(source,target,replacements):
    """replacements maps old full mip payload SHA256 to replacement DDS Path."""
    original=source.read_bytes();offset=0
    def read(n):
        nonlocal offset
        b=original[offset:offset+n];offset+=n
        if len(b)!=n: raise ValueError('truncated packet')
        return b
    def u32():return struct.unpack('<I',read(4))[0]
    magic,version= u32(),u32()
    if magic!=0x32514c43 or version not in range(2,7):raise ValueError('unsupported packet')
    read(12)
    if version>=3:read(28 if version>=4 else 24)
    count=u32();out=bytearray(original[:offset]);changed=[]
    target.parent.mkdir(parents=True,exist_ok=True)
    links=Path(str(target)+'.blobs');links.mkdir(exist_ok=True)
    for p in Path(str(source)+'.blobs').iterdir():
        if not (links/p.name).exists():os.link(p,links/p.name)
    for index in range(count):
        begin=offset;w,h,fmt,n=[u32() for _ in range(4)];digest=hashlib.sha256()
        for _ in range(n):
            u32();size=u32()
            if size&0x80000000:
                key=read(64).decode();b=(links/key).read_bytes()
                if len(b)!=(size&0x7fffffff) or sha(b)!=key:raise ValueError('invalid mip reference')
            else:b=read(size)
            digest.update(b)
        old=digest.hexdigest()
        if old not in replacements:out.extend(original[begin:offset]);continue
        replacement=dds(replacements[old])
        if (w,h,fmt,n)!=(replacement['width'],replacement['height'],replacement['format'],len(replacement['mips'])):
            raise ValueError('diagnostic replacement must preserve texture layout')
        out.extend(struct.pack('<4I',w,h,fmt,n))
        for pitch,b in replacement['mips']:
            key=sha(b);p=links/key
            if not p.exists():p.write_bytes(b)
            elif p.read_bytes()!=b:raise ValueError('corrupt output blob')
            out.extend(struct.pack('<II',pitch,len(b)|0x80000000));out.extend(key.encode())
        changed.append({'texture_index':index+1,'old_payload_sha256':old,'new_payload_sha256':replacement['payload_sha256']})
    if {c['old_payload_sha256'] for c in changed} != set(replacements):raise ValueError('replacement target absent')
    tail=original[offset:];out.extend(tail);target.write_bytes(out)
    return {'changed_textures':changed,'all_geometry_buffers_draws_bindings_unchanged':True,
            'unchanged_packet_tail_sha256':sha(tail),'source_packet_sha256':sha(original),'output_packet_sha256':sha(out)}
