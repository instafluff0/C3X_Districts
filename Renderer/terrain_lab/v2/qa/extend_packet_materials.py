"""Add explicit material textures to unused non-feature draw slots in a copy.

Diagnostic adapter. Never overwrite a binding, geometry, shadows or source file.
"""
import os
from pathlib import Path
import struct
from rebind_packet_textures import dds,sha


def extend(source,target,channels):
    if target.exists():raise ValueError('preserved extended packet exists')
    data=source.read_bytes();offset=0
    def read(n):
        nonlocal offset
        out=data[offset:offset+n];offset+=n
        if len(out)!=n:raise ValueError('truncated packet')
        return out
    def u32():return struct.unpack('<I',read(4))[0]
    def skip_blob():
        size=u32();read(64 if size&0x80000000 else size)
    magic,version=u32(),u32()
    if magic!=0x32514c43 or version not in (5,6):raise ValueError('packet v5/v6 required')
    read(40);count_offset=offset;count=u32()
    for _ in range(count):
        read(12);n=u32()
        for _ in range(n):u32();skip_blob()
    tail_start=offset;tail=bytearray(data[tail_start:]);extra=bytearray()
    target.parent.mkdir(parents=True,exist_ok=True)
    blobs=Path(str(target)+'.blobs');blobs.mkdir(exist_ok=True)
    for p in Path(str(source)+'.blobs').iterdir():
        if not (blobs/p.name).exists():os.link(p,blobs/p.name)
    identities=[];bindings={}
    for slot,path in sorted(channels.items()):
        if not 0<=slot<128:raise ValueError('invalid texture slot')
        t=dds(path);bindings[slot]=count+len(bindings)+1
        extra.extend(struct.pack('<4I',t['width'],t['height'],t['format'],len(t['mips'])))
        for pitch,b in t['mips']:
            key=sha(b);p=blobs/key
            if not p.exists():p.write_bytes(b)
            elif p.read_bytes()!=b:raise ValueError('corrupt existing mip')
            extra.extend(struct.pack('<II',pitch,len(b)|0x80000000));extra.extend(key.encode())
        identities.append({'slot':slot,'texture_index':bindings[slot],
            'dds_sha256':sha(path.read_bytes()),'payload_sha256':t['payload_sha256']})
    for _ in range(u32()):skip_blob()
    changes=[]
    for draw in range(u32()):
        values=[u32() for _ in range(7)];read(40 if version==6 else 16);read(u32()*8)
        for slot in range(128):
            position=offset;old=u32()
            if values[4] or slot not in bindings:continue
            if old:raise ValueError('refuse to replace occupied material binding')
            struct.pack_into('<I',tail,position-tail_start,bindings[slot])
            changes.append({'draw':draw,'slot':slot,'offset_in_tail':position-tail_start,'texture_index':bindings[slot]})
    if offset!=len(data) or not changes:raise ValueError('packet tail contract')
    # Independently undo only recorded bindings; every remaining byte must match.
    restored=bytearray(tail)
    for change in changes:struct.pack_into('<I',restored,change['offset_in_tail'],0)
    if restored!=data[tail_start:]:raise ValueError('non-binding packet mutation')
    prefix=bytearray(data[:tail_start]);struct.pack_into('<I',prefix,count_offset,count+len(bindings))
    target.write_bytes(prefix+extra+tail)
    return {'source_packet_sha256':sha(data),'output_packet_sha256':sha(target.read_bytes()),
        'original_tail_sha256':sha(restored),'all_non_binding_tail_bytes_identical':True,
        'added_channels':identities,'binding_changes':changes}
