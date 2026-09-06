#!/usr/bin/env python3
"""Prepare a Q6 diagnostic from the existing generic normalized city bundle.
No source-format conversion or production placement policy; all output is local.
"""
import hashlib,json,math,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'fixtures/lighting/generated'
PACK=ROOT/'Renderer/packs/CityComponentsNormalized'

def read_bundle(path):
    b=path.read_bytes(); cursor=8
    if b[:8]!=b'C3XVEG1\0': raise ValueError('generic bundle magic')
    def take(fmt):
        nonlocal cursor
        r=struct.unpack_from('<'+fmt,b,cursor);cursor+=struct.calcsize('<'+fmt);return r
    def string():
        nonlocal cursor
        n,=take('I');s=b[cursor:cursor+n].decode();cursor+=n;return s
    version,nt,na,ng=take('4I')
    if version!=1 or not 0<nt<=8 or not 0<na<=256: raise ValueError('bundle bounds')
    textures=[string() for _ in range(nt)]; assets=[]
    for _ in range(na):
        name=string();tex,nv,ni=take('3I');verts=[take('8f') for _ in range(nv)];inds=take(str(ni)+'I')
        if tex>=nt or any(i>=nv for i in inds): raise ValueError('mesh bounds')
        assets.append((name,tex,verts,inds))
    groups={}
    for _ in range(ng):
        name=string();n,=take('I');groups[name]=[take('IffIIIIff') for _ in range(n)]
    if cursor!=len(b):raise ValueError('bundle trailing data')
    return textures,assets,groups

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    textures,assets,groups=read_bundle(PACK/'city_runtime.bin')
    vertices=[]; placements=[]
    # Existing L17 modern metropolis recipe: eleven slots, common scale,
    # golden-angle placement. Two neighboring replicas provide a dense witness.
    group=groups['modern']
    for city,origin in enumerate([(-.48,0),(.48,.10)]):
        for slot in range(11):
            placement=group[(city+slot)%len(group)];name,tex,vs,indices=assets[placement[0]]
            angle=slot*2.39996322973;radius=0 if slot==0 else .41*math.sqrt(slot/10)
            cx=origin[0]+math.cos(angle)*radius;cy=origin[1]+math.sin(angle)*radius*.78
            yaw=angle+.55;scale=placement[1]*1.08
            placements.append(dict(asset=name,city=city,slot=slot,xy=[cx,cy],yaw=yaw,scale=scale,texture=tex))
            for i in indices:
                x,y,z,nx,ny,nz,u,v=vs[i];c=math.cos(yaw);s=math.sin(yaw)
                # One uniform normalized scale, never deform geometry to fake light.
                x,y,z=cx+(x*c-y*s)*scale,cy+(x*s+y*c)*scale,z*scale
                nx,ny=nx*c-ny*s,nx*s+ny*c
                vertices.extend((0,0,0,u,v,nx,ny,nz,x,y,z,float(tex)))
    # Ground is a diagnostic neutral receiver, not BIQ terrain.
    for x,y in [(-2,-2),(-2,2),(2,-2),(2,-2),(-2,2),(2,2)]:
        vertices.extend((0,0,0,0,0,0,0,1,x,y,-.002,8))
    data=bytearray()
    def u(*x):data.extend(struct.pack('<'+'I'*len(x),*x))
    def blob(b):u(len(b));data.extend(b)
    u(0x32514c43,2,768,512,1,len(textures))
    source_hashes={'city_runtime.bin':hashlib.sha256((PACK/'city_runtime.bin').read_bytes()).hexdigest()}
    for path in textures:
        path=path.replace('\\','/');b=(PACK/path).read_bytes();source_hashes[path]=hashlib.sha256(b).hexdigest()
        h,w=struct.unpack_from('<II',b,12);fmt,=struct.unpack_from('<I',b,128);nm,=struct.unpack_from('<I',b,28)
        if b[:4]!=b'DDS ' or b[84:88]!=b'DX10' or fmt not in [71,72]:raise ValueError('city DDS contract')
        u(w,h,72,max(1,nm));offset=148
        for _ in range(max(1,nm)):
            pitch=((w+3)//4)*8;length=pitch*((h+3)//4);u(pitch);blob(b[offset:offset+length]);offset+=length;w=max(1,w//2);h=max(1,h//2)
        if offset!=len(b):raise ValueError('DDS mip closure')
    u(2);blob(struct.pack('<'+'f'*len(vertices),*vertices));blob(bytes(160))
    u(1);u(0,1,len(vertices)//12,48,0,1,1)
    u(5)
    for components,offset in [(3,0),(2,12),(3,20),(3,32),(1,44)]:u(components,offset)
    for i in range(128):u(i+1 if i<4 else (i-3 if 8<=i<12 else 0))
    (OUT/'city.packet').write_bytes(data)
    report={'fixture_class':'synthetic_dense_lighting_layout_with_existing_normalized_components','terrain':'neutral authored receiver; not test.biq','recipe':'L17 modern metropolis eleven slots, two adjacent diagnostic replicas','source_hashes':source_hashes,'placements':placements,'geometry_sha256':hashlib.sha256(data).hexdigest(),'triangle_count':len(vertices)//36,'bounds_z':[min(vertices[10::12]),max(vertices[10::12])],'emissive_binding':'generic runtime atlas entries 4..7, confirmed source channel; inherited L17 sRGB view','local_light_attachments':'disabled; source binding unresolved'}
    (OUT/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['placements','source_hashes']},indent=2))
if __name__=='__main__':prepare()
