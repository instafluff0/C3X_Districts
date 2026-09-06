#!/usr/bin/env python3
"""Source-reuse contact witnesses; uniform transforms, unchanged UVs/geometry.
Diagnostic receiver only. Does not define Q7 production placement or scale.
"""
import hashlib,json,math,re,struct
from pathlib import Path
from prepare_city import ROOT,V2,read_bundle

DEFINITIONS=[('trees','Civ5EnvironmentVegetation','vegetation','forest'),('rocks','ShoreNormalized','shore','river_rock'),('units','UnitFamilyLab','unit_infantry','idle_0'),('improvements','ImprovementsNormalized','mine','mine_0')]
def prepare():
 for kind,packname,bundlename,groupname in DEFINITIONS:
    pack=ROOT/'Renderer/packs'/packname;bundle=pack/(bundlename+'_runtime.bin');paths,assets,groups=read_bundle(bundle)
    group=groups[groupname]
    if kind=='trees':group=[p for p in group if 'leafy_v1_' in assets[p[0]][0]][:3]
    if kind=='rocks':group=group[:3]
    slots=[];vertices=[];emission={}
    if kind in ['trees','rocks']:
        instances=[([p],(-.6+index*.55, .25*(index%2))) for index,p in enumerate(group)]
    else:instances=[(group,(-.5,0)),(group,(.5,.1))]
    for instance,(parts,(cx,cy)) in enumerate(instances):
      for part in parts:
        name,texture,vs,indices=assets[part[0]];scale=part[1];yaw=.55;c=math.cos(yaw);s=math.sin(yaw)
        slots.append(dict(asset=name,translation=[cx,cy,0],uniform_scale=scale,yaw=yaw,base_texture=texture))
        code=re.search(r':e([12])$',name)
        if kind=='improvements' and code:emission[texture]=5+int(code[1])
        for index in indices:
          x,y,z,nx,ny,nz,u,v=vs[index];vertices.extend((0,0,0,u,v,nx*c-ny*s,nx*s+ny*c,nz,cx+(x*c-y*s)*scale,cy+(x*s+y*c)*scale,z*scale,float(texture)))
    for x,y in [(-2,-2),(-2,2),(2,-2),(2,-2),(-2,2),(2,2)]:vertices.extend((0,0,0,0,0,0,0,1,x,y,-.002,8))
    out=V2/f'fixtures/lighting/generated/{kind}';out.mkdir(parents=True,exist_ok=True)
    data=bytearray()
    def u(*x):data.extend(struct.pack('<'+'I'*len(x),*x))
    def blob(b):u(len(b));data.extend(b)
    u(0x32514c43,2,768,512,1,len(paths));hashes={bundle.name:hashlib.sha256(bundle.read_bytes()).hexdigest()}
    for path in paths:
      path=path.replace('\\','/');b=(pack/path).read_bytes();hashes[path]=hashlib.sha256(b).hexdigest()
      h,w=struct.unpack_from('<II',b,12);fmt,=struct.unpack_from('<I',b,128);nm,=struct.unpack_from('<I',b,28)
      if b[:4]!=b'DDS ' or b[84:88]!=b'DX10' or fmt not in [71,72,77,78]:raise ValueError('source DDS contract')
      fmt=72 if fmt in [71,72] else 78;block=8 if fmt==72 else 16
      u(w,h,fmt,max(1,nm));offset=148
      for _ in range(max(1,nm)):
        pitch=((w+3)//4)*block;length=pitch*((h+3)//4);u(pitch);blob(b[offset:offset+length]);offset+=length;w=max(1,w//2);h=max(1,h//2)
      if offset!=len(b):raise ValueError('mip closure')
    u(2);blob(struct.pack('<'+'f'*len(vertices),*vertices));blob(bytes(160));u(1);u(0,1,len(vertices)//12,48,0,1,1);u(5)
    for components,offset in [(3,0),(2,12),(3,20),(3,32),(1,44)]:u(components,offset)
    for index in range(128):u(index+1 if index<(6 if kind=='improvements' else len(paths)) else (emission[index-8]+1 if index-8 in emission else 0))
    packet=out/'scene.packet';packet.write_bytes(data)
    report={'classification':'source_reuse','diagnostic_receiver':'diagnostic_proxy','fixture_class':'source-backed category on flat synthetic receiver','category':kind,'pack':str(pack.relative_to(ROOT)),'source_hashes':hashes,'source_geometry_and_uv':'unchanged; rigid rotation/uniform inherited scale','placements':slots,'geometry_sha256':hashlib.sha256(data).hexdigest(),'triangles':len(vertices)//36,'source_emission_bindings':emission,'clock':'shared EnvironmentState','scope':'representative contact/receiver witness, not complete category or gameplay approval'}
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    f=json.loads((V2/'fixtures/lighting/city_linear.fixture.json').read_text());f.update(id='lighting-'+kind);f['scenarios']={'lighting_geometry':str(packet.relative_to(ROOT))};f['packs'][kind]=str(pack.relative_to(ROOT));(V2/f'fixtures/lighting/{kind}.fixture.json').write_text(json.dumps(f,indent=2)+'\n')
    for control,macro in [('contact_off','Q6_CONTACT 0'),('shadows_off','Q6_SHADOWS 0')]:
      sh=V2/f'shaders/lighting/linear_{control}.hlsl';sh.write_text(f'#define Q6_LINEAR 1\n#define {macro}\n#include "city.hlsl"\n')
      m=json.loads((V2/'systems/lighting/city_linear.module.json').read_text());m.update(id='lighting-linear-'+control,shader=str(sh.relative_to(ROOT)));mp=V2/f'systems/lighting/linear_{control}.module.json';mp.write_text(json.dumps(m,indent=2)+'\n')
      cf=dict(f,id='lighting-'+kind+'-'+control,modules=[str(mp.relative_to(ROOT))]);(V2/f'fixtures/lighting/{kind}_{control}.fixture.json').write_text(json.dumps(cf,indent=2)+'\n')
    print(kind,len(vertices)//36,'source triangles')
if __name__=='__main__':prepare()
