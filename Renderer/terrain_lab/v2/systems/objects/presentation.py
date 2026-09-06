"""Generic normalized-pack Q7 layout experiments; all outputs stay Q7-owned."""
from __future__ import annotations
import argparse, hashlib, json, math, struct, statistics
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from clearance_adapter import intersects,local_witness

ROOT=Path(__file__).resolve().parents[5]
V2=Path('Renderer/terrain_lab/v2')
OWN=V2/'systems/objects'
FIX=V2/'fixtures/objects'
AUD=V2/'audits/objects'
PACK=Path('Renderer/packs/CityComponentsNormalized')
WORLD_DRAWS=defaultdict(list)
ACTIVE_CORRIDORS=None
def read(p): return json.loads((ROOT/p).read_text())
def write(p,x):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def rotate(p,a):
    c,s=math.cos(a),math.sin(a)
    return [p[0]*c-p[1]*s,p[0]*s+p[1]*c,p[2]]
def project(p):return [(p[0]-p[1])*64,(p[0]+p[1])*32-p[2]*80.9543]
def world_at(p,center,viewport):
    difference=(center[0]-viewport[0]*.5)/64
    total=(center[1]-viewport[1]*.5)/32
    return [p[0]+(total+difference)*.5,p[1]+(total-difference)*.5,p[2]]

@lru_cache(None)
def component(asset,pack=PACK):
    manifest=read(pack/'manifest.json'); l=read(pack/manifest['assets'][asset]['landmark'])
    parts=[]
    for b in l['draw_bindings']:
        if 'worked' not in b['states']:continue
        mesh=read(pack/l['components']['geometry'][b['geometry']])
        mat=read(pack/l['components']['materials'][b['material']])
        for channel in mat['channels'].values():
            if isinstance(channel,dict) and 'texture' in channel:channel['texture']=str(pack/channel['texture'])
        parts.append((mesh,mat))
    points=[v['position'] for mesh,_ in parts for v in mesh['vertices']]
    lo=[min(v[i] for v in points) for i in range(3)]
    hi=[max(v[i] for v in points) for i in range(3)]
    return dict(id=asset,parts=parts,lo=lo,hi=hi,sockets=l.get('attachment_points',[]))

def layout(assets,size,recipe='stable',factor=1,exclusions=None):
    """Stable slots: largest skyline body first, later growth only appends."""
    counts=[4,7,11];result=[]
    # Uniform per-pool scale; preserve relative source building proportions.
    span=max(max(a['hi'][0]-a['lo'][0],a['hi'][1]-a['lo'][1]) for a in assets)
    typical_height=statistics.median(a['hi'][2]-a['lo'][2] for a in assets)
    # Long low compounds must not shrink every cottage to a few pixels. This
    # modest per-pool uniform floor is capped by the largest source footprint.
    scale=max(.205/span,min(12/(80.9543*typical_height),.32/span))*factor
    order=sorted(assets,key=lambda a: (-(a['hi'][2]-a['lo'][2]),a['id']))
    placed=[]
    def box(a,x,y,rotation,sc,screen=False):
        points=[]
        for mesh,_ in a['parts']:
            for v in mesh['vertices']:
                p=[v['position'][0]-(a['lo'][0]+a['hi'][0])/2,v['position'][1]-(a['lo'][1]+a['hi'][1])/2,v['position'][2]-a['lo'][2]]
                p=[t*sc for t in rotate(p,rotation)];p[0]+=x;p[1]+=y
                points.append(project(p) if screen else p)
        return [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
    def area(b):return (b[2]-b[0])*(b[3]-b[1])
    def overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    # Grow the same deterministic sequence only as far as requested. Future
    # unbuildable slots cannot reject an otherwise legal town.
    for i in range(counts[size]):
        a=order[i%len(order)]
        if recipe=='golden':
            if i>=counts[size]:break
            angle=i*2.39996323+.2;r=[.25,.33,.41][size]*math.sqrt(i/max(1,counts[size]-1))
            x,y=math.cos(angle)*r,math.sin(angle)*r*.78
            rotation=angle+.55;sc=scale*[.92,1,1.08][size]
        else:
            sc=scale;rotation=0 if i%2==0 else math.pi/2
            local=box(a,0,0,rotation,sc);proj=box(a,0,0,rotation,sc,True)
            candidates=[]
            for ix in range(-7,8):
                for iy in range(-7,8):
                    x,y=ix*.07,iy*.07
                    if abs(x)+abs(y)>(.91 if recipe=='compact' else .77):continue
                    b=[local[0]+x,local[1]+y,local[2]+x,local[3]+y]
                    expanded=[b[0]-.012,b[1]-.012,b[2]+.012,b[3]+.012]
                    if any(overlap(expanded,q[0])>0 for q in placed):continue
                    if exclusions and intersects(expanded,exclusions):continue
                    dx,dy=project([x,y,0]);pb=[proj[0]+dx,proj[1]+dy,proj[2]+dx,proj[3]+dy]
                    occlusion=sum(overlap(pb,q[1])/min(area(pb),area(q[1])) for q in placed)
                    score=x*x+y*y+occlusion*(.065 if recipe=='compact' else .22)+(x+y)*.025
                    if recipe=='compact' and placed:
                        # Favor one settlement on a legal side of a corridor.
                        # Physical source clearance still precedes this score.
                        gap=min(max(0,b[0]-q[0][2],q[0][0]-b[2])**2+max(0,b[1]-q[0][3],q[0][1]-b[3])**2 for q in placed)
                        score+=gap*3
                    if not placed:score=(x+.14)**2+(y+.14)**2
                    candidates.append((score,x,y,b,pb))
            if not candidates:raise ValueError('city footprint cannot fit')
            _,x,y,b,pb=min(candidates);placed.append((b,pb))
        result.append(dict(asset=a,slot=i,x=x,y=y,rotation=rotation,scale=sc))
    result=result[:counts[size]]
    return result

def emit_city(draws,records,pool,size,center,viewport,recipe,factor):
    assets=[component(a) for a in read(PACK/'city_catalog.json')['pools'][pool]['components']]
    for inst in layout(assets,size,recipe,factor,ACTIVE_CORRIDORS):
        a=inst['asset'];screen=[];ground=[]
        for mesh,mat in a['parts']:
            channels=mat['channels'];base=channels['base_color']['texture']
            em=channels['emissive']['texture'] if 'emissive' in channels else ''
            verts=[];world=[]
            for v in mesh['vertices']:
                pos=[v['position'][0]-(a['lo'][0]+a['hi'][0])/2,v['position'][1]-(a['lo'][1]+a['hi'][1])/2,v['position'][2]-a['lo'][2]]
                p=[x*inst['scale'] for x in rotate(pos,inst['rotation'])]
                p[0]+=inst['x'];p[1]+=inst['y'];px,py=project(p)
                sx,sy=center[0]+px,center[1]+py
                screen.append([sx,sy]);ground.append(p)
                depth=.94-(center[1]+(p[0]+p[1])*32)/viewport[1]*.75-p[2]*.20732
                normal=rotate(v['normal'],inst['rotation'])
                world.append([*world_at(p,center,viewport),*v['uv0'],*normal])
                verts.append([sx/viewport[0]*2-1,1-sy/viewport[1]*2,depth,*v['uv0'],*normal,-float(inst['slot']+1)])
            draws[(base,em)].extend(verts[i] for i in mesh['topology']['indices'])
            WORLD_DRAWS[(base,em)].extend(world[i] for i in mesh['topology']['indices'])
        records.append(dict(pool=pool,size=size,recipe=recipe,slot=inst['slot'],asset=a['id'],scale=inst['scale'],rotation=inst['rotation'],translation=[inst['x'],inst['y']],
            bounds=[min(p[0] for p in screen),min(p[1] for p in screen),max(p[0] for p in screen),max(p[1] for p in screen)],
            footprint=[min(p[0] for p in ground),min(p[1] for p in ground),max(p[0] for p in ground),max(p[1] for p in ground)],ground_min=min(p[2] for p in ground),socket_count=len(a['sockets'])))

def bundle(path,group):
    """Read the existing generic FeatureBundle without changing its pack."""
    data=(ROOT/path).read_bytes();offset=8
    def unpack(fmt):
        nonlocal offset
        v=struct.unpack_from('<'+fmt,data,offset);offset+=struct.calcsize('<'+fmt);return v
    def string():
        nonlocal offset
        n,=unpack('I');s=data[offset:offset+n].decode();offset+=n;return s
    version,nt,na,ng=unpack('4I');assert data[:8]==b'C3XVEG1\0' and version==1
    textures=[string() for _ in range(nt)];assets=[]
    for _ in range(na):
        aid=string();ti,nv,ni=unpack('3I')
        verts=[]
        for _ in range(nv):
            v=unpack('8f');verts.append(dict(position=list(v[:3]),normal=list(v[3:6]),uv0=list(v[6:])))
        inds=list(unpack(str(ni)+'I'))
        assets.append((dict(vertices=verts,topology={'indices':inds}),dict(channels={'base_color':{'texture':str(path.parent/textures[ti])}}),aid))
    groups={}
    for _ in range(ng):
        gn=string();n,=unpack('I');groups[gn]=[unpack('IffIIIIff') for _ in range(n)]
    assert offset==len(data)
    parts=[]
    for placement in groups[group]:
        mesh,mat,aid=assets[placement[0]]
        # Placement scale is part of the baked compound's normalized transform.
        mesh=dict(mesh,vertices=[dict(v,position=[x*placement[1] for x in v['position']]) for v in mesh['vertices']])
        parts.append((mesh,mat))
    pts=[v['position'] for mesh,_ in parts for v in mesh['vertices']]
    return dict(id=group,parts=parts,lo=[min(v[i] for v in pts) for i in range(3)],hi=[max(v[i] for v in pts) for i in range(3)],sockets=[])

def emit_object(draws,records,a,category,center,viewport,height,rotation=0):
    # Exactly one scale for the entire compound, never per attachment or axis.
    scale=height/(a['hi'][2]-a['lo'][2])/80.9543
    if category in ('fish','mine'):
        pts=[project(rotate(v['position'],rotation)) for mesh,_ in a['parts'] for v in mesh['vertices']]
        scale=min(scale,(32 if category=='fish' else 42)/(max(p[0] for p in pts)-min(p[0] for p in pts)))
    screen=[]
    for mesh,mat in a['parts']:
        base=mat['channels']['base_color']['texture'];vertices=[];world=[]
        for v in mesh['vertices']:
            p=[v['position'][0]-(a['lo'][0]+a['hi'][0])/2,v['position'][1]-(a['lo'][1]+a['hi'][1])/2,v['position'][2]-a['lo'][2]]
            p=[x*scale for x in rotate(p,rotation)];px,py=project(p)
            sx,sy=center[0]+px,center[1]+py;screen.append([sx,sy])
            depth=.94-(center[1]+(p[0]+p[1])*32)/viewport[1]*.75-p[2]*.20732
            world.append([*world_at(p,center,viewport),*v['uv0'],*rotate(v['normal'],rotation)])
            vertices.append([sx/viewport[0]*2-1,1-sy/viewport[1]*2,depth,*v['uv0'],*rotate(v['normal'],rotation),-100.0 if category=='mine' else 29.0])
        draws[(base,'')].extend(vertices[i] for i in mesh['topology']['indices'])
        WORLD_DRAWS[(base,'')].extend(world[i] for i in mesh['topology']['indices'])
    records.append(dict(category=category,asset=a['id'],scale=scale,rotation=rotation,ground_min=0,bounds=[min(p[0] for p in screen),min(p[1] for p in screen),max(p[0] for p in screen),max(p[1] for p in screen)]))

def emit_walls(draws,records,pool,size,center,viewport):
    city=[r for r in records if r.get('pool')==pool and r.get('size')==size and 'footprint' in r]
    # Shared enclosure bounds; no per-axis deformation of any source wall piece.
    x0=min(r['footprint'][0] for r in city)-.06;x1=max(r['footprint'][2] for r in city)+.06
    y0=min(r['footprint'][1] for r in city)-.06;y1=max(r['footprint'][3] for r in city)+.06
    era=pool.split('/')[-1];era=era if era in ('ancient','medieval') else 'industrial'
    pack=Path('Renderer/packs/CityAdjunctsNormalized')
    seg=component('city/walls/'+era+'/segment_01',pack);gate=component('city/walls/'+era+'/gate',pack)
    scale=1.3;length=(seg['hi'][1]-seg['lo'][1])*scale;gatewidth=(gate['hi'][0]-gate['lo'][0])*scale
    def place(a,x,y,angle):
        px,py=project([x,y,0]);height=(a['hi'][2]-a['lo'][2])*scale*80.9543
        emit_object(draws,records,a,'wall',[center[0]+px,center[1]+py],viewport,height,angle)
    # Back and side walls are source segments. Front wall has an actual gate.
    for axis,fixed,lo,hi in [('y',x0,y0,y1),('y',x1,y0,y1),('x',y0,x0,x1),('x',y1,x0,x1)]:
        intervals=[(lo,hi)]
        if axis=='x' and (fixed==y1 or ACTIVE_CORRIDORS):
            mid=0 if ACTIVE_CORRIDORS else (lo+hi)/2;place(gate,mid,fixed,0)
            intervals=[(lo,mid-gatewidth/2),(mid+gatewidth/2,hi)]
        for start,end in intervals:
            n=max(1,math.ceil((end-start)/length));spacing=(end-start)/n
            for i in range(n):
                t=start+(i+.5)*spacing
                place(seg,fixed if axis=='y' else t,t if axis=='y' else fixed,0 if axis=='y' else math.pi/2)


def generate(name='micro',factor=1,all_pools=False,context=False,recipe='stable',objects=False,pool=None,walls=False,corridor=False,relief=False):
    global ACTIVE_CORRIDORS
    ACTIVE_CORRIDORS=local_witness() if corridor else None
    WORLD_DRAWS.clear()
    pools=sorted(read(PACK/'city_catalog.json')['pools']) if all_pools else ['city/pool/european/medieval']
    if pool:pools=['city/pool/'+pool]
    viewport=[768,len(pools)*320 if all_pools else 384]
    if context:viewport=[592,376]
    if objects:viewport=[768,384]
    draws=defaultdict(list);records=[];cells=[]
    for row,pool in enumerate(pools if not context and not objects else []):
        recipes=['stable'] if all_pools else ['golden','stable']
        for ri,recipe in enumerate(recipes):
            for size in range(3):
                center=[128+size*256,(row*320+190) if all_pools else (ri*180+125)]
                emit_city(draws,records,pool,size,center,viewport,recipe,factor)
                cells.append(dict(pool=pool,size=size,recipe=recipe,center=center))
                if walls:emit_walls(draws,records,pool,size,center,viewport)
    if context:
        for size,center in [(0,[296,91]),(2,[360,251])]:
            emit_city(draws,records,pools[0],size,center,viewport,recipe,factor)
            cells.append(dict(pool=pools[0],size=size,recipe=recipe,center=center))
            if walls:emit_walls(draws,records,pools[0],size,center,viewport)
        emit_object(draws,records,bundle(Path('Renderer/packs/ResourceNormalized/resource_runtime.bin'),'horses'),'horse-resource',[168,155],viewport,18,math.pi/2)
        emit_object(draws,records,bundle(Path('Renderer/packs/CompoundUnitLab/unit_horseman_runtime.bin'),'idle_0'),'mounted-unit',[296,187],viewport,34,math.pi/2)
    if context and relief:
        emit_object(draws,records,bundle(Path('Renderer/packs/ImprovementsNormalized/mine_runtime.bin'),'mine_0'),'mine',[296,219],viewport,22)
    if objects:
        sources=[('horse-resource','Renderer/packs/ResourceNormalized/resource_runtime.bin','horses',18),('mounted-unit','Renderer/packs/CompoundUnitLab/unit_horseman_runtime.bin','idle_0',34),('infantry','Renderer/packs/UnitFamilyLab/unit_infantry_runtime.bin','idle_0',30),('fish','Renderer/packs/ResourceNormalized/resource_runtime.bin','fish',10)]
        for i,(cat,path,group,height) in enumerate(sources):
            emit_object(draws,records,bundle(Path(path),group),cat,[96+i*192,100],viewport,height)
        for facing in range(8):
            emit_object(draws,records,bundle(Path(sources[1][1]),'idle_0'),'facing-'+str(facing),[48+96*facing,270],viewport,34,facing*math.pi/4)
    payload=bytearray(struct.pack('<II',0x37515043,len(draws)))
    for (base,em),vertices in sorted(draws.items()):
        for s in (base,em):b=s.encode();payload+=struct.pack('<I',len(b))+b
        payload+=struct.pack('<I',len(vertices))
        for v in vertices:payload+=struct.pack('<9f',*v)
    out=FIX/'generated'/name; (ROOT/out).mkdir(parents=True,exist_ok=True)
    (ROOT/out/'geometry.bin').write_bytes(payload)
    write(out/'world.json',dict(schema='c3x.q7.world_triangles.v1',classification='source_adaptation',vertex_fields=['world_x','world_y','world_z','u','v','normal_x','normal_y','normal_z'],projection={'tile_pixels':[128,64],'z_pixels_per_tile':80.9543},draws=[dict(base_color=base,emissive=em,vertices=vertices) for (base,em),vertices in sorted(WORLD_DRAWS.items())]))
    if ACTIVE_CORRIDORS:write(out/'clearance.json',ACTIVE_CORRIDORS)
    terrain='C3X_BIQ_TERRAIN_WINDOW_V2,4,4,16,0,0,60,60,0\n'+''.join(f'{x},{y},{x+y},{10+x-y},2,2,0,0,0\n' for y in range(4) for x in range(4))
    if relief:terrain=terrain.replace('1,1,2,10,2,2,0,0,0','1,1,2,10,2,6,0,0,0')
    (ROOT/out/'terrain.csv').write_text(terrain)
    baseline=read(V2/'tests/platform/micro.fixture.json')
    baseline.update(id=name,track='Q7-presentation',viewport=viewport,terrain=str(out/'terrain.csv'),modules=[str(OWN/'presentation.module.json')],references=['civ6.city_object_scale','civ3.real_gameplay_layout'],isolations=['features'],scenarios={'geometry':str(out/'geometry.bin')})
    baseline['packs']['cities']=str(PACK)
    if walls:baseline['packs']['walls']='Renderer/packs/CityAdjunctsNormalized'
    if relief:baseline['packs']['improvements']='Renderer/packs/ImprovementsNormalized'
    if context:
        baseline['modules'].insert(0,str(OWN/'terrain.module.json'))
    if context or objects:
        for key,path in [('resources','ResourceNormalized'),('units','UnitFamilyLab'),('compound_units','CompoundUnitLab')]:baseline['packs'][key]='Renderer/packs/'+path
    write(out/'fixture.json',baseline)
    write(out/'layout.json',dict(schema='c3x.q7.layout.v1',projection={'tile_pixels':[128,64],'z_pixels_per_tile':80.9543},factor=factor,cells=cells,components=records,source_catalog_sha256=sha(PACK/'city_catalog.json'),geometry_sha256=sha(out/'geometry.bin'),provisional=True))
    print(out/'fixture.json')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='micro');p.add_argument('--factor',type=float,default=1);p.add_argument('--all-pools',action='store_true');p.add_argument('--context',action='store_true');p.add_argument('--objects',action='store_true');p.add_argument('--pool');p.add_argument('--walls',action='store_true');p.add_argument('--corridor',action='store_true');p.add_argument('--relief',action='store_true');p.add_argument('--recipe',default='stable',choices=['stable','golden']);a=p.parse_args()
    generate(a.name,a.factor,a.all_pools,a.context,a.recipe,a.objects,a.pool,a.walls,a.corridor,a.relief)
