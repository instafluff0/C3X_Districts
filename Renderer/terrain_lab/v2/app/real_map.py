#!/usr/bin/env python3
"""Verified offline BIQ registry and owner-local replay recipes; never edits BIQ."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
from cache import canonical, digest, file_hash

ROOT = Path(__file__).resolve().parents[4]
APP = Path(__file__).resolve().parent
REGISTRY = APP.parent / 'shared/real_map/registry_v1.json'
NAMES = ['desert','plains','grassland','tundra','floodplain','hills','mountain','forest','jungle','marsh','volcano','coast','sea','ocean']
PROFILE = 'q0_real_map_v1'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))

def lookup(data, x, y):
    if data['wrap_x']: x %= data['width']
    if data['wrap_y']: y %= data['height']
    return data['_tiles'].get((x,y))

def prepare(data):
    data['_tiles'] = {(t['sourceX'],t['sourceY']):t for t in data['tiles']}
    if len(data['_tiles']) != data['width']*data['height']//2:
        raise ValueError('missing or duplicate source tiles')
    return data

def region_tiles(data, origin, extent, halo):
    x,y=origin; cols,rows=extent
    result=[]
    for r in range(-halo,rows+halo):
        for c in range(-halo,cols+halo):
            t=lookup(data,x+c+r,y+c-r)
            if t is None:
                raise ValueError('region/halo extends beyond non-wrapping map edge')
            result.append(dict(t,column=c,row=r,visible=0<=c<cols and 0<=r<rows))
    return result

def metrics(tiles):
    tiles=[t for t in tiles if t['visible']]
    by={(t['column'],t['row']):t for t in tiles}
    transitions=coast=0
    for (c,r),t in by.items():
        for q in ((c+1,r),(c,r+1)):
            n=by.get(q)
            if n:
                transitions+=n['real']!=t['real']
                coast+=(n['base']>=11)!=(t['base']>=11)
    counts=Counter(t['real'] for t in tiles)
    return dict(terrain={NAMES[k] if k<len(NAMES) else str(k):v for k,v in sorted(counts.items())},river_tiles=sum(bool(t['riverMask']) for t in tiles),land_water_edges=coast,terrain_edges=transitions)

def csv_bytes(data, region):
    tiles=region_tiles(data,region['origin'],region['extent'],region['halo'])
    ordered=[t for t in tiles if t['visible']]+[t for t in tiles if not t['visible']]
    c,r=region['extent']; x,y=region['origin']
    lines=[f'C3X_BIQ_TERRAIN_WINDOW_V2,{c},{r},{c*r},{x//2},{y},{data["width"]},{data["height"]},{len(tiles)-c*r}']
    lines += [','.join(str(t[k]) for k in ('column','row','sourceX','sourceY','base','real','bonus','overlays','riverMask')) for t in ordered]
    return ('\n'.join(lines)+'\n').encode()

def load_registry(path=REGISTRY):
    reg=json.loads(path.read_text())
    dataset=ROOT/reg['dataset_cache']
    if not dataset.is_file() or file_hash(dataset)!=reg['dataset_payload_sha256']:
        raise ValueError('missing/corrupt cached dataset; explicitly refresh verified source')
    data=json.loads(dataset.read_text())
    if data['source']!=reg['source'] or data['parser']!=reg['parser']:
        raise ValueError('dataset identity or parser contract drift')
    return reg,prepare(data)

def publish(source):
    if source.name.lower()!='test.biq':
        raise ValueError('real registry source must be the explicitly selected test.biq')
    source_hash=file_hash(source)
    tmp=APP/'.local/real_map/import.json'
    subprocess.run(['node',str(APP/'import_biq.js'),str(source),str(tmp)],check=True,cwd=ROOT)
    data=json.loads(tmp.read_text())
    if data['source']['sha256']!=source_hash or file_hash(source)!=source_hash:
        raise ValueError('source identity changed while importing')
    payload=canonical(data)
    dataset=APP/'.local/real_map'/source_hash/(digest(payload)+'.json')
    save(dataset,data)
    prepare(data)
    choices=[]
    for y in range(data['height']):
        for x in range(y%2,data['width'],2):
            region=dict(origin=[x,y],extent=[4,4],halo=2)
            try: m=metrics(region_tiles(data,region['origin'],region['extent'],2))
            except ValueError: continue
            choices.append((region,m))
    def score(m):
        return len(m['terrain'])*30+m['terrain_edges']+m['land_water_edges']*10+m['river_tiles']*5
    selections=[]
    for name,allowed in [('mixed',lambda r,m:True),('relief',lambda r,m:any(k in m['terrain'] for k in ('mountain','hills'))),('wrap',lambda r,m:r['origin'][0]+6>=data['width'])]:
        candidates=[p for p in choices if allowed(*p)]
        if not candidates: continue
        region,m=max(candidates,key=lambda p:score(p[1]))
        selections.append(dict(region,id=name,coverage=m,role='primary'))
        # Fixed adjacent holdout selected before any visual tuning. Full halo required.
        for dx,dy in [(4,4),(-4,-4),(4,-4),(-4,4)]:
            origin=[region['origin'][0]+dx,region['origin'][1]+dy]
            if data['wrap_x']: origin[0]%=data['width']
            if data['wrap_y']: origin[1]%=data['height']
            try: hm=metrics(region_tiles(data,origin,[4,4],2))
            except ValueError: continue
            selections.append(dict(region,id=name+'-holdout',origin=origin,coverage=hm,role='fixed_neighbor_holdout'))
            break
    for r in selections:
        r['terrain_sha256']=digest(csv_bytes(data,r))
        r['coordinate_basis']='raw_BIQ_xy; column=(+1,+1), row=(+1,-1)'
        r['wrap']={'x':data['wrap_x'],'y':data['wrap_y']}
        r['camera']={'viewport':[768,512],'zooms':[1,2],'hours':[12,18,0,6]}
    inventory=Counter(t['real'] for t in data['tiles'])
    reg=dict(schema='c3x.lab_v2.real_map_registry.v1',profile=PROFILE,source=data['source'],parser=data['parser'],source_resolution='one distinct installed test.biq; read-only copied through configured VM dispatcher',historical_source='Ancient Treasures export is a separate immutable v1 reference, not this dataset',dataset_cache=dataset.relative_to(ROOT).as_posix(),dataset_payload_sha256=digest(payload),dimensions=[data['width'],data['height']],wrap={'x':data['wrap_x'],'y':data['wrap_y']},regions=selections,coverage={name:{'state':'present' if inventory[i] else 'absent','tiles':inventory[i]} for i,name in enumerate(NAMES)},source_section_counts=data['section_counts'],fine_topology_coverage='coves, channels, islands, mouths and rocky/sandy coast classifications pending Q3 audit; do not infer from broad tags')
    save(REGISTRY,reg)
    print(json.dumps({'registry':str(REGISTRY.relative_to(ROOT)),'source':reg['source'],'regions':selections,'coverage':reg['coverage']},indent=2))

def overlay_for(reg,region,data):
    tiles=[t for t in region_tiles(data,region['origin'],region['extent'],0) if t['base']<11]
    legal=[t for t in tiles if t['real'] not in (5,6,9,10)]
    city=legal[len(legal)//2] if legal else None
    land={(t['column'],t['row']):t for t in tiles}
    routes=[]
    for (c,r),t in sorted(land.items()):
        for q in ((c+1,r),(c,r+1)):
            if q in land:
                routes.append({'from':[c,r],'to':list(q),'kind':'road','domain':'land'})
    objects=[] if city is None else [{'kind':'city','tile':[city['column'],city['row']],'domain':'land','stable_id':1}]
    return dict(schema='c3x.lab_v2.augmentation.v1',label='deterministic Lab augmentation, not captured Civ III state',source_sha256=reg['source']['sha256'],region_id=region['id'],terrain_sha256=region['terrain_sha256'],profile=PROFILE,seed=0,objects=objects,routes=routes)

def resolved_region(reg,data,rid,halo=None):
    source=next((r for r in reg['regions'] if r['id']==rid),None)
    if source is None: raise ValueError('unknown registered region')
    region=dict(source)
    if halo is not None:
        if type(halo)!=int or not source['halo']<=halo<=8:
            raise ValueError('halo must preserve registered neighbors and be at most eight')
        region['halo']=halo
    region['terrain_sha256']=digest(csv_bytes(data,region))
    return region

def validate_provenance(f):
    p=f['real_map']; reg,data=load_registry()
    if p['source_sha256']!=reg['source']['sha256'] or p['dataset_payload_sha256']!=reg['dataset_payload_sha256'] or p['parser']!=reg['parser'] or p['profile']!=PROFILE:
        raise ValueError('stale real-map source/parser/profile identity')
    region=resolved_region(reg,data,p['region_id'],p['region']['halo'])
    if region is None or region!=p['region']:
        raise ValueError('unknown or stale region/halo contract')
    if f['tile_count']!=region['extent'][0]*region['extent'][1]:raise ValueError('real-map fixture extent/count mismatch')
    expected=csv_bytes(data,region)
    if digest(expected)!=region['terrain_sha256'] or (ROOT/f['terrain']).read_bytes()!=expected:
        raise ValueError('real-map terrain differs from verified source/halo')
    if p.get('overlay'):
        overlay_path=ROOT/p['overlay']
        overlay=json.loads(overlay_path.read_text())
        if file_hash(overlay_path)!=p['overlay_sha256']:
            raise ValueError('corrupt real-map augmentation')
        if (overlay.get('schema') != 'c3x.lab_v2.augmentation.v1' or
            overlay.get('source_sha256') != reg['source']['sha256'] or
            overlay.get('region_id') != region['id'] or
            overlay.get('terrain_sha256') != region['terrain_sha256'] or
            not overlay.get('profile') or not isinstance(overlay.get('seed'),int)):
            raise ValueError('stale/mismatched real-map augmentation')
        if overlay['profile'] != PROFILE:
            from runner import owned
            if overlay.get('owner') != f['track']:
                raise ValueError('augmentation profile owner mismatch')
            owned(overlay_path, f['track'])
        elif overlay != overlay_for(reg,region,data):
            raise ValueError('Q0 augmentation recipe drift')
        if set(p['scenario_hashes']) != set(f['scenarios']):
            raise ValueError('every augmented scenario requires a hash')
        tiles={(t['column'],t['row']):t for t in region_tiles(data,region['origin'],region['extent'],0)}
        for key,h in p['scenario_hashes'].items():
            path=ROOT/f['scenarios'][key]
            if file_hash(path)!=h:
                raise ValueError('augmentation scenario hash mismatch')
            lines=path.read_text().splitlines(); header=lines[0].split(',')
            if len(header)!=6 or list(map(int,header[1:3]))!=region['extent'] or int(header[3])!=len(lines)-1 or header[4]!=region['terrain_sha256']:
                raise ValueError('scenario source/extent/count mismatch')
            for line in lines[1:]:
                v=list(map(int,line.split(',')))
                t=tiles.get(tuple(v[:2]))
                if t is None: raise ValueError('augmentation placement outside region')
                if key in ('roads','railroads'):
                    other=tiles.get(tuple(v[2:4]))
                    if other is None or t['base']>=11 or other['base']>=11 or abs(v[0]-v[2])+abs(v[1]-v[3])!=1:
                        raise ValueError('illegal real-region route domain/adjacency')
                elif key in ('units','resources'):
                    water_kind=4 if key=='units' else 7
                    if (v[2]==water_kind)!=(t['base']>=11):
                        raise ValueError('illegal unit/resource terrain domain')
                elif t['base']>=11 or (key=='cities' and t['real'] in (5,6,9,10)):
                    raise ValueError('illegal object terrain domain')
    return p

def export(region_id,output,owner,augment,halo=None):
    from runner import owned
    output=output.resolve(); owned(output/'fixture.json',owner)
    reg,data=load_registry()
    region=resolved_region(reg,data,region_id,halo)
    output.mkdir(parents=True,exist_ok=True)
    terrain=output/'terrain.csv'; terrain.write_bytes(csv_bytes(data,region))
    fixture=json.loads((APP.parent/'tests/platform/micro.fixture.json').read_text())
    fixture.update(id='real-'+region_id+('-augmented' if augment else '-terrain'),track=owner,terrain=terrain.relative_to(ROOT).as_posix(),tile_count=region['extent'][0]*region['extent'][1],viewport=region['camera']['viewport'])
    p=dict(source_sha256=reg['source']['sha256'],dataset_payload_sha256=reg['dataset_payload_sha256'],parser=reg['parser'],profile=PROFILE,region_id=region_id,region=region,fixture_class='unaltered_real_terrain')
    fixture['real_map']=p
    if augment:
        overlay=overlay_for(reg,region,data); overlay_path=output/'augmentation.json'; save(overlay_path,overlay)
        p.update(overlay=overlay_path.relative_to(ROOT).as_posix(),overlay_sha256=file_hash(overlay_path),fixture_class='real_terrain_with_lab_augmentations',scenario_hashes={})
        baseline=json.loads((APP.parent/'tests/platform/complete.fixture.json').read_text())
        fixture['packs']=baseline['packs']; fixture['scenarios']={}
        for key,old in baseline['scenarios'].items():
            magic=(ROOT/old).read_text().split(',')[0]
            lines=[]
            if key=='roads':
                lines=[','.join(map(str,[*r['from'],*r['to'],0,1,0,0])) for r in overlay['routes']]
            if key=='cities':
                lines=[','.join(map(str,[*o['tile'],0,0,0,0,0,0,1,0])) for o in overlay['objects']]
            target=output/(key+'.csv')
            target.write_text(f'{magic},{region["extent"][0]},{region["extent"][1]},{len(lines)},{region["terrain_sha256"]},{"lab_augmentation_absolute_time_samples" if key=="units" else "lab_augmentation"}\n'+''.join(s+'\n' for s in lines))
            fixture['scenarios'][key]=target.relative_to(ROOT).as_posix(); p['scenario_hashes'][key]=file_hash(target)
        module=dict(schema='c3x.lab_v2.module.v1',id='real-map-overlay-replay',owner=owner,provider='frozen_l21',contract=1,scene='complete',fit_viewport=True,suppress_territory=True,shader='Renderer/terrain_lab/v2/shaders/common/frozen_l21.hlsl')
        save(output/'module.json',module); fixture['modules']=[(output/'module.json').relative_to(ROOT).as_posix()]
    save(output/'fixture.json',fixture)
    validate_provenance(fixture)
    print((output/'fixture.json').relative_to(ROOT))

def register(request_path):
    reg,data=load_registry(); request=json.loads(request_path.read_text())
    if request['source_sha256'] != reg['source']['sha256']:
        raise ValueError('region request source mismatch')
    for item in request['regions']:
        rid=item['requested_id']
        if not rid or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in rid):
            raise ValueError('invalid region id')
        if any(r['id']==rid for r in reg['regions']):
            raise ValueError('region IDs are immutable; use a new versioned ID')
        r={k:item[k] for k in ('origin','extent','halo')}
        if type(r['halo'])!=int or r['halo']<2 or r['halo']>8 or len(r['extent'])!=2 or any(type(x)!=int or x<4 or x>24 for x in r['extent']) or not 16<=r['extent'][0]*r['extent'][1]<=192:
            raise ValueError('registry requires extent 4..24, at most192 tiles and halo2..8')
        locations=[('',list(r['origin']))]
        if item.get('role')!='user_evaluation':locations.append(('-holdout',[r['origin'][0]+r['extent'][0],r['origin'][1]+r['extent'][1]]))
        for suffix,origin in locations:
            if data['wrap_x']: origin[0]%=data['width']
            if data['wrap_y']: origin[1]%=data['height']
            m=metrics(region_tiles(data,origin,r['extent'],r['halo']))
            n=dict(r,id=rid+suffix,origin=origin,coverage=m,role='fixed_neighbor_holdout' if suffix else item.get('role','primary'),coordinate_basis='raw_BIQ_xy; column=(+1,+1), row=(+1,-1)',wrap=reg['wrap'],camera=item.get('camera',{'viewport':[768,512],'zooms':[1,2],'hours':[12,18,0,6]}),request_sha256=file_hash(request_path))
            n['terrain_sha256']=digest(csv_bytes(data,n));reg['regions'].append(n)
    save(REGISTRY,reg)
    print('Registered: '+', '.join(i['requested_id'] for i in request['regions']))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    imp=sub.add_parser('import'); imp.add_argument('--source',type=Path,default=Path(os.environ.get('C3X_LAB_TEST_BIQ',str(APP/'.local/real_map/test.biq'))))
    exp=sub.add_parser('export'); exp.add_argument('region'); exp.add_argument('--output',type=Path,required=True); exp.add_argument('--owner',default='Q0-platform'); exp.add_argument('--augment',action='store_true');exp.add_argument('--halo',type=int,default=6)
    add=sub.add_parser('register');add.add_argument('--request',type=Path,required=True)
    a=p.parse_args()
    if a.command=='import': publish(a.source.resolve())
    elif a.command=='register': register(a.request)
    else: export(a.region,a.output,a.owner,a.augment,a.halo)
if __name__=='__main__': main()
