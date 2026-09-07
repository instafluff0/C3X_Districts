"""Freeze a source-selected, wholly unseen material regression region."""
import json
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'app'))
import real_map
from coastal_pass import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',choices=['freshground','freshcache','freshwater'],default='freshground')
    args=parser.parse_args()
    name=args.name
    campaign={'freshground':'surface-decals-foundation','freshcache':'cached-normal-foundation',
              'freshwater':'water-natural-foundation'}[name]
    region_id=f'beauty-{name}-100-v1'
    out=V2/'fixtures/beauty'/campaign/name
    if (out/'BENCHMARKS.json').exists():raise ValueError('holdout already frozen')
    reg,data=real_map.load_registry();seen=set();origins=[]
    for r in reg['regions']:
        if 'beauty' in r['id'] and r['id'] != region_id:
            origins.append(r['origin'])
            seen|={(t['sourceX'],t['sourceY']) for t in real_map.region_tiles(data,r['origin'],r['extent'],0)}
    candidates=[]
    for x in range(0,100,2):
        for y in range(16,84,2):
            tiles=real_map.region_tiles(data,[x,y],[10,10],0);m=real_map.metrics(tiles);c=m['terrain']
            if name=='freshwater':
                water=sum(c.get(t,0) for t in ('coast','sea','ocean'))
                if not 30<=water<=75 or c.get('coast',0)<20 or c.get('sea',0)+c.get('ocean',0)<15:continue
            elif c.get('grassland',0)<15 or c.get('plains',0)<15 or c.get('hills',0)<3:continue
            if any((t['sourceX'],t['sourceY']) in seen for t in tiles):continue
            distance=min(min(abs(x-ox),100-abs(x-ox))**2+(y-oy)**2 for ox,oy in origins)
            score=c.get('coast',0)*10000+water*100 if name=='freshwater' else distance
            candidates.append((score,x,y))
    score,x,y=max(candidates)
    request={'source_sha256':reg['source']['sha256'],'regions':[{
        'requested_id':region_id,'origin':[x,y],'extent':[10,10],
        'halo':6,'role':'user_evaluation','camera':{'viewport':[1360,800],'zooms':[1,2],'hours':[12,0]}}]}
    save(out/'region-request.json',request)
    existing=next((r for r in reg['regions'] if r['id']==region_id),None)
    if existing:
        for key in ('origin','extent','halo','camera'):
            if existing[key]!=request['regions'][0][key]:raise ValueError('registered holdout differs')
        if existing['request_sha256']!=real_map.file_hash(out/'region-request.json'):
            raise ValueError('registered request differs')
    else:real_map.register(out/'region-request.json')
    real_map.export(region_id,out,'Q8-beauty',False)
    exported=json.loads((out/'fixture.json').read_text())
    old=V2/'fixtures/beauty/river-corridor-r3/coastal'
    f=json.loads((old/'fixture.json').read_text());m=json.loads((old/'terrain.module.json').read_text())
    for key in ('real_map','terrain','tile_count','viewport','id'):f[key]=exported[key]
    for key,path in f['scenarios'].items():
        header=(ROOT/path).read_text().strip().split(',');header[4]=f['real_map']['region']['terrain_sha256']
        save(out/(key+'.csv'),','.join(header)+'\n');f['scenarios'][key]=(out/(key+'.csv')).relative_to(ROOT).as_posix()
    shader=(ROOT/m['shader']).read_text().replace('ORIGIN_X 56.5',f'ORIGIN_X {(x+y)/2-.5}').replace('ORIGIN_Y 18.5',f'ORIGIN_Y {(x-y)/2-.5}')
    save(out/'combined.hlsl',shader);m['shader']=(out/'combined.hlsl').relative_to(ROOT).as_posix()
    m['id']=campaign+'-'+name;f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    save(out/'terrain.module.json',m)
    # export() created an intermediate fixture; replace it with the composed recipe.
    (out/'fixture.json').write_text(json.dumps(f,indent=2)+'\n')
    real_map.validate_provenance(f)
    selection=('Before viewing: maximize coast count, then total water count, then origin coordinates among even-coordinate 10x10 crops with 30-75 water, >=20 coast, >=15 sea/ocean and ALL 100 tiles outside prior beauty regions.' if name=='freshwater' else
        'Before viewing: max minimum wrapped origin distance among even-coordinate 10x10 crops with >=15 grassland, >=15 plains, >=3 hills and ALL 100 tiles outside prior beauty regions.')
    save(out/'BENCHMARKS.json',{'region':f['real_map']['region'],'projection':m['projection'],
        'selection':selection,
        'previously_unseen_tiles':100,'selection_score':score,'gameplay_crop':[360,220,1000,540]})
    print(json.dumps({'origin':[x,y],'coverage':f['real_map']['region']['coverage'],'unseen_tiles':100}))


if __name__=='__main__':main()
