"""Matched source mountain/volcano size pass; terrain identities remain fixed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'qa'))
from coastal_pass import save
sys.path.insert(0,str(V2/'app'))
import real_map

def fresh_relief():
    out=V2/'fixtures/beauty/relief-size-foundation/freshrelief'
    if (out/'fixture.json').exists():return out
    reg,_=real_map.load_registry()
    request={'source_sha256':reg['source']['sha256'],'regions':[{
        'requested_id':'beauty-freshrelief-100-v1','origin':[46,46],'extent':[10,10],
        'halo':6,'role':'user_evaluation',
        'camera':{'viewport':[1360,800],'zooms':[1,2],'hours':[12,0]}}]}
    path=out/'region-request.json';save(path,request)
    if not any(r['id']=='beauty-freshrelief-100-v1' for r in reg['regions']):real_map.register(path)
    real_map.export('beauty-freshrelief-100-v1',out,'Q8-beauty',False)
    exported=json.loads((out/'fixture.json').read_text())
    source=V2/'fixtures/beauty/coast-pass-rocks-r8/coastal'
    f=json.loads((source/'fixture.json').read_text())
    for key in ('real_map','terrain','tile_count','viewport','id'):f[key]=exported[key]
    for key,old in f['scenarios'].items():
        header=(ROOT/old).read_text().strip().split(',')
        header[4]=f['real_map']['region']['terrain_sha256']
        p=out/(key+'.csv');save(p,','.join(header)+'\n');f['scenarios'][key]=p.relative_to(ROOT).as_posix()
    m=json.loads((source/'terrain.module.json').read_text())
    shader=(source/'combined.hlsl').read_text().replace('ORIGIN_X 56.5','ORIGIN_X 45.5').replace('ORIGIN_Y 18.5','ORIGIN_Y -0.5')
    save(out/'combined.hlsl',shader);m['shader']=(out/'combined.hlsl').relative_to(ROOT).as_posix()
    save(out/'terrain.module.json',m);f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    (out/'fixture.json').write_text(json.dumps(f,indent=2)+'\n');real_map.validate_provenance(f)
    save(out/'BENCHMARKS.json',{'region':f['real_map']['region'],'projection':m['projection'],
        'gameplay_crop':[360,220,1000,540],
        'selection':'Four source mountains; selected by coverage before size-pass rendering; no region-specific tuning.'})
    return out

def prepare(region,revision):
    scale={'baseline':1,'r1':1.30,'r2':1.30}[revision]
    volcano_scale=1.6 if revision=='r2' else scale
    source=fresh_relief() if region=='freshrelief' else V2/'fixtures/beauty'/(
        'volcano-witness-r2/inland' if region=='volcano' else 'coast-pass-rocks-r8/'+region)
    f=json.loads((source/'fixture.json').read_text())
    m=json.loads((ROOT/f['modules'][0]).read_text())
    out=V2/'fixtures/beauty'/('relief-size-'+revision)/region
    shader=(ROOT/m['shader']).read_text()
    if scale>1:shader=f'#define Q4_BROAD_RELIEF 1\n#define Q4_VOLCANO_FOOTPRINT {0.62/volcano_scale:.9f}\n'+shader
    f['id']='relief-size-'+revision+'-'+region;m['id']=f['id']
    if scale>1:m['relief_scale']=scale
    if revision=='r2':m['volcano_scale']=volcano_scale
    m['omit_replaced_shadow_surface']=1
    m['shader']=(out/'combined.hlsl').relative_to(ROOT).as_posix()
    f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    save(out/'combined.hlsl',shader);save(out/'terrain.module.json',m);save(out/'fixture.json',f)
    adaptation={'classification':'source_height_adaptation',
        'uniform_body_scale':scale,'maximum_neighbor_skirt_tiles':.25,
        'source_height_samples_unchanged':True,'physical_source_reconstruction_proven':False,
        'synthetic':region=='volcano','previous_fixture':(source/'fixture.json').relative_to(ROOT).as_posix(),
        'notes':'User requested larger bodies with modest neighboring overlap. Source material and height use aligned volcano coordinates; camera and terrain are preserved.'}
    if revision=='r2':adaptation['uniform_volcano_scale']=volcano_scale
    save(out/'adaptation.json',adaptation)
    return out/'fixture.json'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['coastal','inland','wilderness','longcoast','freshcoast','freshrelief','volcano','all'],required=True)
    p.add_argument('--revision',choices=['baseline','r1','r2'],default='r1')
    p.add_argument('--hours',nargs='+',type=int,choices=[0,6,12,18],default=[12,0])
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--output-root',type=Path)
    a=p.parse_args()
    for region in (['coastal','inland','wilderness','longcoast','freshcoast','volcano'] if a.region=='all' else [a.region]):
        f=prepare(region,a.revision)
        if a.prepare_only:continue
        out=(a.output_root or V2/'audits/beauty/out'/('relief-size-'+a.revision))/region
        if (out/'report.json').exists():raise ValueError('Preserved render already exists')
        subprocess.run([sys.executable,str(V2/'app/runner.py'),'compose','--fixture',str(f),
            '--candidate','relief-size-'+a.revision,'--output',str(out),'--hours',*map(str,a.hours)],check=True,cwd=ROOT)
        for bmp in out.glob('h*-z*-pan00.bmp'):
            subprocess.run(['sips','-s','format','png',str(bmp),'--out',str(bmp.with_suffix('.png'))],check=True,stdout=subprocess.DEVNULL)

if __name__=='__main__':main()
