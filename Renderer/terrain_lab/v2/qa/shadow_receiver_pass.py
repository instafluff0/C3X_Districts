"""Compose bounded receiver-offset correction against the retained relief pass."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'app'));sys.path.insert(0,str(V2/'qa'))
import real_map
from coastal_pass import save

REGIONS=['coastal','inland','wilderness','longcoast','freshcoast','freshrelief','combinedvolcano','freshshadow']

def foundation():
    out=V2/'fixtures/beauty/shadow-receiver-foundation/freshshadow'
    if (out/'fixture.json').exists():return out
    reg,_=real_map.load_registry()
    request={'source_sha256':reg['source']['sha256'],'regions':[{
        'requested_id':'beauty-freshshadow-100-v1','origin':[54,64],'extent':[10,10],
        'halo':6,'role':'user_evaluation',
        'camera':{'viewport':[1360,800],'zooms':[1,2],'hours':[12,0]}}]}
    path=out/'region-request.json';save(path,request)
    if not any(r['id']=='beauty-freshshadow-100-v1' for r in reg['regions']):real_map.register(path)
    real_map.export('beauty-freshshadow-100-v1',out,'Q8-beauty',False)
    exported=json.loads((out/'fixture.json').read_text())
    source=V2/'fixtures/beauty/relief-size-r3/coastal'
    f=json.loads((source/'fixture.json').read_text())
    for key in ('real_map','terrain','tile_count','viewport','id'):f[key]=exported[key]
    for key,old in f['scenarios'].items():
        header=(ROOT/old).read_text().strip().split(',');header[4]=f['real_map']['region']['terrain_sha256']
        p=out/(key+'.csv');save(p,','.join(header)+'\n');f['scenarios'][key]=p.relative_to(ROOT).as_posix()
    m=json.loads((source/'terrain.module.json').read_text())
    shader=(source/'combined.hlsl').read_text().replace('ORIGIN_X 56.5','ORIGIN_X 58.5').replace('ORIGIN_Y 18.5','ORIGIN_Y -5.5')
    save(out/'combined.hlsl',shader);m['shader']=(out/'combined.hlsl').relative_to(ROOT).as_posix()
    save(out/'terrain.module.json',m);f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    (out/'fixture.json').write_text(json.dumps(f,indent=2)+'\n');real_map.validate_provenance(f)
    save(out/'BENCHMARKS.json',{'region':f['real_map']['region'],'projection':m['projection'],
        'gameplay_crop':[360,220,1000,540],
        'selection':'Selected before viewing: 16 desert, 11 hill, 13 mountain and 12 forest tiles; largest origin distance from existing six benchmarks among qualifying regions. No region-specific tuning.'})
    return out

def prepare(region,baseline=False):
    source=foundation() if region=='freshshadow' else V2/'fixtures/beauty/relief-size-r3'/region
    f=json.loads((source/'fixture.json').read_text());m=json.loads((ROOT/f['modules'][0]).read_text())
    name='shadow-receiver-baseline' if baseline else 'shadow-receiver-r1'
    out=V2/'fixtures/beauty'/name/region
    shader=(ROOT/m['shader']).read_text()
    if not baseline:shader='#define Q6_TEXEL_RECEIVER_OFFSET 1\n'+shader
    f['id']=name+'-'+region;m['id']=f['id'];m['shader']=(out/'combined.hlsl').relative_to(ROOT).as_posix()
    f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    save(out/'combined.hlsl',shader);save(out/'terrain.module.json',m);save(out/'fixture.json',f)
    return out/'fixture.json'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=REGIONS+['all'],required=True)
    p.add_argument('--baseline',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--output-root',type=Path)
    a=p.parse_args();name='shadow-receiver-baseline' if a.baseline else 'shadow-receiver-r1'
    for region in REGIONS if a.region=='all' else [a.region]:
        f=prepare(region,a.baseline)
        if a.prepare_only:continue
        out=(a.output_root or V2/'audits/beauty/out'/name)/region
        if (out/'report.json').exists():raise ValueError('Preserved render already exists')
        subprocess.run([sys.executable,str(V2/'app/runner.py'),'compose','--fixture',str(f),
            '--candidate',name,'--output',str(out),'--hours','12','0'],check=True,cwd=ROOT)
        for bmp in out.glob('h*-z*-pan00.bmp'):
            subprocess.run(['sips','-s','format','png',str(bmp),'--out',str(bmp.with_suffix('.png'))],check=True,stdout=subprocess.DEVNULL)

if __name__=='__main__':main()
