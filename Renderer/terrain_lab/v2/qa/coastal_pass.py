"""Matched coast/relief pass on preserved gameplay terrain and one new long coast."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'app'))
import real_map

def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    text=value if isinstance(value,str) else json.dumps(value,indent=2)+'\n'
    if p.exists() and p.read_text()!=text:
        raise ValueError('Immutable recipe differs: '+str(p.relative_to(ROOT)))
    p.write_text(text)

def long_coast():
    out=V2/'fixtures/beauty/coast-pass-foundation/longcoast'
    fixture=out/'fixture.json'
    if fixture.exists():return fixture
    reg,_=real_map.load_registry()
    request={'source_sha256':reg['source']['sha256'],'regions':[{
        'requested_id':'beauty-longcoast-128-v1','origin':[64,38],
        'extent':[16,8],'halo':6,'role':'user_evaluation',
        'camera':{'viewport':[1616,888],'zooms':[1,2],'hours':[12,18,0,6]}}]}
    path=out.parent/'region-request.json';save(path,request)
    if not any(r['id']=='beauty-longcoast-128-v1' for r in reg['regions']):real_map.register(path)
    real_map.export('beauty-longcoast-128-v1',out,'Q8-beauty',False)
    exported=json.loads(fixture.read_text())
    source=V2/'fixtures/beauty/gameplay-100-candidate-v5/coastal'
    f=json.loads((source/'fixture.json').read_text())
    for k in ('real_map','terrain','tile_count','viewport','id'):f[k]=exported[k]
    for key,old in f['scenarios'].items():
        header=(ROOT/old).read_text().splitlines()[0].split(',')
        header[1:5]=['16','8','0',f['real_map']['region']['terrain_sha256']]
        p=out/(key+'.csv');save(p,','.join(header)+'\n');f['scenarios'][key]=p.relative_to(ROOT).as_posix()
    m=json.loads((source/'terrain.module.json').read_text())
    m['projection']={'origin':[104,316],'half_width':64}
    shader=(source/'combined.hlsl').read_text().replace('ORIGIN_X 56.5','ORIGIN_X 50.5').replace('ORIGIN_Y 18.5','ORIGIN_Y 12.5')
    save(out/'combined.hlsl',shader)
    m['shader']=(out/'combined.hlsl').relative_to(ROOT).as_posix()
    save(out/'terrain.module.json',m)
    f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    fixture.write_text(json.dumps(f,indent=2)+'\n')
    real_map.validate_provenance(f)
    return fixture

def fresh_coast():
    """Untuned coast selected by terrain coverage, before viewing this pass there."""
    out=V2/'fixtures/beauty/coast-pass-foundation/freshcoast'
    fixture=out/'fixture.json'
    if fixture.exists():return fixture
    reg,_=real_map.load_registry()
    request={'source_sha256':reg['source']['sha256'],'regions':[{
        'requested_id':'beauty-freshcoast-100-v1','origin':[20,30],
        'extent':[10,10],'halo':6,'role':'user_evaluation',
        'camera':{'viewport':[1360,800],'zooms':[1,2],'hours':[12,0]}}]}
    path=out/'region-request.json';save(path,request)
    if not any(r['id']=='beauty-freshcoast-100-v1' for r in reg['regions']):real_map.register(path)
    real_map.export('beauty-freshcoast-100-v1',out,'Q8-beauty',False)
    exported=json.loads(fixture.read_text())
    source=V2/'fixtures/beauty/gameplay-100-candidate-v5/coastal'
    f=json.loads((source/'fixture.json').read_text())
    for k in ('real_map','terrain','tile_count','viewport','id'):f[k]=exported[k]
    for key,old in f['scenarios'].items():
        header=(ROOT/old).read_text().splitlines()[0].split(',')
        header[1:5]=['10','10','0',f['real_map']['region']['terrain_sha256']]
        p=out/(key+'.csv');save(p,','.join(header)+'\n');f['scenarios'][key]=p.relative_to(ROOT).as_posix()
    m=json.loads((source/'terrain.module.json').read_text())
    shader=(source/'combined.hlsl').read_text().replace('ORIGIN_X 56.5','ORIGIN_X 24.5').replace('ORIGIN_Y 18.5','ORIGIN_Y -5.5')
    save(out/'combined.hlsl',shader)
    m['shader']=(out/'combined.hlsl').relative_to(ROOT).as_posix()
    save(out/'terrain.module.json',m)
    f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    fixture.write_text(json.dumps(f,indent=2)+'\n');real_map.validate_provenance(f)
    save(out/'BENCHMARKS.json',{'id':f['real_map']['region_id'],
        'source_biq_sha256':f['real_map']['source_sha256'],'region':f['real_map']['region'],
        'projection':m['projection'],'gameplay_crop':[360,220,1000,540],
        'selection':'Selected by terrain coverage before rendering r8; no region-specific tuning.',
        'classification':'unaltered_real_terrain'})
    return fixture

def prepare(region,revision):
    original=(long_coast() if region=='longcoast' else fresh_coast() if region=='freshcoast'
        else V2/'fixtures/beauty/gameplay-100-candidate-v5'/region/'fixture.json')
    f=json.loads(original.read_text());m=json.loads((ROOT/f['modules'][0]).read_text())
    if region=='longcoast':
        save(V2/'fixtures/beauty/coast-pass-foundation/BENCHMARKS.json',{
            'id':'beauty-longcoast-128-v1','source_biq_sha256':f['real_map']['source_sha256'],
            'region':f['real_map']['region'],'projection':m['projection'],
            'gameplay_crop':[520,390,1160,710],
            'selection':'New region selected before shoreline/rock tuning; now a regression witness.',
            'classification':'unaltered_real_terrain','full_images_required':True,
            'synthetic_volcano_is_separate':True})
    shader=(ROOT/m['shader']).read_text()
    out=V2/'fixtures/beauty'/('coast-pass-'+revision)/region
    f['id']='coast-pass-'+revision+'-'+region
    m['id']=f['id']
    if revision in ('shore-r1','rocks-r2','rocks-r3','rocks-r4','rocks-r5','rocks-r6','rocks-r7','rocks-r8'):m['hydrology_hooks']['initialize']='q3_scene::initialize_varied'
    if revision in ('rocks-r2','rocks-r3','rocks-r4','rocks-r5','rocks-r6','rocks-r7','rocks-r8'):
        sys.path.insert(0,str(V2/'systems/relief'))
        from prepare_coast_rocks import prepare as prepare_rocks
        m['coastal_rocks']=prepare_rocks(selected=revision in ('rocks-r5','rocks-r6','rocks-r7','rocks-r8'))
        if revision in ('rocks-r3','rocks-r4','rocks-r5'):m['coastal_rocks']['placement_version']=2
        if revision=='rocks-r6':m['coastal_rocks']['placement_version']=3
        if revision in ('rocks-r7','rocks-r8'):
            m['coastal_rocks']['placement_version']=4
            m['omit_replaced_shadow_surface']=1
        m['hydrology_hooks']['coast_segment']='q3_scene::coast_segment'
        shader='#define Q4_COASTAL_ROCKS 1\n#define Q3_COAST_DETAIL 1\n'+shader
    if revision in ('rocks-r4','rocks-r5','rocks-r6','rocks-r7','rocks-r8'):
        m['hydrology_hooks']['initialize']='q3_scene::initialize_articulated'
        m['direct_hill_source']=1
    if revision=='rocks-r8':shader='#define Q3_SOURCE_WATER_NORMALS 1\n'+shader
    m['shader']=(out/'combined.hlsl').relative_to(ROOT).as_posix()
    f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    save(out/'combined.hlsl',shader);save(out/'terrain.module.json',m);save(out/'fixture.json',f)
    return out/'fixture.json'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['coastal','inland','wilderness','longcoast','freshcoast','all'],default='longcoast')
    p.add_argument('--revision',choices=['baseline-v5','shore-r1','rocks-r2','rocks-r3','rocks-r4','rocks-r5','rocks-r6','rocks-r7','rocks-r8'],required=True)
    p.add_argument('--hours',nargs='+',type=int,choices=[0,6,12,18],default=[12])
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--output-root',type=Path,help='Fresh replay root; completed reports stay immutable')
    a=p.parse_args()
    for region in (['coastal','inland','wilderness','longcoast'] if a.region=='all' else [a.region]):
        f=prepare(region,a.revision)
        out=(a.output_root or V2/'audits/beauty/out'/('coast-pass-'+a.revision))/region
        if a.prepare_only:continue
        if (out/'report.json').exists():raise ValueError('Preserved render already exists')
        subprocess.run([sys.executable,str(V2/'app/runner.py'),'compose','--fixture',str(f),
            '--candidate',f.stem+'-'+region,'--output',str(out),'--hours',*map(str,a.hours)],check=True,cwd=ROOT)
        for bmp in out.glob('h*-z*-pan00.bmp'):
            subprocess.run(['sips','-s','format','png',str(bmp),'--out',str(bmp.with_suffix('.png'))],check=True,stdout=subprocess.DEVNULL)

if __name__=='__main__':main()
