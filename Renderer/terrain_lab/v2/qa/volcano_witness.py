"""Render a labeled synthetic volcano witness beside the unchanged real map.

test.biq contains no volcano. Only this supplemental copy changes one terrain
classification; it is never registered or presented as unaltered BIQ evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'qa'))
from coastal_pass import save

def prepare(revision='r1'):
    source=V2/'fixtures/beauty/gameplay-100-candidate-v5/inland'
    out=V2/'fixtures/beauty'/('volcano-witness-'+revision)/'inland'
    f=json.loads((source/'fixture.json').read_text())
    original=f.pop('real_map')
    terrain=ROOT/f['terrain'];lines=terrain.read_text().splitlines()
    changes=[]
    for i,line in enumerate(lines[1:],1):
        row=list(map(int,line.split(',')))
        if row[:2]==[5,5]:
            assert row[5]==6
            after=row.copy();after[5]=10
            changes.append({'before':row,'after':after})
            lines[i]=','.join(map(str,after))
    assert len(changes)==1
    csv='\n'.join(lines)+'\n';save(out/'terrain.csv',csv)
    terrain_hash=hashlib.sha256(csv.encode()).hexdigest()
    f['terrain']=(out/'terrain.csv').relative_to(ROOT).as_posix()
    f['id']='synthetic-volcano-witness-'+revision
    for key,old in f['scenarios'].items():
        header=(ROOT/old).read_text().strip().split(',')
        assert header[3]=='0'
        header[4]=terrain_hash
        p=out/(key+'.csv');save(p,','.join(header)+'\n')
        f['scenarios'][key]=p.relative_to(ROOT).as_posix()
    m=json.loads((source/'terrain.module.json').read_text())
    m['id']=f['id']
    shader=(source/'combined.hlsl').read_text()
    if revision=='r2':
        m['volcano_source_mapping']=1
        shader='#define Q4_VOLCANO_SOURCE_MAPPING 1\n'+shader
    save(out/'combined.hlsl',shader)
    m['shader']=(out/'combined.hlsl').relative_to(ROOT).as_posix()
    save(out/'terrain.module.json',m)
    f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    save(out/'fixture.json',f)
    save(out/'provenance.json',{
        'classification':'synthetic_supplemental_witness',
        'source_biq_sha256':original['source_sha256'],
        'source_terrain':terrain.relative_to(ROOT).as_posix(),
        'source_terrain_sha256':original['region']['terrain_sha256'],
        'terrain_sha256':terrain_hash,'changes':changes,
        'benchmark':False,'geometry_classification':
        'source-height reconstruction with unproven physical coordinate calibration',
        'reason':'Verified source has zero volcano tiles; one mountain is replaced only in this copy.'})
    return out/'fixture.json'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--revision',choices=['r1','r2'],default='r1')
    a=p.parse_args();f=prepare(a.revision)
    if a.prepare_only:return
    out=V2/'audits/beauty/out'/('volcano-witness-'+a.revision)/'inland'
    if (out/'report.json').exists():raise ValueError('Preserved render already exists')
    subprocess.run([sys.executable,str(V2/'app/runner.py'),'compose','--fixture',str(f),
        '--candidate','synthetic-volcano-witness-'+a.revision,'--output',str(out),'--hours','12','0'],cwd=ROOT,check=True)
    for bmp in out.glob('h*-z*-pan00.bmp'):
        subprocess.run(['sips','-s','format','png',str(bmp),'--out',str(bmp.with_suffix('.png'))],check=True,stdout=subprocess.DEVNULL)

if __name__=='__main__':main()
