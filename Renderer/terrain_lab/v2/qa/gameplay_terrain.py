"""Reproduce the fixed 100-tile terrain benchmarks without VM or object layers.

Use --region wilderness first to check the previously untuned dry region.
Every revision has its own inputs and outputs; source terrain stays immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def prepare(region,revision):
    fixed=V2/'fixtures/beauty/gameplay-100-v1'/region
    dest=V2/'fixtures/beauty'/('gameplay-100-'+revision)/region
    dest.mkdir(parents=True,exist_ok=True)
    f=json.loads((fixed/'fixture.json').read_text())
    m=json.loads((fixed/'terrain.module.json').read_text())
    f['id']='gameplay-100-'+region+'-'+revision
    shader=(fixed/'combined.hlsl').read_text()
    if revision.startswith('candidate-'):
        shader='#define Q4_COMBINED_ROCK_PROJECTION 1\n#define Q3_STATIC_OPTICS_V2 1\n'+shader
        hill=V2/'fixtures/relief/selected-source/hills_height_lod0.dds'
        m['hill_source']={'path':hill.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(hill.read_bytes()).hexdigest(),
            'height_multiplier':10/14}
    if revision in ('candidate-v2','candidate-v3','candidate-v4','candidate-v5'):
        shader='#define Q4_BIQ_DUNE_COVERAGE 1\n'+shader
        m['complete_materials']=1
    if revision in ('candidate-v3','candidate-v4','candidate-v5'):
        shader='#define Q6_GAMEPLAY_NIGHT 1\n#define Q4_BIQ_CONTINUOUS_DESERT 1\n'+shader
        m['continuous_desert']={'candidate-v3':1,'candidate-v4':2,'candidate-v5':3}[revision]
    m['shader']=(dest/'combined.hlsl').relative_to(ROOT).as_posix()
    f['modules']=[(dest/'terrain.module.json').relative_to(ROOT).as_posix()]
    # Stable recipe regeneration is allowed; never silently change a revision.
    for name,value in [('terrain.module.json',m),('fixture.json',f)]:
        target=dest/name
        if target.exists() and json.loads(target.read_text())!=value:
            raise ValueError('Existing revision differs; create a new revision')
        save(target,value)
    target=dest/'combined.hlsl'
    if target.exists() and target.read_text()!=shader:
        raise ValueError('Existing shader differs; create a new revision')
    target.write_text(shader)
    return dest/'fixture.json'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['coastal','inland','wilderness','all'],default='all')
    p.add_argument('--revision',choices=['baseline-v1','candidate-v1','candidate-v2','candidate-v3','candidate-v4','candidate-v5'],required=True)
    p.add_argument('--hours',nargs='+',type=int,choices=[0,6,12,18],default=[12])
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--output-root',type=Path,
                   help='Fresh replay root; completed benchmark reports are never overwritten')
    a=p.parse_args()
    for region in (['wilderness','coastal','inland'] if a.region=='all' else [a.region]):
        fixture=prepare(region,a.revision)
        out=(a.output_root or V2/'audits/beauty/out'/('gameplay-100-'+a.revision))/region
        if not a.prepare_only:
            if (out/'report.json').exists():
                raise ValueError('Preserved render already exists; select a fresh --output-root')
            subprocess.run([sys.executable,str(V2/'app/runner.py'),'compose',
                '--fixture',str(fixture),'--candidate',region+'-'+a.revision,
                '--output',str(out),'--hours',*map(str,a.hours)],cwd=ROOT,check=True)
            for bmp in out.glob('h*-z*-pan00.bmp'):
                subprocess.run(['sips','-s','format','png',str(bmp),'--out',str(bmp.with_suffix('.png'))],
                               check=True,stdout=subprocess.DEVNULL)

if __name__=='__main__':main()
