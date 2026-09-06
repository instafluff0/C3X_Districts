#!/usr/bin/env python3
"""Reproduce Q1 real-map augmentation, off controls and existing attack poses.

Reads Q0's verified local registry; never changes the BIQ or shared metadata.
Exported terrain is ignored local data. Layer recipes remain portable.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'app'))
import real_map

BASE=V2/'fixtures/sampling'
SOURCE='a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e'

def save(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rel(path):return path.relative_to(ROOT).as_posix()

def linear_fixture(path):
    f=json.loads(path.read_text())
    module=json.loads((ROOT/f['modules'][0]).read_text())
    module.update(id=f['id']+'-scene-linear-v1',linear_adapter=1,
        color_branch='q6_scene_linear_premultiplied_v1',shader_owner='Q6-lighting',
        shader=rel(V2/'shaders/lighting/generated/scene_linear_v1.hlsl'))
    target=path.with_name('linear.module.json');save(target,module)
    f['modules']=[rel(target)];f['id']+='-scene-linear-v1'
    f['settings'].update(samples=4,postprocess=dict(shader=rel(V2/'shaders/sampling/linear_reconstruct.hlsl'),owner='Q1-sampling',contract=2))
    save(path.with_name('linear.fixture.json'),f)

def layer_fixture(source,out,off=False,pose=None):
    out.mkdir(parents=True,exist_ok=True)
    f=copy.deepcopy(source);f['id']=source['id']+'-q1' if out.name.endswith('-q1') else out.name
    metadata=f['real_map']
    overlay=json.loads((ROOT/metadata['overlay']).read_text())
    overlay.update(profile='q1_sampling_context_v1',owner='Q1-sampling',seed=731)
    if off:
        overlay['objects']=[];overlay['routes']=[]
        overlay['label']='augmentation-off control; immutable real terrain'
    else:
        for obj in overlay['objects']:obj['era']=3
        tile=[3,2] if metadata['region_id']=='mixed' else [1,0]
        overlay['objects'].append(dict(kind='unit',stable_id=2,tile=tile,domain='land',family=7,era=3))
        overlay['routes'] += [dict(r,kind='railroad') for r in overlay['routes'][:2]]
    for key,old in list(f['scenarios'].items()):
        lines=(ROOT/old).read_text().splitlines();header=lines[0].split(',');rows=lines[1:]
        if off:rows=[]
        elif key=='cities':
            rows=[','.join(row.split(',')[:2]+['3']+row.split(',')[3:]) for row in rows]
        elif key=='units':
            action=4 if pose is not None else 0
            rows=[','.join(map(str,[*tile,7,0,2,action,pose or 0,1,0,0,0,0]))]
        elif key=='railroads':
            roads=(ROOT/source['scenarios']['roads']).read_text().splitlines()[1:3]
            rows=[','.join(r.split(',')[:5]+['4']+r.split(',')[6:]) for r in roads]
        header[3]=str(len(rows));target=out/(key+'.csv')
        target.write_text(','.join(header)+'\n'+''.join(row+'\n' for row in rows))
        f['scenarios'][key]=rel(target);metadata['scenario_hashes'][key]=sha(target)
    if pose is not None:
        overlay['animation']=dict(action='attack',phase=pose,source='existing normalized unit pose; fixed anchor and daylight')
    path=out/'augmentation.json';save(path,overlay)
    metadata.update(overlay=rel(path),overlay_sha256=sha(path))
    save(out/'fixture.json',f)
    real_map.validate_provenance(f)
    linear_fixture(out/'fixture.json')

def main():
    registry,_=real_map.load_registry()
    if registry['source']['sha256']!=SOURCE:raise ValueError('Q1 source changed; create a new candidate revision')
    for region,stem in [('mixed','real-mixed'),('mixed-holdout','real-holdout')]:
        directory=BASE/stem
        # Pin the initial witness's halo and coordinates; later halo revisions
        # require new evidence, never silently relabel old rendered pixels.
        real_map.export(region,directory,'Q1-sampling',True,2)
        f=json.loads((directory/'fixture.json').read_text())
        layer_fixture(f,BASE/(stem+'-q1'))
        layer_fixture(f,BASE/(stem+'-off'),off=True)
        if region=='mixed':
            for pose in range(4):layer_fixture(f,BASE/f'real-animation-{pose}',pose=pose)
    print('Q1 contexts and source-pose fixtures validated')

if __name__=='__main__':main()
