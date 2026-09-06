#!/usr/bin/env python3
"""Q1 one-variable experiments using Q0's cached builder and shared Metal batch.

Fixtures pin either verified test.biq regions or explicitly synthetic terrain.
All source-backed paths use Q0's shared packet builder and viewport contract.
Only the historical gallery experiment uses the owned crop adapter.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[5]
V2 = ROOT / 'Renderer/terrain_lab/v2'
sys.path.insert(0, str(V2 / 'app'))
import runner as q0
from cache import Cache, canonical, file_hash

BASE = V2 / 'audits/sampling'
FIXTURE = V2 / 'fixtures/sampling/mixed.fixture.json'


def prepare(cache, f, scene, hour, zoom):
    if f['id']=='sampling-mixed-proxy-v1':
        module=q0.read_json(q0.local(f['modules'][0]),'c3x.lab_v2.module.v1')
        return q0.packet(cache,f,module,scene,hour,zoom,q0.pack_identity(cache,f['packs']))
    if 'real_map' in f:
        module=q0.read_json(q0.local(f['modules'][0]),'c3x.lab_v2.module.v1')
        return q0.packet(cache,f,module,scene,hour,zoom,q0.pack_identity(cache,f['packs']))
    if f['id']=='sampling-frozen-crop-v1':
        parent,module=q0.fixture(V2/'tests/platform/complete.fixture.json')
        source=q0.packet(cache,parent,module,scene,hour,1,q0.pack_identity(cache,parent['packs']))
        obj=q0.compile_cpp(cache,Path(__file__).with_name('crop_packet.cpp'))
        exe=cache.artifact('sampling-crop-exe',{'object':file_hash(obj)},lambda out:q0.run(['clang++',obj,'-o',out]));exe.chmod(0o755)
        return cache.artifact('geometry',{'q1_crop':1,'source':file_hash(source),'builder':file_hash(exe),'xy':[1200,320],
            'viewport':f['viewport'],'zoom':zoom,'margin':128},lambda out:q0.run([exe,source,out,1200,320,*f['viewport'],zoom,128]))
    module=q0.read_json(q0.local(f['modules'][0]),'c3x.lab_v2.module.v1')
    return q0.packet(cache,f,module,scene,hour,zoom,q0.pack_identity(cache,f['packs']))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',default='c001')
    ap.add_argument('--mode',choices=['quick','ab','post','pair','pan','matrix'],default='quick')
    ap.add_argument('--post',choices=['off','linear_box','mitchell','sharpen','bounded','scene_linear_box'],default='off')
    ap.add_argument('--fixture',type=Path,default=FIXTURE)
    a=ap.parse_args()
    if not a.candidate.replace('-','').isalnum():raise ValueError('invalid candidate')
    f,module=q0.fixture(a.fixture.resolve())
    expanded_shader=q0.shader_source(q0.local(module['shader']))
    linear=module.get('color_branch')=='q6_scene_linear_premultiplied_v1'
    if a.post=='scene_linear_box' and not linear:raise ValueError('scene-linear reconstruction requires linear input')
    if linear and a.post not in ('off','scene_linear_box'):raise ValueError('legacy filters cannot consume scene-linear input')
    if linear and a.mode=='post':raise ValueError('legacy post experiment requires legacy input')
    out=BASE/'out/Q1/Q1-sampling'/a.candidate;out.mkdir(parents=True,exist_ok=True)
    cache=Cache(V2/'app/.cache')
    scene,metal=q0.executables(cache)
    caps=json.loads(q0.run([metal,'--capabilities']))
    variants=[('baseline',8,0,1,1,'off')]
    if a.mode=='ab':
        variants += [('aniso16',16,0,1,1,'off')]
        variants += [(f'bias{abs(b):g}',8,b,1,1,'off') for b in [-.35,-.65,-1]]
        variants += [('msaa4',8,0,4,1,'off'),('ssaa2',8,0,1,2,'off')]
    if a.post!='off':
        variants += [(a.post,8,0,1,1,a.post)]
    if a.mode=='post':
        variants += [(name,8,0,1,1,name) for name in ['linear_box','mitchell','sharpen','bounded'] if name!=a.post]
    if a.mode in ('pair','pan','matrix'):
        variants += [('finalist_control',8,0,4,1,'off')]
        if a.post!='off':variants += [('finalist',8,0,4,1,a.post)]
    phases=[12,18,0,6] if a.mode=='matrix' else [12]
    zooms=[1] if a.mode=='quick' else [1,2]
    offsets=[[0,0]] if a.mode!='pan' else [[0,0],[.25,.25],[.5,.5],[1,1],[2,1],[4,2],[8,4],[16,8],[32,16],[64,32],[0,0]]
    jobs=[];evidence=[];unsupported=[];compiled_shaders={}
    for name,anis,bias,samples,scale,post in variants:
        if samples not in caps['sample_counts']:
            unsupported.append({'variant':name,'samples':samples});continue
        shaderdir=out/f'shaders-{bias:g}';shaderdir.mkdir(exist_ok=True)
        sh=q0.shaders(cache,q0.local(module['shader']),bias,module.get('msl_version',20100))
        compiled_shaders[str(bias)]={entry:file_hash(path) for entry,path in sh.items()}
        for entry,path in sh.items():shutil.copyfile(path,shaderdir/(entry+'.msl'))
        postpath='box'
        if post!='off':
            filename='linear_reconstruct' if post=='scene_linear_box' else post
            postpath=str(q0.post_shader(cache,V2/f'shaders/sampling/{filename}.hlsl',out))
        for hour in phases:
            for zoom in zooms:
                packet=prepare(cache,f,scene,hour,zoom)
                for i,offset in enumerate(offsets):
                    stem=f'{name}-h{hour:02}-z{zoom}-p{i:02}'
                    target=out/(stem+'.bmp');cost=out/(stem+'.cost.json')
                    post_contract=2 if post=='scene_linear_box' else 1
                    jobs.append(list(map(str,[packet,shaderdir,target,cost,samples,anis,scale,*offset,2,postpath,post_contract])))
                    evidence.append({'variant':name,'image':q0.relative(target),'cost':q0.relative(cost),
                        'packet':q0.relative(packet),'hour':hour,'zoom':zoom,'offset':offset,
                        'mip_bias':bias,'anisotropy':anis,'samples':samples,'post':post,'post_contract':post_contract,
                        'internal_size':[x*scale for x in f['viewport']],
                        'output_size':[x//zoom for x in f['viewport']]})
    batch=out/'batch.json';batch.write_bytes(canonical(jobs))
    q0.run([metal,'--batch',batch])
    for e in evidence:
        p=q0.local(e['image']);e['sha256']=file_hash(p)
        e['repeat_identical']=e['sha256']==file_hash(str(p)+'.repeat1.bmp')
        if not e['repeat_identical']:raise ValueError('nondeterministic output')
    report={'schema':'c3x.q1.sampling.study.v1','provisional':True,'fixture':q0.relative(a.fixture),
        'fixture_sha256':file_hash(a.fixture),'contract':file_hash(q0.CONTRACT),'caps':caps,
        'backend_source':file_hash(V2/'backends/metal.mm'),'builder_source':file_hash(V2/'shared/frozen_scene.cpp'),
        'shader_source':file_hash(q0.local(module['shader'])),'unsupported':unsupported,'outputs':evidence,
        'module':module,'fixture_manifest':f,'pack_identity':q0.pack_identity(cache,f['packs']),
        'expanded_shader_sha256':hashlib.sha256(expanded_shader.encode()).hexdigest(),
        'compiled_msl_sha256':compiled_shaders,
        'source_changed_during_run':expanded_shader!=q0.shader_source(q0.local(module['shader'])),
        'real_map':f.get('real_map'),
        'visual_review':'pending direct inspection','device_loads':1}
    (out/'report.json').write_bytes(canonical(report))
    print(f'Rendered {len(evidence)} comparisons, repeats identical: {q0.relative(out)}')


if __name__=='__main__':main()
