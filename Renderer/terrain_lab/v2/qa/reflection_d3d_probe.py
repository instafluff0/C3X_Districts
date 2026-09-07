"""Replay the preserved reflection candidate through the approved Windows dispatcher.

This is standalone Lab evidence, not native renderer promotion. Each destination
is immutable; baseline controls use the preserved sky-only shader and identical
packets. Existing parity thresholds are applied without modification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2';OUT=V2/'audits/beauty/out'
sys.path.insert(0,str(V2/'app'))
from parity import compare
from runner import shader_source
sys.path.insert(0,str(ROOT/'Renderer/tools'))
import renderer_dev

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path):return path.relative_to(ROOT).as_posix()
def dispatch(args):
    command=' '.join('"'+str(a).replace('/','\\')+'"' for a in args)
    result=renderer_dev.native_command_result('.',command)
    if result['status']!='pass':raise RuntimeError('D3D replay failed: '+result['output_tail'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region',choices=['coastal','inland','wilderness','longcoast','freshwater'],required=True)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--repeat',action='store_true')
    parser.add_argument('--resume',action='store_true',help='continue missing cases after verifying preserved completed outputs')
    a=parser.parse_args()
    if os.name!='nt':os.environ.setdefault('C3X_RENDERER_WINDOWS_ROOT',str(renderer_dev.windows_live_target()))
    if a.build:
        result=renderer_dev.native_command_result('Renderer/terrain_lab/v2/backends','call build_d3d11.bat')
        if result['status']!='pass':raise RuntimeError('D3D build failed')
    out=OUT/'water-reflection-d3d-r5'/a.region
    if out.exists() and not a.resume:raise ValueError('preserved D3D probe exists')
    out.mkdir(parents=True,exist_ok=a.resume)
    source=OUT/'water-reflection-r5'/a.region/'combined'
    jobs=json.loads((source/'batch.json').read_text())
    post=out/'post.hlsl';post_source=shader_source(V2/'shaders/sampling/linear_reconstruct.hlsl')
    if post.exists():
        if post.read_text()!=post_source:raise ValueError('postprocess changed since preserved run')
    else:post.write_text(post_source)
    evidence=out/'evidence.json'
    rows=json.loads(evidence.read_text())['results'] if a.resume and evidence.exists() else []
    for job in jobs:
        packet=Path(job[0]);frame=Path(job[2]).name
        for mode in ['sky-only','reflected']:
            shader=(source/'shaders/source.hlsl' if mode=='reflected' else
                    OUT/'water-natural-r6'/a.region/'phase-0/shaders/source.hlsl')
            target=out/(mode+'-'+frame)
            saved=next((r for r in rows if r['mode']==mode and r['frame']==frame),None)
            if saved:
                assert saved['packet_sha256']==sha(packet) and saved['shader_sha256']==sha(shader)
                assert saved['d3d11_sha256']==sha(target)
                if a.repeat:assert sha(target)==sha(out/(mode+'-repeat-'+frame))
                continue
            if target.exists():raise ValueError('unrecorded output exists; preserve and inspect before resuming')
            args=[relative(V2/'backends/build/d3d11.exe'),relative(packet),relative(shader),relative(target),
                  job[4],relative(post),job[7],job[8],job[6]]
            if mode=='reflected':args+=[relative(source/'shaders/reflection/source.hlsl')]
            dispatch(args)
            metal=Path(job[2]) if mode=='reflected' else OUT/'water-natural-r6'/a.region/'phase-0'/frame
            metrics=compare(metal,target)
            row={'mode':mode,'frame':frame,'packet_sha256':sha(packet),'shader_sha256':sha(shader),
                 'metal':relative(metal),'d3d11':relative(target),'d3d11_sha256':sha(target),'metrics':metrics}
            if a.repeat:
                again=out/(mode+'-repeat-'+frame);args[3]=relative(again);dispatch(args)
                row['repeat_identical']=sha(again)==sha(target)
            rows.append(row)
            (out/'evidence.json').write_text(json.dumps({'region':a.region,'results':rows},indent=2)+'\n')
            print(json.dumps(row),flush=True)
    if not all(r['metrics']['pass'] and r.get('repeat_identical',True) for r in rows):
        raise ValueError('parity or repeat gate failed; preserved evidence.json contains results')

if __name__=='__main__':main()
