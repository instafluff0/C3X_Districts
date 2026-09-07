"""Standalone Windows check of a preserved city render, including HDR glow.

Uses the approved renderer dispatcher. No injected code, native city promotion,
or visual approval is implied by backend parity.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'qa'))
from reflection_d3d_probe import dispatch, relative, sha
sys.path.insert(0,str(V2/'app'))
from parity import compare

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Verify completed frame identities and continue only missing outputs')
    parser.add_argument('--render-only',action='store_true',help='Render independently while Metal is compiling; parity remains explicitly pending until a normal resume')
    a=parser.parse_args();source=a.source.resolve();out=a.output.resolve()
    out.relative_to(V2/'audits/beauty/out')
    if out.exists() and not a.resume:raise ValueError('preserve existing Windows city evidence')
    out.mkdir(parents=True,exist_ok=a.resume)
    jobs=json.loads((source/'batch.json').read_text())
    shader=source/'shaders/source.hlsl';reflected=source/'shaders/reflection/source.hlsl'
    post=source/'postprocess/source.hlsl'
    if not all(p.is_file() for p in (shader,reflected,post)):
        raise ValueError('source shader closure is not fully prepared yet')
    evidence=out/'evidence.json'
    results=json.loads(evidence.read_text())['results'] if a.resume and evidence.exists() else []
    frames=[Path(job[2]).name for job in jobs]
    if len({r['frame'] for r in results})!=len(results) or any(r['frame'] not in frames for r in results):
        raise ValueError('saved Windows results do not match requested frame set')
    for job in jobs:
        packet=Path(job[0]);metal=Path(job[2]);target=out/metal.name
        saved=next((r for r in results if r['frame']==metal.name),None)
        if saved:
            for key,path in [('packet_sha256',packet),('shader_sha256',shader),('post_sha256',post),
                             ('reflection_sha256',reflected),('d3d11_sha256',target)]:
                if saved[key]!=sha(path):raise ValueError('preserved Windows input/output changed: '+key)
            if saved.get('metrics') is None and not a.render_only:
                saved['metrics']=compare(metal,target)
                evidence.write_text(json.dumps({'promotion':False,'results':results},indent=2)+'\n')
            continue
        if target.exists():raise ValueError('unrecorded Windows output exists; inspect before resuming')
        args=[relative(V2/'backends/build/d3d11.exe'),relative(packet),relative(shader),relative(target),
              job[4],relative(post),job[7],job[8],job[6],relative(reflected)]
        dispatch(args)
        row={'frame':metal.name,'packet_sha256':sha(packet),'shader_sha256':sha(shader),
             'post_sha256':sha(post),'reflection_sha256':sha(reflected),
             'metrics':None if a.render_only else compare(metal,target),'d3d11_sha256':sha(target)}
        results.append(row)
        evidence.write_text(json.dumps({'promotion':False,'results':results},indent=2)+'\n')
        print(json.dumps(row),flush=True)
    if not a.render_only and not all(r['metrics']['pass'] for r in results):raise ValueError('city backend parity failed')

if __name__=='__main__':main()
