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
    a=parser.parse_args();source=a.source.resolve();out=a.output.resolve()
    out.relative_to(V2/'audits/beauty/out')
    if out.exists():raise ValueError('preserve existing Windows city evidence')
    out.mkdir(parents=True)
    jobs=json.loads((source/'batch.json').read_text())
    shader=source/'shaders/source.hlsl';reflected=source/'shaders/reflection/source.hlsl'
    post=source/'postprocess/source.hlsl'
    results=[]
    for job in jobs:
        packet=Path(job[0]);metal=Path(job[2]);target=out/metal.name
        args=[relative(V2/'backends/build/d3d11.exe'),relative(packet),relative(shader),relative(target),
              job[4],relative(post),job[7],job[8],job[6],relative(reflected)]
        dispatch(args)
        row={'frame':metal.name,'packet_sha256':sha(packet),'shader_sha256':sha(shader),
             'post_sha256':sha(post),'reflection_sha256':sha(reflected),
             'metrics':compare(metal,target),'d3d11_sha256':sha(target)}
        results.append(row)
        (out/'evidence.json').write_text(json.dumps({'promotion':False,'results':results},indent=2)+'\n')
        print(json.dumps(row),flush=True)
    if not all(r['metrics']['pass'] for r in results):raise ValueError('city backend parity failed')

if __name__=='__main__':main()
