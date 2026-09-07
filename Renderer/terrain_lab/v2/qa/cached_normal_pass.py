"""Compose source-kernel cached normals with all other terrain layers intact."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from extend_packet_materials import extend

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['coastal','inland','wilderness','freshground'],required=True)
    a=p.parse_args();region=a.region
    recipe='surface-decals-foundation' if region=='freshground' else 'river-corridor-r3'
    baseline='surface-decals-foundation-v2' if region=='freshground' else recipe
    source=V2/'audits/beauty/out'/baseline/region
    inputs=V2/'audits/beauty/out/cached-normal-r1-input'/region
    out=V2/'audits/beauty/out/cached-normal-r1'/region
    if (inputs/'report.json').exists() or (out/'report.json').exists():raise ValueError('preserved diagnostic exists')
    report=json.loads((source/'report.json').read_text());jobs=json.loads((source/'batch.json').read_text())
    normals=V2/'fixtures/beauty/source-normal-cache-r1'
    channels={116+i:normals/(name+'.dds') for i,name in enumerate(['grassland','plains','desert','marsh','tundra'])}
    inputs.mkdir(parents=True,exist_ok=True);e=[]
    for i,job in enumerate(jobs):
        target=inputs/f'packet-{i}';e.append(extend(Path(job[0]),target,channels))
        job[0]=str(target);report['outputs'][i]['packet']=target.relative_to(ROOT).as_posix()
    (inputs/'report.json').write_text(json.dumps(report)+'\n')
    (inputs/'batch.json').write_text(json.dumps(jobs)+'\n')
    (inputs/'bindings.json').write_text(json.dumps(e,indent=2)+'\n')
    fixture=V2/'fixtures/beauty/cached-normal-r1'/region;fixture.mkdir(parents=True,exist_ok=True)
    shader=fixture/'combined.hlsl'
    shader.write_text('#define Q2_CACHED_NORMAL 1\n'+f'#include "../../{recipe}/{region}/combined.hlsl"\n')
    subprocess.run([sys.executable,str(V2/'systems/lighting/prepare_linear_scene.py')],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(inputs/'report.json'),
        '--shader',str(shader),'--output',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
