"""Compose a source alpha-weighted terrain diagnostic on preserved packets."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['coastal','inland','wilderness','freshground'],required=True)
    a=p.parse_args()
    recipe='surface-decals-foundation' if a.region=='freshground' else 'river-corridor-r3'
    baseline='surface-decals-foundation-v2' if a.region=='freshground' else recipe
    fixture=V2/'fixtures/beauty/source-blend-r1'/a.region
    out=V2/'audits/beauty/out/source-blend-r1'/a.region
    if (out/'report.json').exists():raise ValueError('preserved diagnostic exists')
    fixture.mkdir(parents=True,exist_ok=True)
    shader=fixture/'combined.hlsl'
    shader.write_text('#define Q2_SOURCE_ALPHA_BLEND 1\n'+f'#include "../../{recipe}/{a.region}/combined.hlsl"\n')
    subprocess.run([sys.executable,str(V2/'systems/lighting/prepare_linear_scene.py')],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(V2/'audits/beauty/out'/baseline/a.region/'report.json'),
        '--shader',str(shader),'--output',str(out)],check=True,cwd=ROOT)
