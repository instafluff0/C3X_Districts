"""Bounded water-only exploration using fixed combined gameplay packets."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['coastal','inland','wilderness','longcoast','freshwater'],required=True)
    p.add_argument('--phase',type=float,default=0)
    p.add_argument('--natural',type=int,help='Preserved static natural-water revision (no surf or clock)')
    p.add_argument('--coordinate-shift',type=float,default=0,help='Water-only periodic-coordinate diagnostic')
    p.add_argument('--disabled',action='store_true',help='Default-off control with identical packets')
    a=p.parse_args()
    if a.natural is None and (a.coordinate_shift or a.disabled):p.error('natural revision required for controls')
    if a.natural is not None and a.phase:p.error('static natural-water captures have no animated phase')
    baseline='shadow-receiver-r1' if a.region=='longcoast' else 'river-corridor-r3'
    if a.region=='freshwater':baseline='water-natural-foundation'
    tag='phase-'+format(a.phase,'g').replace('.','p')
    if a.coordinate_shift:tag+='-shift-'+format(a.coordinate_shift,'g').replace('.','p')
    if a.disabled:tag+='-disabled'
    campaign=f'water-natural-r{a.natural}' if a.natural is not None else 'water-effects-r2'
    fixture=V2/'fixtures/beauty'/campaign/a.region/tag
    out=V2/'audits/beauty/out'/campaign/a.region/tag
    if (out/'report.json').exists():raise ValueError('preserved diagnostic exists')
    fixture.mkdir(parents=True,exist_ok=True)
    shader=fixture/'combined.hlsl'
    defines='#define Q3_NATURAL_WATER 1\n' if a.natural is not None else '#define Q3_WATER_EFFECTS 1\n'+f'#define Q3_WATER_TIME {a.phase:.6f}\n'
    if a.natural is not None:defines+=f'#define Q3_NATURAL_COORD_SHIFT {a.coordinate_shift:.6f}\n'
    if a.disabled:defines=''
    shader.write_text(defines+
        f'#include "../../../{baseline}/{a.region}/combined.hlsl"\n')
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',
        str(V2/'audits/beauty/out'/baseline/a.region/'report.json'),
        '--shader',str(shader),'--output',str(out)],check=True,cwd=ROOT)
