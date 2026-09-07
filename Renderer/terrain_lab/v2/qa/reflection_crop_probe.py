"""Shift a frozen water packet to expose offscreen-reflector/edge defects.

The fixed benchmark is untouched. This probe changes only replay viewport offset,
with a matched sky-only control to separate reflection errors from existing crop
differences in terrain/shadow/material sampling.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2';OUT=V2/'audits/beauty/out'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['freshwater','longcoast'],default='freshwater')
    p.add_argument('--shift-y',type=int,default=-120)
    p.add_argument('--revision',type=int,default=5)
    a=p.parse_args();region=a.region
    base='water-natural-foundation' if region=='freshwater' else 'shadow-receiver-r1'
    original=OUT/base/region
    out=OUT/f'water-reflection-crops-r{a.revision}'/region/f'y{a.shift_y}'
    if out.exists():raise ValueError('preserved crop probe exists')
    inputs=out/'input';inputs.mkdir(parents=True)
    report=json.loads((original/'report.json').read_text())
    jobs=json.loads((original/'batch.json').read_text());shifts=[]
    for j in jobs:
        down=struct.unpack_from('<I',Path(j[0]).read_bytes(),16)[0]
        j[8]=str(a.shift_y/down)
        shifts.append({'frame':Path(j[2]).name,'internal_pixel_offset':[0,a.shift_y],
                       'final_pixel_offset':[0,a.shift_y/down]})
    report['diagnostic_camera_shifts']=shifts
    report['effective']['settings']['camera_offsets']=[r['final_pixel_offset'] for r in shifts]
    (inputs/'report.json').write_text(json.dumps(report)+'\n')
    (inputs/'batch.json').write_text(json.dumps(jobs)+'\n')
    (inputs/'camera.json').write_text(json.dumps(shifts,indent=2)+'\n')
    fixture=V2/f'fixtures/beauty/water-reflection-r{a.revision}'/region
    for mode in ['reflected','sky-only']:
        shader=fixture/'combined.hlsl' if mode=='reflected' else OUT/'water-natural-r6'/region/'phase-0/shaders/source.hlsl'
        cmd=[sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(inputs/'report.json'),
             '--shader',str(shader),'--output',str(out/mode)]
        if mode=='reflected':cmd+=['--reflection-shader',str(fixture/'reflection.hlsl')]
        subprocess.run(cmd,check=True,cwd=ROOT)

if __name__=='__main__':main()
