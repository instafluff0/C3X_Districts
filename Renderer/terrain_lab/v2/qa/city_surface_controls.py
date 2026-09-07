"""Render isolated material controls using the candidate's exact shared packets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Completed combined render directory')
    parser.add_argument('--mode',choices=['off','normal'],required=True)
    a=parser.parse_args();source=a.source.resolve();source.relative_to(V2/'audits/beauty/out')
    target=source.parent/f'surface-control-{a.mode}'
    if target.exists():raise ValueError('preserve existing material control')
    target.mkdir()
    for filename,input_path in (('control.hlsl',source/'shaders/source.hlsl'),
                                ('reflection.hlsl',source/'shaders/reflection/source.hlsl')):
        text=input_path.read_text()
        for feature in (['SPECULAR','SURFACE'] if a.mode=='off' else ['SPECULAR']):
            marker=f'#define Q8_CITY_SOURCE_{feature} 1'
            if text.count(marker)!=1:raise ValueError('source shader does not identify the requested material feature')
            text=text.replace(marker,f'#define Q8_CITY_SOURCE_{feature} 0')
        (target/filename).write_text(text)
    report=json.loads((source/'report.json').read_text())
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(ROOT/report['source_report']),
                    '--shader',str(target/'control.hlsl'),'--reflection-shader',str(target/'reflection.hlsl'),
                    '--post-shader',str(source/'postprocess/source.hlsl'),'--output',str(target/'render')],cwd=ROOT,check=True)


if __name__=='__main__':main()
