"""Rebuild city shadows on an explicit prior light grid for matched comparison."""
import argparse
import json
from pathlib import Path
import subprocess
from city_scene_pass import ROOT,V2,OUT,Cache,executable,compact_packet,rel,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='City output directory containing report.json')
    p.add_argument('--reference',type=Path,required=True,help='Reference city output directory containing report.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shader-render',type=Path,help='Use this preserved render closure, including composed local lights and ground')
    a=p.parse_args();source=ROOT/a.source;reference=ROOT/a.reference;output=ROOT/a.output
    if output.exists():raise ValueError('preserve existing frame comparison')
    output.mkdir(parents=True)
    report=json.loads((source/'report.json').read_text())
    jobs=json.loads((source/'batch.json').read_text())
    if len(jobs)!=len(report['outputs']):raise ValueError('source city batch/report mismatch')
    prior=json.loads((reference/'report.json').read_text())
    frames={(r['hour'],r['zoom']):r for r in prior['outputs']}
    exe=executable(V2/'systems/lighting/scene_shadow.cpp',Cache(V2/'app/.cache'))
    comparisons=[]
    for index,row in enumerate(report['outputs']):
        r=frames[row['hour'],row['zoom']];packet=output/f'combined-{index}.packet'
        subprocess.run([str(exe),str(ROOT/row['packet']),str(packet),str(row['hour']),
                        str(source/'report.json'),str(ROOT/r['packet'])],check=True)
        compact_packet(packet,V2/'app/.cache/content')
        comparisons.append({'current_packet':row['packet'],'reference_packet':r['packet'],'output_packet':rel(packet)})
        row['packet']=rel(packet)
        jobs[index][0]=str(packet)
    save(output/'report.json',report);save(output/'batch.json',jobs);save(output/'frame-control.json',comparisons)
    fixture=V2/'fixtures/beauty'/source.parent.name/source.name
    closure=ROOT/a.shader_render if a.shader_render else None
    subprocess.run(['python3',str(V2/'qa/replay_shader.py'),'--report',str(output/'report.json'),
                    '--shader',str(closure/'shaders/source.hlsl' if closure else fixture/'combined.hlsl'),
                    '--reflection-shader',str(closure/'shaders/reflection/source.hlsl' if closure else fixture/'reflection.hlsl'),
                    '--post-shader',str(closure/'postprocess/source.hlsl' if closure else V2/'shaders/common/hdr_glow_tiled.hlsl'),
                    '--output',str(output/'render')],check=True)


if __name__=='__main__':main()
