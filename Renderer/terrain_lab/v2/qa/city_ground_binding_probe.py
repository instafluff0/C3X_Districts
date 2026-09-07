"""Replay a ground-only era-atlas hypothesis with every other input frozen."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from city_scene_pass import ROOT,V2,Cache,executable,compact_packet,rel,save
from cache import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-render',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=ROOT/a.source_render;output=ROOT/a.output
    output.resolve().relative_to(V2/'audits/beauty/out')
    if output.exists():raise ValueError('preserve previous ground binding result')
    if shutil.disk_usage(V2).free<8*1024**3:raise ValueError('preserve 8 GiB free space')
    mapping=json.loads((ROOT/a.mapping).read_text())
    if mapping['schema']!='c3x.lab.ground_binding_override.v1':raise ValueError('unknown binding mapping')
    for key in ('expected','replacement'):
        if file_hash(ROOT/mapping[key]['texture'])!=mapping[key]['sha256']:raise ValueError('ground texture fingerprint changed')
    source_report=json.loads((source/'report.json').read_text())
    input_path=ROOT/source_report['source_report'];report=json.loads(input_path.read_text())
    jobs=json.loads((input_path.parent/'batch.json').read_text())
    if len(jobs)!=len(report['outputs']):raise ValueError('batch/report mismatch')
    output.mkdir(parents=True)
    exe=executable(V2/'qa/city_ground_binding_probe.cpp',Cache(V2/'app/.cache'))
    evidence=[]
    for i,(job,row) in enumerate(zip(jobs,report['outputs'])):
        packet=output/f'combined-{i}.packet';original=ROOT/row['packet']
        result=subprocess.run([str(exe),str(original),str(ROOT/mapping['expected']['texture']),
                               str(ROOT/mapping['replacement']['texture']),str(packet)],capture_output=True,text=True,check=True)
        compact_packet(packet,V2/'app/.cache/content')
        evidence.append({'original':rel(original),'original_sha256':file_hash(original),
                         'output':rel(packet),'output_sha256':file_hash(packet),**json.loads(result.stdout)})
        row['packet']=rel(packet);job[0]=str(packet)
    save(output/'report.json',report);save(output/'batch.json',jobs)
    save(output/'binding.json',{'classification':'Era grounding atlas hypothesis; source height/state application remains unproven',
                              'mapping':a.mapping.as_posix(),'mapping_sha256':file_hash(ROOT/a.mapping),
                              'previous_render':a.source_render.as_posix(),'packets':evidence})
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(output/'report.json'),
                    '--shader',str(source/'shaders/source.hlsl'),'--reflection-shader',str(source/'shaders/reflection/source.hlsl'),
                    '--post-shader',str(source/'postprocess/source.hlsl'),'--output',str(output/'render')],check=True)


if __name__=='__main__':main()
