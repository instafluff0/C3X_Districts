"""Disable local irradiance only on the added paving; keep terrain lights active."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil
from city_scene_pass import ROOT,V2,save,rel,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve();out.relative_to(V2/'audits/beauty/out')
    if out.exists():raise ValueError('preserve prior paving light control')
    if shutil.disk_usage(V2).free<8*2**30:raise ValueError('preserve 8 GiB free space')
    out.mkdir(parents=True)
    marker='ground.rgb*q6_receiver_illumination(p,normalize(p.geometry_normal),1,1)'
    replacement='ground.rgb*(q6_receiver_illumination(p,normalize(p.geometry_normal),1,1)-(p.material_index>61.5?q8_local_irradiance(p.q6_world,normalize(p.geometry_normal),1):float3(0,0,0)))'
    for name,path in [('combined',source/'shaders/source.hlsl'),('reflection',source/'shaders/reflection/source.hlsl')]:
        text=path.read_text()
        if text.count(marker)!=1:raise ValueError('paving receiver marker changed')
        (out/(name+'.hlsl')).write_text(text.replace(marker,replacement))
    save(out/'control.json',{'classification':'Local irradiance disabled only on settlement underlay; all packets and other receivers unchanged',
        'source':rel(source),'source_sha256':file_hash(source/'report.json')})
    report=json.loads((source/'report.json').read_text())
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(ROOT/report['source_report']),
        '--shader',str(out/'combined.hlsl'),'--reflection-shader',str(out/'reflection.hlsl'),
        '--post-shader',str(source/'postprocess/source.hlsl'),'--output',str(out/'render')],check=True)


if __name__=='__main__':main()
