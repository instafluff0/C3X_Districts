"""Bounded shader-only experiment on a verified existing render packet.

This intentionally reuses identical geometry and shadow resources. It does not
replace fixture replay or qualify a candidate for promotion.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

V2=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(V2/'app'))
import runner
from cache import Cache,canonical,file_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--shader',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.report.read_text());out=a.output.resolve()
    runner.owned(out,'Q8-beauty');out.mkdir(parents=True,exist_ok=True)
    settings=r['effective']['settings'];cache=Cache(V2/'app/.cache')
    _,metal=runner.executables(cache)
    shaderdir=out/'shaders';shaderdir.mkdir(exist_ok=True)
    compiled=runner.shaders(cache,a.shader.resolve(),settings['mip_bias'])
    for name,path in compiled.items():shutil.copyfile(path,shaderdir/(name+'.msl'))
    (shaderdir/'source.hlsl').write_text(runner.shader_source(a.shader.resolve()))
    jobs=json.loads((a.report.parent/'batch.json').read_text())
    identities=[]
    for job in jobs:
        packet=Path(job[0]);identities.append({'path':runner.relative(packet),'sha256':file_hash(packet)})
        job[1]=str(shaderdir);job[2]=str(out/Path(job[2]).name)
        job[3]=str(out/Path(job[3]).name);job[9]='1'
    batch=out/'batch.json';batch.write_bytes(canonical(jobs))
    runner.run([metal,'--batch',batch])
    record={'kind':'shader_only_diagnostic','promotion':False,'source_report':runner.relative(a.report),
        'source_report_sha256':file_hash(a.report),'shader_closure_sha256':file_hash(shaderdir/'source.hlsl'),
        'packets':identities,'outputs':[]}
    for job in jobs:
        bmp=Path(job[2]);runner.run(['sips','-s','format','png',bmp,'--out',bmp.with_suffix('.png')])
        record['outputs'].append({'image':runner.relative(bmp),'sha256':file_hash(bmp)})
    (out/'report.json').write_bytes(canonical(record))
    print('PASS shader-only replay:',runner.relative(out))

if __name__=='__main__':main()
