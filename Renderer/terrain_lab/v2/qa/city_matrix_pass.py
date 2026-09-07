"""Preserve the full 5-culture/4-era/3-size city matrix on fixed coastal terrain.

Every case keeps the same anchor, terrain, camera, source scale policy and noon/
midnight views. This is coverage and growth evidence, not visual acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',type=int,required=True)
    parser.add_argument('--start-index',type=int,default=0)
    parser.add_argument('--max-cases',type=int,default=6,help='Bounded incremental capture; default six cases')
    a=parser.parse_args();output=V2/f'audits/beauty/out/city-scene-r{a.revision}'
    if not 0<=a.start_index<60 or not 1<=a.max_cases<=60:raise ValueError('matrix case bounds')
    if output.exists():raise ValueError('matrix output exists; preserve it')
    output.mkdir(parents=True)
    dependencies=[V2/'qa/city_scene_pass.py',V2/'qa/append_city_scene.cpp',V2/'systems/objects/presentation.py',
                  V2/'shaders/objects/city_scene_material.hlsl',V2/'shaders/common/hdr_glow_tiled.hlsl']
    hashes={p:sha(p) for p in dependencies}
    catalog=json.loads((ROOT/'Renderer/packs/CityStudyAuxiliaryUV/city_catalog.json').read_text())
    pools=sorted(catalog['pools']);rows=[]
    cases=[(size,pool) for size in (1,0,2) for pool in pools]
    for size,pool in cases[a.start_index:a.start_index+a.max_cases]:
            if shutil.disk_usage(output).free<8*1024**3:raise ValueError('capture stopped: preserve at least 8 GiB free disk space')
            assert all(sha(p)==v for p,v in hashes.items()),'matrix implementation changed during capture'
            args=[sys.executable,V2/'qa/city_scene_pass.py','--revision',str(a.revision),'--pool',pool.removeprefix('city/pool/'),
                  '--size',str(size),'--factor','1.5','--expanded','--authored-ground','--emissive-uv','2','--emissive-gain','8','--glow']
            job=subprocess.run([str(x) for x in args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            row={'pool':pool,'size':size,'status':'pass' if job.returncode==0 else 'failed','output_tail':job.stdout[-2400:]}
            # Keep portable evidence even when a diagnostic fails before rendering.
            row['output_tail']=row['output_tail'].replace(str(ROOT)+'/', '')
            rows.append(row)
            if job.returncode==0:
                row['packet_storage']='directly replayable shared-resource packet; no duplicated terrain payloads'
            (output/'matrix.json').write_text(json.dumps({'promotion':False,'implementation':{str(p.relative_to(ROOT)):v for p,v in hashes.items()},'results':rows},indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='output_tail'}),flush=True)
    if any(x['status']!='pass' for x in rows):raise ValueError('preserved matrix contains failed layout/render cases')

if __name__=='__main__':main()
