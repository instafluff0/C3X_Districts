"""Render recovered decal geometry/materials on immutable combined scene packets.

This is a diagnostic replay, not full fixture composition or promotion.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from rebind_packet_textures import rebind, dds

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
PAIRS=[('Grass_Decal_B','base_color_c996c6a9d015eebe.dds'),
       ('Grass_Decal_H','height_31eb0f0117ea3beb.dds'),
       ('Plains_Decal_B','base_color_211cf603f50c6f54.dds'),
       ('Plains_Decal_H','height_3ba76b3b97d571a8.dds')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['coastal','inland','wilderness','freshcanopy','freshground'],required=True)
    p.add_argument('--revision',choices=['r3','r4'],default='r4')
    a=p.parse_args();name='surface-decals-'+a.revision
    foundation='surface-decals-foundation-v2' if a.region=='freshground' else 'river-corridor-r3'
    recipe='surface-decals-foundation' if a.region=='freshground' else 'river-corridor-r3'
    source=V2/'audits/beauty/out'/foundation/a.region
    out=V2/'audits/beauty/out'/name/a.region
    if (out/'report.json').exists():raise ValueError('preserved diagnostic already exists')
    inputs=V2/'audits/beauty/out'/(name+'-input')/a.region
    inputs.mkdir(parents=True,exist_ok=True)
    replacements={dds(ROOT/'Renderer/packs/DecalsNormalized/textures/decals'/old)['payload_sha256']:
                  V2/'fixtures/beauty/source-ground-decals-r1'/(new+'.dds') for new,old in PAIRS}
    report=json.loads((source/'report.json').read_text());jobs=json.loads((source/'batch.json').read_text());evidence=[]
    for index,job in enumerate(jobs):
        target=inputs/f'packet-{index}'
        evidence.append(rebind(Path(job[0]),target,replacements));job[0]=str(target)
        report['outputs'][index]['packet']=target.relative_to(ROOT).as_posix()
    report['diagnostic_texture_rebinding']=True
    (inputs/'batch.json').write_text(json.dumps(jobs)+'\n')
    (inputs/'report.json').write_text(json.dumps(report)+'\n')
    (inputs/'rebinding.json').write_text(json.dumps(evidence,indent=2)+'\n')
    fixture=V2/'fixtures/beauty'/name/a.region;fixture.mkdir(parents=True,exist_ok=True)
    shader=fixture/'combined.hlsl'
    settings='#define Q2_GROUND_PATCH_SCALE .55\n#define Q2_GROUND_CELL_COUNT 32\n' if a.revision=='r4' else ''
    shader.write_text(settings+'#define Q2_GROUND_DECAL_SOURCE_ALPHA 1\n#define Q2_SOURCE_GROUND_DECALS 1\n'
                      '#include "../../source-ground-decals-r1/geometry.hlsl"\n'
                      f'#include "../../{recipe}/{a.region}/combined.hlsl"\n')
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(inputs/'report.json'),
                    '--shader',str(shader),'--output',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
