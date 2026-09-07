"""Compose flat/high ground and complete rock channels over displaced ground."""
import json
from pathlib import Path
import subprocess
import sys
from extend_packet_materials import extend

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2'


def main():
    source=V2/'audits/beauty/out/continental-ground-r2/inland'
    inputs=V2/'audits/beauty/out/continental-material-r1-input/inland'
    out=V2/'audits/beauty/out/continental-material-r1/inland'
    if (out/'report.json').exists():raise ValueError('preserved material diagnostic exists')
    r=json.loads((source/'report.json').read_text());jobs=json.loads((source/'batch.json').read_text())
    pack=ROOT/r['effective']['fixture']['packs']['terrain']
    rock=json.loads((pack/'materials/mountains.json').read_text());channels={}
    for i,name in enumerate(('snow','desert_stripe_1','desert_stripe_2','desert_stripe_3')):
        for j,c in enumerate(('height','specular')):channels[108+i*2+j]=pack/rock['authored_layers'][name][c]['texture']
    for i,name in enumerate(('grassland','plains')):
        m=json.loads((pack/'materials'/(name+'.json')).read_text())
        for j,c in enumerate(('base_color','height','specular')):channels[116+i*3+j]=pack/m['elevated'][c]['texture']
    inputs.mkdir(parents=True,exist_ok=True);e=[]
    for i,job in enumerate(jobs):
        target=inputs/f'packet-{i}';e.append(extend(Path(job[0]),target,channels))
        job[0]=str(target);r['outputs'][i]['packet']=target.relative_to(ROOT).as_posix()
    (inputs/'report.json').write_text(json.dumps(r)+'\n');(inputs/'batch.json').write_text(json.dumps(jobs)+'\n')
    (inputs/'bindings.json').write_text(json.dumps(e,indent=2)+'\n')
    fixture=V2/'fixtures/beauty/continental-material-r1/inland';fixture.mkdir(parents=True,exist_ok=True)
    shader=fixture/'combined.hlsl';shader.write_text('#define Q2_CONTINENTAL_MATERIAL 1\n'
        '#include "../../rock-channels-r2/inland/combined.hlsl"\n')
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(inputs/'report.json'),
        '--shader',str(shader),'--output',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
