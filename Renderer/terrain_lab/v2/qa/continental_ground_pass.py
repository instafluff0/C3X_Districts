"""Compose source continental ground on fixed real-map benchmark recipes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hashlib
from coastal_pass import save

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['inland','coastal','wilderness','freshground'],required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();name='continental-ground-r1'
    sys.path.insert(0,str(V2/'app'));import real_map
    registry,data=real_map.load_registry()
    cells=[0xbb]*(data['width']*data['height'])
    for t in data['tiles']:cells[t['sourceY']*data['width']+t['sourceX']]=t['base']|(t['real']<<4)
    context=V2/'fixtures/beauty/source-continental-r1/map_context.h'
    context.write_text('// Same authoritative BIQ material context.\nnamespace q2_continental {\n'
        +f'inline int const map_width={data["width"]},map_height={data["height"]};\n'
        +'inline unsigned char const map_terrain[]={'+','.join(map(str,cells))+'};\n}\n')
    (context.parent/'.gitignore').write_text('*.dds\nground_fields.h\nmap_context.h\n')
    save(context.parent/'map_context.json',{'source_biq_sha256':registry['source']['sha256'],
        'compiled_header_sha256':hashlib.sha256(context.read_bytes()).hexdigest(),
        'dimensions':[data['width'],data['height']],'source_tiles':len(data['tiles'])})
    base='surface-decals-foundation' if a.region=='freshground' else 'river-corridor-r3'
    old=V2/'fixtures/beauty'/base/a.region;out=V2/'fixtures/beauty'/name/a.region
    f=json.loads((old/'fixture.json').read_text());m=json.loads((old/'terrain.module.json').read_text())
    m['terrain_hooks']['header']=(out.parent/'owner_hooks.h').relative_to(ROOT).as_posix()
    m['terrain_hooks']['ground_height']='q2_continental::ground_height'
    m['terrain_hooks']['initialize']='q2_continental::initialize'
    m['id']=name+'-'+a.region;f['id']=m['id']
    # Preserve the region shader and all existing geometry/material settings.
    save(out/'terrain.module.json',m)
    f['modules']=[(out/'terrain.module.json').relative_to(ROOT).as_posix()]
    save(out/'fixture.json',f)
    target=V2/'audits/beauty/out'/name/a.region
    if not a.prepare_only:
        if (target/'report.json').exists():raise ValueError('preserved candidate exists')
        subprocess.run([sys.executable,str(V2/'app/runner.py'),'compose','--fixture',str(out/'fixture.json'),
            '--candidate',name,'--output',str(target),'--hours','12','0'],cwd=ROOT,check=True)


if __name__=='__main__':main()
