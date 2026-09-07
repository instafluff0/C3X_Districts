"""Compose a centered-capital fixture with the validated paving and facade lights."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from city_scene_pass import ROOT,V2,save,rel,file_hash
from city_facade_light_probe import derive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.relative_to(V2/'audits/beauty/out')
    if out.exists():raise ValueError('preserve existing capital composition')
    ap=next((V2/f'fixtures/beauty/city-scene-r{a.revision}').glob('*/augmentation.json'))
    augmentation=json.loads(ap.read_text());surface=json.loads((ap.parent/'surface.json').read_text())
    if augmentation['capital']['composition']!='central_surrounded':raise ValueError('requires a central capital fixture')
    out.mkdir(parents=True)
    data=derive(augmentation,surface,128,source_facade_slots=('capital',))
    data.update(gain=4,local_building_occlusion=True,object_reflections=True,augmentation_sha256=file_hash(ap),
                facade_placement='Capital sampled facade planes; ordinary houses retain their prior light approximation')
    save(out/'lights.json',data)
    raw=V2/f'audits/beauty/out/city-scene-r{a.revision}'/ap.parent.name/'combined'
    def run(tool,*args):subprocess.run([sys.executable,str(V2/'qa'/tool),*map(str,args)],check=True)
    run('city_light_buffer_probe.py','--source-render',raw,'--lights',out/'lights.json','--output',out/'lights')
    run('settlement_ground_probe.py','--source-render',out/'lights/render','--augmentation',rel(ap),
        '--ground-parts',augmentation['compound_ground']['mapping'],
        '--binding',rel(V2/'fixtures/beauty/city-ground-binding-r1/modern.json'),
        '--capital-footprint','source-hull','--output',out/'ground')
    run('city_environment_probe.py','--source',out/'ground/render','--output',out/'environment','--enable-bound-metalness')


if __name__=='__main__':main()
