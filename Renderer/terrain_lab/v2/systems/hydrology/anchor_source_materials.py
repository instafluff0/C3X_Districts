"""Pin source material coordinates to CSV raw origin, independent of crop/camera."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
def anchor(fixture,base):
 f=json.loads(fixture.read_text());m=json.loads(base.read_text());lines=(ROOT/f['terrain']).read_text().splitlines();header=lines[0].split(',');tile=next(list(map(int,line.split(','))) for line in lines[1:] if line.startswith('0,0,'));rx,ry=tile[2:4]
 wrap=f.get('real_map',{}).get('region',{}).get('wrap',{}).get('x',m.get('hydrology_hooks',{}).get('initialize')!='q3_scene::initialize_no_wrap')
 identifier=f['id'];shader=V2/'shaders/hydrology'/f'{identifier}_anchored.hlsl'
 shader.write_text(f'// Generated from exact source tile (0,0); no camera state.\n#define Q3_MATERIAL_ORIGIN_X {(rx+ry)*.5-.5:.9f}\n#define Q3_MATERIAL_ORIGIN_Y {(rx-ry)*.5-.5:.9f}\n#define Q3_MATERIAL_WRAP_WIDTH {int(header[6]) if wrap else 0}\n'+('#define Q6_WORLD_SHADOWS 1\n' if m.get('packet_postprocessor') else '')+('#define Q3_BED_ONLY 1\n' if 'bed-only' in identifier else '')+'#include "scene_linear.hlsl"\n')
 m.update(id=identifier+'-anchored',shader=str(shader.relative_to(ROOT)))
 dest=V2/'systems/hydrology'/f'{identifier}_anchored.module.json';dest.write_text(json.dumps(m,indent=2)+'\n');f['modules']=[str(dest.relative_to(ROOT))];fixture.write_text(json.dumps(f,indent=2)+'\n')
 return {'fixture':str(fixture.relative_to(ROOT)),'raw_origin':[rx,ry],'wrap_x':wrap,'shader':str(shader.relative_to(ROOT))}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);p.add_argument('base_module',type=Path);a=p.parse_args();print(json.dumps(anchor(a.fixture.resolve(),a.base_module.resolve())))
