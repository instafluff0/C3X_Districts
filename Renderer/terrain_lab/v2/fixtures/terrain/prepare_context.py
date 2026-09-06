"""Generate Q2 synthetic open-countryside crop and controlled shader fixtures."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
V2=HERE.parents[1]
ROOT=V2.parents[2]
def relative(p):return p.relative_to(ROOT).as_posix()
rows=[]
for halo in (False,True):
 for y in range(-2,10):
  for x in range(-2,10):
   if (not(0<=x<8 and 0<=y<8)) != halo:continue
   real=0 if x<3 and y<5 else 1 if x+y<9 else 2
   rows.append(f'{x},{y},{(94+x+y)%100},{50+x-y},{real},{real},0,0,0')
(HERE/'context.csv').write_text('C3X_BIQ_TERRAIN_WINDOW_V2,8,8,64,0,0,100,100,80\n'+'\n'.join(rows)+'\n')
f=json.loads((HERE/'micro.fixture.json').read_text());f.update(tile_count=64,viewport=[512,256],terrain=relative(HERE/'context.csv'))
f['settings']['camera_offsets']=[[0,0],[1,0],[16,8],[0,0]]
module=json.loads((V2/'systems/terrain/base.module.json').read_text())
for control in ['detail','off','baseline','weights','normal','height','roughness','albedo']:
 shader=V2/f'shaders/terrain/{control}.hlsl'
 shader.write_text(f'#define Q2_DETAIL {0 if control in ("off","baseline") else 1}\n#define Q2_ISOLATION '+str({'weights':1,'normal':2,'height':3,'roughness':4,'albedo':5}.get(control,0))+'\n#include "base.hlsl"\n')
 m=dict(module,shader=relative(shader));mp=V2/f'systems/terrain/{control}.module.json';mp.write_text(json.dumps(m,indent=2)+'\n')
 fixture=dict(f,id=f'q2-context-{control}',modules=[relative(mp)])
 (HERE/f'context-{control}.fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
(HERE/'context.provenance.json').write_text(json.dumps({'schema':'c3x.q2.context.v1','class':'synthetic_stress_cases','purpose':'provisional open countryside base-only gameplay-scale crop; no object placement or source BIQ claim','seed':0,'basis_pixels':[128,64],'normal_viewport':[512,256],'reduced_viewport':[256,128],'camera_center_lattice':[4,4],'halo':2,'raw_origin':[94,50],'missing_context':['Q8 settlements/routes','Q3 shore and water','Q4 relief','verified test.biq']},indent=2)+'\n')
