"""User-requested large shore overview; synthetic topology, diagnostic relief."""
import json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PREFIX=HERE.relative_to(ROOT).as_posix()
w,h=12,10
cells={}
for y in range(-2,h+2):
 for x in range(-2,w+2):
  # Connected irregular island with two asymmetrical bays, promontories and
  # smaller offshore islands; ocean surrounds every visible land boundary.
  xx,yy=x*20/w,y*16/h
  d=((xx-9.4)/6.8)**2+((yy-7.3)/5.1)**2
  bay=.65*math.exp(-((xx-14.8)**2/5+(yy-6.7)**2/3))+.48*math.exp(-((xx-7.0)**2/4+(yy-11.5)**2/4))
  land=d+bay<1 or ((xx-4.3)**2+(yy-4.4)**2<2.3) or ((xx-16)**2+(yy-11.5)**2<1.5)
  hill=land and ((xx<7 and yy<8) or (xx>12 and yy<6) or ((xx-11)**2+(yy-8)**2<5))
  base=2 if land else (11 if d<1.8 else 12 if d<2.8 else 13)
  cells[x,y]=[base,5 if hill else base,0]
# This overview isolates shorelines; river cases retain separate fixtures.
lines=[f'C3X_BIQ_TERRAIN_WINDOW_V2,{w},{h},{w*h},40,40,160,160,{(w+4)*(h+4)-w*h}']
for (x,y),(base,real,river) in sorted(cells.items(),key=lambda kv:(not (0<=kv[0][0]<w and 0<=kv[0][1]<h),kv[0][1],kv[0][0])):lines.append(f'{x},{y},{40+x+y},{40+x-y},{base},{real},0,0,{river}')
(HERE/'island-overview.csv').write_text('\n'.join(lines)+'\n')
c=dict(mode=0,legacy=0,scene_linear=1,wrap_x=0,tile_halfwidth=80,material_root='Renderer/packs/Civ5EnvironmentSkin/provenance/terrain_textures')
(HERE/'island-overview.controls.json').write_text(json.dumps(c,indent=2)+'\n')
f=json.loads((HERE/'context-linear.fixture.json').read_text());f.update(id='island-overview',tile_count=w*h,viewport=[2048,1280],terrain=PREFIX+'/island-overview.csv');f['scenarios']['controls']=PREFIX+'/island-overview.controls.json';f['isolations']=['large_synthetic_shore_overview_diagnostic_relief'];f['settings']['samples']=4
(HERE/'island-overview.fixture.json').write_text(json.dumps(f,indent=2)+'\n')
print('Prepared120tile synthetic island overview with fully surrounding ocean.')
