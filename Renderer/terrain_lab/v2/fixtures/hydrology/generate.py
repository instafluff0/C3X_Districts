"""Explicit synthetic stress fixtures; never represented as captured BIQ terrain."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PREFIX=HERE.relative_to(ROOT).as_posix()
PACKS={"terrain":"Renderer/packs/Civ5EnvironmentSkin","vegetation":"Renderer/packs/Civ5EnvironmentVegetation","decals":"Renderer/packs/DecalsNormalized","relief":"Renderer/packs/TerrainElementsNormalized","shore":"Renderer/packs/ShoreNormalized"}
def make(name,kind,mode=0,legacy=0):
 n=6
 lines=[f'C3X_BIQ_TERRAIN_WINDOW_V2,{n},{n},{n*n},15,30,120,120,64']
 for y in range(-2,n+2):
  for x in range(-2,n+2):
   water=x>=4
   hill=(y>=3 and x<4)
   if kind=='coves': water=x>=4 or (x>=3 and y==1) or (2<=x<=3 and y==2);hill=y>=3 and x==3
   if kind=='islands': water=not ((x==2 and y==2) or (3<=x<=4 and 3<=y<=4));hill=x>=3 and y>=3
   if kind=='channel': water=x==3;hill=x==2 and y>=3
   if kind=='rivers': hill=x==2 and y>=2
   base=11+(y%3) if water else 2
   river=0
   if kind=='rivers':
    if x==1 and -2<=y<=2:river|=8
    if x==2 and -2<=y<=2:river|=128
    if y==2 and 1<=x<=4:river|=2
    if y==3 and 1<=x<=4:river|=32
   lines.append(f'{x},{y},{(30+x+y)%120},{30+x-y},{base},{5 if hill and not water else base},0,0,{river}')
 (HERE/(name+'.csv')).write_text('\n'.join(lines)+'\n')
 controls={'mode':mode,'legacy':legacy,'material_root':'Renderer/packs/Civ5EnvironmentSkin/provenance/terrain_textures'}
 (HERE/(name+'.controls.json')).write_text(json.dumps(controls,indent=2)+'\n')
 fixture={'schema':'c3x.lab_v2.fixture.v1','id':name,'track':'Q3-hydrology','campaign':'Q1','tile_count':n*n,'viewport':[960,640],'terrain':PREFIX+'/'+name+'.csv','modules':['Renderer/terrain_lab/v2/systems/hydrology/static.module.json'],'packs':PACKS,'references':['civ6.sea_and_shore','civ6.river','civ6.rocky_hill_coast'],'isolations':['beach-only','bed-only','classification'],'settings':{'anisotropy':8,'mip_bias':0,'samples':1,'render_scale':1,'postprocess':'box','camera_offsets':[[0,0]]},'scenarios':{'controls':PREFIX+'/'+name+'.controls.json'}}
 (HERE/(name+'.fixture.json')).write_text(json.dumps(fixture,indent=2)+'\n')
for name,kind,mode,legacy in [('coast','coast',0,0),('coast-before','coast',0,1),('coves','coves',0,0),('islands','islands',0,0),('channel','channel',0,0),('rivers','rivers',0,0),('beach-only','coves',1,0),('bed-only','coves',2,0),('classification','coast',3,0)]: make(name,kind,mode,legacy)
