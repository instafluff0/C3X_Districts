"""Export rendered Q3 corridors to Q0's shared sidecar; source is read-only."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'Renderer/terrain_lab/v2/shared'))
from scene_exchange import validate

def digest(data):return hashlib.sha256(data).hexdigest()
def convex(points):
 points=sorted(set(points));cross=lambda o,a,b:(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
 lower=[];upper=[]
 for p in points:
  while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
  lower.append(p)
 for p in reversed(points):
  while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
  upper.append(p)
 return [list(p) for p in lower[:-1]+upper[:-1]]
def to_world(p):return [64*(p[0]+p[1]),64*(p[0]-p[1])]
def publish(fixture,output):
 f=json.loads(fixture.read_text());terrain=ROOT/f['terrain'];header=terrain.read_text().splitlines()[0].split(',')
 wrap=f.get('real_map',{}).get('region',{}).get('wrap',{'x':True})['x']
 with tempfile.TemporaryDirectory() as tmp:
  exe=Path(tmp)/'export'
  subprocess.run(['clang++','-std=c++17','-O2',str(ROOT/'Renderer/terrain_lab/v2/systems/hydrology/export.cpp'),'-o',str(exe)],check=True)
  data=json.loads(subprocess.check_output([str(exe),str(terrain),'1' if wrap else '0']))
 gh=digest(json.dumps(data['river_edges'],sort_keys=True,separators=(',',':')).encode());envelopes=[];scale=64*math.sqrt(2)
 for index,c in enumerate(data['exclusion_capsules']):
  for kind,key,margin in [('river','water_radius',0),('bank','bank_radius',.04*scale)]:
   radius=c[key]*scale/math.cos(math.pi/16);points=[]
   for end in [to_world(c['a']),to_world(c['b'])]:
    points += [(end[0]+radius*math.cos(i*math.pi/8),end[1]+radius*math.sin(i*math.pi/8)) for i in range(16)]
   envelopes.append({'id':f"q3-{c['edge_id']}-{index%32}-{kind}",'kind':kind,'polygon':convex(points),'height_range':[-32,32],'clearance':margin,'source_geometry_sha256':gh})
 result={'schema':'c3x.lab_v2.corridors.v1','coordinate_space':'civ3_raw_delta_pixels_v1','terrain_sha256':digest(terrain.read_bytes()),'region_id':f.get('real_map',{}).get('region_id',f['id']),'provider':'Q3-hydrology','revision':'static-07','wrap_period':[64*int(header[6]) if wrap else 0,0],'envelopes':envelopes,'units':'normal_zoom_pixels','coordinate_mapping':'X=64*(column+row), Y=64*(column-row), Z=64*local_height; origin is raw tile (0,0)','height_policy':'informational proxy range; footprint exclusion applies to overhangs at all heights','polygonization':'circumscribed 16-gon capsule hulls; radial inflation sec(pi/16)-1; no undercoverage','source_implementation_sha256':digest((ROOT/'Renderer/terrain_lab/v2/systems/hydrology/field.h').read_bytes()),'crossings_local_tile_lattice':data['crossing_witnesses']}
 validate(result);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')
 return {'path':str(output.relative_to(ROOT)),'sha256':digest(output.read_bytes()),'schema':result['schema'],'owner':'Q3-hydrology'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(publish(a.fixture.resolve(),a.output.resolve()),indent=2))
