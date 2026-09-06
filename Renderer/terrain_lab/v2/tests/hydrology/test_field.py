import subprocess
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
class HydrologyTests(unittest.TestCase):
 def test_portable_field(self):
  with tempfile.TemporaryDirectory() as tmp:
   binary=Path(tmp)/'hydrology-test'
   subprocess.run(['clang++','-std=c++17','-O2','Renderer/terrain_lab/v2/tests/hydrology/test_field.cpp','-o',str(binary)],cwd=ROOT,check=True)
   subprocess.run([str(binary)],cwd=ROOT,check=True)
 def test_shared_corridor_polygons(self):
  import json,sys,math
  sys.path.insert(0,str(ROOT/'Renderer/terrain_lab/v2/shared'))
  from scene_exchange import validate,inside,intersects
  p=ROOT/'Renderer/terrain_lab/v2/fixtures/hydrology/rivers.corridors.json'
  data=validate(json.loads(p.read_text()))
  self.assertEqual(data['coordinate_space'],'civ3_raw_delta_pixels_v1')
  self.assertEqual(len({e['id'] for e in data['envelopes']}),len(data['envelopes']))
  raw=json.loads((ROOT/'Renderer/terrain_lab/v2/audits/hydrology/crossing_witness_v1.json').read_text())
  self.assertEqual(len(data['envelopes']),2*len(raw['exclusion_capsules']))
  for i,c in enumerate(raw['exclusion_capsules']):
   # Circumscribed polygons must cover the exact radius at many angles.
   poly=data['envelopes'][i*2+1]['polygon'];r=c['bank_radius']*64*math.sqrt(2)
   for end in [c['a'],c['b']]:
    x,y=64*(end[0]+end[1]),64*(end[0]-end[1])
    for k in range(64):
     point=[x+r*math.cos(k*math.pi/32),y+r*math.sin(k*math.pi/32)]
     self.assertTrue(inside(point,poly) or min(__import__('scene_exchange').point_segment(point,a,b) for a,b in zip(poly,poly[1:]+poly[:1]))<1e-8)
