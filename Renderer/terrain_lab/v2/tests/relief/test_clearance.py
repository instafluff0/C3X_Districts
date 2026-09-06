import importlib.util
from pathlib import Path
import unittest
P=Path(__file__).resolve().parents[2]/'systems/relief/clearance.py'
s=importlib.util.spec_from_file_location('clearance',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ClearanceTests(unittest.TestCase):
 def test_crown_hit_when_origin_clear(self):
  f=m.source_footprint({'minimum':[-.4,-.4,0],'maximum':[.4,.4,1]},0,.4,1,0)
  self.assertFalse(m.clear(f,[{'points':[[-2,0],[2,0]],'half_width':.08}]))
 def test_diagonal_rotated_corner(self):
  f=m.source_footprint({'minimum':[-.3,-.1,0],'maximum':[.3,.1,1]},0,0,1,.7)
  self.assertFalse(m.clear(f,[{'points':[[-1,-1],[1,1]],'half_width':.03}]))
 def test_removal_is_local(self):
  bounds={'minimum':[-.1,-.1,0],'maximum':[.1,.1,1]};e=[{'points':[[-2,0],[2,0]],'half_width':.05}]
  before=[i for i in range(5) if m.clear(m.source_footprint(bounds,0,i,1,0),[])]
  after=[i for i in range(5) if m.clear(m.source_footprint(bounds,0,i,1,0),e)]
  self.assertEqual(before,[0,1,2,3,4]);self.assertEqual(after,[1,2,3,4])
 def test_tree_inside_city_polygon_is_rejected(self):
  f=m.source_footprint({'minimum':[-.1,-.1,0],'maximum':[.1,.1,1]},.5,.5,1,0)
  self.assertFalse(m.clear(f,[{'shape':'polygon','points':[[0,0],[1,0],[1,1],[0,1],[0,0]],'half_width':.04}]))
 def test_city_raw_anchor_conversion(self):
  import json
  witness=Path(__file__).resolve().parents[2]/'audits/objects/CITY_VEGETATION_WITNESS.json'
  if not witness.is_file():self.skipTest('Q7 local candidate unavailable')
  data=json.loads(witness.read_text());region=data['regions'][1]
  meta={'region':{'origin':region['origin_raw']},'source_sha256':region['source_sha256']}
  envelopes=m.city_envelopes(data,meta);self.assertEqual(len(envelopes),4)
  self.assertTrue(all(e['raw_city_anchor']==region['raw_city_anchor'] for e in envelopes))
  x,y=region['polygons'][0]['raw_delta_pixel_polygon'][0];a,b=envelopes[0]['points'][0]
  self.assertAlmostEqual(64*((a-1.5)+(b-1.5)),x)
  self.assertAlmostEqual(64*((a-1.5)-(b-1.5)),y)
 def test_segment_extension_is_not_a_hit(self):
  f=m.source_footprint({'minimum':[0,0,0],'maximum':[1,1,1]},0,0,1,0)
  self.assertTrue(m.clear(f,[{'points':[[2,0],[3,0]],'half_width':.01}]))
if __name__=='__main__':unittest.main()
