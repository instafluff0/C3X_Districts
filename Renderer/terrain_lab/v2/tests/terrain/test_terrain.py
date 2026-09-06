"""Fixture and real Q2 kernel tests; these never claim visual acceptance."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
SOURCE=ROOT/'Renderer/terrain_lab/v2/systems/terrain/fixture_matrix.py'
spec=importlib.util.spec_from_file_location('matrix',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TerrainTests(unittest.TestCase):
 def test_coverage_and_provenance(self):
  recipe=m.matrix();cases=list(m.cases());self.assertEqual(len(recipe['pairs']),105)
  self.assertEqual(len(cases),1340);self.assertEqual(len({c['id'] for c in cases}),len(cases))
  self.assertEqual(recipe['provenance'],'synthetic_lab_only')
  self.assertEqual({p['rule'] for p in recipe['pairs']},{'smooth','shoulder','shore-mediated'})
 def test_csv_aliases_adjacency_and_halo(self):
  seen={}
  for case in m.cases():
   csv=m.csv_fixture(case);lines=csv.splitlines();self.assertEqual(len(lines),65)
   rows=[list(map(int,l.split(','))) for l in lines[1:]]
   self.assertEqual(len({tuple(r[:2]) for r in rows}),64)
   for c,r,x,y,base,real,*_ in rows:
    self.assertEqual((x-y)%2,0);self.assertTrue(0<=x<100);self.assertTrue(0<=base<=13 and 0<=real<=13)
   key=(tuple(case['families']),case['axis'],case['reverse'],case.get('base_override'))
   if key in seen:self.assertEqual(csv,seen[key])
   seen[key]=csv
 def test_surface_kernel(self):
  with tempfile.TemporaryDirectory() as temp:
   exe=Path(temp)/'surface-test'
   subprocess.run(['clang++','-std=c++17','-O2',str(Path(__file__).with_name('surface_test.cpp')),'-o',str(exe)],check=True)
   report=json.loads(subprocess.check_output([str(exe)],text=True));self.assertGreater(report['samples'],13000)
   self.assertLess(report['max_wrap_weight_delta'],1e-6)
 def test_actual_neighbor_recapture(self):
  with tempfile.TemporaryDirectory() as temp:
   exe=Path(temp)/'recapture-test'
   subprocess.run(['clang++','-std=c++17','-O2',str(Path(__file__).with_name('recapture_test.cpp')),'-o',str(exe)],check=True)
   report=json.loads(subprocess.check_output([str(exe)],text=True,cwd=ROOT))
   self.assertEqual(report['actual_neighbor_samples'],771)
   self.assertLessEqual(report['max_weight_delta'],1e-12)
   self.assertLessEqual(report['max_uv_delta'],1e-6)
if __name__=='__main__': unittest.main()
