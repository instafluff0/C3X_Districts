"""Offline numerical/source fidelity gates; visual acceptance remains separate."""
import importlib.util
import json
from pathlib import Path
import unittest
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
P=ROOT/'Renderer/terrain_lab/v2/systems/relief/build_fixture.py'
sys.path.insert(0,str(P.parent))
spec=importlib.util.spec_from_file_location('relief',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@unittest.skipUnless((ROOT/'Renderer/packs/Civ5EnvironmentSkin/textures/grassland_base_color.dds').is_file(), 'local source-pack integration tests')
class ReliefTests(unittest.TestCase):
    def test_selected_skin_height_payload(self):
        import hashlib
        audit=json.loads((ROOT/'Renderer/terrain_lab/v2/audits/relief/SELECTED_SOURCE_AUDIT.json').read_text())
        f=m.Fixture('source-rock',15)
        channel=next(c for c in audit['channels'] if c['entry']=='ART_DEF_TERRAIN_ELEMENT_HILL' and c['lod']==0)
        payload=(ROOT/channel['owned_output']).read_bytes()[148:]
        self.assertEqual(hashlib.sha256(payload).hexdigest(),channel['source_payload_sha256'])
        self.assertFalse(channel['normalized_matches_selected_source'])
        np.testing.assert_array_equal(f.hillfield,np.frombuffer(payload,dtype=np.uint8).reshape(512,512)/255)
    def test_uniform_source_mesh_uvs(self):
        f=m.Fixture('source-rock',7)
        pack=ROOT/'Renderer/packs/ShoreNormalized'
        f.source_mesh(pack,'cliff_large_02',3,3,.52,.4,z=.018)
        got=f.draws[0][0];src=json.loads((pack/'meshes/features/cliff_large_02.json').read_text());ix=src['topology']['indices']
        uv=np.array([v['uv0'] for v in src['vertices']])[ix]
        np.testing.assert_allclose(got[:,6:8],uv,rtol=0,atol=3e-8)
        original=np.array([v['position'] for v in src['vertices']])[ix]
        actual=got[:,8:11];d1=np.linalg.norm(original[1:]-original[:-1],axis=1);d2=np.linalg.norm(actual[1:]-actual[:-1],axis=1)
        np.testing.assert_allclose(d2,d1*.52,rtol=0,atol=5e-7)
        self.assertTrue(f.draws[0][2]);self.assertTrue(f.draws[0][3])
    def test_hills_continuous_across_tile_edges(self):
        f=m.Fixture('coast-source',7)
        for x in range(1,5):
            y=np.linspace(.2,5.5,100);a=f.ground(np.full(100,x-1e-5),y);b=f.ground(np.full(100,x+1e-5),y)
            self.assertLess(np.max(np.abs(a-b)),.0001)
    def test_no_generated_coast_rock_surface(self):
        f=m.Fixture('coast-source',7);x,y=np.meshgrid(np.linspace(0,6,180),np.linspace(0,6,180))
        self.assertEqual(float(np.max(f.fields(x,y)[2])),0)
        f.dressing();self.assertGreater(len(f.transforms),8)
        self.assertTrue(all(t['kind']=='normalized_mesh' and 'ShoreNormalized' in t['mesh'] for t in f.transforms))
    def test_source_height_samples_preserved(self):
        f=m.Fixture('range',7);f.bodies()
        for (cx,cy,s,a,field,aspect),(v,*_) in zip(f.masses,f.draws):
            uv=v[:,6:8];p=v[:,8:11];expected=m.sample(field,uv[:,0],uv[:,1])*aspect*s+.07
            np.testing.assert_allclose(p[:,2],expected,rtol=0,atol=1e-5)
    def test_no_empty_draws(self):
        f=m.Fixture('range',7);f.surface();self.assertTrue(all(len(v)>0 for v,*_ in f.draws))
if __name__=='__main__':unittest.main()
