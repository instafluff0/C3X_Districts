"""Source provenance and executable numerical contracts for the r13 port."""
from pathlib import Path
import hashlib,json,subprocess,tempfile,unittest
from .prepare import ROOT,HERE,LAB,function,terrain_boundaries
class Contract(unittest.TestCase):
    def test_selected_provider_equations(self):
        for name,category in [('terrain','relief'),('mountain','relief'),('objects','objects')]:
            selected=(LAB/f'shaders/{category}/beauty_{name}.hlsl').read_text()
            if name=='terrain':selected=terrain_boundaries(selected)
            adapted=(HERE/f'{name}.hlsl').read_text()
            self.assertEqual(function(selected,'shade'),function(adapted,'shade'))
            self.assertIn('float3 light_direction = ShadowL.xyz;',adapted)
            self.assertNotIn('#include',adapted)
    def test_selected_source_pins(self):
        p=json.loads((HERE/'provenance.json').read_text())
        self.assertEqual(p['authority'],'source-fidelity-r13/inland')
        self.assertEqual((p['trees'],p['recipes'],p['count_weight']),(22,25,180))
        for relative,expected in p['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),expected,relative)
    def test_executable_light_and_river_contract(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'probe'
            subprocess.run(['c++','-std=c++17','-O2',str(HERE/'contract_probe.cpp'),str(HERE.parent/'environment_runtime.cpp'),'-o',str(out)],check=True,capture_output=True)
            r=subprocess.run([str(out),str(LAB/'fixtures/beauty/gameplay-100-v1/inland/terrain.csv')],check=True,capture_output=True,text=True)
            self.assertIn('samples=9216 exact equality',r.stdout)
            self.assertIn('phases=24 noon up-right',r.stdout)
if __name__=='__main__':unittest.main()
