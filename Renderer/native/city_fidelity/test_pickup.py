"""Compare the promoted city pack with the selected culture/era Lab recipes."""
from pathlib import Path
import hashlib
import json
import os
import unittest
ROOT=Path(__file__).resolve().parents[3]
PACK=ROOT/os.environ.get('C3X_TEST_CITY_DIRECTORY','Renderer/packs/CityCompositionRuntime')

class Pickup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack=json.loads((PACK/'manifest.json').read_text())
        cls.recipes=json.loads((PACK/'recipes.json').read_text())['designs']

    def test_source_inputs_and_runtime_texture_closure(self):
        for path,digest in self.pack['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,path)
        for material in self.pack['materials']:
            for path in material['textures']:
                if not path:continue
                prefix='Renderer/packs/CityCompositionRuntime/'
                self.assertTrue(path.startswith(prefix) and '..' not in path and ':' not in path)
                texture=PACK/path.removeprefix(prefix)
                self.assertEqual(texture.stem,hashlib.sha256(texture.read_bytes()).hexdigest())

    def test_exact_selected_instances_for_all_states(self):
        templates=self.pack['templates'];self.assertEqual(len(templates),480)
        self.assertEqual(len(self.recipes),60)
        for t in templates:
            design=next(d for d in self.recipes if (d['culture'],d['era'],d['variant'])==(t['culture'],t['era'],t['variant']))
            tier=design['tier_designs'][t['size']]
            capital=t['capital']
            items=list(tier.get('capital_houses',tier['houses']) if capital else tier['houses'])
            if not capital or not design.get('capital_replaces_centerpiece'):
                items.append(tier.get('capital_centerpiece',tier['base_centerpiece']) if capital else tier['base_centerpiece'])
            if capital:items.append(tier['palace'])
            items.extend(tier.get('capital_decorations' if capital else 'decorations',[]))
            if t['walled']:
                self.assertEqual(t['size'],0)
                items.extend(design['wall_instances'])
            self.assertEqual(len(items),len(t['instances']))
            self.assertTrue(t['owns_walls'] and t['anchor_layout'])
            self.assertTrue(t['authority'].startswith('lab-fixed-'))
            for actual,expected in zip(t['instances'],items):
                model=self.pack['models'][actual['model']]
                self.assertEqual(model['asset'],expected['asset'])
                self.assertEqual(model['pack'],expected['pack'])
                for key in ['scale','rotation','offset']:self.assertEqual(actual[key],expected[key])
            self.assertIsNone(t['paving'])
            self.assertNotIn('foundation',t)

    def test_native_selection_matrix_and_bounded_lighting(self):
        keys={(t['culture'],t['era'],t['size'],int(t['capital']),int(t['walled']),t['variant']) for t in self.pack['templates']}
        expected={(c,e,s,k,w,v) for c in range(5) for e in range(4) for s in range(3)
                  for k in range(2) for w in range(2 if s==0 else 1) for v in range(3)}
        self.assertEqual(keys,expected)
        for t in self.pack['templates']:
            self.assertLessEqual(sum(len(i['lights']) for i in t['instances']),128)
            self.assertLessEqual(len(t['instances']),128)
            self.assertEqual(sum(i['slot']=='capital' for i in t['instances']),int(t['capital']))
        self.assertEqual(self.pack['gaps'],[])
        for gap in self.pack['gaps']:
            self.assertIn(gap['channel'],['ambient_occlusion','emissive'])
            self.assertTrue(gap['reason'].startswith('selected Lab derivative lacks uv'))

if __name__=='__main__':unittest.main()
