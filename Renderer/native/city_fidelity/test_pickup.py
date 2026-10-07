"""Check the promoted city pack: the readability recomposition of the frozen
pre-readability library (Renderer/lab/studies/city_readability/recompose.py)."""
from pathlib import Path
import hashlib
import json
import os
import unittest

from Renderer.lab.studies.city_readability import runtime_pack as rp

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / os.environ.get('C3X_TEST_CITY_DIRECTORY', 'Renderer/packs/CityCompositionRuntime')
PREFIX = 'Renderer/packs/CityCompositionRuntime/'
GENERIC_PALACE = 'city/palace/root/c9fb1862f0f18efe'


class Pickup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((PACK / 'manifest.json').read_text())
        cls.library = rp.decode(PACK / 'city.bin')

    def test_recorded_inputs_and_texture_closure(self):
        self.assertEqual(self.manifest['builder'], 'Renderer/lab/studies/city_readability/recompose.py')
        self.assertEqual(self.library['version'], 5)
        self.assertIn('Renderer/packs/CityCompositionFrozen/city.bin', self.manifest['source_sha256'])
        for path, digest in self.manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        for material in self.library['materials']:
            for path in material['textures']:
                if not path:
                    continue
                self.assertTrue(path.startswith(PREFIX) and '..' not in path and ':' not in path)
                texture = PACK / path.removeprefix(PREFIX)
                self.assertEqual(texture.stem, hashlib.sha256(texture.read_bytes()).hexdigest())

    def test_native_selection_matrix_and_bounded_instances(self):
        templates = self.library['templates']
        keys = {(t['culture'], t['era'], t['size'], int(t['capital']), int(t['walled']), t['variant'])
                for t in templates}
        expected = {(c, e, s, k, w, v) for c in range(5) for e in range(4) for s in range(3)
                    for k in range(2) for w in range(2 if s == 0 else 1) for v in range(3)}
        self.assertEqual(len(templates), 480)
        self.assertEqual(keys, expected)
        for t in templates:
            self.assertLessEqual(len(t['instances']), 128)
            self.assertLessEqual(sum(len(i['lights']) for i in t['instances']), 128)
            self.assertEqual(sum(i['capital'] for i in t['instances']), int(t['capital']))
            self.assertTrue(t['authority'].startswith('lab-readable-'))
            self.assertTrue(t['paving'] and t['paving']['vertices'])
            for i in t['instances']:
                self.assertLess(i['model'], len(self.library['models']))
                self.assertLessEqual(len(i['effects']), 16)

    def test_look_effects_and_culture_palaces(self):
        library = self.library
        self.assertTrue(any(library['look']))
        effect = library['materials'][library['effect_material']]
        self.assertEqual(effect['ground'], 1)
        assets = [m['asset'] for m in self.manifest['models']]
        self.assertEqual(len(assets), len(library['models']))
        late_capitals = 0
        for t in library['templates']:
            for i in t['instances']:
                if i['capital'] and t['era'] >= 2:
                    late_capitals += 1
                    # Every late-era capital has its culture's palace, not the generic one.
                    self.assertNotEqual(assets[i['model']], GENERIC_PALACE)
                for e in i['effects']:
                    self.assertIn(int(e[3]), (rp.FLAME, rp.SMOKE, rp.NIGHT_LIGHT))
        self.assertGreater(late_capitals, 0)


if __name__ == '__main__':
    unittest.main()
