"""Independent comparisons against preserved selected city evidence."""
from pathlib import Path
import hashlib,json,math,unittest
ROOT=Path(__file__).resolve().parents[3]
LAB=ROOT/'Renderer/terrain_lab/v2'
PACK=ROOT/'Renderer/packs/CityCompositionRuntime'
def read(p):return json.loads(p.read_text())

class Pickup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pack=read(PACK/'manifest.json')

    def test_source_files_have_not_changed(self):
        for path,digest in self.pack['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,path)

    def test_all_twenty_source_pools_have_stable_growth_prefixes(self):
        templates=[t for t in self.pack['templates'] if t['authority']=='generic-source-growth']
        self.assertEqual(len(templates),60)
        self.assertEqual(self.pack['gaps'],[])
        for culture in range(5):
            for era in range(4):
                stages=sorted([t for t in templates if t['culture']==culture and t['era']==era],key=lambda t:t['size'])
                self.assertEqual([t['size'] for t in stages],[0,1,2])
                for a,b in zip(stages,stages[1:]):
                    self.assertEqual(a['instances'],b['instances'][:len(a['instances'])])
                    if a['paving']:self.assertEqual(a['paving']['period'],b['paving']['period'])

    def test_selected_body_transforms_are_exact(self):
        for revision in [60,61,64,101,111,112]:
            path=next((LAB/f'fixtures/beauty/city-scene-r{revision}').glob('*/augmentation.json'))
            source=read(path)
            template=next(t for t in self.pack['templates'] if t['authority']==f'selected-r{revision}' and t['size']==1)
            self.assertEqual(len(template['instances']),len(source['instances']))
            self.assertEqual(template['clearance'],[.05,2.5,source.get('vegetation_clearance') or 0,
                (source.get('river_exclusion') or {}).get('threshold_pixels',0)])
            for actual,expected in zip(template['instances'],source['instances']):
                self.assertEqual(self.pack['models'][actual['model']]['asset'],expected['asset'])
                for key in ['slot','scale','rotation','offset']:self.assertEqual(actual[key],expected[key])
                self.assertEqual(actual['bounds'],expected['local_bounds'])

    def test_central_facade_lights_match_preserved_combined_evidence(self):
        for revision,site in [(111,'inland'),(112,'holdout')]:
            path=next((LAB/f'fixtures/beauty/city-scene-r{revision}').glob('*/augmentation.json'))
            augmentation=read(path);surface=read(path.parent/'surface.json')
            expected=read(LAB/f'audits/beauty/out/city-central-capital-r2/{site}/lights.json')['lights']
            template=next(t for t in self.pack['templates'] if t['authority']==f'selected-r{revision}' and t['size']==1)
            actual=[]
            for owner,(instance,source_instance) in enumerate(zip(template['instances'],augmentation['instances'])):
                s=surface['samples'][source_instance['sample_start']]
                origin=[s['column']+s['u'],-(s['row']+1-s['v']),s['height']/112*.648266978876]
                for light in instance['lights']:
                    actual.append({**light,'owner':owner,'position':[p+o for p,o in zip(light['position'],origin)]})
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):
                self.assertEqual(a['owner'],b['owner'])
                for key in ['range','intensity']:self.assertAlmostEqual(a[key],b[key],places=8)
                for key in ['position','direction','color_linear']:
                    for x,y in zip(a[key],b[key]):self.assertAlmostEqual(x,y,places=8)

    def test_selected_paving_uses_palace_hull_and_source_density(self):
        for revision,site in [(111,'inland'),(112,'holdout')]:
            t=next(t for t in self.pack['templates'] if t['authority']==f'selected-r{revision}' and t['size']==1)
            paving=t['paving'];e=read(LAB/f'audits/beauty/out/city-central-capital-r2/{site}/ground/settlement.json')
            self.assertEqual(paving['period'],e['tile_period']);self.assertEqual(paving['atlas_uv'],e['atlas_uv'])
            self.assertEqual(len(paving['coverage_polygons']),1);self.assertEqual(len(paving['coverage_boxes']),7)
            # Translation-independent polygon area compares the real selected
            # palace hull, not its enclosing rectangle or an inferred disk.
            def area(p):return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(p,p[1:]+p[:1])))*.5
            self.assertAlmostEqual(area(paving['coverage_polygons'][0]),area(e['coverage_polygons'][0]),places=9)
            self.assertEqual(self.pack['materials'][paving['material']]['textures'][0],e['atlas']['texture'])

    def test_no_light_budget_truncation_or_path_leakage(self):
        for t in self.pack['templates']:
            self.assertLessEqual(sum(len(i['lights']) for i in t['instances']),128)
            self.assertLessEqual(len(t['instances']),32)
        for revision in [60,64]:
            t=next(t for t in self.pack['templates'] if t['authority']==f'selected-r{revision}' and t['size']==1)
            self.assertEqual(sum(len(i['lights']) for i in t['instances']),53)
        for m in self.pack['materials']:
            for path in m['textures']:
                if path:self.assertTrue(path.startswith('Renderer/') and '..' not in path and ':' not in path)

if __name__=='__main__':unittest.main()
