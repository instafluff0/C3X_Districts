import json,math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'systems/objects'))
import presentation as p

def footprint(inst):
    a=inst['asset'];points=[]
    for mesh,_ in a['parts']:
        for v in mesh['vertices']:
            v=[v['position'][0]-(a['lo'][0]+a['hi'][0])/2,v['position'][1]-(a['lo'][1]+a['hi'][1])/2,0]
            v=p.rotate(v,inst['rotation']);points.append([v[0]*inst['scale']+inst['x'],v[1]*inst['scale']+inst['y']])
    return [min(v[0] for v in points),min(v[1] for v in points),max(v[0] for v in points),max(v[1] for v in points)]

class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pools=p.read(p.PACK/'city_catalog.json')['pools']
        cls.assets={key:[p.component(a) for a in row['components']] for key,row in cls.pools.items()}
        cls.layouts={key:[p.layout(assets,size) for size in range(3)] for key,assets in cls.assets.items()}
    def test_all_twenty_pools_keep_full_worked_bindings(self):
        self.assertEqual(len(self.pools),20)
        manifest=p.read(p.PACK/'manifest.json')
        for aid in {a for pool in self.pools.values() for a in pool['components']}:
            source=p.read(p.PACK/manifest['assets'][aid]['landmark'])
            self.assertEqual(len(p.component(aid)['parts']),sum('worked' in b['states'] for b in source['draw_bindings']))
    def test_all_growth_prefixes_preserve_identity_position_scale_facing(self):
        for key,stages in self.layouts.items():
            for small,large in zip(stages,stages[1:]):
                self.assertEqual(small,large[:len(small)],key)
    def test_all_city_footprints_clear_and_source_front_quadrants(self):
        for key,stages in self.layouts.items():
            boxes=[footprint(i) for i in stages[-1]]
            for i,a in enumerate(boxes):
                self.assertIn(stages[-1][i]['rotation'],[0,math.pi/2])
                for b in boxes[i+1:]:
                    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                    self.assertLess(area,1e-10,key)
    def test_route_reservation_tests_full_extent_and_growth(self):
        envelope=p.local_witness()
        self.assertTrue(p.intersects([-.10,-.1,.10,.1],envelope))
        self.assertFalse(p.intersects([.2,-.1,.3,.1],envelope))
        for key,assets in self.assets.items():
            large=p.layout(assets,2,exclusions=envelope)
            for i in large:self.assertFalse(p.intersects(footprint(i),envelope),key)
            self.assertEqual(p.layout(assets,0,exclusions=envelope),large[:4])
    def test_uniform_transform_preserves_pair_distance_and_uv(self):
        for assets in self.assets.values():
            inst=p.layout(assets,0)[0];mesh=inst['asset']['parts'][0][0]
            a,b=mesh['vertices'][0],mesh['vertices'][-1]
            distance=lambda a,b:math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
            ta=[v*inst['scale'] for v in p.rotate(a['position'],inst['rotation'])]
            tb=[v*inst['scale'] for v in p.rotate(b['position'],inst['rotation'])]
            self.assertAlmostEqual(distance(ta,tb),distance(a['position'],b['position'])*inst['scale'],places=10)
    def test_mounted_source_compound_and_aquatic_bounds(self):
        horse=p.bundle(Path('Renderer/packs/CompoundUnitLab/unit_horseman_runtime.bin'),'idle_0')
        self.assertGreater(len(horse['parts']),1)
        records=[];draws=p.defaultdict(list)
        fish=p.bundle(Path('Renderer/packs/ResourceNormalized/resource_runtime.bin'),'fish')
        p.emit_object(draws,records,fish,'fish',[100,100],[256,256],10)
        b=records[0]['bounds'];self.assertLessEqual(b[2]-b[0],32.00001)
    def test_compact_real_town_cohesion_preserves_corridor_clearance(self):
        from clearance_adapter import projected_plane
        corridor=dict(schema='c3x.q5.route_clearance.v1',world_wrap=[0,0],halo_complete=True,entries=[dict(id='entry',kind='road',shape='capsule_chain',points=[projected_plane([-.03,0]),projected_plane([.8,0])],occupied_radius=14,clearance_radius=18)])
        inst=p.layout(self.assets['city/pool/american/ancient'],0,'compact',exclusions=corridor)
        boxes=[footprint(x) for x in inst]
        self.assertEqual(len(inst),4)
        for i,b in enumerate(boxes):
            self.assertFalse(p.intersects(b,corridor))
            if i:
                gap=min(math.hypot(max(0,b[0]-a[2],a[0]-b[2]),max(0,b[1]-a[3],a[1]-b[3])) for a in boxes[:i])
                self.assertLess(gap,.08)

if __name__=='__main__':unittest.main()
