"""Exercise constrained placements that a fixed-order central greedy layout strands."""
import unittest
from Renderer.lab.shared.cities import growth


def body(name,width,height):
    vertices=[{'position':[x,y,z]} for x in (-width/2,width/2)
              for y in (-height/2,height/2) for z in (0,.2)]
    return {'id':name,'lo':[-width/2,-height/2,0],'hi':[width/2,height/2,.2],
            'parts':[({'vertices':vertices},{})]}


def inside(box,region):
    return box[0]>=region[0] and box[1]>=region[1] and box[2]<=region[2] and box[3]<=region[3]


class GrowthPlacement(unittest.TestCase):
    def test_central_landmark_is_surrounded_and_keeps_growth_prefix(self):
        house=body('house',.14,.14);palace=[-.2,-.2,.2,.2]
        land=lambda b:inside(b,[-.65,-.65,.65,.65]) and not growth.overlaps(b,palace)
        result,stats=growth.solve([house],7,1,.65,land,neighbor_gap=.1,connected_prefixes=(4,7),
                                 fixed_neighbors=[palace],surround_center=[0,0])
        self.assertEqual(stats['status'],'fit')
        sides=[next(s for s in range(4) if growth.surround_sector(s,item['x'],item['y'],[0,0])) for item in result]
        self.assertEqual(set(sides[:4]),set(range(4)));self.assertEqual(len(set(sides[4:])),3)
        boxes=[[r['x']-.07,r['y']-.07,r['x']+.07,r['y']+.07] for r in result]
        for n in (4,7):self.assertTrue(growth.connected([palace]+boxes[:n],.1))

    def test_surrounded_capital_cannot_hide_a_one_sided_site(self):
        house=body('house',.1,.1);palace=[-.2,-.2,.2,.2]
        result,stats=growth.solve([house],4,1,.6,lambda b:b[0]>.25,
                                 neighbor_gap=.15,fixed_neighbors=[palace],surround_center=[0,0])
        self.assertIsNone(result)

    def test_staged_search_keeps_each_prefix_and_landmark_connected(self):
        house=body('house',.1,.1);palace=[-.45,-.12,-.24,.12]
        land=lambda b:inside(b,[-.5,-.3,.5,.3]) and not growth.overlaps(b,palace)
        result,stats=growth.solve([house],6,1,.6,land,neighbor_gap=.12,connected_prefixes=(3,6),
                                 fixed_neighbors=[palace],staged_connection=True)
        self.assertEqual(stats['status'],'fit')
        boxes=[[r['x']-.05,r['y']-.05,r['x']+.05,r['y']+.05] for r in result]
        for n in (3,6):
            self.assertTrue(growth.connected(boxes[:n],.12))
            self.assertTrue(any(growth.shares_frontage(palace,b,.12) for b in boxes[:n]))

    def test_landmark_connection_requires_frontage_not_a_diagonal_corner(self):
        palace=[-.3,-.3,.3,.3]
        self.assertFalse(growth.shares_frontage(palace,[.34,.34,.5,.5],.08))
        self.assertTrue(growth.shares_frontage(palace,[.1,.34,.25,.5],.08))

    def test_fixed_landmark_cannot_disguise_isolated_house_groups(self):
        house=body('house',.1,.1)
        land=lambda b:inside(b,[-.45,-.08,-.25,.08]) or inside(b,[.25,-.08,.45,.08])
        result,stats=growth.solve([house],2,1,.5,land,neighbor_gap=.12,
                                 fixed_neighbors=[[-.2,-.1,.2,.1]],connected_prefixes=(1,2))
        self.assertEqual(stats['status'],'no_grid_solution')
        self.assertIsNone(result)

    def test_disconnected_islands_are_not_accepted_as_one_neighborhood(self):
        house=body('house',.1,.1)
        land=lambda b:inside(b,[-.45,-.08,-.25,.08]) or inside(b,[.25,-.08,.45,.08])
        self.assertEqual(growth.solve([house],2,1,.5,land)[1]['status'],'fit')
        result,stats=growth.solve([house],2,1,.5,land,neighbor_gap=.08)
        self.assertIsNone(result)
        self.assertEqual(stats['status'],'no_grid_solution')

    def test_connectivity_allows_a_chain_not_only_direct_contact_to_center(self):
        boxes=[[0,0,.1,.1],[.15,0,.25,.1],[.3,0,.4,.1]]
        self.assertTrue(growth.connected(boxes,.06))
        self.assertFalse(growth.connected(boxes,.04))

    def test_later_bridge_cannot_hide_disconnected_earlier_growth(self):
        house=body('house',.1,.1)
        preserved=[{'slot':i,'asset':'house','scale':1,'rotation':0,'offset':[x,0],
                    'local_bounds':[-.05,-.05,.05,.05]} for i,x in enumerate((-.21,.21))]
        land=lambda b:inside(b,[-.28,-.08,.28,.08])
        result,stats=growth.solve([house],3,1,.3,land,preserved,neighbor_gap=.12)
        self.assertEqual(stats['status'],'fit')
        result,stats=growth.solve([house],3,1,.3,land,preserved,neighbor_gap=.12,connected_prefixes=(2,3))
        self.assertIsNone(result)

    def setUp(self):
        self.order=[body('small',.12,.12),body('long',.7,.2)]
        self.land=lambda b:inside(b,[-.5,-.15,.5,.15]) or inside(b,[-.1,.21,.1,.45])

    def test_constrained_body_keeps_room_for_small_body(self):
        result,stats=growth.solve(self.order,2,1,.6,self.land)
        self.assertEqual(stats['status'],'fit')
        self.assertEqual([r['asset']['id'] for r in result],['small','long'])
        self.assertGreater(result[0]['y'],.2)
        self.assertAlmostEqual(result[1]['y'],0)
        self.assertEqual(result,growth.solve(self.order,2,1,.6,self.land)[0])

    def test_preserved_central_body_reports_impossible_grid_without_moving_it(self):
        old={'slot':0,'asset':'small','scale':1,'rotation':0,'offset':[0,0],
             'local_bounds':[-.06,-.06,.06,.06]}
        result,stats=growth.solve(self.order,2,1,.6,self.land,[old])
        self.assertIsNone(result)
        self.assertEqual(stats['candidate_counts']['1'],0)
        self.assertEqual(old['offset'],[0,0])
        self.assertEqual(stats['status'],'no_grid_solution')

    def test_quarter_turn_fits_without_rescaling(self):
        result,stats=growth.solve([body('wide',.5,.1)],1,1,.6,
                                 lambda b:inside(b,[-.09,-.35,.09,.35]))
        self.assertEqual(stats['status'],'fit')
        self.assertAlmostEqual(result[0]['rotation'],growth.math.pi/2)
        self.assertEqual(result[0]['scale'],1)

    def test_budget_is_distinct_from_no_solution(self):
        result,stats=growth.solve(self.order,2,1,.6,self.land,node_limit=1)
        self.assertIsNone(result)
        self.assertEqual(stats['status'],'budget_exhausted')

    def test_changed_preserved_geometry_is_rejected(self):
        old={'slot':0,'asset':'small','scale':1,'rotation':0,'offset':[0,0],
             'local_bounds':[-.03,-.06,.03,.06]}
        with self.assertRaisesRegex(ValueError,'geometry changed'):
            growth.solve(self.order,2,1,.6,self.land,[old])

    def test_early_stage_cannot_use_land_only_available_in_later_envelope(self):
        land=lambda b:inside(b,[.3,-.2,.6,.2])
        self.assertEqual(growth.solve([body('house',.12,.12)],1,1,.7,land)[1]['status'],'fit')
        result,stats=growth.solve([body('house',.12,.12)],1,1,.7,land,slot_extents=[.25])
        self.assertIsNone(result)
        self.assertEqual(stats['status'],'no_grid_solution')

    def test_preserved_prefix_must_respect_its_own_stage_envelope(self):
        old={'slot':0,'asset':'small','scale':1,'rotation':0,'offset':[.42,0],
             'local_bounds':[-.06,-.06,.06,.06]}
        with self.assertRaisesRegex(ValueError,'terrain clearance'):
            growth.solve(self.order,2,1,.7,lambda b:True,[old],slot_extents=[.3,.7])

    def test_individual_house_palette_can_exceed_eleven_bodies(self):
        # Twelve isolated sites require twelve separate houses, with no scale
        # reduction or overlap hidden by the larger stage budget.
        sites=[(x*.14,y*.14) for x in (-2,-1,0,1) for y in (-1,0,1)]
        land=lambda b:any(inside(b,[x-.04,y-.04,x+.04,y+.04]) for x,y in sites)
        result,stats=growth.solve([body('house',.04,.04)],12,1,.4,land)
        self.assertEqual(stats['status'],'fit')
        self.assertEqual(len({(i['x'],i['y']) for i in result}),12)
        self.assertTrue(all(i['scale']==1 for i in result))
        with self.assertRaisesRegex(ValueError,'bounded city growth'):
            growth.solve([body('house',.04,.04)],33,1,.4,land)


if __name__=='__main__':unittest.main()
