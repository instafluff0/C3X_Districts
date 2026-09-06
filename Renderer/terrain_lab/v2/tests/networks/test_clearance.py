import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'systems/networks'))
from network import *
from clearance import publish,footprint_intersects,point_segment
from source_routes import build_source_routes
class Clearance(unittest.TestCase):
    def setUp(self):
        self.g=Graph([Node(0,(0,0),(0,0)),Node(1,(2,0),(128,0)),Node(2,(2,2),(128,128))],[Edge(0,0,1,True),Edge(1,1,2)])
        self.env=publish(self.g,lambda x,y:0)
    def test_actual_source_ribbon_vertices_are_enclosed(self):
        m=Mesh(lambda x,y:0);build_source_routes(self.g,m)
        for v in m.vertices:
            p=v[:2]
            self.assertTrue(any(point_segment(p,a,b)<=e['occupied_radius']+1e-7 for e in self.env['entries'] for a,b in zip(e['points'],e['points'][1:])))
    def test_overhang_intersects_even_when_center_is_outside(self):
        self.assertTrue(footprint_intersects(self.env,[(48,-36),(62,-36),(62,-18),(48,-18)]))
        self.assertFalse(footprint_intersects(self.env,[(48,-40),(62,-40),(62,-25),(48,-25)]))
    def test_corner_and_junction_not_cleared_only_at_tile_centers(self):
        self.assertTrue(footprint_intersects(self.env,[(100,-12),(117,-12),(117,4),(100,4)]))
    def test_periodic_aliases_agree(self):
        self.env['world_wrap']=(6400,0)
        p=[(48,-36),(62,-36),(62,-18),(48,-18)]
        self.assertEqual(footprint_intersects(self.env,p),footprint_intersects(self.env,[(x+6400,y) for x,y in p]))
    def test_city_footprint_enclosing_entire_route_rejected(self):
        self.assertTrue(footprint_intersects(self.env,[(-50,-50),(200,-50),(200,180),(-50,180)]))
    def test_margin_does_not_clear_whole_tile(self):
        self.assertFalse(footprint_intersects(self.env,[(30,-24),(40,-24),(40,-16),(30,-16)]))
    def test_source_bridge_width_and_height_included(self):
        c=Crossing(0,2,(64,0),(1,0),20,4)
        g=Graph([Node(0,(0,0),(0,0)),Node(1,(2,0),(128,0))],[Edge(0,0,1,True,crossing=c)])
        env=publish(g,lambda x,y:0,[{'crossing_id':0,'source_bounds':[[-1,-.7,-.1],[1,.7,1]],'uniform_scale':16,'asset_id':'test'}])
        self.assertTrue(footprint_intersects(env,[(60,12),(68,12),(68,15),(60,15)]))
        bridge=env['entries'][-1];self.assertEqual(bridge['height_range'],[2.4,20])
    def test_source_bridge_grade_never_tunnels(self):
        surface=lambda x,y:x*.12+y*.07
        anchor=Crossing(0,2,(64,0),(1,0),20,1)
        fitted=fit_crossing_grade(anchor,surface)
        self.assertEqual(fitted.xy,anchor.xy);self.assertEqual(fitted.hydrology_edge,anchor.hydrology_edge)
        nodes=[Node(0,(0,0),(0,0)),Node(1,(2,0),(128,0))]
        m=Mesh(surface);build_source_routes(Graph(nodes,[Edge(0,0,1,True,crossing=fitted)]),m)
        self.assertTrue(all(v[2]>=surface(v[0],v[1])+.34 for v in m.vertices))
        with self.assertRaises(ValueError):build_source_routes(Graph(nodes,[Edge(0,0,1,crossing=anchor)]),Mesh(surface))
    def test_two_crossings_keep_one_gameplay_edge(self):
        a=Crossing(1,11,(32,0),(1,0),14,4);b=Crossing(2,12,(96,0),(1,0),14,5)
        nodes=[Node(0,(0,0),(0,0)),Node(1,(2,0),(128,0))]
        e=Edge(0,0,1,True,crossing=a,additional_crossings=(b,));g=Graph(nodes,[e]);m=Mesh(lambda x,y:0)
        build_source_routes(g,m);self.assertEqual(len(g.edges),1);self.assertEqual([c.id for c in g.crossings(e)],[1,2])
        self.assertEqual(g.centerline(e)[0][0],(0,0));self.assertEqual(g.centerline(e)[-1][0],(128,0))
if __name__=='__main__':unittest.main()
