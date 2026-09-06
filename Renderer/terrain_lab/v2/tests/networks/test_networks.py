"""Q5 behavioral tests: exhaustive reachable masks and adversarial geometry."""
import math, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'systems/networks'))
from network import *

class Networks(unittest.TestCase):
    def star(self,mask):
        nodes=[Node(0,(0,0),(0,0))];edges=[]
        for i,d in enumerate(DIRS):
            if mask>>i&1:
                nodes.append(Node(i+1,d,mul(d,64)));edges.append(Edge(i,0,i+1,rail=True))
        return Graph(nodes,edges)
    def test_all_256_masks_exact_nodes_and_finite_mesh(self):
        for mask in range(256):
            g=self.star(mask);m=Mesh(lambda x,y:5*math.sin(x*.007)*math.cos(y*.009))
            build_routes(g,m)
            self.assertEqual(len(g.edges),bin(mask).count('1'))
            for e in g.edges:
                line=g.centerline(e)
                self.assertEqual(line[0][0],g.nodes[e.a].xy);self.assertEqual(line[-1][0],g.nodes[e.b].xy)
            self.assertTrue(all(math.isfinite(x) for v in m.vertices for x in v))
    def test_degree_two_tangent_coincidence_all_pairs(self):
        for a in range(8):
            for b in range(a+1,8):
                g=self.star((1<<a)|(1<<b));ta=g.tangent(g.edges[0],0);tb=g.tangent(g.edges[1],0)
                self.assertAlmostEqual(dot(ta,tb),-1,places=10)
    def test_reversed_input_keeps_geometry(self):
        g=self.star(0xff);m=Mesh(lambda x,y:0);build_routes(g,m)
        other=Graph(list(reversed(list(g.nodes.values()))),list(reversed(g.edges)))
        n=Mesh(lambda x,y:0);build_routes(other,n)
        self.assertEqual(sorted(m.vertices),sorted(n.vertices))
    def test_non_neighbor_and_reciprocal_rejected(self):
        with self.assertRaises(ValueError):Graph([Node(0,(0,0),(0,0)),Node(1,(4,0),(256,0))],[Edge(0,0,1)])
        g=self.star(1)
        with self.assertRaises(ValueError):Graph(list(g.nodes.values()),[g.edges[0],Edge(9,1,0)])
    def test_wrapping_uses_near_translated_image(self):
        g=Graph([Node(0,(98,50),(6272,3200)),Node(1,(0,50),(0,3200))],[Edge(0,0,1,wrap=(6400,0))],(100,0))
        line=g.centerline(g.edges[0]);self.assertEqual(line[-1][0],(6400,3200))
        self.assertLess(max(length(sub(b[0],a[0])) for a,b in zip(line,line[1:])),3)
        self.assertEqual(length(sub(line[-1][0],line[0][0])),128)
    def test_supplied_crossing_pins_straight_deck(self):
        c=Crossing(7,99,(64,0),(1,0),20,4)
        g=Graph([Node(0,(0,0),(0,0)),Node(1,(2,0),(128,0))],[Edge(0,0,1,crossing=c)])
        line=g.centerline(g.edges[0]);bridge=[p for p,b in line if b]
        self.assertTrue(bridge);self.assertTrue(all(abs(p[1])<1e-9 for p in bridge))
        self.assertEqual(bridge[-1],(80,0))
        m=Mesh(lambda x,y:0);build_routes(g,m)
        samples=[s for s in m.route_samples if 48<=s[1]<=80]
        self.assertTrue(all(abs(s[3]-4.24)<1e-8 for s in samples))
    def test_bad_crossing_does_not_move_hydrology(self):
        nodes=[Node(0,(0,0),(0,0)),Node(1,(2,0),(128,0))]
        for c in (Crossing(1,2,(64,9),(1,0),20,4),Crossing(1,2,(64,0),(0,1),20,4),Crossing(1,2,(64,0),(1,0),120,4)):
            with self.assertRaises(ValueError):Graph(nodes,[Edge(0,0,1,crossing=c)])
    def test_ground_contact_uses_lateral_surface_samples(self):
        g=self.star(0x55);surface=lambda x,y:17*math.sin(x*.008)+12*math.cos(y*.01)
        m=Mesh(surface);build_routes(g,m)
        road=[v for v in m.vertices if v[8]>.13] # Check every mesh vertex instead below.
        for v in m.vertices:self.assertGreaterEqual(v[2]-surface(v[0],v[1]),.1-1e-6)
        for s in m.route_samples:self.assertAlmostEqual(s[3]-surface(s[1],s[2]),.24,places=9)
    def test_empty_graph_is_exact_no_op(self):
        m=Mesh(lambda x,y:0);m.vertices=[(1,2,3)]
        build_routes(Graph([],[]),m);self.assertEqual(m.vertices,[(1,2,3)])
    def test_stages_pillage_and_coexistence(self):
        nodes=[Node(0,(0,0),(0,0)),Node(1,(2,0),(128,0))]
        counts=[]
        for rail in (False,True):
            for pillaged in (False,True):
                for stage in range(4):
                    m=Mesh(lambda x,y:0);build_routes(Graph(nodes,[Edge(0,0,1,rail,stage,pillaged)]),m)
                    self.assertGreater(len(m.vertices),0);counts.append(len(m.vertices))
        self.assertGreater(len(set(counts)),2)

if __name__=='__main__':unittest.main()
