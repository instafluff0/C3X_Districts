"""Analytic settlement coverage and categorical vegetation clipping checks."""
import unittest
from Renderer.lab.shared.cities.ground import coverage,rectangle_distance,convex_hull,polygon_distance,footprint_alignment
from Renderer.lab.shared.cities.clipping import clip_ground_to_land_cells


class SettlementGroundTests(unittest.TestCase):
    def test_baked_thirty_degree_rotation_is_removed_without_deforming_mesh(self):
        import math
        angle=-math.pi/6;c=math.cos(angle);s=math.sin(angle)
        rectangle=[(x*c-y*s,x*s+y*c) for x in (-2,2) for y in (-1,1)]
        self.assertAlmostEqual(footprint_alignment(rectangle),-angle)
        self.assertEqual(footprint_alignment([(-2,-1),(2,-1),(2,1),(-2,1)]),0)

    def test_rotated_foundation_does_not_pave_bounding_box_corners(self):
        hull=convex_hull([(0,1),(1,0),(0,-1),(-1,0),(0,0),(0,1)])
        self.assertEqual(len(hull),4)
        self.assertEqual(coverage(.8,.8,[],.1,.025,[hull]),0)
        self.assertEqual(coverage(.8,.8,[(-1,-1,1,1)],.1,.025),1)
        self.assertEqual(coverage(.45,.45,[],.1,.025,[hull]),1)
        self.assertAlmostEqual(polygon_distance(0,0,hull),-2**-.5)
        self.assertAlmostEqual(polygon_distance(1.1,0,hull),.1)

    def test_mesh_footprint_preserves_rectangle_union_and_feather(self):
        hull=convex_hull([(0,0),(1,0),(1,1),(0,1)])
        for x,y in [(.5,.5),(1.06,1.08),(1.09,1.09),(1.08,.4),(-.2,.5)]:
            self.assertAlmostEqual(coverage(x,y,[(0,0,1,1)],.1,.025),coverage(x,y,[],.1,.025,[hull]))

    def test_overlap_is_a_union_and_small_gap_connects(self):
        boxes=[(0,0,1,1),(.9,0,2,1)]
        self.assertEqual(coverage(.95,.5,boxes,.1,.025),1)
        self.assertEqual(coverage(1.05,.5,[(0,0,1,1),(1.1,0,2,1)],.1,.025),1)
        self.assertEqual(coverage(1.2,.5,[(0,0,1,1),(1.4,0,2,1)],.1,.025),0)

    def test_rounded_corner_does_not_fill_the_diagonal_square(self):
        self.assertAlmostEqual(rectangle_distance(1.06,1.08,(0,0,1,1)),.1)
        self.assertEqual(coverage(1.09,1.09,[(0,0,1,1)],.1,.025),0)
        self.assertGreater(coverage(1.06,1,[(0,0,1,1)],.1,.025),0)

    def test_dry_forest_cell_is_excluded_without_inventing_shore_distance(self):
        vertices=[[.5,.25,0,0],[1.5,.25,1,0],[.5,.75,0,1]]
        clipped=clip_ground_to_land_cells(vertices,[-1,-1,-1],{(0,0):2,(1,0):2},
                                         world_axes=(0,1),excluded_cells={(1,0)})
        area=sum(abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))*.5
                 for a,b,c in zip(clipped[::3],clipped[1::3],clipped[2::3]))
        self.assertAlmostEqual(area,.1875)
        for x,y,u,v in clipped:
            self.assertLessEqual(x,1);self.assertAlmostEqual(u,x-.5);self.assertAlmostEqual(v,(y-.25)*2)


if __name__=='__main__':unittest.main()
