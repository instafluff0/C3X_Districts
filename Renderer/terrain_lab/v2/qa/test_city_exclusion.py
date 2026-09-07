import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'systems/objects'))
from city_exclusion import Exclusion


class ExclusionTests(unittest.TestCase):
    def test_thin_corridor_crosses_box_without_touching_sample_points(self):
        e=Exclusion([[[-1,.22],[1,.22],[1,.24],[-1,.24]]])
        self.assertTrue(e.blocks([-.5,-.5,.5,.5]))
        self.assertFalse(e.blocks([-.5,-.5,.5,.2]))

    def test_triangle_bbox_is_not_the_exclusion(self):
        e=Exclusion([[[0,0],[1,0],[0,1]]])
        self.assertFalse(e.blocks([.7,.7,.9,.9]))
        self.assertTrue(e.blocks([.1,.1,.2,.2]))
        self.assertFalse(e.blocks([-1,0,0,1]))

    def test_reverse_winding_and_degenerate_polygons(self):
        e=Exclusion([[[0,1],[1,0],[0,0]],[[0,0],[0,0],[0,0]]])
        self.assertTrue(e.blocks([.1,.1,.2,.2]))
        self.assertEqual(len(e.polygons),1)

    def test_invalid_polygon(self):
        for polygon in ([[0,0],[1,0],[float('nan'),1]],[[0,0],[1,0],[.3,.3],[1,1],[0,1]]):
            with self.assertRaises(ValueError):Exclusion([polygon])


if __name__=='__main__':unittest.main()
