"""Analytic geometry checks for shoreline clipping, independent of GPU output."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'systems/objects'))
from city_ground_geometry import clip_ground_triangle


class GroundBoundaryTests(unittest.TestCase):
    def test_uniform_land_or_water(self):
        vertices = [[0, 0, 0, 0], [2, 0, 1, 0], [0, 2, 0, 1]]
        self.assertEqual(clip_ground_triangle(vertices, [-1, -1, -1]), vertices)
        self.assertEqual(clip_ground_triangle(vertices, [1, 1, 1]), [])

    def test_oblique_atlas_triangle_keeps_area_and_uv_parameterization(self):
        vertices = [[0, 0, 0, 0], [2, 0, 1, 0], [0, 2, 0, 1]]
        # Clip x <= 1 from a right triangle of area 2: remove area 1/2.
        clipped = clip_ground_triangle(vertices, [-1, 1, -1], boundary=0)
        area = sum(abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
                   for a, b, c in zip(clipped[::3], clipped[1::3], clipped[2::3]))
        self.assertEqual(area, 1.5)
        for x, y, u, v in clipped:
            self.assertLessEqual(x, 1)
            self.assertEqual(u, x/2)
            self.assertEqual(v, y/2)


if __name__ == '__main__':
    unittest.main()
