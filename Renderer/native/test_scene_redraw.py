"""Verify damaged HDR color/depth reconstruction preserves every MSAA sample."""
import unittest
from Renderer.native.test_composition_recording import run


class SceneRedrawTests(unittest.TestCase):
    def test_preserves_all_color_depth_samples(self):
        result = run('call BUILD.bat scene-redraw', timeout=180)
        self.assertEqual(result['status'], 'pass', result)
        self.assertIn('every_color_depth_sample_exact=1', result['output_tail'])
        self.assertIn('negative_control=1', result['output_tail'])


if __name__ == '__main__':
    unittest.main()
