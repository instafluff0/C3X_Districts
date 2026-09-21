"""Verify static backup storage against the original per-sample D3D path."""
import unittest
from Renderer.native.test_composition_recording import run


class LinearBackupTests(unittest.TestCase):
    def test_preserves_all_color_depth_samples(self):
        result = run('call BUILD.bat linear-backup', timeout=180)
        self.assertEqual(result['status'], 'pass', result)
        self.assertIn('every_color_depth_sample_exact=1', result['output_tail'])
        self.assertIn('negative_control=1', result['output_tail'])


if __name__ == '__main__':
    unittest.main()
