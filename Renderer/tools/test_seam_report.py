import unittest

import numpy as np

from Renderer.tools.seam_report import seams, BAND


def terrain(height=600, width=900, seed=3):
    # Smooth, varied texture with diagonal structure, like the isometric map.
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    image = np.zeros((height, width, 3), np.float32)
    for channel in range(3):
        for _ in range(6):
            fx, fy, phase = rng.uniform(0.01, 0.08), rng.uniform(0.01, 0.08), rng.uniform(0, 6.3)
            image[:, :, channel] += 20 * np.sin(fx * x + fy * y + phase)
        image[:, :, channel] += 120 + rng.normal(0, 4, (height, width))
    return np.clip(image, 0, 255).astype(np.uint8)


class SeamTests(unittest.TestCase):
    def test_clean_map_has_no_seam(self):
        self.assertEqual(seams(terrain()), [])

    def test_stale_trailing_strip_is_a_seam(self):
        # A trailing strip from another camera (performance review, section 24).
        image = terrain()
        image[:, :200] = terrain(seed=9)[:, :200]
        found = seams(image)
        self.assertIn(('x', 199), [(axis, position) for axis, position, _ in found])

    def test_repeated_slice_is_a_seam(self):
        image = terrain()
        image[:, 700:] = image[:, 650:850].copy()
        self.assertIn(('x', 699), [(axis, position) for axis, position, _ in seams(image)])

    def test_horizontal_strip_is_a_seam(self):
        image = terrain()
        top = int(BAND[0] * image.shape[0])
        image[:top + 100] = terrain(seed=11)[:top + 100]
        self.assertIn(('y', top + 99), [(axis, position) for axis, position, _ in seams(image)])

    def test_short_edges_and_interface_outside_the_band_are_not_seams(self):
        image = terrain()
        image[250:300, 400:520] = (140, 20, 20)  # a label box: short edges only
        bottom = int(BAND[1] * image.shape[0])
        image[bottom:, 300:] = 0  # the bottom panel and its straight edge
        self.assertEqual(seams(image), [])


if __name__ == '__main__':
    unittest.main()
