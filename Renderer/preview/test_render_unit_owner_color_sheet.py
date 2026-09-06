from __future__ import annotations

import unittest

from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_unit_owner_color_sheet import changed_pixel_count, manifest_unit_ids


class UnitOwnerColorSheetTests(unittest.TestCase):
    def test_changed_pixel_count_applies_a_screen_space_rgb_threshold(self) -> None:
        neutral = Canvas(3, 1, (10, 10, 10))
        tinted = Canvas(3, 1, (10, 10, 10))
        tinted.pixels[0] = (15, 10, 10)
        tinted.pixels[1] = (10, 16, 10)
        tinted.pixels[2] = (10, 10, 30)
        self.assertEqual(2, changed_pixel_count(neutral, tinted, 6))

    def test_changed_pixel_count_requires_matching_canvases(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            changed_pixel_count(Canvas(1, 1), Canvas(2, 1), 6)

    def test_manifest_unit_discovery_accepts_arbitrary_logical_units(self) -> None:
        self.assertEqual(
            ["unit/modded_scout", "unit/transport_alpha"],
            manifest_unit_ids(
                {"units": {"unit/transport_alpha": {}, "unit/modded_scout": {}}}
            ),
        )


if __name__ == "__main__":
    unittest.main()
