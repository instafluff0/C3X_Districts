import json
from pathlib import Path
import re
import unittest

from PIL import Image

from Renderer.lab.studies.settler_desirability.preview import (
    CUSTOM_TILE_WIDTHS, WHITE, GREEN, color, compose, diamond, grade, invented_evaluation,
)

ROOT = Path(__file__).resolve().parents[4]


class SettlerDesirabilityStudyTests(unittest.TestCase):
    def test_scales_follow_c3x_source_and_sprite_sheet(self):
        source = (ROOT / "injected_code.c").read_text()
        match = re.search(r"int levels\[3\]\s*=\s*\{([^}]+)\}", source)
        self.assertIsNotNone(match)
        self.assertEqual(CUSTOM_TILE_WIDTHS, tuple(int(value.strip()) for value in match.group(1).split(",")))
        self.assertRegex(source, r"Sprite_slice_pcx\s*\(&is->tile_highlights\[n\].*128\*n,\s*0,\s*128,\s*64")
        with Image.open(ROOT / "Art/TileHighlights.pcx") as sprites:
            self.assertEqual((11 * 128, 64), sprites.size)
        category = json.loads((ROOT / "Renderer/lab/categories/interface/settler-desirability/standard.json").read_text())
        self.assertEqual(list(CUSTOM_TILE_WIDTHS), category["recipe"]["zooms"])
        self.assertEqual(((320, 208), (384, 240), (320, 272), (256, 240)),
                         diamond(16, 16, 128, (640, 480)))
        with self.assertRaises(ValueError):
            compose(Image.new("RGB", (640, 480)), 64, "outlined")

    def test_c3x_grade_and_eligibility(self):
        self.assertIsNone(grade(0))
        self.assertIsNone(grade(-10))
        self.assertEqual([0, 4, 5, 5, 6, 10],
                         [grade(v) for v in (999_940, 999_994, 999_995,
                                             1_000_000, 1_000_005, 1_000_100)])

    def test_palette_monotonically_moves_white_to_green_without_red(self):
        self.assertEqual(WHITE, color(0))
        self.assertEqual(GREEN, color(10))
        colors = [color(i) for i in range(11)]
        self.assertTrue(all(a[0] >= b[0] and a[1] >= b[1] and a[2] >= b[2]
                            for a, b in zip(colors, colors[1:])))
        self.assertTrue(all(r <= g + 3 for r, g, _ in colors[1:]))

    def test_fixture_has_ineligible_gaps_and_coastal_omission(self):
        for case in ("wash", "outlined", "coastal"):
            self.assertEqual(0, invented_evaluation(12, 16, case))
            self.assertGreater(invented_evaluation(16, 16, case), 0)
        self.assertGreater(invented_evaluation(20, 16, "wash"), 0)
        self.assertEqual(0, invented_evaluation(20, 16, "coastal"))
        self.assertEqual(10, grade(invented_evaluation(14, 16, "wash")))
        levels = {grade(invented_evaluation(x, y, "wash"))
                  for y in range(9, 24) for x in range(8 + y % 2, 24, 2)}
        self.assertEqual(set(range(11)), levels - {None})

    def test_projection_and_omitted_tile_survive_each_zoom(self):
        for zoom in CUSTOM_TILE_WIDTHS:
            background = Image.new("RGB", (640, 480), (100, 80, 60))
            result = compose(background, zoom, "wash")
            center = diamond(16, 16, zoom, background.size)
            inside = (round(center[0][0]), round((center[0][1] + center[2][1]) / 2))
            self.assertNotEqual(background.getpixel(inside), result.getpixel(inside))
            tint = color(grade(invented_evaluation(16, 16, "wash")))
            for original, painted, target in zip(background.getpixel(inside), result.getpixel(inside), tint):
                self.assertGreater(painted, min(original, target))
                self.assertLess(painted, max(original, target))
            gap = diamond(16, 14, zoom, background.size)
            omitted = (round(gap[0][0]), round((gap[0][1] + gap[2][1]) / 2))
            self.assertEqual(background.getpixel(omitted), result.getpixel(omitted))
            self.assertEqual(background.size, result.size)
            self.assertEqual(result.tobytes(), compose(background, zoom, "wash").tobytes())

    def test_coastal_ineligible_tile_does_not_receive_overlay(self):
        background = Image.new("RGB", (640, 480), (100, 80, 60))
        tile = diamond(20, 16, 128, background.size)
        center = (round(tile[0][0]), round((tile[0][1] + tile[2][1]) / 2))
        self.assertEqual(background.getpixel(center), compose(background, 128, "coastal").getpixel(center))

    def test_outlined_treatment_is_distinct_and_coastal_uses_it(self):
        background = Image.new("RGB", (640, 480), (100, 80, 60))
        self.assertNotEqual(compose(background, 128, "wash").tobytes(),
                            compose(background, 128, "outlined").tobytes())
        self.assertEqual(compose(background, 128, "coastal").getpixel((320, 240)),
                         compose(background, 128, "outlined").getpixel((320, 240)))


if __name__ == "__main__":
    unittest.main()
