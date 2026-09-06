from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import write_png
from Renderer.tools.visual_qa import animation_metrics, compare_day_night, compare_zoom, image_metrics, read_image


class VisualQaTests(unittest.TestCase):
    def _frames(self, root: Path) -> tuple[Path, Path, Path]:
        paths = []
        for index, color in enumerate(((100, 80, 60), (220, 180, 80), (100, 80, 60))):
            canvas = Canvas(32, 24)
            canvas.fill_polygon([(8 + index, 18), (16 + index, 6), (24 + index, 18)], color)
            path = root / f"frame_{index}.png"
            write_png(canvas, path)
            paths.append(path)
        return tuple(paths)

    def test_png_bmp_metrics_cover_silhouette_ground_spill_and_civ_color(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canvas = Canvas(32, 24)
            canvas.fill_polygon([(8, 18), (16, 6), (24, 18)], (200, 20, 20))
            png, bmp = root / "image.png", root / "image.bmp"
            write_png(canvas, png)
            canvas.write_bmp(bmp)
            for image in (read_image(png), read_image(bmp)):
                metrics = image_metrics(*image, (37, 43, 46), allowed_bounds=[9, 0, 23, 24], ground_y=19, civ_colors=[(200, 20, 20)])
                self.assertGreater(metrics["subject_pixels"], 50)
                self.assertGreater(metrics["neighbor_spill_pixels"], 0)
                self.assertGreater(metrics["civ_color_fraction_basis_points"], 9000)
                self.assertEqual(0, metrics["grounding_gap_px"])

    def test_day_night_zoom_and_animation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._frames(Path(directory))
            frames = [read_image(path) for path in paths]
            day_night = compare_day_night(frames[0], frames[1])
            self.assertGreater(day_night["emissive_pixels"], 0)
            motion = animation_metrics(frames, (37, 43, 46))
            self.assertTrue(motion["animated"])
            normal = image_metrics(*frames[0], (37, 43, 46))
            self.assertTrue(compare_zoom(normal, normal)["consistent"])


if __name__ == "__main__":
    unittest.main()
