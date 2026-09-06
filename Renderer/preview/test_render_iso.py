#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_iso


def sample_pack() -> dict:
    return {
        "projection": {
            "tile_width_px": 128,
            "tile_height_px": 64,
            "height_scale_px": 54,
        },
        "terrains": {
            "grassland": {"preview_color": [83, 143, 79]},
            "plains": {"preview_color": [178, 157, 88]},
            "desert": {"preview_color": [202, 176, 103]},
            "tundra": {"preview_color": [132, 151, 139]},
        },
        "relief": {
            "mountains": {
                "variants": [
                    {"preview_height": 0.8, "preview_color": [110, 112, 104]},
                    {"preview_height": 0.9, "preview_color": [118, 118, 108]},
                ]
            }
        },
    }


class PreviewRenderTests(unittest.TestCase):
    def test_render_is_nonblank(self) -> None:
        canvas = render_iso.render(sample_pack(), 640, 480, 8, 123)
        self.assertGreater(canvas.non_background_pixels(), 1000)

    def test_output_is_deterministic(self) -> None:
        a = render_iso.render(sample_pack(), 640, 480, 8, 123)
        b = render_iso.render(sample_pack(), 640, 480, 8, 123)
        self.assertEqual(a.pixels, b.pixels)

    def test_grassland_only_pack_without_mountains_renders(self) -> None:
        pack = sample_pack()
        pack["terrains"] = {"grassland": {"preview_color": [80, 140, 80]}}
        pack["relief"]["mountains"]["variants"] = []
        canvas = render_iso.render(pack, 640, 480, 8, 123, "grassland")
        self.assertGreater(canvas.non_background_pixels(), 1000)

    def test_writes_bmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview.bmp"
            pack_path = Path(tmp) / "manifest.json"
            pack_path.write_text(json.dumps(sample_pack()), encoding="utf-8")
            canvas = render_iso.render(sample_pack(), 320, 240, 4, 123)
            canvas.write_bmp(path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 54)


if __name__ == "__main__":
    unittest.main()
