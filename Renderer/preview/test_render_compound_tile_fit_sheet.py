from __future__ import annotations

import unittest
from pathlib import Path

from Renderer.preview.render_compound_tile_fit_sheet import render_sheet


PACK = Path(__file__).resolve().parents[1] / "packs/FutureGateCandidates/manifest.json"


class CompoundTileFitSheetTests(unittest.TestCase):
    @unittest.skipUnless(PACK.is_file(), "local future-gate candidate pack unavailable")
    def test_radar_candidate_renders_all_facing_zoom_light_cells_deterministically(self) -> None:
        first, report = render_sheet(PACK, "candidate/infrastructure/radar_observatory_body")
        repeated, repeated_report = render_sheet(PACK, "candidate/infrastructure/radar_observatory_body")
        self.assertEqual(32, len(report["cells"]))
        self.assertEqual(first.pixels, repeated.pixels)
        self.assertEqual(report["calibration_hash"], repeated_report["calibration_hash"])
        self.assertGreater(first.non_background_pixels(), 5000)


if __name__ == "__main__":
    unittest.main()
