from __future__ import annotations

import unittest

from Renderer.tools.asset_compiler.tile_fit_calibrator import (
    FACINGS,
    calibrate_positions,
)


class TileFitCalibratorTests(unittest.TestCase):
    def test_eight_facings_two_zooms_ground_and_fit(self) -> None:
        positions = [
            [-0.8, -0.25, 0.2], [0.8, -0.25, 0.2], [0.8, 0.25, 0.2], [-0.8, 0.25, 0.2],
            [0.0, 0.0, 1.6],
        ]
        result = calibrate_positions(positions, "fixture/tower")
        self.assertEqual(16, len(result["cells"]))
        self.assertEqual(list(FACINGS), result["facings"])
        self.assertEqual(-0.2, result["grounding"]["translation_tile"][2])
        self.assertTrue(all(cell["fits_limits"] for cell in result["cells"]))
        by_facing = {}
        for cell in result["cells"]:
            by_facing.setdefault(cell["facing"], {})[cell["zoom"]] = cell["uniform_scale"]
        self.assertTrue(all(value["normal"] == value["reduced"] for value in by_facing.values()))

    def test_calibration_is_deterministic(self) -> None:
        points = [[-0.2, -0.4, -0.1], [0.3, 0.5, 0.9]]
        self.assertEqual(
            calibrate_positions(points, "fixture/a")["calibration_hash"],
            calibrate_positions(points, "fixture/a")["calibration_hash"],
        )


if __name__ == "__main__":
    unittest.main()
