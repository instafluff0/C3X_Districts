"""The shared-lighting diagnostic must exercise more than isolated noon terrain."""
from pathlib import Path
import tempfile
import unittest
from Renderer import renderer


class ShadowFixture(unittest.TestCase):
    def test_focused_shadows_exercise_both_dynamic_paths(self):
        names = {case[0] for case in renderer.integration_replay_cases("shadows")}
        self.assertTrue({"terrain-edit", "resource-playback", "unit-actions-day", "unit-actions-night"} <= names)

    def test_relief_foliage_and_flat_receivers_at_all_four_phases(self):
        value = renderer.standard("shadows")
        self.assertEqual(set(value["recipe"]["hours"]), {0, 6, 12, 18})
        self.assertEqual(set(value["recipe"]["zooms"]), {64, 128})
        self.assertIn("units", renderer.asset_jobs_for(["shadows"]))
        self.assertIn("resource-animation", renderer.asset_jobs_for(["shadows"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.csv"
            for case in value["recipe"]["cases"]:
                renderer.scene("shadows", case, path)
                rows = [list(map(int, line.split(","))) for line in path.read_text().splitlines()[1:]]
                tiles = {(row[0], row[1]): row[3] for row in rows}
                self.assertEqual(tiles[14, 14], 6)
                self.assertEqual(tiles[18, 14], 7)
                self.assertEqual(tiles[16, 18], 2)
                self.assertEqual(tiles[20, 16], 2)
