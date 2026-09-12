"""Ordinary volcano fixtures preserve terrain identity and bounded context."""
from pathlib import Path
import tempfile
import unittest

from Renderer import renderer


class VolcanoFixtureTests(unittest.TestCase):
    def rows(self, case):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'scene.csv'
            renderer.scene('volcanoes', case, target)
            return target.read_text()

    def test_activity_control_changes_no_geometry(self):
        self.assertEqual(self.rows('detail'), self.rows('active'))

    def test_each_case_contains_one_ordinary_volcano_and_ground(self):
        import csv
        import io
        for case in renderer.standard('volcanoes')['recipe']['cases']:
            rows = list(csv.reader(io.StringIO(self.rows(case))))[1:]
            # V3 fields: x, y, base, real, river, bonus, flags.
            self.assertEqual(sum(int(row[3]) == 10 for row in rows), 1, case)
            terrain = {(int(row[0]), int(row[1])): int(row[3]) for row in rows}
            self.assertEqual([terrain[xy] for xy in ((15, 15), (14, 14))], [6, 6], case)
            self.assertGreater(sum(int(row[3]) in (0, 1, 2, 3) for row in rows), 100, case)

    def test_shared_changes_select_volcanoes_and_terrain_witness(self):
        for category in ('day-night', 'shadows', 'transitions'):
            self.assertIn('volcanoes', renderer.affected(category))
        self.assertIn('terrain-edit', [case[0] for case in renderer.integration_replay_cases('volcanoes')])


if __name__ == '__main__':
    unittest.main()
