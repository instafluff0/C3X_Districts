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
            pair = ((13, 13), (12, 12)) if case == "isolated" else ((15, 15), (14, 14))
            self.assertEqual([terrain[xy] for xy in pair], [6, 6], case)
            self.assertGreater(sum(int(row[3]) in (0, 1, 2, 3) for row in rows), 100, case)

    def test_volcano_ownership_follows_the_shared_relief(self):
        # Volcanoes are stamps of the shared mountain relief: their tiles use
        # the joined relief grid, and each vertex takes the volcano's texture
        # position, coverage, activity and lava channel from that same shape
        # (behaviour and wrapped occurrences: test_mountain_shape.cpp). The
        # retired ground provider adds no second cone (test_relief.cpp).
        terrain = (renderer.ROOT / 'Renderer/native/source_fidelity/terrain_mesh_body.h').read_text()
        self.assertIn('unified_mountain_surface|=real==6 || real==10;', terrain)
        self.assertNotIn('VolcanoCenter', terrain)
        relief = (renderer.ROOT / 'Renderer/lab/shared/natural/relief_mesh_body.h').read_text()
        for line in ('out.relief_owner_u=sample.volcano_u-.5f;out.relief_owner_v=sample.volcano_v-.5f;',
                     'out.relief_owner_coverage=sample.volcano;',
                     'out.relief_owner_state=2.f*float(sample.activity)+sample.channel;'):
            self.assertIn(line, relief)

    def test_shared_changes_select_volcanoes_and_terrain_witness(self):
        for category in ('day-night', 'shadows', 'transitions'):
            self.assertIn('volcanoes', renderer.affected(category))
        for witness in ('terrain-edit', 'volcano-lifecycle'):
            self.assertIn(witness, [case[0] for case in renderer.integration_replay_cases('volcanoes')])


if __name__ == '__main__':
    unittest.main()
