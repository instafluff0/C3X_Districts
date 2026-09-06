import csv
import tempfile
import unittest
from pathlib import Path

from Renderer.terrain_lab.build_l18_mine_scenario import write_scenario


ROOT = Path(__file__).resolve().parents[2]


class BuildL18MineScenarioTests(unittest.TestCase):
    def test_checked_in_fixture_rebuilds_byte_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mines.csv"
            write_scenario(
                ROOT / "Renderer/preview/out/terrain_lab/test_biq_l13_rivers_192.csv",
                ROOT / "Renderer/terrain_lab/fixtures/l14_roads_192.csv",
                ROOT / "Renderer/terrain_lab/fixtures/l16_resources_192.csv",
                ROOT / "Renderer/terrain_lab/fixtures/l17_cities_192.csv",
                output,
            )
            expected = ROOT / "Renderer/terrain_lab/fixtures/l18_mines_192.csv"
            self.assertEqual(expected.read_bytes(), output.read_bytes())

    def test_fixture_has_twenty_visible_mines(self):
        path = ROOT / "Renderer/terrain_lab/fixtures/l18_mines_192.csv"
        rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(20, len(rows) - 1)
        self.assertTrue(all(int(row[4]) == 1 for row in rows[1:]))


if __name__ == "__main__":
    unittest.main()
