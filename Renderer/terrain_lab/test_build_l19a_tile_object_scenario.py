import csv
import tempfile
import unittest
from pathlib import Path

from Renderer.terrain_lab.build_l19a_tile_object_scenario import build


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "Renderer/terrain_lab/fixtures"


class BuildL19ATileObjectScenarioTests(unittest.TestCase):
    def test_checked_in_fixture_rebuilds_byte_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "objects.csv"
            build(
                ROOT / "Renderer/preview/out/terrain_lab/test_biq_l13_rivers_192.csv",
                FIXTURES / "l16_resources_192.csv",
                FIXTURES / "l17_cities_192.csv",
                FIXTURES / "l18_mines_192.csv",
                FIXTURES / "l19_farms_192.csv",
                output,
            )
            self.assertEqual(
                (FIXTURES / "l19a_tile_objects_192.csv").read_bytes(),
                output.read_bytes(),
            )

    def test_fixture_covers_visibility_buckets_eras_and_extraterritorial_colonies(self):
        rows = list(csv.reader(
            (FIXTURES / "l19a_tile_objects_192.csv").read_text().splitlines()
        ))
        self.assertEqual("C3X_LAB_TILE_OBJECT_SCENARIO_V0", rows[0][0])
        records = [tuple(map(int, row)) for row in rows[1:]]
        huts = [row for row in records if row[2] == 0]
        colonies = [row for row in records if row[2] == 1]
        self.assertEqual(set(range(8)), {row[3] for row in huts if row[7] == 1})
        self.assertEqual(set(range(4)), {row[4] for row in colonies if row[7] == 1})
        self.assertEqual(set(range(4)), {row[5] for row in colonies if row[7] == 1})
        self.assertTrue(all(row[5] != row[6] for row in colonies))
        self.assertEqual(2, sum(row[7] == 0 for row in records))
        self.assertTrue(all(row[8] < 7 for row in colonies))


if __name__ == "__main__":
    unittest.main()
