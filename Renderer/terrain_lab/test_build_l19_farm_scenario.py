import csv
import tempfile
import unittest
from pathlib import Path

from Renderer.terrain_lab.build_l19_farm_scenario import (
    write_scenario,
    write_tundra_viewport,
)


ROOT = Path(__file__).resolve().parents[2]


class BuildL19FarmScenarioTests(unittest.TestCase):
    def test_checked_in_fixtures_rebuild_byte_identically(self):
        viewport = ROOT / "Renderer/preview/out/terrain_lab/test_biq_l13_rivers_192.csv"
        cities = ROOT / "Renderer/terrain_lab/fixtures/l17_cities_192.csv"
        with tempfile.TemporaryDirectory() as directory:
            farms = Path(directory) / "farms.csv"
            tundra = Path(directory) / "tundra.csv"
            write_scenario(viewport, cities, farms)
            write_tundra_viewport(viewport, farms, tundra)
            self.assertEqual(
                (ROOT / "Renderer/terrain_lab/fixtures/l19_farms_192.csv").read_bytes(),
                farms.read_bytes(),
            )
            self.assertEqual(
                (ROOT / "Renderer/terrain_lab/fixtures/l19_tundra_witness_192.csv").read_bytes(),
                tundra.read_bytes(),
            )

    def test_fixture_covers_all_masks_eras_and_terrain_families(self):
        path = ROOT / "Renderer/terrain_lab/fixtures/l19_farms_192.csv"
        rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))[1:]
        records = [tuple(map(int, row)) for row in rows]
        self.assertEqual(set(range(16)), {row[3] for row in records})
        self.assertEqual(set(range(4)), {row[2] for row in records})
        self.assertEqual(set(range(4)), {row[4] for row in records})
        self.assertEqual(1, sum(row[5] == 0 for row in records))

    def test_tundra_witness_has_irrigated_and_unirrigated_cells(self):
        farms = list(csv.reader((ROOT / "Renderer/terrain_lab/fixtures/l19_farms_192.csv").read_text().splitlines()))[1:]
        farm_cells = {(int(row[0]), int(row[1])) for row in farms if int(row[5]) == 1}
        tundra_rows = list(csv.reader((ROOT / "Renderer/terrain_lab/fixtures/l19_tundra_witness_192.csv").read_text().splitlines()))
        columns, height = int(tundra_rows[0][1]), int(tundra_rows[0][2])
        tundra_cells = {(int(row[0]), int(row[1])) for row in tundra_rows[1:1 + columns * height] if int(row[4]) == 3}
        self.assertTrue(tundra_cells & farm_cells)
        self.assertTrue(tundra_cells - farm_cells)


if __name__ == "__main__":
    unittest.main()
