#!/usr/bin/env python3
import csv
import tempfile
import unittest
from pathlib import Path

from Renderer.terrain_lab.build_l19b_infrastructure_scenario import build


ROOT = Path(__file__).resolve().parents[2]


class L19BScenarioBuilderTests(unittest.TestCase):
    def test_real_192_tile_fixture_is_deterministic_and_complete(self) -> None:
        args = [
            ROOT / "Renderer/preview/out/terrain_lab/test_biq_l13_rivers_192.csv",
            ROOT / "Renderer/terrain_lab/fixtures/l14_roads_192.csv",
            ROOT / "Renderer/terrain_lab/fixtures/l16_resources_192.csv",
            ROOT / "Renderer/terrain_lab/fixtures/l17_cities_192.csv",
            ROOT / "Renderer/terrain_lab/fixtures/l18_mines_192.csv",
            ROOT / "Renderer/terrain_lab/fixtures/l19_farms_192.csv",
            ROOT / "Renderer/terrain_lab/fixtures/l19a_tile_objects_192.csv",
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            build(*args, first)
            build(*args, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            rows = list(csv.reader(first.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(["16", "12", "36"], rows[0][1:4])
            kinds = [int(row[2]) for row in rows[1:]]
            self.assertEqual(set(range(8)), set(kinds))
            self.assertEqual(3, sum(int(row[6]) for row in rows[1:] if int(row[2]) == 4))
            self.assertEqual({0, 1, 2, 3}, {int(row[4]) for row in rows[1:]})


if __name__ == "__main__":
    unittest.main()
