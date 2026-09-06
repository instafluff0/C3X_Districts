import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("build_l14_road_scenario.py")
SPEC = importlib.util.spec_from_file_location("build_l14_road_scenario", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
MAGIC = MODULE.MAGIC
write_scenario = MODULE.write_scenario


class RoadScenarioTests(unittest.TestCase):
    def test_dense_connected_network_is_deterministic_and_keeps_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "terrain.csv"
            output = root / "roads.csv"
            rows = [["C3X_BIQ_TERRAIN_WINDOW_V2", 16, 12, 192, 0, 0, 64, 64, 0]]
            for y in range(12):
                for x in range(16):
                    real = 5 if (x + y) % 9 == 0 else 2
                    river = 8 if (x * 3 + y) % 17 == 0 else 0
                    rows.append([x, y, x * 2, y * 2, 2, real, 0, 0, river])
            with source.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream, lineterminator="\n").writerows(rows)
            before = source.read_bytes()
            write_scenario(source, output)
            first = output.read_bytes()
            write_scenario(source, output)
            self.assertEqual(first, output.read_bytes())
            self.assertEqual(before, source.read_bytes())
            records = list(csv.reader(first.decode().splitlines()))
            self.assertEqual(records[0][0], MAGIC)
            self.assertEqual(records[0][5], "lab_augmentation")
            self.assertGreater(int(records[0][3]), 50)
            self.assertTrue(any(int(row[4]) for row in records[1:]))
            self.assertTrue(any(int(row[7]) for row in records[1:]))


if __name__ == "__main__":
    unittest.main()
