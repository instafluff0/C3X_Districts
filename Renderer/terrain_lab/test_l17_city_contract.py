import csv
import hashlib
import unittest
from pathlib import Path


SCENARIO = Path(__file__).with_name("fixtures") / "l17_cities_192.csv"
CPP = Path(__file__).with_name("terrain_lab.cpp").read_text(encoding="utf-8")
HLSL = Path(__file__).with_name("terrain_lab.hlsl").read_text(encoding="utf-8")
RUN = Path(__file__).with_name("RUN_L17.bat").read_text(encoding="utf-8")


class L17CityContractTests(unittest.TestCase):
    def test_matrix_covers_era_size_culture_owner_walls_and_capitals(self):
        rows = list(csv.reader(SCENARIO.read_text(encoding="utf-8").splitlines()))
        self.assertEqual("C3X_LAB_CITY_SCENARIO_V0", rows[0][0])
        self.assertEqual("lab_augmentation", rows[0][5])
        records = [tuple(map(int, row)) for row in rows[1:]]
        self.assertEqual(set(range(4)), {row[2] for row in records})
        self.assertEqual(set(range(3)), {row[3] for row in records})
        self.assertEqual(set(range(5)), {row[4] for row in records})
        self.assertEqual(set(range(4)), {row[5] for row in records})
        self.assertGreaterEqual(sum(row[6] for row in records), 4)
        self.assertGreaterEqual(sum(row[7] for row in records), 2)

    def test_city_pass_uses_source_emissive_and_shared_lighting(self):
        self.assertIn("add_city_scene", CPP)
        self.assertIn("city_runtime.bin", CPP)
        self.assertIn("wall_runtime.bin", CPP)
        self.assertIn("environment_night_activation", HLSL)
        self.assertIn("city_emissive_views", CPP)
        self.assertIn("Civ5EnvironmentSkin", RUN)
        self.assertNotIn("bloom", RUN.lower())

    def test_checked_in_fixture_hash_is_stable(self):
        self.assertEqual("ef7b24cb0cec97c7b6de7971552ebdea9725c09ec666f3eb57a5423aeab8ed47",
                         hashlib.sha256(SCENARIO.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
