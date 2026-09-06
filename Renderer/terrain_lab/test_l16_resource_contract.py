import csv
import hashlib
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = Path(__file__).with_name("fixtures") / "l16_resources_192.csv"
CPP = Path(__file__).with_name("terrain_lab.cpp").read_text(encoding="utf-8")
HLSL = Path(__file__).with_name("terrain_lab.hlsl").read_text(encoding="utf-8")
RUN = Path(__file__).with_name("RUN_L16.bat").read_text(encoding="utf-8")


class L16ResourceContractTests(unittest.TestCase):
    def test_fixture_covers_three_classes_land_water_and_hidden(self):
        rows = list(csv.reader(SCENARIO.read_text(encoding="utf-8").splitlines()))
        self.assertEqual("C3X_LAB_RESOURCE_SCENARIO_V0", rows[0][0])
        self.assertEqual("lab_augmentation", rows[0][5])
        self.assertEqual(192, int(rows[0][1]) * int(rows[0][2]))
        self.assertEqual(int(rows[0][3]), len(rows) - 1)
        visible = Counter(int(row[2]) for row in rows[1:] if int(row[3]))
        self.assertTrue({0, 1, 2, 3, 4, 5, 6, 7}.issubset(visible))
        self.assertGreaterEqual(sum(visible.values()), 24)
        self.assertEqual(1, sum(not int(row[3]) for row in rows[1:]))

    def test_runtime_uses_normalized_source_art_and_shared_lighting(self):
        self.assertIn("add_resource_scene", CPP)
        self.assertIn("resource_runtime.bin", CPP)
        self.assertIn("resource_base_texture_7", HLSL)
        self.assertIn("frame_illumination", HLSL)
        self.assertIn("Civ5EnvironmentSkin", RUN)
        self.assertIn("resources_hidden", RUN)
        self.assertNotIn("smoke", RUN.lower())

    def test_land_resources_share_face_and_cast_shadow_contract(self):
        self.assertIn("float resource_weight", HLSL)
        self.assertIn("resource_weight * 0.95", HLSL)
        self.assertIn("resource_weight * 0.18", HLSL)
        self.assertIn("output, 7.0f, 1.05f", CPP)
        self.assertIn("instance.resource == 7u ? submerged_shadows : shadows", CPP)

    def test_checked_in_fixture_hash_is_stable(self):
        self.assertEqual(
            "4bee2c83af59ffdbb076796c7747f4c067b5b3e964aa30d9687b9a5751c527ee",
            hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
