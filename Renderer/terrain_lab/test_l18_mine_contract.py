import csv
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "Renderer/terrain_lab/fixtures/l18_mines_192.csv"
RUNTIME = ROOT / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
CPP = (ROOT / "Renderer/terrain_lab/terrain_lab.cpp").read_text(encoding="utf-8")
HLSL = (ROOT / "Renderer/terrain_lab/terrain_lab.hlsl").read_text(encoding="utf-8")
RUN = (ROOT / "Renderer/terrain_lab/RUN_L18.bat").read_text(encoding="utf-8")


class L18MineContractTests(unittest.TestCase):
    def test_scenario_covers_eras_variants_resources_and_relief(self):
        rows = list(csv.reader(SCENARIO.read_text(encoding="utf-8").splitlines()))
        self.assertEqual("C3X_LAB_MINE_SCENARIO_V0", rows[0][0])
        records = [tuple(map(int, row)) for row in rows[1:]]
        self.assertEqual(set(range(4)), {row[2] for row in records})
        self.assertEqual(set(range(3)), {row[3] for row in records})
        self.assertGreaterEqual(sum(row[5] for row in records), 4)

    def test_runtime_and_pass_are_source_backed_and_isolated(self):
        self.assertTrue(RUNTIME.read_bytes().startswith(b"C3XVEG1\0"))
        self.assertIn("add_mine_scene", CPP)
        self.assertIn("mine_runtime.bin", CPP)
        self.assertIn("mine_emissive_views", CPP)
        self.assertIn("mine_weight", HLSL)
        self.assertIn("Civ5EnvironmentSkin", RUN)
        self.assertNotIn("smoke", RUN.lower())

    def test_checked_in_hashes_are_stable(self):
        self.assertEqual("3cf0b60852e1caa4934febbacc649d18c7953e2da07b72f823b4ec1ffafe62a7",
                         hashlib.sha256(SCENARIO.read_bytes()).hexdigest())
        self.assertEqual("d833335ec1a4b1e4c9c4da637f81b46f7828ff038a722a9a5a4673c76ba751f1",
                         hashlib.sha256(RUNTIME.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
