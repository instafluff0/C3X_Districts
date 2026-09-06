import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "Renderer/terrain_lab/fixtures/l19_farms_192.csv"
TUNDRA = ROOT / "Renderer/terrain_lab/fixtures/l19_tundra_witness_192.csv"
RUNTIME = ROOT / "Renderer/packs/ImprovementsNormalized/farm_runtime.bin"
CPP = (ROOT / "Renderer/terrain_lab/terrain_lab.cpp").read_text(encoding="utf-8")
HLSL = (ROOT / "Renderer/terrain_lab/terrain_lab.hlsl").read_text(encoding="utf-8")
RUN = (ROOT / "Renderer/terrain_lab/RUN_L19.bat").read_text(encoding="utf-8")


class L19FarmContractTests(unittest.TestCase):
    def test_source_bundle_topology_and_tundra_paths_exist(self):
        self.assertTrue(RUNTIME.read_bytes().startswith(b"C3XVEG1\0"))
        self.assertIn("add_farm_scene", CPP)
        self.assertIn("adjacency_mask", CPP)
        self.assertIn("farm_runtime.bin", CPP)
        self.assertIn("tundra_base_color.dds", CPP)
        self.assertIn("material_tundra", HLSL)
        self.assertIn("feature_base_texture_5", HLSL)
        self.assertIn("Civ5EnvironmentSkin", RUN)
        self.assertNotIn("smoke", RUN.lower())

    def test_checked_in_inputs_are_stable(self):
        self.assertEqual(
            "746cbf2e6ea2c7d8dfd44a44d7c5e647f3c250766c8ad9a6228a991446be2d7e",
            hashlib.sha256(RUNTIME.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            "b1ec2a17879dbaa61a4d7b3c9e31d2dbc235c1a59444dc0ec603ffeed83c32c6",
            hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            "25c1839ddbaefbb79ee108f79aa06430d90693900af7e7abdfd1f890c3e1133f",
            hashlib.sha256(TUNDRA.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
