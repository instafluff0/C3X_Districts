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
            "c82b8d17ad7256f11f1807ca2ccf3e7ced402a63a4d99e275e67d22ee6ed769a",
            hashlib.sha256(RUNTIME.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            "8ab4f84e86b6778283c193def06ee681560026b3f56bb41895f0b5fd9085bb9d",
            hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            "6da892dd9807905c93e38c01cf6e4f40d5c4977e2a06bb8ff04352caf057e8e6",
            hashlib.sha256(TUNDRA.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
