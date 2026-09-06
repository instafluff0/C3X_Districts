import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "Renderer/terrain_lab/fixtures/l19a_tile_objects_192.csv"
RUNTIME = ROOT / "Renderer/packs/TileObjectsNormalized/tile_object_runtime.bin"
CPP = (ROOT / "Renderer/terrain_lab/terrain_lab.cpp").read_text(encoding="utf-8")
HLSL = (ROOT / "Renderer/terrain_lab/terrain_lab.hlsl").read_text(encoding="utf-8")
RUN = (ROOT / "Renderer/terrain_lab/RUN_L19A.bat").read_text(encoding="utf-8")


class L19ATileObjectContractTests(unittest.TestCase):
    def test_runtime_and_viewer_owner_paths_are_source_backed(self):
        self.assertTrue(RUNTIME.read_bytes().startswith(b"C3XVEG1\0"))
        self.assertIn("add_tile_object_scene", CPP)
        self.assertIn("hut_bucket_to_variant", CPP)
        self.assertIn("matching_resource", CPP)
        self.assertIn("tile_object_weight", HLSL)
        self.assertIn("Civ5EnvironmentSkin", RUN)
        self.assertNotIn("smoke", RUN.lower())
        self.assertNotIn("particle", RUN.lower())

    def test_checked_in_inputs_are_stable(self):
        self.assertEqual(
            "16e1acdb3835b25cd929ad51221e08ce8b303c0caff4050f98aded15bcb41ec3",
            hashlib.sha256(RUNTIME.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            "b4c5ec90004960237411106104276b5639e31e8c6151688e01f76128d0310fe1",
            hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
