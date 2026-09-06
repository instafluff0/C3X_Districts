import hashlib
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "Renderer" / "canonical"
RUN_L13 = (ROOT / "Renderer" / "terrain_lab" / "RUN_L13.bat").read_text(
    encoding="utf-8"
)

EXPECTED = {
    "daynight.png": ((1608, 1368), "5e210083753c9938478c636815bf50ae6d2f5d82f4fe9cd2a45c33357db65c5a"),
    "desert.png": ((1008, 620), "e270f22003589504e22de2b0f4a7365de8a109e437c44e12b5f21e079ef8b61f"),
    "forest.png": ((660, 502), "4c94f820423486ef056b4a7fc3e41d3f985efc022ff48f3a09dc5bf3b8dee36f"),
    "hills.png": ((600, 382), "c0f10967a42915efef0d7b518f453439dd10c3b8cb2b46cad11430b7a1b98c1f"),
    "jungle.png": ((1036, 598), "2a665669fdd56176e034f127459c4054e3a3dca8ab0073bd1607039fe9b9c2c1"),
    "marsh.png": ((398, 258), "1f1ad37936013598e74261ea868c27bd7940ffad4007d1d39be5bea0614367b3"),
    "mountain.png": ((584, 506), "18bc5ea32b6276786a5db444a928fec153b4ada25ff21015eefa677dc05b4661"),
    "river.png": ((1954, 970), "ebfbcc875f4469601b961b609188d4026afbeadac9fdb4c03ff9e0d7726a9766"),
    "roads.png": ((824, 780), "2452e917565a69eadbef8bd972f7a98be09104292be3ac438e778cf15b090817"),
    "sea_and_shore.png": ((3430, 1750), "8a2b7561d5b1fb9a33276196cd5a520c3953373381d2083232d923f75155e3e8"),
}


class CanonicalReferenceContractTests(unittest.TestCase):
    def test_l13_defaults_to_alternate_environment_packs(self) -> None:
        self.assertIn("packs\\Civ5EnvironmentSkin", RUN_L13)
        self.assertIn("packs\\Civ5EnvironmentVegetation", RUN_L13)
        self.assertNotIn("set \"C3X_LAB_PACK=..\\packs\\TerrainNormalized\"", RUN_L13)

    @unittest.skipUnless(CANONICAL.is_dir(), "local canonical screenshots are optional")
    def test_local_reference_set_is_complete_and_unchanged(self) -> None:
        self.assertEqual(set(EXPECTED), {path.name for path in CANONICAL.glob("*.png")})
        for name, (dimensions, digest) in EXPECTED.items():
            data = (CANONICAL / name).read_bytes()
            self.assertEqual(b"\x89PNG\r\n\x1a\n", data[:8])
            self.assertEqual(dimensions, struct.unpack(">II", data[16:24]))
            self.assertEqual(digest, hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
