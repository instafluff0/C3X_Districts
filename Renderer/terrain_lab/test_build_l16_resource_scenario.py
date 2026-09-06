import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_l16_resource_scenario.py")
SPEC = importlib.util.spec_from_file_location("build_l16_resource_scenario", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ResourceScenarioBuilderTests(unittest.TestCase):
    def test_decoded_viewport_produces_deterministic_non_mutating_fixture(self):
        source = Path(__file__).resolve().parents[1] / "preview/out/terrain_lab/test_biq_l13_rivers_192.csv"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "resources.csv"
            before = source.read_bytes()
            MODULE.write_scenario(source, target)
            first = target.read_bytes()
            MODULE.write_scenario(source, target)
            self.assertEqual(first, target.read_bytes())
            self.assertEqual(before, source.read_bytes())
            self.assertTrue(first.startswith(MODULE.MAGIC.encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
