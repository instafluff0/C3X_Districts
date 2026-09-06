import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_l15_railroad_scenario.py")
SPEC = importlib.util.spec_from_file_location("build_l15_railroad_scenario", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RailroadScenarioBuilderTests(unittest.TestCase):
    def test_accepted_road_graph_produces_deterministic_connected_subset(self):
        source = Path(__file__).with_name("fixtures") / "l14_roads_192.csv"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "railroads.csv"
            before = source.read_bytes()
            MODULE.write_scenario(source, target)
            first = target.read_bytes()
            MODULE.write_scenario(source, target)
            self.assertEqual(first, target.read_bytes())
            self.assertEqual(before, source.read_bytes())
            self.assertTrue(first.startswith(MODULE.MAGIC.encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
