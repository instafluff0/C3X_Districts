import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_l17_city_scenario.py")
SPEC = importlib.util.spec_from_file_location("build_l17_city_scenario", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CityScenarioBuilderTests(unittest.TestCase):
    def test_city_matrix_is_deterministic_and_non_mutating(self):
        root = Path(__file__).resolve().parents[1]
        viewport = root / "preview/out/terrain_lab/test_biq_l13_rivers_192.csv"
        roads = Path(__file__).with_name("fixtures") / "l14_roads_192.csv"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "cities.csv"
            before = viewport.read_bytes(), roads.read_bytes()
            MODULE.write_scenario(viewport, roads, target)
            first = target.read_bytes()
            MODULE.write_scenario(viewport, roads, target)
            self.assertEqual(first, target.read_bytes())
            self.assertEqual(before, (viewport.read_bytes(), roads.read_bytes()))


if __name__ == "__main__":
    unittest.main()
