import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_route_bridge_runtime.py")
SPEC = importlib.util.spec_from_file_location("build_route_bridge_runtime", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RouteBridgeRuntimeTests(unittest.TestCase):
    def test_local_normalized_pack_builds_deterministically(self):
        source = Path("Renderer/packs/RouteDoodadsNormalized").resolve()
        if not source.is_dir():
            self.skipTest("local normalized route doodad pack is unavailable")
        first = MODULE.build(source).read_bytes()
        second = MODULE.build(source).read_bytes()
        self.assertEqual(first, second)
        self.assertTrue(first.startswith(MODULE.MAGIC))
        self.assertGreater(len(first), 100_000)


if __name__ == "__main__":
    unittest.main()
