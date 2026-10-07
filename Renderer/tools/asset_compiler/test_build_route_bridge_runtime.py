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

    def test_tunnel_keeps_only_the_entrance(self):
        # The portal's block and the rubble on it stood out of the mountain
        # as a brown box at its foot; only the facade, wing walls, floor and
        # bore behind the arch remain.
        def vertex(x, y, z):
            return {"position": [x, y, z], "normal": [0.0, -1.0, 0.0], "uv0": [0.0, 0.0]}
        triangles = {
            "facade": [(-0.05, -0.025, 0.0), (0.04, -0.025, 0.0), (0.0, -0.025, 0.07)],
            "wing wall": [(-0.05, -0.08, 0.0), (-0.05, -0.025, 0.0), (-0.05, -0.025, 0.045)],
            "bore": [(-0.03, -0.025, 0.0), (-0.03, 0.033, 0.0), (-0.03, 0.033, 0.04)],
            "block": [(-0.05, -0.025, 0.057), (-0.05, 0.085, 0.057), (0.04, 0.085, 0.057)],
            "rubble": [(0.0, 0.02, 0.071), (0.01, 0.03, 0.075), (0.0, 0.03, 0.077)],
        }
        vertices = [vertex(*point) for points in triangles.values() for point in points]
        mesh = {"vertices": vertices, "topology": {"primitive": "triangles", "indices": list(range(len(vertices)))}}
        entrance = MODULE.tunnel_entrance(mesh)
        kept = {tuple(vertex["position"]) for vertex in entrance["vertices"]}
        for name, points in triangles.items():
            self.assertEqual(all(point in kept for point in points), name in ("facade", "wing wall", "bore"), name)
        self.assertEqual(len(entrance["topology"]["indices"]), 9)
        self.assertTrue(all(index < len(entrance["vertices"]) for index in entrance["topology"]["indices"]))


if __name__ == "__main__":
    unittest.main()
