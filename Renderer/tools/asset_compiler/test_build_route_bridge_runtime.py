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



    def test_tunnel_rock_greys_like_the_mountains(self):
        # The portal's block and cap were Civ VI's brown rock against grey
        # mountains. Their rock turns grey; the red signal stripes stay red;
        # every BC1 block keeps its colour mode.
        import struct
        header = bytearray(148)
        header[:4] = b"DDS "
        header[84:88] = b"DX10"
        struct.pack_into("<I", header, 128, 72)
        brown = MODULE._rgb565((150, 110, 80)), MODULE._rgb565((90, 70, 50))
        red = MODULE._rgb565((220, 40, 30)), MODULE._rgb565((120, 20, 15))
        # Brown in four-colour order, then three-colour order; red; then a
        # pair whose order flips once greyed.
        blocks = [(max(brown), min(brown), 0x1B1B1B1B), (min(brown), max(brown), 0x1B1B1B1B),
                  (max(red), min(red), 0), (MODULE._rgb565((60, 120, 120)), MODULE._rgb565((140, 90, 80)), 0x55555555)]
        data = bytes(header) + b"".join(struct.pack("<HHI", *block) for block in blocks)
        out = MODULE.grey_texture(data)
        for index, (c0, c1, indices) in enumerate(blocks):
            n0, n1, ni = struct.unpack_from("<HHI", out, 148 + 8 * index)
            self.assertEqual(n0 > n1, c0 > c1, index)  # colour mode kept
            if index == 2:
                self.assertEqual((n0, n1), (c0, c1))  # strong red kept
            else:
                for c in (n0, n1):
                    r, g, b = MODULE._unpack565(c)
                    self.assertLess(max(r, g, b) - min(r, g, b), 16, index)
        # The flipped pair (three-colour order) swapped its endpoints, so its
        # indices swapped too: every 1 became 0.
        self.assertEqual(struct.unpack_from("<HHI", out, 148 + 24)[2], 0)

    def test_tunnel_portal_lengths_stretch_only_behind_the_facade(self):
        # The block stretches behind the facade; the wing walls before it stay.
        def vertex(y, z):
            return {"position": [0.0, y, z]}
        mesh = {"vertices": [vertex(-0.08, 0.0), vertex(-0.025, 0.07), vertex(0.085, 0.07)]}
        stretched = MODULE.stretch_block(mesh, 3.0)
        ys = [vertex["position"][1] for vertex in stretched["vertices"]]
        self.assertEqual(ys[:2], [-0.08, -0.025])
        self.assertAlmostEqual(ys[2], -0.025 + 0.11 * 3.0)

    def test_tunnel_cutting_centres_on_the_rail(self):
        # The rail ran into a wing wall: the cutting and arch centre 0.007 off
        # the mesh axis, where the rail runs.
        mesh = {"vertices": [{"position": [-0.0393, -0.05, 0.0]}, {"position": [0.0252, -0.05, 0.0]}]}
        xs = [vertex["position"][0] for vertex in MODULE.centre_on_cutting(mesh)["vertices"]]
        self.assertAlmostEqual(xs[0] + xs[1], 0.0, places=3)

if __name__ == "__main__":
    unittest.main()
