"""Owner-colour coverage measurement and garment mask painting (synthetic packs)."""
import json
import struct

import numpy as np
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import unit_owner_coverage as coverage


def dds(path, width, height, dxgi, blocks):
    header = b"DDS " + struct.pack("<7I", 124, 0, height, width, 0, 0, 1) + b"\0" * 52
    header += struct.pack("<8I", 32, 4, 0, 0, 0, 0, 0, 0)[:32]
    header = header[:84] + b"DX10" + header[88:]
    header = header.ljust(128, b"\0") + struct.pack("<5I", dxgi, 3, 0, 1, 0)
    path.write_bytes(header + blocks)


def quad(path, x0, x1, height, normal=(0, -1, 0)):
    """A vertical quad in the source XZ plane with UVs over the texture."""
    vertices = [(x0, 0, 0, 0, 1), (x1, 0, 0, 1, 1), (x1, 0, height, 1, 0), (x0, 0, height, 0, 0)]
    blob = b"C3XANM1\0" + struct.pack("<6I", 1, 4, 6, 1, 1, 0)
    for x, y, z, u, v in vertices:
        blob += struct.pack("<8f4I4f", x, y, z, *normal, u, v, 0, 0, 0, 0, 1, 0, 0, 0)
    blob += struct.pack("<6I", 0, 1, 2, 0, 2, 3)
    blob += struct.pack("<16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    path.write_bytes(blob)


def hull(path, length, height, width, rows=8, z0=0.0):
    """A closed box hull's four side walls, subdivided into horizontal strips.
    The two long walls carry the texture; the short ends map to its margin."""
    vertices, triangles = [], []
    walls = (((-length / 2, -width / 2), (length / 2, -width / 2), (0, -1)),
             ((length / 2, width / 2), (-length / 2, width / 2), (0, 1)),
             ((length / 2, -width / 2), (length / 2, width / 2), (1, 0)),
             ((-length / 2, width / 2), (-length / 2, -width / 2), (-1, 0)))
    for w, ((x0, y0), (x1, y1), (nx, ny)) in enumerate(walls):
        base = len(vertices)
        u0, u1 = (0, 1) if w < 2 else (.98, 1)
        for r in range(rows + 1):
            z, v = z0 + height * r / rows, 1 - r / rows
            vertices += [(x0, y0, z, nx, ny, u0, v), (x1, y1, z, nx, ny, u1, v)]
        for r in range(rows):
            a = base + 2 * r
            triangles += [a, a + 1, a + 3, a, a + 3, a + 2]
    blob = b"C3XANM1\0" + struct.pack("<6I", 1, len(vertices), len(triangles), 1, 1, 0)
    for x, y, z, nx, ny, u, v in vertices:
        blob += struct.pack("<8f4I4f", x, y, z, nx, ny, 0, u, v, 0, 0, 0, 0, 1, 0, 0, 0)
    blob += struct.pack(f"<{len(triangles)}I", *triangles)
    blob += struct.pack("<16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    path.write_bytes(blob)


def mesh(path, points, triangles, joints=None, frames=((np.eye(4),),), uv=None):
    """A skinned payload: points (x, y, z), one joint per vertex, one 4x4
    palette per bone and frame (row-major, translation in the last row)."""
    joints = joints or [0] * len(points)
    uv = uv or [(0.5, 0.5)] * len(points)
    bones = len(frames[0])
    blob = b"C3XANM1\0" + struct.pack("<6I", 1, len(points), len(triangles), bones, len(frames), 0)
    for (x, y, z), j, (u, v) in zip(points, joints, uv):
        blob += struct.pack("<8f4I4f", x, y, z, 0, 0, 1, u, v, j, 0, 0, 0, 1, 0, 0, 0)
    blob += struct.pack(f"<{len(triangles)}I", *triangles)
    for frame in frames:
        for matrix in frame:
            blob += struct.pack("<16f", *np.asarray(matrix, float).reshape(-1))
    path.write_bytes(blob)


def turn(degrees):
    a = np.radians(degrees)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = np.cos(a), np.sin(a), -np.sin(a), np.cos(a)
    return m


class OwnerCoverageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.pack = Path(temporary.name)
        (self.pack / "clips").mkdir()
        (self.pack / "textures").mkdir()

    def test_bc3_alpha_block_decodes_interpolated_and_explicit_values(self):
        # a0=255, a1=0 (eight-value mode); indices 0,1,2,... across the block.
        bits = sum((k % 8) << (3 * k) for k in range(16))
        block = bytes([255, 0]) + bits.to_bytes(6, "little") + b"\0" * 8
        dds(self.pack / "a.dds", 4, 4, 78, block)
        alpha = coverage.dds_alpha(self.pack / "a.dds")
        expected = [255, 0, 6 * 255 / 7, 5 * 255 / 7, 4 * 255 / 7, 3 * 255 / 7, 2 * 255 / 7, 255 / 7]
        flat = [alpha[k // 4, k % 4] * 255 for k in range(16)]
        for k in range(16):
            self.assertAlmostEqual(flat[k], expected[k % 8], places=3)

    def make(self, marked_width):
        # Opaque BC1 texture: owner mode 1 (inverse alpha) contributes nothing.
        dds(self.pack / "textures/opaque.dds", 4, 4, 72, bytes([0xff, 0xff, 0, 0]) + b"\0" * 4)
        # A figure: taller than its footprint is long.
        quad(self.pack / "clips/big.bin", -.12, .12, .6)
        quad(self.pack / "clips/side.bin", .12, .12 + marked_width, .6)
        def action():
            return {"part_count": 2,
                    "part0": {"mesh": "clips/big.bin", "texture": "textures/opaque.dds", "owner_mask": 0, "owner_strength": 0},
                    "part1": {"mesh": "clips/side.bin", "texture": "textures/opaque.dds", "owner_mask": 1, "owner_strength": .35}}
        bindings = {"unit0": {"key_count": 1, "key0": "PRTO_A", "scale": 1.0, "offset_z": 0.0,
                              "idle": action(), "move": action()}, "unit_count": 1}
        components = {"unit0": {"idle": ["c/big", "c/marked"], "move": ["c/big", "c/marked"]}}
        return bindings, components

    def test_sparse_unit_takes_largest_component_when_marked_part_is_small(self):
        bindings, components = self.make(.02)
        measured = coverage.measure(self.pack, bindings["unit0"])
        chosen, share, marked = coverage.choose_garment(bindings["unit0"]["idle"], components["unit0"]["idle"],
                                                        measured["part_share"], .15)
        self.assertEqual((chosen, marked), ("c/big", False))

    def test_large_marked_component_is_preferred(self):
        bindings, components = self.make(.6)
        measured = coverage.measure(self.pack, bindings["unit0"])
        chosen, share, marked = coverage.choose_garment(bindings["unit0"]["idle"], components["unit0"]["idle"],
                                                        measured["part_share"], .15)
        self.assertEqual((chosen, marked), ("c/marked", True))

    def test_covered_unit_is_unchanged(self):
        bindings, components = self.make(.05)
        bindings["unit0"]["idle"]["part0"].update(owner_mask=2, owner_strength=.82)
        before = json.dumps(bindings, sort_keys=True)
        self.assertEqual(coverage.paint_garments(bindings, self.pack, components, .10, .15, .25, .9), {})
        self.assertEqual(json.dumps(bindings, sort_keys=True), before)

    def test_bc3_rewrite_keeps_colour_and_writes_alpha_on_every_mip(self):
        # 8x8 BC1 with two mips: a four-colour block, a three-colour block (c0<c1).
        four = struct.pack("<HHI", 0xF800, 0x001F, 0x1B1B1B1B)
        three = struct.pack("<HHI", 0x001F, 0xF800, 0x24924924)
        top = four + three + four + three
        dds(self.pack / "c.dds", 8, 8, 72, top + four)
        data = bytearray((self.pack / "c.dds").read_bytes()); struct.pack_into("<I", data, 28, 2)
        (self.pack / "c.dds").write_bytes(bytes(data))
        before = coverage.dds_rgb(self.pack / "c.dds")
        # One value per 4x4 block is exact in BC3 (a block's own endpoints).
        alpha = np.kron(np.array([[0.0, .5], [.25, 1.0]]), np.ones((4, 4)))
        (self.pack / "o.dds").write_bytes(coverage.bc3_with_alpha(self.pack / "c.dds", alpha))
        after = coverage.dds_rgb(self.pack / "o.dds")
        decoded = coverage.dds_alpha(self.pack / "o.dds")
        self.assertLess(np.abs(decoded - alpha).max(), 1 / 255 + 1e-6)
        # Four-colour blocks are copied exactly; three-colour blocks keep their endpoints.
        self.assertTrue(np.array_equal(after[:4, :4], before[:4, :4]))
        for colour in (before[0:4, 4:8][0, 0], before[0:4, 4:8][0, 1]):
            self.assertTrue(any(np.allclose(colour, c) for c in after[0:4, 4:8].reshape(-1, 3)))
        self.assertEqual(struct.unpack_from("<I", (self.pack / "o.dds").read_bytes(), 128)[0], 78)
        self.assertEqual(len((self.pack / "o.dds").read_bytes()), 148 + 4 * 16 + 16)

    def test_paint_garments_reaches_target_from_the_top(self):
        bindings, components = self.make(.02)
        white = struct.pack("<HHI", 0xFFFF, 0x0000, 0)  # paintable neutral light texels
        dds(self.pack / "textures/opaque.dds", 32, 32, 72, white * 64)
        report = coverage.paint_garments(bindings, self.pack, components, .10, .15, .25, .9)
        result = report["PRTO_A"]
        self.assertEqual(result["component"], "c/big")
        self.assertAlmostEqual(result["coverage_after"], .25 * .9, delta=.08)
        for action in ("idle", "move"):
            part = bindings["unit0"][action]["part0"]
            self.assertEqual((part["owner_mask"], part["owner_strength"]), (1, .9))
            self.assertNotEqual(part["texture"], "textures/opaque.dds")
            self.assertTrue((self.pack / part["texture"]).is_file())
        # The painted alpha sits on the upper part of the quad (texture V=0 is the top).
        alpha = coverage.dds_alpha(self.pack / bindings["unit0"]["idle"]["part0"]["texture"])
        self.assertLess(alpha[0].mean(), alpha[-1].mean())

    def test_long_body_gets_a_side_stripe_not_its_top(self):
        # A hull: long footprint, low height, sides facing -y. Two stacked
        # quads give it thickness so its footprint is elongated, not a line.
        bindings, components = self.make(.05)
        white = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
        dds(self.pack / "textures/opaque.dds", 32, 32, 72, white * 64)
        hull(self.pack / "clips/big.bin", 2.0, .5, .8)
        side, low, high, centre, half_width = coverage.hull_band(self.pack, bindings["unit0"], {0}, 1.0, 0.0)
        self.assertAlmostEqual(half_width, .4, delta=.02)
        self.assertGreater(abs(side[1]), .2)
        self.assertAlmostEqual(low, 0.0, delta=.03)  # spans its full length at every height
        self.assertAlmostEqual(high, .5, delta=.03)
        self.assertLess(coverage.measure(self.pack, bindings["unit0"])["aspect"], 1.15)
        report = coverage.paint_garments(bindings, self.pack, components, .10, .15, .25, .9, .14)
        self.assertEqual(report["PRTO_A"]["style"], "side stripe")
        alpha = coverage.dds_alpha(self.pack / bindings["unit0"]["idle"]["part0"]["texture"])
        rows = 1 - alpha.mean(1)
        # A band on the upper hull (texture V=0 is the top edge), not the top
        # edge and not the whole side.
        self.assertLess(rows[0], .05)
        self.assertGreater(rows.max(), .5)
        self.assertTrue(2 < (rows > .5).sum() < 24)

    def test_centreline_barrel_stays_unmarked_beside_a_striped_hull(self):
        white = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
        for name in ("hull", "barrel"):
            dds(self.pack / f"textures/{name}.dds", 32, 32, 72, white * 64)
        hull(self.pack / "clips/hull.bin", 2.0, .5, .8)
        hull(self.pack / "clips/barrel.bin", 2.6, .06, .06, rows=2, z0=.3)  # gun along the centreline
        def action():
            return {"part_count": 2,
                    "part0": {"mesh": "clips/hull.bin", "texture": "textures/hull.dds", "owner_mask": 0, "owner_strength": 0},
                    "part1": {"mesh": "clips/barrel.bin", "texture": "textures/barrel.dds", "owner_mask": 0, "owner_strength": 0}}
        bindings = {"unit0": {"key_count": 1, "key0": "PRTO_A", "scale": 1.0, "offset_z": 0.0,
                              "idle": action(), "move": action()}, "unit_count": 1}
        components = {"unit0": {"idle": ["c/body", "c/body"], "move": ["c/body", "c/body"]}}
        report = coverage.paint_garments(bindings, self.pack, components, .10, .15, .25, .9, .14)
        self.assertEqual(report["PRTO_A"]["style"], "side stripe")
        hull_part, barrel_part = bindings["unit0"]["move"]["part0"], bindings["unit0"]["move"]["part1"]
        self.assertNotEqual(hull_part["texture"], "textures/hull.dds")
        self.assertGreater((coverage.dds_alpha(self.pack / hull_part["texture"]) < .5).mean(), .05)
        # The barrel keeps its own texture and no owner colour.
        self.assertEqual((barrel_part["texture"], barrel_part["owner_mask"]), ("textures/barrel.dds", 0))

    def white(self, *names):
        block = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
        for name in names:
            dds(self.pack / f"textures/{name}.dds", 32, 32, 72, block * 64)

    def test_hull_band_starts_at_the_waterline_below_long_yards(self):
        # A short hull under a yard twice its length (a lateen rig): the yard is
        # the only slice spanning the body's length, but the hull is below it.
        hull(self.pack / "clips/hull.bin", 1.0, .2, .4)
        hull(self.pack / "clips/yard.bin", 2.0, .1, .05, rows=2, z0=.6)
        binding = {"scale": 1.0, "offset_z": 0.0, "idle": {"part_count": 2,
                   "part0": {"mesh": "clips/hull.bin"}, "part1": {"mesh": "clips/yard.bin"}}}
        side, low, high, centre, half_width = coverage.hull_band(self.pack, binding, {0, 1}, 1.0, 0.0)
        self.assertAlmostEqual(low, 0.0, delta=.01)
        self.assertLessEqual(high, .62)

    def test_symmetry_axis_follows_the_fuselage(self):
        # A plane seen from above: fuselage along x, wings and tailplane along y.
        fuselage = [(x, y) for x in np.linspace(-1, 1, 41) for y in (-.08, .08)]
        wings = [(x, y) for x in (-.1, .1) for y in np.linspace(-1.2, 1.2, 49)]
        tail = [(x, y) for x in (-.95, -.85) for y in np.linspace(-.4, .4, 17)]
        axis = coverage.symmetry_axis(np.array(fuselage + wings + tail))
        self.assertGreater(abs(axis[0]), .99)

    def test_idle_motion_marks_a_rotor_not_a_banking_body(self):
        body = [(x, y, z) for x in (-1, -.3, .3, 1) for y in (-.2, .2) for z in (0, .2)]
        rotor = [(x, y, .5) for x in (-.8, .8) for y in (-.05, .05)]
        triangles = [0, 1, 3, 0, 3, 2, 16, 17, 19, 16, 19, 18]
        joints = [0] * 16 + [1] * 4
        spinning = [(np.eye(4), np.eye(4))] + [(np.eye(4), turn(30 * k)) for k in range(1, 4)]
        mesh(self.pack / "clips/heli.bin", body + rotor, triangles, joints, spinning)
        binding = {"idle": {"part_count": 1, "part0": {"mesh": "clips/heli.bin"}}}
        moving = coverage.idle_motion(self.pack, binding, [0])[0]
        self.assertEqual(list(moving), [False] * 16 + [True] * 4)
        banking = [(turn(10 * k), turn(10 * k)) for k in range(4)]
        mesh(self.pack / "clips/heli.bin", body + rotor, triangles, joints, banking)
        self.assertFalse(coverage.idle_motion(self.pack, binding, [0])[0].any())

    def test_hull_marks_stripe_the_hull_and_cap_the_mast_but_clear_the_sail(self):
        self.white("hull", "mast", "sail")
        hull(self.pack / "clips/hull.bin", 2.0, .4, .6)
        hull(self.pack / "clips/mast.bin", .06, 1.6, .06, rows=16, z0=.4)
        quad(self.pack / "clips/sail.bin", -.6, .6, 1.0, normal=(1, 0, 0))
        def part(name):
            return {"mesh": f"clips/{name}.bin", "texture": f"textures/{name}.dds", "owner_mask": 1, "owner_strength": .82}
        action = {"part_count": 3, "part0": part("hull"), "part1": part("mast"), "part2": part("sail")}
        bindings = {"unit0": {"key_count": 1, "key0": "PRTO_Ship", "scale": 1.0, "offset_z": 0.0, "idle": action}, "unit_count": 1}
        components = {"unit0": {"idle": ["c/ship"] * 3}}
        targets = {"stripe": .12, "accents": .02, "accent_floor": .1, "tips": .05, "tail": .02}
        report = coverage.paint_marks(bindings, self.pack, components, {"PRTO_Ship": "hull"}, targets, .9)
        self.assertEqual(report["PRTO_Ship"]["style"], "hull")
        hull_part, mast_part, sail_part = (action[f"part{i}"] for i in range(3))
        rows = 1 - coverage.dds_alpha(self.pack / hull_part["texture"]).mean(1)
        self.assertGreater(rows[:16].max(), .5)     # a band on the upper hull (V=0 is the deck edge)
        self.assertLess(rows[-6:].max(), .05)       # clear of the waterline
        self.assertTrue(2 < (rows > .5).sum() < 24)  # not the whole side
        mast = 1 - coverage.dds_alpha(self.pack / mast_part["texture"]).mean(1)
        self.assertGreater(mast[:3].mean(), .3)     # the mast's top (texture V=0)
        self.assertLess(mast[6:28].max(), .02)      # nothing between the hull band and the masthead
        # The sail sits below the accent floor: no mark, and no broad tint left.
        self.assertEqual((sail_part["texture"], sail_part["owner_mask"]), ("textures/sail.dds", 0))

    def test_extremity_marks_wing_tips_not_the_wing_root(self):
        self.white("wing", "body")
        # Wing along y (texture U runs tip to tip), fuselage along x.
        span = np.linspace(-1.2, 1.2, 25)
        wing = [(x, y, .1) for y in span for x in (-.15, .15)]
        uv = [(float((y + 1.2) / 2.4), v) for y in span for v in (0.0, 1.0)]
        triangles = [i for k in range(24) for i in (2 * k, 2 * k + 1, 2 * k + 3, 2 * k, 2 * k + 3, 2 * k + 2)]
        mesh(self.pack / "clips/wing.bin", wing, triangles, uv=uv)
        hull(self.pack / "clips/body.bin", 2.0, .2, .2)
        def part(name):
            return {"mesh": f"clips/{name}.bin", "texture": f"textures/{name}.dds", "owner_mask": 1, "owner_strength": .82}
        action = {"part_count": 2, "part0": part("wing"), "part1": part("body")}
        bindings = {"unit0": {"key_count": 1, "key0": "PRTO_Plane", "scale": 1.0, "offset_z": 0.0, "idle": action}, "unit_count": 1}
        components = {"unit0": {"idle": ["c/plane"] * 2}}
        targets = {"stripe": .12, "accents": .02, "accent_floor": .1, "tips": .05, "tail": .02}
        coverage.paint_marks(bindings, self.pack, components, {"PRTO_Plane": "extremities"}, targets, .9)
        columns = 1 - coverage.dds_alpha(self.pack / action["part0"]["texture"]).mean(0)
        self.assertGreater(min(columns[0], columns[-1]), .5)   # both wing tips
        self.assertLess(columns[12:20].mean(), .02)            # not the wing root


if __name__ == "__main__":
    unittest.main()
