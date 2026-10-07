"""Connection-mask route sheets compile to exact, generic centerline joins."""
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Renderer.tools.asset_compiler import build_route_pattern_runtime as patterns


def encode_pcx(image):
    height, width = image.shape
    header = bytearray(128)
    header[0:4] = bytes((10, 5, 1, 8))
    struct.pack_into("<4H", header, 4, 0, 0, width - 1, height - 1)
    header[65] = 1
    struct.pack_into("<H", header, 66, width)
    body = bytearray()
    for row in image:
        x = 0
        while x < width:
            value, run = int(row[x]), 1
            while x + run < width and run < 63 and row[x + run] == value:
                run += 1
            if run > 1 or value >= 0xC0:
                body += bytes((0xC0 | run, value))
            else:
                body.append(value)
            x += run
    palette = bytes(range(256)) * 3
    return bytes(header) + bytes(body) + b"\x0c" + palette


def synthetic_sheet(rows=16, ladder=False):
    """Each cell paints a 3-pixel path from the center to every connected join.

    A ladder paints two rails with sleepers, leaving key-colored gaps between
    them, as a railroad sheet does. Rows past 16 repeat the full mask.
    """
    image = np.full((64 * rows, 2048), 255, np.uint8)
    yy, xx = np.mgrid[0:64, 0:128]
    image[:] = 254  # second key color outside every diamond
    diamond = np.abs(xx + 0.5 - 64) / 64 + np.abs(yy + 0.5 - 32) / 32 <= 1
    for index in range(16 * rows):
        mask = min(index, 255)
        row, column = divmod(index, 16)
        cell = image[row * 64:(row + 1) * 64, column * 128:(column + 1) * 128]
        cell[diamond] = 255
        segments = [((56, 32), (72, 32)), ((64, 26), (64, 38))] if mask == 0 else [
            ((64, 32), patterns.CONNECT[d]) for d in range(8) if mask >> d & 1]
        for start, end in segments:
            length = math.dist(start, end)
            nx, ny = (start[1] - end[1]) / length, (end[0] - start[0]) / length
            for t in np.linspace(0, 1, 160):
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                if ladder:
                    rails = [(x + nx * side, y + ny * side) for side in (-1.6, 1.6)]
                    if int(t * length) % 3 == 0:  # a sleeper across both rails
                        rails += [(x + nx * side, y + ny * side) for side in (-0.8, 0, 0.8)]
                    for px, py in rails:
                        cell[(xx + 0.5 - px) ** 2 + (yy + 0.5 - py) ** 2 <= 0.36] = 40
                else:
                    near = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= 2.25
                    cell[near] = 40
    return image


class RoutePatternRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = encode_pcx(synthetic_sheet())
        cls.compiled = patterns.compile_sheet(cls.data)

    def test_every_connection_ends_exactly_on_its_shared_join(self):
        for mask, lines in enumerate(self.compiled):
            joined = {}
            for start, end, points in lines:
                self.assertGreaterEqual(len(points), 2)
                for side, direction in ((0, start), (-1, end)):
                    if direction >= 0:
                        self.assertTrue(mask >> direction & 1, (mask, direction))
                        self.assertEqual(tuple(points[side]), patterns.CONNECT[direction])
                        joined[direction] = joined.get(direction, 0) + 1
            expected = {d: 1 for d in range(8) if mask >> d & 1}
            self.assertEqual(joined, expected, mask)
            for _, _, points in lines:
                self.assertTrue(np.all(points[:, 0] >= -0.5) and np.all(points[:, 0] <= 128.5))
                self.assertTrue(np.all(points[:, 1] >= -0.5) and np.all(points[:, 1] <= 64.5))

    def test_centerline_follows_the_painted_path(self):
        # A straight painted road stays on its axis after thinning and smoothing.
        for start, end, points in self.compiled[2]:  # east corner only
            self.assertEqual(sorted((start, end)), [-1, 1])
            distances = [abs(y - 32) for x, y in points]
            self.assertLess(max(distances), 1.0)
        self.assertGreater(len(self.compiled[0]), 0)  # isolated mark survives

    def test_tile_coordinates_match_the_renderer_diamond(self):
        corners = np.array([[64, 0], [128, 32], [64, 64], [0, 32], [64, 32]], float)
        uv = patterns.to_tile(corners)
        np.testing.assert_allclose(uv, [[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]], atol=1e-9)

    def test_runtime_payload_round_trips(self):
        payload = patterns.encode(self.compiled)
        self.assertEqual(payload[:8], patterns.MAGIC)
        masks, line_count, point_count = struct.unpack_from("<III", payload, 8)
        self.assertEqual(masks, 256)
        offsets = struct.unpack_from("<257I", payload, 20)
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], line_count)
        self.assertEqual(len(payload), 20 + 257 * 4 + line_count * 8 + point_count * 8)
        first, count, start, end = struct.unpack_from("<IHbb", payload, 20 + 257 * 4 + offsets[1] * 8)
        self.assertEqual(sorted((start, end)), [-1, 0])  # mask 1: one NE join
        join = first if start == 0 else first + count - 1
        u, v = struct.unpack_from("<2f", payload, 20 + 257 * 4 + line_count * 8 + join * 8)
        self.assertTrue(math.isclose(u, .5, abs_tol=1e-6) and math.isclose(v, 0, abs_tol=1e-6))

    def test_build_writes_pack_and_reports_its_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "roads.pcx").write_bytes(self.data)
            (source / "railroads.pcx").write_bytes(encode_pcx(synthetic_sheet(17, ladder=True)))
            consumed = {}
            original_root = patterns.ROOT
            try:
                patterns.ROOT = root
                consumed = patterns.build(root / "out", source)
            finally:
                patterns.ROOT = original_root
            self.assertEqual(sorted(consumed), ["source/railroads.pcx", "source/roads.pcx"])
            self.assertTrue((root / "out/road_patterns.bin").read_bytes().startswith(patterns.MAGIC))
            rail = (root / "out/railroad_patterns.bin").read_bytes()
            self.assertEqual(struct.unpack_from("<I", rail, 8)[0], 272)

    def test_railroad_ladder_closes_into_one_centerline(self):
        # A railroad sprite paints two rails and sleepers with gaps between
        # them; thinned directly, it became a mesh of tiny paths.
        ladder = synthetic_sheet(17, ladder=True)
        cell = lambda index: ladder[(index // 16) * 64:(index // 16 + 1) * 64, (index % 16) * 128:(index % 16 + 1) * 128]
        # One path (plus at most the usual short connector to its exact join).
        self.assertLessEqual(len(patterns.extract(cell(1), 1, closing=1)), 2)
        self.assertGreater(len(patterns.extract(cell(1), 1, closing=0)), 10)
        compiled = patterns.compile_sheet(encode_pcx(ladder), closing=1)
        self.assertEqual(len(compiled), 272)
        # Cells past 255 are further variants of the fully connected mask.
        for lines in compiled[255:]:
            joined = {direction for start, end, _ in lines for direction in (start, end) if direction >= 0}
            self.assertEqual(joined, set(range(8)))

    def test_straight_paths_stay_within_the_geometry_budget(self):
        # A dense late-game save sits at its tile-cache cap; the first, finely
        # sampled pack overflowed it and left the map black. A straight painted
        # connection must collapse to bounded segments.
        for start, end, points in self.compiled[2]:  # center to the east corner, about 64 px
            self.assertLessEqual(len(points), int(64 / patterns.MAX_SEGMENT) + 3)

    def test_installed_sheets_leave_no_junction_stubs(self):
        # Thinning a closed railroad band left clusters of junction pixels;
        # each tiny connector between them drew as a short crosswise rail.
        missing = [name for name, *_ in patterns.SHEETS.values() if not (patterns.SOURCE / name).is_file()]
        if missing:
            self.skipTest("Local route sheets are not imported: " + ", ".join(missing))
        for kind, (name, closing, spur) in patterns.SHEETS.items():
            compiled = patterns.compile_sheet((patterns.SOURCE / name).read_bytes(), closing, spur)
            for index, lines in enumerate(compiled):
                for start, end, points in lines:
                    length = float(np.hypot(*np.diff(points, axis=0).T).sum())
                    if start < 0 and end < 0 and len(lines) > 1:
                        self.assertGreater(length, patterns.CLUSTER, (kind, index))  # separate junctions only

    def test_unpainted_connection_is_rejected(self):
        image = synthetic_sheet()
        image[0:64, 128:256][image[0:64, 128:256] == 40] = 255  # mask 1 loses its path
        with self.assertRaises(ValueError):
            patterns.compile_sheet(encode_pcx(image))


if __name__ == "__main__":
    unittest.main()
